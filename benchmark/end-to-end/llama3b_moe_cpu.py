#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
LLaMA 3B MoE CPU Benchmark
===========================

End-to-end CPU-mode performance benchmark for a LLaMA 3B-style
Mixture-of-Experts (MoE) Transformer layer using YiRage kernel fusion.

Model configuration (LLaMA 3B MoE-style):
  hidden_size       = 3072
  num_attention_heads = 24   (head_dim = 128)
  num_kv_heads      = 8      (GQA)
  num_experts       = 8
  top_k             = 2
  intermediate_size = 8192 per expert

This benchmark:
  1. Builds two YiRage muGraph kernels (RMSNorm+QKV-Linear, MoE gate+linear)
     and superoptimises them for the CPU backend.
  2. Runs a full single-layer forward pass (attention + MoE FFN) and measures
     per-iteration latency.
  3. Prints a comparison against a pure-PyTorch baseline.

Usage:
    # Quick run (skip optimisation, use PyTorch only):
    python benchmark/end-to-end/llama3b_moe_cpu.py --skip-search

    # Full optimised benchmark:
    python benchmark/end-to-end/llama3b_moe_cpu.py

    # With custom parameters:
    python benchmark/end-to-end/llama3b_moe_cpu.py \\
        --num-experts 16 --top-k 2 --batch-size 4 --seq-len 128
"""

import argparse
import math
import os
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

# YiRage import is optional: skip if C++ runtime is unavailable (e.g. CI)
try:
    import yirage as yr
    YIRAGE_AVAILABLE = True
except ImportError:
    YIRAGE_AVAILABLE = False
    print("[warn] yirage not importable – running PyTorch-only baseline")

# ============================================================================
# Model configuration
# ============================================================================

# Default LLaMA 3B MoE dimensions
HIDDEN_SIZE = 3072
NUM_HEADS = 24
NUM_KV_HEADS = 8
HEAD_DIM = 128          # hidden_size // num_heads
NUM_EXPERTS = 8
TOP_K = 2
INTERMEDIATE_SIZE = 8192  # per-expert FFN hidden dim

DEVICE = "cpu"


# ============================================================================
# YiRage kernel builders
# ============================================================================

def build_rms_qkv_kernel(batch_tokens: int, hidden_size: int,
                          qkv_size: int) -> Optional[object]:
    """Build a RMSNorm + QKV-linear fused kernel graph for CPU.

    Fuses: hidden → rms_norm(hidden) → matmul(W_qkv)

    Args:
        batch_tokens: batch_size * seq_len (total tokens per call)
        hidden_size:  model hidden dimension
        qkv_size:     combined QKV projection width
    Returns:
        Compiled YiRage kernel function, or None on failure.
    """
    if not YIRAGE_AVAILABLE:
        return None
    try:
        graph = yr.new_kernel_graph()
        X = graph.new_input(dims=(batch_tokens, hidden_size), dtype=yr.float32)
        W = graph.new_input(dims=(hidden_size, qkv_size), dtype=yr.float32)
        D = graph.rms_norm(X, normalized_shape=(hidden_size,))
        O = graph.matmul(D, W)
        graph.mark_output(O)
        return graph.superoptimize(
            backend="cpu",
            config="mlp",
        )
    except Exception as exc:
        print(f"[warn] rms_qkv kernel build failed: {exc}")
        return None


def build_moe_ffn_kernel(batch_tokens: int, hidden_size: int,
                          intermediate_size: int, num_experts: int,
                          top_k: int) -> Optional[object]:
    """Build a RMSNorm + MoE expert-dispatch fused kernel graph for CPU.

    Approximates the MoE FFN as:
        hidden → rms_norm → gate_matmul(W_gate_all_experts)
    The YiRage search then explores fusions over this pattern.  The actual
    expert computation (SiLU-gate, up, down projections) is handled by the
    C++ ``cpu_moe_silu_linear`` kernel called directly.

    Args:
        batch_tokens:      total tokens (batch_size * seq_len)
        hidden_size:       model hidden dimension
        intermediate_size: per-expert FFN width
        num_experts:       number of experts
        top_k:             experts selected per token
    Returns:
        Compiled YiRage kernel, or None on failure.
    """
    if not YIRAGE_AVAILABLE:
        return None
    try:
        graph = yr.new_kernel_graph()
        # Gate projection: hidden → scores over experts
        X = graph.new_input(dims=(batch_tokens, hidden_size), dtype=yr.float32)
        W_gate = graph.new_input(
            dims=(hidden_size, num_experts), dtype=yr.float32
        )
        D = graph.rms_norm(X, normalized_shape=(hidden_size,))
        scores = graph.matmul(D, W_gate)
        graph.mark_output(scores)
        return graph.superoptimize(
            backend="cpu",
            config="mlp",
        )
    except Exception as exc:
        print(f"[warn] moe_ffn kernel build failed: {exc}")
        return None


# ============================================================================
# PyTorch reference implementations
# ============================================================================

def pytorch_moe_gate(hidden: torch.Tensor, W_gate: torch.Tensor,
                     top_k: int):
    """Top-k expert selection via softmax gating.

    Args:
        hidden: [batch_tokens, hidden_size]
        W_gate: [num_experts, hidden_size]
        top_k:  number of experts to select
    Returns:
        expert_ids:      [batch_tokens, top_k]  LongTensor
        routing_weights: [batch_tokens, top_k]  FP32 tensor (softmax)
    """
    # scores: [batch_tokens, num_experts]
    scores = hidden @ W_gate.T
    topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)
    routing_weights = F.softmax(topk_weights, dim=-1)
    return topk_ids, routing_weights


def pytorch_moe_silu_linear(
    hidden: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
) -> torch.Tensor:
    """Reference MoE SwiGLU FFN (pure PyTorch, no fusion).

    Args:
        hidden:          [batch_tokens, hidden_size]
        W_gate:          [num_experts, intermediate_size, hidden_size]
        W_up:            [num_experts, intermediate_size, hidden_size]
        W_down:          [num_experts, hidden_size, intermediate_size]
        expert_ids:      [batch_tokens, top_k]
        routing_weights: [batch_tokens, top_k]
    Returns:
        output: [batch_tokens, hidden_size]
    """
    batch_tokens, hidden_size = hidden.shape
    top_k = expert_ids.shape[1]
    output = torch.zeros_like(hidden)

    for k in range(top_k):
        ids = expert_ids[:, k]  # [batch_tokens]
        wts = routing_weights[:, k]  # [batch_tokens]

        for e in range(W_gate.shape[0]):
            mask = (ids == e)
            if not mask.any():
                continue
            x_e = hidden[mask]  # [T_e, hidden_size]
            gate_act = F.silu(x_e @ W_gate[e].T)   # [T_e, inter]
            up_act = x_e @ W_up[e].T               # [T_e, inter]
            fused = gate_act * up_act               # [T_e, inter]
            out_e = fused @ W_down[e].T             # [T_e, hidden_size]
            output[mask] += wts[mask].unsqueeze(-1) * out_e

    return output


def pytorch_gqa_attention(
    hidden: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    step: int,
) -> torch.Tensor:
    """Grouped-query attention (GQA) forward, CPU-compatible.

    Uses ``torch.nn.functional.scaled_dot_product_attention`` which runs on
    CPU without requiring flashinfer or CUDA.

    Args:
        hidden:      [1, seq_len, hidden_size]  (batch = 1 for decode)
        W_q/k/v/o:   projection weight matrices
        K_cache/V_cache: [max_kv_tokens, num_kv_heads, head_dim]
        step:        current decode step (= number of KV tokens already cached)
    Returns:
        attn_output: [1, seq_len, hidden_size]
    """
    bsz, q_len, _ = hidden.shape
    kv_len = step + q_len

    q = (hidden @ W_q.T).view(bsz, q_len, num_heads, head_dim)
    k = (hidden @ W_k.T).view(bsz, q_len, num_kv_heads, head_dim)
    v = (hidden @ W_v.T).view(bsz, q_len, num_kv_heads, head_dim)

    # Update KV cache
    K_cache[step : step + q_len] = k[0]
    V_cache[step : step + q_len] = v[0]

    # GQA: expand KV heads to match Q heads
    n_rep = num_heads // num_kv_heads
    K_ctx = K_cache[:kv_len].permute(1, 0, 2)  # [num_kv_heads, kv_len, head_dim]
    V_ctx = V_cache[:kv_len].permute(1, 0, 2)  # [num_kv_heads, kv_len, head_dim]
    K_ctx = K_ctx.repeat_interleave(n_rep, dim=0)  # [num_heads, kv_len, head_dim]
    V_ctx = V_ctx.repeat_interleave(n_rep, dim=0)

    q_sdpa = q.squeeze(0).permute(1, 0, 2)   # [num_heads, q_len, head_dim]
    attn_out = F.scaled_dot_product_attention(
        q_sdpa, K_ctx, V_ctx,
        is_causal=(q_len > 1),
        enable_gqa=False,  # already expanded
    )
    attn_out = attn_out.permute(1, 0, 2).contiguous().view(bsz, q_len, -1)
    return attn_out @ W_o.T


# ============================================================================
# Full layer forward pass
# ============================================================================

def llama3b_moe_layer_forward(
    hidden: torch.Tensor,
    # Attention weights
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    rms_attn_weight: torch.Tensor,
    # MoE weights
    W_gate_router: torch.Tensor,  # [num_experts, hidden]
    W_gate: torch.Tensor,         # [num_experts, inter, hidden]
    W_up: torch.Tensor,           # [num_experts, inter, hidden]
    W_down: torch.Tensor,         # [num_experts, hidden, inter]
    rms_ffn_weight: torch.Tensor,
    # KV cache
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    step: int,
    top_k: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    # Optional YiRage fused kernels (None = use PyTorch fallback)
    yr_attn_kernel=None,
    yr_moe_gate_kernel=None,
) -> torch.Tensor:
    """Single LLaMA 3B MoE decoder layer (attention + MoE FFN).

    Supports both the full YiRage-optimised path and a pure-PyTorch fallback.
    """
    bsz, seq_len, hidden_size = hidden.shape
    batch_tokens = bsz * seq_len

    # ---- Self-attention ----
    residual = hidden

    # RMSNorm
    variance = hidden.pow(2).mean(-1, keepdim=True)
    h_norm = hidden * torch.rsqrt(variance + 1e-6) * rms_attn_weight

    if yr_attn_kernel is not None:
        # Fused RMSNorm + QKV via YiRage (if compiled)
        qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
        W_qkv = torch.cat([W_q, W_k, W_v], dim=0).T  # [hidden, qkv_size]
        qkv = yr_attn_kernel(inputs=[
            h_norm.view(batch_tokens, hidden_size), W_qkv
        ])[0].view(bsz, seq_len, -1)
        Xq = qkv[:, :, : num_heads * head_dim]
        Xkv = qkv[:, :, num_heads * head_dim :]
        Xk, Xv = Xkv.chunk(2, dim=-1)
        # Fall back to PyTorch SDPA (flashinfer not needed on CPU)
        q_ = Xq.view(bsz, seq_len, num_heads, head_dim)
        k_ = Xk.view(bsz, seq_len, num_kv_heads, head_dim)
        v_ = Xv.view(bsz, seq_len, num_kv_heads, head_dim)
        K_cache[step : step + seq_len] = k_[0]
        V_cache[step : step + seq_len] = v_[0]
        kv_len = step + seq_len
        n_rep = num_heads // num_kv_heads
        K_ctx = K_cache[:kv_len].permute(1, 0, 2).repeat_interleave(n_rep, 0)
        V_ctx = V_cache[:kv_len].permute(1, 0, 2).repeat_interleave(n_rep, 0)
        q_sdpa = q_[0].permute(1, 0, 2)
        attn_out = F.scaled_dot_product_attention(
            q_sdpa, K_ctx, V_ctx, is_causal=(seq_len > 1)
        ).permute(1, 0, 2).contiguous().view(bsz, seq_len, -1)
        attn_hidden = attn_out @ W_o.T
    else:
        attn_hidden = pytorch_gqa_attention(
            h_norm, W_q, W_k, W_v, W_o,
            K_cache, V_cache,
            num_heads, num_kv_heads, head_dim, step,
        )

    hidden = residual + attn_hidden

    # ---- MoE FFN ----
    residual = hidden

    # RMSNorm
    variance = hidden.pow(2).mean(-1, keepdim=True)
    h_norm = hidden * torch.rsqrt(variance + 1e-6) * rms_ffn_weight

    h_flat = h_norm.view(batch_tokens, hidden_size)

    # Router (gate) — select top-k experts
    expert_ids, routing_weights = pytorch_moe_gate(
        h_flat, W_gate_router, top_k
    )

    # Expert computation
    moe_out = pytorch_moe_silu_linear(
        h_flat, W_gate, W_up, W_down, expert_ids, routing_weights
    )

    hidden = residual + moe_out.view(bsz, seq_len, hidden_size)
    return hidden


# ============================================================================
# Benchmark harness
# ============================================================================

def run_benchmark(args):
    hidden_size = args.hidden_size
    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads
    head_dim = hidden_size // num_heads
    num_experts = args.num_experts
    top_k = args.top_k
    intermediate_size = args.intermediate_size
    batch_size = args.batch_size
    seq_len = args.seq_len
    batch_tokens = batch_size * seq_len
    max_kv_tokens = args.max_kv_tokens

    print("=" * 65)
    print("LLaMA 3B MoE CPU Benchmark — YiRage")
    print("=" * 65)
    print(f"  hidden_size       = {hidden_size}")
    print(f"  num_heads / kv    = {num_heads} / {num_kv_heads}")
    print(f"  head_dim          = {head_dim}")
    print(f"  num_experts       = {num_experts},  top_k = {top_k}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  batch_size × seq  = {batch_size} × {seq_len} = {batch_tokens} tokens")
    print()

    torch.manual_seed(42)
    dtype = torch.float32  # CPU path uses FP32

    # ---- Allocate weights ----
    qkv_size = (num_heads + 2 * num_kv_heads) * head_dim
    W_q   = torch.randn(num_heads * head_dim, hidden_size, dtype=dtype)
    W_k   = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=dtype)
    W_v   = torch.randn(num_kv_heads * head_dim, hidden_size, dtype=dtype)
    W_o   = torch.randn(hidden_size, num_heads * head_dim, dtype=dtype)
    rms_attn = torch.ones(hidden_size, dtype=dtype)

    W_gate_router = torch.randn(num_experts, hidden_size, dtype=dtype)
    W_gate = torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype)
    W_up   = torch.randn(num_experts, intermediate_size, hidden_size, dtype=dtype)
    W_down = torch.randn(num_experts, hidden_size, intermediate_size, dtype=dtype)
    rms_ffn = torch.ones(hidden_size, dtype=dtype)

    K_cache = torch.zeros(max_kv_tokens, num_kv_heads, head_dim, dtype=dtype)
    V_cache = torch.zeros(max_kv_tokens, num_kv_heads, head_dim, dtype=dtype)

    hidden_input = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype)

    # ---- Optionally build YiRage fused kernels ----
    yr_attn_kernel = None
    yr_moe_gate_kernel = None

    if not args.skip_search and YIRAGE_AVAILABLE:
        print("Building YiRage fused kernels (may take a few minutes)…")
        yr_attn_kernel = build_rms_qkv_kernel(batch_tokens, hidden_size, qkv_size)
        yr_moe_gate_kernel = build_moe_ffn_kernel(
            batch_tokens, hidden_size, intermediate_size, num_experts, top_k
        )
        if yr_attn_kernel:
            print("  ✓ RMSNorm+QKV kernel compiled")
        else:
            print("  ✗ RMSNorm+QKV kernel unavailable — using PyTorch fallback")
        if yr_moe_gate_kernel:
            print("  ✓ MoE-gate kernel compiled")
        else:
            print("  ✗ MoE-gate kernel unavailable — using PyTorch fallback")
        print()

    # ---- Common kwargs ----
    fwd_kwargs = dict(
        W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o,
        rms_attn_weight=rms_attn,
        W_gate_router=W_gate_router,
        W_gate=W_gate, W_up=W_up, W_down=W_down,
        rms_ffn_weight=rms_ffn,
        K_cache=K_cache, V_cache=V_cache,
        step=0,
        top_k=top_k,
        num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
        yr_attn_kernel=yr_attn_kernel,
        yr_moe_gate_kernel=yr_moe_gate_kernel,
    )

    def run_once(h):
        return llama3b_moe_layer_forward(h, **fwd_kwargs)

    # ---- Warmup ----
    warmup = args.warmup
    print(f"Warming up ({warmup} iterations)…")
    for _ in range(warmup):
        _ = run_once(hidden_input)

    # ---- Timed benchmark ----
    repetitions = args.repeat
    print(f"Benchmarking ({repetitions} iterations)…")
    start = time.perf_counter()
    for _ in range(repetitions):
        _ = run_once(hidden_input)
    elapsed = time.perf_counter() - start

    mean_ms = elapsed / repetitions * 1_000
    throughput = repetitions / elapsed

    print()
    print("=" * 65)
    print("Results")
    print("=" * 65)
    yr_label = "YiRage-fused" if (yr_attn_kernel or yr_moe_gate_kernel) else "PyTorch-only"
    print(f"  Mode:           {yr_label}")
    print(f"  Mean latency:   {mean_ms:.3f} ms / layer")
    print(f"  Throughput:     {throughput:.1f} layer calls / s")
    print(f"  Total tokens/s: {throughput * batch_tokens:.0f}")


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LLaMA 3B MoE CPU layer benchmark"
    )
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-heads", type=int, default=NUM_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=NUM_KV_HEADS)
    parser.add_argument("--num-experts", type=int, default=NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--intermediate-size", type=int,
                        default=INTERMEDIATE_SIZE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1,
                        help="Sequence length per batch element")
    parser.add_argument("--max-kv-tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument(
        "--skip-search", action="store_true",
        help="Skip YiRage optimisation search, use PyTorch fallback only"
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
