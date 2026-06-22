#!/usr/bin/env python3
"""Qwen2.5-3B LLM Inference Demo — YiRage MPS-optimized vs PyTorch baseline.

Demonstrates YiRage kernel optimization on Apple Silicon for a real LLM
workload.  Benchmarks RMS norm, attention projections, MLP projections,
and full-layer latency.

Architecture: Qwen2.5-3B (2048 hidden, 16 heads, 36 layers, GQA-less)
Usage::

    python demo/mps/llm_inference.py
    python demo/mps/llm_inference.py --device mps --superoptimize
    python demo/mps/llm_inference.py --device cpu   # smoke on Linux/CI
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import yirage as yr

# ---------------------------------------------------------------------------
# Model config (Qwen2.5-3B)
# ---------------------------------------------------------------------------
class Qwen3BConfig:
    hidden_size = 2048
    intermediate_size = 11008
    num_attention_heads = 16
    num_key_value_heads = 16
    num_hidden_layers = 36
    head_dim = hidden_size // num_attention_heads  # 128
    vocab_size = 151936
    max_seq_len = 2048
    rope_theta = 1000000.0


CFG = Qwen3BConfig()

# Set by configure_runtime() before benchmarks run.
DEVICE = "cpu"
DTYPE = torch.float16
YR_DTYPE = yr.float16
USE_SUPEROPTIMIZE = False
_ROPE_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def resolve_device(requested: str) -> str:
    """Resolve execution device: auto | mps | cpu."""
    req = (requested or "auto").lower()
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if req == "auto":
        return "mps" if mps_ok else "cpu"
    if req == "mps":
        if not mps_ok:
            raise RuntimeError(
                "MPS requested but not available. Use --device cpu or run on Apple Silicon."
            )
        return "mps"
    if req == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device {requested!r}; use auto, mps, or cpu")


def configure_runtime(device: str, superoptimize: bool = False) -> None:
    """Initialize globals used by kernels and benchmarks."""
    global DEVICE, DTYPE, YR_DTYPE, USE_SUPEROPTIMIZE, _ROPE_CACHE
    DEVICE = resolve_device(device)
    DTYPE = torch.float16
    YR_DTYPE = yr.float16
    USE_SUPEROPTIMIZE = superoptimize
    _ROPE_CACHE = {}


def sync():
    if DEVICE == "mps":
        torch.mps.synchronize()
    elif DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def bench_ms(fn, warmup=10, reps=100):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t0) / reps * 1000


def gflops(ms, ops):
    return ops / (ms * 1e6) if ms > 0 else 0


def yirage_linear(graph, matmul_out: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Apply optional linear bias after a YiRage matmul result."""
    if bias is None:
        return matmul_out
    return matmul_out + bias


def maybe_superoptimize(graph):
    """Optionally run superoptimize (slow; opt-in via --superoptimize)."""
    if not USE_SUPEROPTIMIZE:
        return graph
    backend = DEVICE if DEVICE in ("mps", "cuda", "cpu") else "cpu"
    try:
        optimized = graph.superoptimize(
            backend=backend,
            use_ray=False,
            warmup_iters=2,
            profile_iters=10,
            use_persistent_cache=True,
        )
        return optimized if optimized is not None else graph
    except Exception as exc:
        print(f"  ⚠️  superoptimize failed ({exc}); using unoptimized graph")
        return graph


# ---------------------------------------------------------------------------
# RoPE (lazy, device-aware)
# ---------------------------------------------------------------------------
def precompute_rope(max_seq_len, dim, theta=1000000.0):
    """Precompute rotary position embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=DTYPE, device=DEVICE)
    sin = emb.sin().to(dtype=DTYPE, device=DEVICE)
    return cos, sin


def get_rope_tables():
    key = (CFG.max_seq_len, CFG.head_dim, CFG.rope_theta, DEVICE)
    if key not in _ROPE_CACHE:
        _ROPE_CACHE[key] = precompute_rope(CFG.max_seq_len, CFG.head_dim, CFG.rope_theta)
    return _ROPE_CACHE[key]


def apply_rope(x, position_ids):
    """Apply rotary position embeddings to query/key tensors."""
    rope_cos, rope_sin = get_rope_tables()
    cos = rope_cos[position_ids].unsqueeze(1)
    sin = rope_sin[position_ids].unsqueeze(1)
    return x * cos + rotate_half(x) * sin


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------------------
# YiRage-optimized kernel builders
# ---------------------------------------------------------------------------
def yirage_rms_norm_kernel(x, weight=None):
    """Build and run a YiRage RMS norm kernel (norm only; affine in PyTorch)."""
    seq, hidden = x.shape
    g = yr.new_kernel_graph()
    X = g.new_input(dims=(seq, hidden), dtype=YR_DTYPE)
    Y = g.rms_norm(X, normalized_shape=(hidden,))
    g.mark_output(Y)
    return maybe_superoptimize(g), [x]


def yirage_matmul_kernel(a_shape, b_shape):
    """Build a YiRage matmul kernel."""
    M, K = a_shape
    K2, N = b_shape
    assert K == K2
    g = yr.new_kernel_graph()
    A = g.new_input(dims=(M, K), dtype=YR_DTYPE)
    B = g.new_input(dims=(K, N), dtype=YR_DTYPE)
    C = g.matmul(A, B)
    g.mark_output(C)
    return maybe_superoptimize(g)


# =============================================================================
# Qwen2.5-3B Transformer Layer (MPS-optimized)
# =============================================================================
class Qwen3BAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = CFG.hidden_size
        self.n_heads = CFG.num_attention_heads
        self.head_dim = CFG.head_dim
        self.q_proj = nn.Linear(self.hidden, self.n_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden, self.n_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden, self.n_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden, bias=False)
        self._yr_matmul: dict[tuple, object] = {}

    def _get_yr_matmul(self, M, K, N, tag: str):
        key = (tag, M, K, N)
        if key not in self._yr_matmul:
            self._yr_matmul[key] = yirage_matmul_kernel((M, K), (K, N))
        return self._yr_matmul[key]

    def _proj_yirage(self, x_2d, linear: nn.Linear, tag: str):
        seq, hidden = x_2d.shape
        out_dim = linear.weight.shape[0]
        graph = self._get_yr_matmul(seq, hidden, out_dim, tag)
        y = graph(inputs=[x_2d, linear.weight.T])[0]
        return yirage_linear(graph, y, linear.bias)

    def forward_yirage(self, hidden_states, position_ids, attention_mask=None):
        if hidden_states.shape[0] != 1:
            raise ValueError("forward_yirage currently supports batch size 1 only")
        seq = hidden_states.shape[1]
        x = hidden_states.squeeze(0)

        q = self._proj_yirage(x, self.q_proj, "q").view(seq, self.n_heads, self.head_dim)
        k = self._proj_yirage(x, self.k_proj, "k").view(seq, self.n_heads, self.head_dim)
        v = self._proj_yirage(x, self.v_proj, "v").view(seq, self.n_heads, self.head_dim)

        q = apply_rope(q, position_ids)
        k = apply_rope(k, position_ids)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
        )
        attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(seq, -1)

        out = self._proj_yirage(attn_out, self.o_proj, "o")
        return out.unsqueeze(0)

    def forward_pytorch(self, hidden_states, position_ids, attention_mask=None):
        x = hidden_states.squeeze(0)
        seq = x.shape[0]
        q = self.q_proj(x).view(seq, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(seq, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(seq, self.n_heads, self.head_dim)

        q = apply_rope(q, position_ids)
        k = apply_rope(k, position_ids)

        q, k, v = q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=(attention_mask is None),
        )
        attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(seq, -1)
        return self.o_proj(attn_out).unsqueeze(0)


class Qwen3BMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = CFG.hidden_size
        self.intermediate = CFG.intermediate_size
        self.gate_proj = nn.Linear(self.hidden, self.intermediate, bias=False)
        self.up_proj = nn.Linear(self.hidden, self.intermediate, bias=False)
        self.down_proj = nn.Linear(self.intermediate, self.hidden, bias=False)
        self._yr_matmul: dict[tuple, object] = {}

    def _get_yr_matmul(self, M, K, N, tag: str):
        key = (tag, M, K, N)
        if key not in self._yr_matmul:
            self._yr_matmul[key] = yirage_matmul_kernel((M, K), (K, N))
        return self._yr_matmul[key]

    def forward_yirage(self, x):
        if x.shape[0] != 1:
            raise ValueError("forward_yirage currently supports batch size 1 only")
        seq, hidden = x.squeeze(0).shape
        x2d = x.squeeze(0)

        gate_g = self._get_yr_matmul(seq, hidden, self.intermediate, "gate")
        up_g = self._get_yr_matmul(seq, hidden, self.intermediate, "up")
        gate = gate_g(inputs=[x2d, self.gate_proj.weight.T])[0]
        up = up_g(inputs=[x2d, self.up_proj.weight.T])[0]
        act = up * F.silu(gate)

        down_g = self._get_yr_matmul(seq, self.intermediate, hidden, "down")
        out = down_g(inputs=[act, self.down_proj.weight.T])[0]
        return out.unsqueeze(0)

    def forward_pytorch(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        act = up * F.silu(gate)
        return self.down_proj(act)


class Qwen3BDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(CFG.hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(CFG.hidden_size, eps=1e-6)
        self.self_attn = Qwen3BAttention()
        self.mlp = Qwen3BMLP()

    def forward_yirage(self, hidden_states, position_ids, attention_mask=None):
        residual = hidden_states
        yr_norm_g, yr_norm_inputs = yirage_rms_norm_kernel(hidden_states.squeeze(0))
        normed = yr_norm_g(inputs=yr_norm_inputs)[0].unsqueeze(0)
        normed = normed * self.input_layernorm.weight
        hidden_states = self.self_attn.forward_yirage(normed, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        yr_norm2_g, yr_norm2_inputs = yirage_rms_norm_kernel(hidden_states.squeeze(0))
        normed = yr_norm2_g(inputs=yr_norm2_inputs)[0].unsqueeze(0)
        normed = normed * self.post_attention_layernorm.weight
        hidden_states = self.mlp.forward_yirage(normed)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_pytorch(self, hidden_states, position_ids, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_pytorch(hidden_states, position_ids, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp.forward_pytorch(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# =============================================================================
# Benchmarks
# =============================================================================
def bench_rms_norm(batch, seq):
    """Benchmark RMS norm kernel."""
    print(f"\n  RMS Norm ({seq}) — FP16:")
    x = torch.randn(batch, seq, CFG.hidden_size, dtype=DTYPE, device=DEVICE)
    w = torch.randn(CFG.hidden_size, dtype=DTYPE, device=DEVICE)
    norm = nn.RMSNorm(CFG.hidden_size, eps=1e-6, elementwise_affine=True).to(DEVICE, DTYPE)
    norm.weight.data = w

    pt_ms = bench_ms(lambda: norm(x))
    yr_g, yr_in = yirage_rms_norm_kernel(x.squeeze(0))
    yr_ms = bench_ms(lambda: yr_g(inputs=yr_in)[0] * w)
    sp = pt_ms / yr_ms if yr_ms > 0 else 0
    print(f"    PyTorch: {pt_ms:.4f} ms   YiRage: {yr_ms:.4f} ms   {sp:.2f}x")
    return pt_ms, yr_ms


def bench_attention_proj(batch, seq):
    """Benchmark QKV projection matmuls."""
    print(f"\n  Attention QKV Projections ({seq}×2048→2048) — FP16:")
    attn = Qwen3BAttention().to(DEVICE, DTYPE)
    x = torch.randn(batch, seq, CFG.hidden_size, dtype=DTYPE, device=DEVICE)
    pos = torch.arange(seq, device=DEVICE)

    pt_ms = bench_ms(lambda: attn.forward_pytorch(x, pos))
    yr_ms = bench_ms(lambda: attn.forward_yirage(x, pos))
    sp = pt_ms / yr_ms if yr_ms > 0 else 0

    ops = 4 * seq * CFG.hidden_size * CFG.hidden_size
    print(f"    PyTorch: {pt_ms:.4f} ms ({gflops(pt_ms, ops):.0f} GFLOPS)")
    print(f"    YiRage:  {yr_ms:.4f} ms ({gflops(yr_ms, ops):.0f} GFLOPS)   {sp:.2f}x")
    return pt_ms, yr_ms


def bench_mlp(batch, seq):
    """Benchmark MLP projections."""
    print(f"\n  MLP ({seq} → {CFG.intermediate_size}) — FP16:")
    mlp = Qwen3BMLP().to(DEVICE, DTYPE)
    x = torch.randn(batch, seq, CFG.hidden_size, dtype=DTYPE, device=DEVICE)

    pt_ms = bench_ms(lambda: mlp.forward_pytorch(x))
    yr_ms = bench_ms(lambda: mlp.forward_yirage(x))
    sp = pt_ms / yr_ms if yr_ms > 0 else 0

    ops = 2 * seq * CFG.hidden_size * CFG.intermediate_size * 2
    print(f"    PyTorch: {pt_ms:.4f} ms ({gflops(pt_ms, ops):.0f} GFLOPS)")
    print(f"    YiRage:  {yr_ms:.4f} ms ({gflops(yr_ms, ops):.0f} GFLOPS)   {sp:.2f}x")
    return pt_ms, yr_ms


def bench_full_layer(batch, seq):
    """Benchmark a full decoder layer."""
    print(f"\n  Full Decoder Layer ({seq}) — FP16:")
    layer = Qwen3BDecoderLayer().to(DEVICE, DTYPE)
    x = torch.randn(batch, seq, CFG.hidden_size, dtype=DTYPE, device=DEVICE)
    pos = torch.arange(seq, device=DEVICE)

    pt_ms = bench_ms(lambda: layer.forward_pytorch(x, pos), warmup=3, reps=30)
    yr_ms = bench_ms(lambda: layer.forward_yirage(x, pos), warmup=3, reps=30)
    sp = pt_ms / yr_ms if yr_ms > 0 else 0

    print(f"    PyTorch: {pt_ms:.4f} ms/layer")
    print(f"    YiRage:  {yr_ms:.4f} ms/layer   {sp:.2f}x")
    return pt_ms, yr_ms


def parse_args():
    p = argparse.ArgumentParser(description="YiRage Qwen2.5-3B MPS inference demo")
    p.add_argument(
        "--device",
        choices=("auto", "mps", "cpu"),
        default="auto",
        help="Execution device (default: auto)",
    )
    p.add_argument(
        "--superoptimize",
        action="store_true",
        help="Run YiRage superoptimize on kernel graphs (slow; real optimized kernels)",
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    args = parse_args()
    configure_runtime(args.device, superoptimize=args.superoptimize)

    device_label = DEVICE.upper()
    print("╔" + "═" * 62 + "╗")
    print("║" + f"  YiRage {device_label} — Qwen2.5-3B LLM Inference Demo  ".center(62) + "║")
    print("╚" + "═" * 62 + "╝")
    print(f"  PyTorch: {torch.__version__}   YiRage: {yr.__version__}")
    print(f"  Device:  {DEVICE}" + ("  (superoptimize on)" if USE_SUPEROPTIMIZE else ""))
    if DEVICE == "cpu":
        print("  Note:    YiRage path uses CPU interpreter unless --superoptimize is set.")
    try:
        brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        print(f"  Chip:    {brand}")
    except Exception:
        pass
    print(f"  Model:   Qwen2.5-3B (h={CFG.hidden_size}, heads={CFG.num_attention_heads}, layers={CFG.num_hidden_layers})")

    all_results = []

    print(f"\n{'='*62}")
    print("  1. Kernel-Level Benchmarks (1-token prefill)")
    print("=" * 62)
    all_results.append(("RMSNorm (1)", *bench_rms_norm(1, 1)))
    all_results.append(("AttentionProj (1)", *bench_attention_proj(1, 1)))
    all_results.append(("MLP (1)", *bench_mlp(1, 1)))
    all_results.append(("Layer (1)", *bench_full_layer(1, 1)))

    print(f"\n{'='*62}")
    print("  2. Prefill Benchmarks (128 tokens)")
    print("=" * 62)
    all_results.append(("RMSNorm (128)", *bench_rms_norm(1, 128)))
    all_results.append(("AttentionProj (128)", *bench_attention_proj(1, 128)))
    all_results.append(("MLP (128)", *bench_mlp(1, 128)))
    all_results.append(("Layer (128)", *bench_full_layer(1, 128)))

    print(f"\n{'='*62}")
    print("  3. Long Prefill Benchmarks (1024 tokens)")
    print("=" * 62)
    all_results.append(("RMSNorm (1024)", *bench_rms_norm(1, 1024)))
    all_results.append(("AttentionProj (1024)", *bench_attention_proj(1, 1024)))
    all_results.append(("MLP (1024)", *bench_mlp(1, 1024)))
    all_results.append(("Layer (1024)", *bench_full_layer(1, 1024)))

    print(f"\n{'='*62}")
    print(f"  Summary — Qwen2.5-3B @ {device_label}")
    print("=" * 62)
    print(f"  {'Kernel':<28s} {'PyTorch ms':>10s} {'YiRage ms':>10s} {'Speedup':>8s}")
    print(f"  {'-'*56}")

    geo = 1.0
    for name, pt, yr_ in all_results:
        sp = pt / yr_ if yr_ > 0 else 0
        print(f"  {name:<28s} {pt:10.4f} {yr_:10.4f} {sp:7.2f}x")
        geo *= sp
    geo = geo ** (1 / len(all_results))
    print(f"  {'-'*56}")
    print(f"  {'Geometric mean':<28s} {'':>10s} {'':>10s} {geo:7.2f}x")

    layer_1_pt, layer_1_yr = all_results[3][1], all_results[3][2]
    layer_128_pt, layer_128_yr = all_results[7][1], all_results[7][2]

    print(f"\n  Projected 36-layer throughput:")
    if layer_1_yr > 0:
        yr_tok_s = 1000 / (layer_1_yr * 36)
        pt_tok_s = 1000 / (layer_1_pt * 36) if layer_1_pt > 0 else 0
        print(f"    1-token decode:  PyTorch {pt_tok_s:.1f} tok/s   YiRage {yr_tok_s:.1f} tok/s")
    if layer_128_yr > 0:
        yr_ms_36 = layer_128_yr * 36
        pt_ms_36 = layer_128_pt * 36
        print(f"    128-token prefill: PyTorch {pt_ms_36:.1f} ms    YiRage {yr_ms_36:.1f} ms")

    print()
