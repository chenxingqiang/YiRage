# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
LLaMA 3B MoE model implementation for CPU inference with YiRage acceleration.

Architecture
------------
Each decoder layer has:
  - RMSNorm + GQA self-attention (with RoPE, KV cache)
  - RMSNorm + Sparse MoE FFN (top-k gating, SwiGLU expert MLPs)

This module is written in pure PyTorch and does not require the native YiRage
C++ runtime.  It is used by `demo.py` as the reference model and can be
optionally patched to call pre-compiled YiRage CPU kernels.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .configuration_llama3b_moe import LLaMA3BMoEConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Root-mean-square layer normalisation (no mean centering)."""
    rms = x.pow(2).mean(-1, keepdim=True).add(eps).sqrt()
    return weight * (x / rms)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings to query / key tensor [B, H, T, D]."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    rope_theta: float,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freq = 1.0 / (
        rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half)
    )
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin  # [T, D/2]


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class LLaMA3BMoEAttention(nn.Module):
    """GQA self-attention with RoPE and KV cache."""

    def __init__(self, config: LLaMA3BMoEConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        H = config.hidden_size
        Nq = self.num_heads * self.head_dim
        Nkv = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(H, Nq, bias=False)
        self.k_proj = nn.Linear(H, Nkv, bias=False)
        self.v_proj = nn.Linear(H, Nkv, bias=False)
        self.o_proj = nn.Linear(Nq, H, bias=False)

        cos, sin = _build_rope_cache(
            config.max_position_embeddings, self.head_dim, config.rope_theta
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,          # [B, T, H]
        k_cache: torch.Tensor,    # [max_seq, num_kv_heads, head_dim]
        v_cache: torch.Tensor,    # [max_seq, num_kv_heads, head_dim]
        step: int,
    ) -> torch.Tensor:
        B, T, H = x.shape
        Dh = self.head_dim

        q = self.q_proj(x).view(B, T, self.num_heads, Dh).transpose(1, 2)   # [B, Nh, T, Dh]
        k = self.k_proj(x).view(B, T, self.num_kv_heads, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, Dh).transpose(1, 2)

        cos = self.rope_cos[step : step + T]   # [T, Dh/2]
        sin = self.rope_sin[step : step + T]
        cos = cos.unsqueeze(0).unsqueeze(0)    # [1, 1, T, Dh/2]
        sin = sin.unsqueeze(0).unsqueeze(0)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Update KV cache (inference mode: one step at a time)
        k_cache[step : step + T] = k[0].transpose(0, 1)  # [T, num_kv_heads, Dh]
        v_cache[step : step + T] = v[0].transpose(0, 1)

        # GQA: repeat KV heads to match query heads
        groups = self.num_heads // self.num_kv_heads
        k_full = k_cache[: step + T].transpose(0, 1).unsqueeze(0)           # [1, Nkv, S, Dh]
        v_full = v_cache[: step + T].transpose(0, 1).unsqueeze(0)
        k_full = k_full.repeat_interleave(groups, dim=1)                     # [1, Nh, S, Dh]
        v_full = v_full.repeat_interleave(groups, dim=1)

        # Scaled dot-product attention (CPU-compatible)
        out = F.scaled_dot_product_attention(q, k_full, v_full)             # [B, Nh, T, Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# MoE FFN
# ---------------------------------------------------------------------------

class LLaMA3BMoEFFN(nn.Module):
    """
    Sparse MoE feed-forward layer with top-k routing and SwiGLU experts.

    Each expert computes:  SiLU(x @ W_gate.T) * (x @ W_up.T) @ W_down.T
    The outputs of the selected top-k experts are averaged using their
    routing weights (after softmax).
    """

    def __init__(self, config: LLaMA3BMoEConfig) -> None:
        super().__init__()
        H = config.hidden_size
        I = config.intermediate_size
        E = config.num_experts
        self.top_k = config.top_k_experts

        self.gate_proj   = nn.Linear(H, E, bias=False)   # router
        self.expert_gate = nn.Parameter(torch.empty(E, I, H))
        self.expert_up   = nn.Parameter(torch.empty(E, I, H))
        self.expert_down = nn.Parameter(torch.empty(E, H, I))

        nn.init.normal_(self.expert_gate, std=0.02)
        nn.init.normal_(self.expert_up,   std=0.02)
        nn.init.normal_(self.expert_down, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        B, T, H = x.shape
        x_flat = x.view(B * T, H)

        # Router
        scores  = self.gate_proj(x_flat)                     # [BT, E]
        topk_w, topk_ids = torch.topk(scores, self.top_k, dim=-1)
        routing = F.softmax(topk_w, dim=-1)                  # [BT, K]

        # Expert computation
        out = torch.zeros_like(x_flat)
        E = self.expert_gate.shape[0]
        for k in range(self.top_k):
            for e in range(E):
                mask = topk_ids[:, k] == e
                if not mask.any():
                    continue
                xe = x_flat[mask]
                g  = F.silu(xe @ self.expert_gate[e].T)
                u  = xe @ self.expert_up[e].T
                proj = (g * u) @ self.expert_down[e].T       # [Te, H]
                out[mask] += routing[mask, k : k + 1] * proj

        return out.view(B, T, H)


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class LLaMA3BMoEDecoderLayer(nn.Module):
    def __init__(self, config: LLaMA3BMoEConfig) -> None:
        super().__init__()
        self.attn = LLaMA3BMoEAttention(config)
        self.ffn  = LLaMA3BMoEFFN(config)
        self.norm_attn = nn.Parameter(torch.ones(config.hidden_size))
        self.norm_ffn  = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.rms_norm_eps

    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        # Attention sub-layer
        residual = x
        x = _rms_norm(x, self.norm_attn, self.eps)
        x = self.attn(x, k_cache, v_cache, step)
        x = x + residual

        # MoE FFN sub-layer
        residual = x
        x = _rms_norm(x, self.norm_ffn, self.eps)
        x = self.ffn(x)
        x = x + residual

        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class LLaMA3BMoEModel(nn.Module):
    """
    LLaMA 3B Mixture-of-Experts causal language model for CPU inference.

    This is the reference PyTorch implementation.  The `demo.py` script
    optionally replaces the inner MoE FFN with YiRage-accelerated kernels.
    """

    def __init__(self, config: LLaMA3BMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LLaMA3BMoEDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.Parameter(torch.ones(config.hidden_size))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # KV caches: one pair per layer
        self._kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    # ------------------------------------------------------------------
    # KV cache management
    # ------------------------------------------------------------------

    def init_kv_cache(
        self,
        max_seq_len: int = 512,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        """Allocate KV caches for all layers."""
        cfg = self.config
        self._kv_caches = [
            (
                torch.zeros(max_seq_len, cfg.num_key_value_heads, cfg.head_dim,
                            dtype=dtype, device=device),
                torch.zeros(max_seq_len, cfg.num_key_value_heads, cfg.head_dim,
                            dtype=dtype, device=device),
            )
            for _ in range(cfg.num_hidden_layers)
        ]

    def reset_kv_cache(self) -> None:
        """Zero all KV caches (call between independent generation runs)."""
        if self._kv_caches is not None:
            for k, v in self._kv_caches:
                k.zero_()
                v.zero_()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,   # [B, T]
        step: int = 0,
    ) -> torch.Tensor:             # [B, T, vocab]
        if self._kv_caches is None:
            raise RuntimeError("Call init_kv_cache() before forward().")

        x = self.embed_tokens(input_ids)   # [B, T, H]

        for i, layer in enumerate(self.layers):
            k_cache, v_cache = self._kv_caches[i]
            x = layer(x, k_cache, v_cache, step)

        x = _rms_norm(x, self.norm, self.config.rms_norm_eps)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
    ) -> torch.Tensor:
        """Greedy-decode `max_new_tokens` tokens."""
        generated = input_ids.clone()
        for step in range(max_new_tokens):
            token_id = generated[:, -1:]
            logits = self(token_id, step=step)         # [B, 1, vocab]
            next_id = logits[:, -1, :].argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=-1)
        return generated
