# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Engine-side layer stubs (vLLM Qwen2-shaped) without importing vLLM.

**Not for Serving cert or pytest.** Real verification uses ``TorchEngineModel``
(``torch_engine.py``) + ``tests/python/test_runtime_fusion_s*.py``.
This module remains for internal/offline reference only; do not wire it back
into ``cpu_cert`` or add new ``*smoke*`` demos. See AGENTS.md § Serving 验证禁令.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .mlp_capsule import mlp_eager_numpy


# HF / vLLM-style weight name map for one decoder MLP fragment (attach hints).
QWEN2_MLP_HF_ATTACH = {
    "rms_weight": "post_attention_layernorm.weight",
    "w_gate": "mlp.gate_proj.weight",
    "w_up": "mlp.up_proj.weight",
    "w_down": "mlp.down_proj.weight",
}


@dataclass
class EngineAttentionMeta:
    """Placeholder for engine attention / paged-KV meta (owned by vLLM)."""

    seq_lens: Optional[Tuple[int, ...]] = None
    block_tables: Any = None


class EngineDecoderLayerStub:
    """Minimal stand-in for one vLLM decoder layer.

    - ``attention_forward``: engine-owned (S2 keeps this path; simple residual linear)
    - ``mlp_forward``: engine eager MLP (fallback when RF skips the Capsule)
    """

    def __init__(
        self,
        layer_id: int,
        *,
        hidden_size: int,
        intermediate_size: int,
        seed: int = 0,
        dtype=np.float32,
    ):
        self.layer_id = int(layer_id)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        rng = np.random.default_rng(seed + layer_id * 17)
        scale = 0.02
        self.attn_w = rng.normal(0.0, scale, size=(hidden_size, hidden_size)).astype(dtype)
        self.rms_weight = np.ones((hidden_size,), dtype=dtype)
        self.w_gate = rng.normal(0.0, scale, size=(hidden_size, intermediate_size)).astype(dtype)
        self.w_up = rng.normal(0.0, scale, size=(hidden_size, intermediate_size)).astype(dtype)
        self.w_down = rng.normal(0.0, scale, size=(intermediate_size, hidden_size)).astype(dtype)
        self.hf_attach = {
            k: f"layers.{layer_id}.{v}" for k, v in QWEN2_MLP_HF_ATTACH.items()
        }

    def attention_forward(
        self,
        hidden: np.ndarray,
        attn_meta: Optional[EngineAttentionMeta] = None,
    ) -> np.ndarray:
        """Engine Attention + residual (not fused by RF in S2)."""
        del attn_meta
        # Toy attention: residual linear projection (stands in for paged attn).
        return hidden + hidden @ self.attn_w

    def mlp_forward(self, hidden: np.ndarray) -> np.ndarray:
        """Engine eager MLP (fallback when FusionCapsule is skipped)."""
        return mlp_eager_numpy(
            hidden,
            rms_weight=self.rms_weight,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
        )

    def forward_engine_full(
        self,
        hidden: np.ndarray,
        attn_meta: Optional[EngineAttentionMeta] = None,
    ) -> np.ndarray:
        h = self.attention_forward(hidden, attn_meta)
        return self.mlp_forward(h)


class EngineModelStub:
    """Stack of :class:`EngineDecoderLayerStub` (vLLM model loop stand-in)."""

    def __init__(
        self,
        num_layers: int,
        *,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        seed: int = 0,
    ):
        self.layers: List[EngineDecoderLayerStub] = [
            EngineDecoderLayerStub(
                i,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                seed=seed,
            )
            for i in range(num_layers)
        ]
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def forward_engine_full(
        self,
        hidden: np.ndarray,
        attn_meta: Optional[EngineAttentionMeta] = None,
    ) -> np.ndarray:
        h = hidden
        for layer in self.layers:
            h = layer.forward_engine_full(h, attn_meta)
        return h
