# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Real PyTorch engine model for RuntimeFusion integration tests.

This replaces ``engine_stub`` for *measured* runs: tensors live on CPU/CUDA and
ops go through ``torch`` matmul / F.silu, not NumPy reference loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .engine_stub import QWEN2_MLP_HF_ATTACH
from .torch_exec import default_device, mlp_torch, require_torch, to_torch


@dataclass
class TorchAttentionMeta:
    seq_lens: Optional[tuple] = None
    block_tables: Any = None


class TorchDecoderLayer:
    """One decoder layer with real ``torch`` weights (Attention toy + MLP)."""

    def __init__(
        self,
        layer_id: int,
        *,
        hidden_size: int,
        intermediate_size: int,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        require_torch()
        import torch

        self.layer_id = int(layer_id)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.device = device or default_device()
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed + layer_id * 17)
        scale = 0.02

        def _w(shape):
            return torch.randn(shape, generator=gen, device="cpu", dtype=torch.float32) * scale

        self.attn_w = _w((hidden_size, hidden_size)).to(self.device)
        self.rms_weight = torch.ones((hidden_size,), device=self.device, dtype=torch.float32)
        self.w_gate = _w((hidden_size, intermediate_size)).to(self.device)
        self.w_up = _w((hidden_size, intermediate_size)).to(self.device)
        self.w_down = _w((intermediate_size, hidden_size)).to(self.device)
        self.hf_attach = {
            k: f"layers.{layer_id}.{v}" for k, v in QWEN2_MLP_HF_ATTACH.items()
        }

    def attention_forward(self, hidden, attn_meta: Optional[TorchAttentionMeta] = None):
        del attn_meta
        return hidden + hidden @ self.attn_w

    def mlp_forward(self, hidden):
        return mlp_torch(
            hidden,
            rms_weight=self.rms_weight,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
        )

    def forward_engine_full(self, hidden, attn_meta: Optional[TorchAttentionMeta] = None):
        h = self.attention_forward(hidden, attn_meta)
        return self.mlp_forward(h)


class TorchEngineModel:
    """Stack of :class:`TorchDecoderLayer` — real torch forward for parity/bench."""

    def __init__(
        self,
        num_layers: int,
        *,
        hidden_size: int = 32,
        intermediate_size: int = 64,
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.device = device or default_device()
        self.layers: List[TorchDecoderLayer] = [
            TorchDecoderLayer(
                i,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                seed=seed,
                device=self.device,
            )
            for i in range(num_layers)
        ]
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    def forward_engine_full(self, hidden, attn_meta: Optional[TorchAttentionMeta] = None):
        h = to_torch(hidden, device=self.device)
        for layer in self.layers:
            h = layer.forward_engine_full(h, attn_meta)
        return h

    def inspect(self) -> Dict[str, Any]:
        return {
            "engine": "TorchEngineModel",
            "device": self.device,
            "num_layers": len(self.layers),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
        }
