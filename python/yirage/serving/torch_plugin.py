# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S8: real PyTorch decoder MLP RF hook (TorchDecoderLayer; no stub/mock).

Use when validating the MLP hook path without an installed ``vllm`` package.
For production vLLM integration see :mod:`yirage.serving.vllm_plugin`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

from .exec_backend import BACKEND_TORCH
from .layer_override import (
    LayerForwardResult,
    RuntimeFusionMlpLayerOverride,
    build_layer_mlp_capsule,
    capsule_name_for_layer,
)
from .runtime_fusion import RuntimeFusion, StepMeta
from .torch_engine import TorchDecoderLayer
from .torch_exec import require_torch


class TorchDecoderMlpRfHook:
    """RuntimeFusion MLP hook backed by real ``torch`` weights on :class:`TorchDecoderLayer`."""

    def __init__(self, layer: TorchDecoderLayer):
        require_torch()
        self.layer = layer
        cap = build_layer_mlp_capsule(layer, backend=BACKEND_TORCH)
        rf = RuntimeFusion([cap])
        self.override = RuntimeFusionMlpLayerOverride(
            layer, rf, capsule_name=capsule_name_for_layer(layer.layer_id)
        )

    def forward_mlp(
        self,
        hidden_after_attn,
        *,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
    ) -> LayerForwardResult:
        return self.override.forward_mlp_only(hidden_after_attn, rf_meta=rf_meta)

    def inspect(self) -> Dict[str, Any]:
        return {
            "plugin": "TorchDecoderMlpRfHook",
            "device": self.layer.device,
            "layer_id": self.layer.layer_id,
            "override": self.override.inspect(),
        }


def build_torch_mlp_rf_hook(layer: TorchDecoderLayer) -> TorchDecoderMlpRfHook:
    """Factory for measured torch MLP RF hook (S8 default validation path)."""
    return TorchDecoderMlpRfHook(layer)
