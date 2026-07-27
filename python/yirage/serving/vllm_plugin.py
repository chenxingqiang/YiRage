# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S8: vLLM Qwen2 MLP RuntimeFusion plugin (duck-typed; no vLLM vendor).

Hook point: after ``self_attn`` + residual, replace ``self.mlp(...)`` with
:meth:`VllmQwen2MlpRfHook.forward_mlp` so Attention/Paged KV stay on vLLM.

When ``vllm`` is not installed, contract tests use a duck-typed mock layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from .exec_backend import BACKEND_TORCH, default_serving_backend
from .layer_override import (
    LayerForwardResult,
    RuntimeFusionMlpLayerOverride,
    build_layer_mlp_capsule,
    capsule_name_for_layer,
)
from .runtime_fusion import RuntimeFusion, StepMeta


VLLM_QWEN2_MLP_ATTACH: Dict[str, str] = {
    "post_attention_layernorm": "post_attention_layernorm.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


def is_vllm_available() -> bool:
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class VllmMlpWeightView:
    """YiRage-shaped MLP weights extracted from a vLLM Qwen2 decoder layer."""

    layer_id: int
    hidden_size: int
    intermediate_size: int
    rms_weight: Any
    w_gate: Any
    w_up: Any
    w_down: Any
    device: Optional[str] = None
    hf_attach: Optional[Dict[str, str]] = None

    def inspect(self) -> Dict[str, Any]:
        return {
            "adapter": "VllmMlpWeightView",
            "layer_id": self.layer_id,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "hf_attach": dict(self.hf_attach or {}),
        }


def _linear_weight_t(linear) -> Any:
    w = linear.weight
    if hasattr(w, "t"):
        return w.t()
    return w.T


def extract_qwen2_mlp_weights(vllm_decoder_layer, *, layer_id: Optional[int] = None) -> VllmMlpWeightView:
    """Extract MLP tensors from a duck-typed vLLM ``Qwen2DecoderLayer``."""
    lid = int(layer_id if layer_id is not None else getattr(vllm_decoder_layer, "layer_id", 0))
    norm = vllm_decoder_layer.post_attention_layernorm
    mlp = vllm_decoder_layer.mlp
    rms = norm.weight
    hidden_size = int(rms.shape[0])
    w_gate = _linear_weight_t(mlp.gate_proj)
    w_up = _linear_weight_t(mlp.up_proj)
    w_down = _linear_weight_t(mlp.down_proj)
    intermediate_size = int(w_gate.shape[1])
    device = str(rms.device) if hasattr(rms, "device") else None
    hf_attach = {k: f"layers.{lid}.{v}" for k, v in VLLM_QWEN2_MLP_ATTACH.items()}
    return VllmMlpWeightView(
        layer_id=lid,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        rms_weight=rms,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
        device=device,
        hf_attach=hf_attach,
    )


class _VllmMlpLayerAdapter:
    """Minimal layer surface for :class:`RuntimeFusionMlpLayerOverride`."""

    def __init__(self, view: VllmMlpWeightView, *, source_layer: Any = None):
        self.layer_id = view.layer_id
        self.hidden_size = view.hidden_size
        self.intermediate_size = view.intermediate_size
        self.rms_weight = view.rms_weight
        self.w_gate = view.w_gate
        self.w_up = view.w_up
        self.w_down = view.w_down
        self.device = view.device
        self.hf_attach = dict(view.hf_attach or {})
        self._source = source_layer

    def attention_forward(self, hidden, attn_meta=None):
        raise RuntimeError("vLLM plugin: use forward_mlp_only after engine Attention")

    def mlp_forward(self, hidden):
        if self._source is not None and hasattr(self._source, "mlp"):
            mlp = self._source.mlp
            if callable(mlp):
                return hidden + mlp(hidden)
        from .torch_exec import mlp_torch

        return mlp_torch(
            hidden,
            rms_weight=self.rms_weight,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
        )


class VllmQwen2MlpRfHook:
    """RuntimeFusion MLP hook for one vLLM Qwen2 decoder layer (S8)."""

    def __init__(
        self,
        vllm_decoder_layer,
        *,
        layer_id: Optional[int] = None,
        backend: Optional[str] = None,
    ):
        view = extract_qwen2_mlp_weights(vllm_decoder_layer, layer_id=layer_id)
        self.view = view
        self.adapter = _VllmMlpLayerAdapter(view, source_layer=vllm_decoder_layer)
        be = backend or default_serving_backend()
        cap = build_layer_mlp_capsule(self.adapter, backend=be)
        rf = RuntimeFusion([cap])
        self.override = RuntimeFusionMlpLayerOverride(
            self.adapter, rf, capsule_name=capsule_name_for_layer(view.layer_id)
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
            "plugin": "VllmQwen2MlpRfHook",
            "vllm_installed": is_vllm_available(),
            "weight_view": self.view.inspect(),
            "override": self.override.inspect(),
        }


def build_vllm_qwen2_mlp_rf_hook(
    vllm_decoder_layer,
    *,
    layer_id: Optional[int] = None,
    backend: Optional[str] = None,
) -> VllmQwen2MlpRfHook:
    """Factory for vLLM model plugin registration."""
    return VllmQwen2MlpRfHook(
        vllm_decoder_layer,
        layer_id=layer_id,
        backend=backend or BACKEND_TORCH,
    )
