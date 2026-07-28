# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S8: vLLM Qwen2 MLP RuntimeFusion plugin (requires installed ``vllm``).

Hook point: after ``self_attn`` + residual, replace ``self.mlp(...)`` with
:meth:`VllmQwen2MlpRfHook.forward_mlp` so Attention/Paged KV stay on vLLM.

Measured torch validation without vLLM: :mod:`yirage.serving.torch_plugin`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from .exec_backend import BACKEND_TORCH
from .layer_override import (
    LayerForwardResult,
    RuntimeFusionMlpLayerOverride,
    build_layer_mlp_capsule,
    capsule_name_for_layer,
)
from .runtime_fusion import RuntimeFusion, StepMeta
from .torch_exec import mlp_torch, require_torch


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


def require_vllm() -> None:
    if not is_vllm_available():
        raise RuntimeError(
            "vLLM plugin requires the vllm package. "
            "Install vllm or use yirage.serving.build_torch_mlp_rf_hook for torch-only validation."
        )


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
    """Extract MLP tensors from an installed vLLM ``Qwen2DecoderLayer``."""
    require_vllm()
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
    """Layer surface for :class:`RuntimeFusionMlpLayerOverride` (engine MLP fallback via torch)."""

    def __init__(self, view: VllmMlpWeightView):
        self.layer_id = view.layer_id
        self.hidden_size = view.hidden_size
        self.intermediate_size = view.intermediate_size
        self.rms_weight = view.rms_weight
        self.w_gate = view.w_gate
        self.w_up = view.w_up
        self.w_down = view.w_down
        self.device = view.device
        self.hf_attach = dict(view.hf_attach or {})

    def attention_forward(self, hidden, attn_meta=None):
        raise RuntimeError("vLLM plugin: use forward_mlp_only after engine Attention")

    def mlp_forward(self, hidden):
        require_torch()
        return mlp_torch(
            hidden,
            rms_weight=self.rms_weight,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
        )


class VllmQwen2MlpRfHook:
    """RuntimeFusion MLP hook for one vLLM Qwen2 decoder layer (requires ``vllm``)."""

    def __init__(
        self,
        vllm_decoder_layer,
        *,
        layer_id: Optional[int] = None,
    ):
        require_vllm()
        view = extract_qwen2_mlp_weights(vllm_decoder_layer, layer_id=layer_id)
        self.view = view
        adapter = _VllmMlpLayerAdapter(view)
        cap = build_layer_mlp_capsule(adapter, backend=BACKEND_TORCH)
        rf = RuntimeFusion([cap])
        self.override = RuntimeFusionMlpLayerOverride(
            adapter, rf, capsule_name=capsule_name_for_layer(view.layer_id)
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
            "vllm_installed": True,
            "weight_view": self.view.inspect(),
            "override": self.override.inspect(),
        }


def build_vllm_qwen2_mlp_rf_hook(
    vllm_decoder_layer,
    *,
    layer_id: Optional[int] = None,
) -> VllmQwen2MlpRfHook:
    """Factory for vLLM model plugin registration (``vllm`` must be installed)."""
    return VllmQwen2MlpRfHook(vllm_decoder_layer, layer_id=layer_id)
