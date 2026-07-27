# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S10: SGLang Qwen2 MLP RuntimeFusion plugin (requires installed ``sglang``).

Hook point: after engine Attention + residual, replace MLP with
:meth:`SglangQwen2MlpRfHook.forward_mlp` while building StepMeta from
``ForwardBatch`` fields via :func:`rf_step_meta_from_forward_batch`.

Measured torch validation without sglang:
:func:`build_sglang_batch_torch_mlp_rf_hook` on :class:`~yirage.serving.torch_engine.TorchDecoderLayer`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np

from .exec_backend import BACKEND_TORCH
from .layer_override import (
    LayerForwardResult,
    RuntimeFusionMlpLayerOverride,
    build_layer_mlp_capsule,
    capsule_name_for_layer,
)
from .radix_meta import build_sglang_rf_step_meta
from .runtime_fusion import RuntimeFusion, StepMeta
from .torch_engine import TorchDecoderLayer
from .torch_exec import mlp_torch, require_torch
from .torch_plugin import TorchDecoderMlpRfHook


SGLANG_QWEN2_MLP_ATTACH: Dict[str, str] = {
    "post_attention_layernorm": "post_attention_layernorm.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
    "gate_up_proj": "mlp.gate_up_proj.weight",
}


def is_sglang_available() -> bool:
    try:
        import sglang  # noqa: F401

        return True
    except ImportError:
        return False


def require_sglang() -> None:
    if not is_sglang_available():
        raise RuntimeError(
            "SGLang plugin requires the sglang package. "
            "Install sglang or use build_sglang_batch_torch_mlp_rf_hook for torch-only validation."
        )


def _to_host_list(value: Any) -> Optional[list]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    arr = np.asarray(value)
    if arr.ndim == 0:
        return [arr.item()]
    return arr.tolist()


def rf_step_meta_from_forward_batch(
    forward_batch: Any,
    *,
    base: Optional[Mapping[str, Any]] = None,
    enabled: Optional[Sequence[str]] = None,
    page_size: Optional[int] = None,
    sm_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Build StepMeta from a duck-typed SGLang ``ForwardBatch`` (no ``sglang`` import)."""
    extend = getattr(forward_batch, "extend_seq_lens", None)
    if extend is None:
        extend = getattr(forward_batch, "extend_lens", None)
    block_tables = getattr(forward_batch, "block_tables", None)
    if block_tables is None:
        block_tables = getattr(forward_batch, "req_to_token", None)
    seq_lens = getattr(forward_batch, "seq_lens", None)
    ps = page_size
    if ps is None:
        ps = int(getattr(forward_batch, "page_size", 16))
    return build_sglang_rf_step_meta(
        base,
        block_tables=_to_host_list(block_tables),
        seq_lens=_to_host_list(seq_lens),
        extend_lens=_to_host_list(extend),
        page_size=int(ps),
        enabled=enabled,
        sm_budget=sm_budget,
    )


def _linear_weight_t(linear) -> Any:
    w = linear.weight
    if hasattr(w, "t"):
        return w.t()
    return w.T


def _split_gate_up_weight(gate_up_w: Any):
    """Split fused gate_up linear weight into gate and up (Qwen2 layout)."""
    if hasattr(gate_up_w, "shape"):
        shape = tuple(gate_up_w.shape)
    else:
        shape = np.asarray(gate_up_w).shape
    if len(shape) != 2:
        raise ValueError(f"gate_up weight must be rank-2, got shape={shape}")
    hidden_a, hidden_b = int(shape[0]), int(shape[1])
    if hidden_a % 2 == 0 and hidden_a > hidden_b:
        inter = hidden_a // 2
        if hasattr(gate_up_w, "chunk"):
            gate, up = gate_up_w.chunk(2, dim=0)
            return gate, up
        arr = np.asarray(gate_up_w)
        return arr[:inter], arr[inter:]
    if hidden_b % 2 == 0:
        inter = hidden_b // 2
        w_t = gate_up_w if not hasattr(gate_up_w, "t") else gate_up_w.t()
        if hasattr(w_t, "chunk"):
            gate, up = w_t.chunk(2, dim=1)
            return gate, up
        arr = np.asarray(w_t)
        return arr[:, :inter], arr[:, inter:]
    raise ValueError(f"cannot infer gate/up split from gate_up shape={shape}")


@dataclass(frozen=True)
class SglangMlpWeightView:
    """YiRage-shaped MLP weights extracted from a SGLang Qwen2 decoder layer."""

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
            "adapter": "SglangMlpWeightView",
            "layer_id": self.layer_id,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "hf_attach": dict(self.hf_attach or {}),
        }


def extract_qwen2_mlp_weights(sglang_decoder_layer, *, layer_id: Optional[int] = None) -> SglangMlpWeightView:
    """Extract MLP tensors from an installed SGLang ``Qwen2DecoderLayer``."""
    require_sglang()
    lid = int(layer_id if layer_id is not None else getattr(sglang_decoder_layer, "layer_id", 0))
    norm = sglang_decoder_layer.post_attention_layernorm
    mlp = sglang_decoder_layer.mlp
    rms = norm.weight
    hidden_size = int(rms.shape[0])
    if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
        w_gate = _linear_weight_t(mlp.gate_proj)
        w_up = _linear_weight_t(mlp.up_proj)
        intermediate_size = int(w_gate.shape[1])
        attach = {
            "post_attention_layernorm": "post_attention_layernorm.weight",
            "gate_proj": "mlp.gate_proj.weight",
            "up_proj": "mlp.up_proj.weight",
            "down_proj": "mlp.down_proj.weight",
        }
    elif hasattr(mlp, "gate_up_proj"):
        fused = _linear_weight_t(mlp.gate_up_proj)
        w_gate, w_up = _split_gate_up_weight(fused)
        intermediate_size = int(w_gate.shape[1]) if len(w_gate.shape) == 2 else int(w_gate.shape[0])
        attach = {
            "post_attention_layernorm": "post_attention_layernorm.weight",
            "gate_up_proj": "mlp.gate_up_proj.weight",
            "down_proj": "mlp.down_proj.weight",
        }
    else:
        raise RuntimeError("SGLang Qwen2 MLP must expose gate/up/down or gate_up/down projections")
    w_down = _linear_weight_t(mlp.down_proj)
    device = str(rms.device) if hasattr(rms, "device") else None
    hf_attach = {k: f"layers.{lid}.{v}" for k, v in attach.items()}
    return SglangMlpWeightView(
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


class _SglangMlpLayerAdapter:
    """Layer surface for :class:`RuntimeFusionMlpLayerOverride` (engine MLP fallback via torch)."""

    def __init__(self, view: SglangMlpWeightView):
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
        raise RuntimeError("SGLang plugin: use forward_mlp after engine Attention")

    def mlp_forward(self, hidden):
        require_torch()
        return mlp_torch(
            hidden,
            rms_weight=self.rms_weight,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
        )


class SglangBatchTorchMlpRfHook:
    """Torch decoder MLP hook that builds RF meta from a ForwardBatch-like object."""

    def __init__(self, layer: TorchDecoderLayer):
        require_torch()
        self.layer = layer
        self._inner = TorchDecoderMlpRfHook(layer)

    def forward_mlp(
        self,
        hidden_after_attn,
        *,
        forward_batch: Any = None,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
    ) -> LayerForwardResult:
        meta = rf_meta
        if forward_batch is not None:
            enabled = {self._inner.override.capsule_name}
            base = dict(rf_meta) if isinstance(rf_meta, Mapping) else None
            meta = rf_step_meta_from_forward_batch(
                forward_batch,
                base=base,
                enabled=enabled,
            )
        return self._inner.forward_mlp(hidden_after_attn, rf_meta=meta)

    def inspect(self) -> Dict[str, Any]:
        return {
            "plugin": "SglangBatchTorchMlpRfHook",
            "device": self.layer.device,
            "layer_id": self.layer.layer_id,
            "inner": self._inner.inspect(),
        }


class SglangQwen2MlpRfHook:
    """RuntimeFusion MLP hook for one SGLang Qwen2 decoder layer (requires ``sglang``)."""

    def __init__(
        self,
        sglang_decoder_layer,
        *,
        layer_id: Optional[int] = None,
    ):
        require_sglang()
        view = extract_qwen2_mlp_weights(sglang_decoder_layer, layer_id=layer_id)
        self.view = view
        adapter = _SglangMlpLayerAdapter(view)
        cap = build_layer_mlp_capsule(adapter, backend=BACKEND_TORCH)
        rf = RuntimeFusion([cap])
        self.override = RuntimeFusionMlpLayerOverride(
            adapter, rf, capsule_name=capsule_name_for_layer(view.layer_id)
        )

    def forward_mlp(
        self,
        hidden_after_attn,
        *,
        forward_batch: Any = None,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
    ) -> LayerForwardResult:
        meta = rf_meta
        if forward_batch is not None:
            enabled = {self.override.capsule_name}
            base = dict(rf_meta) if isinstance(rf_meta, Mapping) else None
            meta = rf_step_meta_from_forward_batch(
                forward_batch,
                base=base,
                enabled=enabled,
            )
        return self.override.forward_mlp_only(hidden_after_attn, rf_meta=meta)

    def inspect(self) -> Dict[str, Any]:
        return {
            "plugin": "SglangQwen2MlpRfHook",
            "sglang_installed": True,
            "weight_view": self.view.inspect(),
            "override": self.override.inspect(),
        }


def build_sglang_batch_torch_mlp_rf_hook(layer: TorchDecoderLayer) -> SglangBatchTorchMlpRfHook:
    """Factory for measured torch MLP hook + SGLang ForwardBatch meta (no ``sglang`` required)."""
    return SglangBatchTorchMlpRfHook(layer)


def build_sglang_qwen2_mlp_rf_hook(
    sglang_decoder_layer,
    *,
    layer_id: Optional[int] = None,
) -> SglangQwen2MlpRfHook:
    """Factory for SGLang model plugin registration (``sglang`` must be installed)."""
    return SglangQwen2MlpRfHook(sglang_decoder_layer, layer_id=layer_id)
