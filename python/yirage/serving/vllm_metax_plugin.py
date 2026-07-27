# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S15: vLLM-metax Qwen2 MLP RuntimeFusion plugin tier.

Requires installed ``vllm`` on a MetaX mcPytorch host (vLLM-metax fork). Measured
torch validation without the fork: :mod:`yirage.serving.torch_plugin` +
:mod:`yirage.serving.maca_serving_e2e`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

from .exec_backend import BACKEND_TORCH
from .layer_override import (
    LayerForwardResult,
    RuntimeFusionMlpLayerOverride,
    build_layer_mlp_capsule,
    capsule_name_for_layer,
)
from .maca_serving_meta import MacaServingRfSpec
from .runtime_fusion import RuntimeFusion, StepMeta
from .vllm_plugin import (
    VLLM_QWEN2_MLP_ATTACH,
    _VllmMlpLayerAdapter,
    extract_qwen2_mlp_weights,
    is_vllm_available,
    require_vllm,
)


def is_metax_torch() -> bool:
    """True when PyTorch reports a MetaX mcPytorch build."""
    try:
        import torch

        return "metax" in torch.__version__.lower()
    except ImportError:
        return False


def is_vllm_metax_available() -> bool:
    """True when vLLM is installed on a MetaX-capable host (vLLM-metax tier)."""
    if not is_vllm_available():
        return False
    if os.environ.get("YIRAGE_MACA_INTEGRATION", "").strip() == "1":
        return True
    if is_metax_torch():
        return True
    try:
        from yirage.backends.maca.config import is_maca_torch_device_available

        return is_maca_torch_device_available()
    except Exception:
        return False


def require_vllm_metax() -> None:
    if not is_vllm_metax_available():
        raise RuntimeError(
            "vLLM-metax plugin requires vllm on a MetaX mcPytorch host "
            "(or YIRAGE_MACA_INTEGRATION=1 for contract smoke). "
            "Use build_torch_mlp_rf_hook + maca_serving_e2e on CPU torch hosts."
        )


def rf_step_meta_for_vllm_metax(
    *,
    spec: Optional[MacaServingRfSpec] = None,
    sm_budget: Optional[int] = None,
    base: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """ForwardBatch-style MACA meta for vLLM-metax MLP RF hooks."""
    spec = spec or MacaServingRfSpec()
    merged = spec.as_rf_meta(sm_budget=sm_budget)
    if base:
        base_extras = dict(base.get("extras") or {})
        out_extras = dict(merged.get("extras") or {})
        out_extras.update(base_extras)
        merged["extras"] = out_extras
        for key in ("enabled", "disabled", "sm_budget", "block_tables", "seq_lens"):
            if key in base and key not in merged:
                merged[key] = base[key]
    return merged


@dataclass(frozen=True)
class VllmMetaxMlpRfHookReport:
    parity_ok: bool
    used_rf_mlp: bool
    maca_meta_attached: bool
    plugin: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parity_ok": self.parity_ok,
            "used_rf_mlp": self.used_rf_mlp,
            "maca_meta_attached": self.maca_meta_attached,
            "plugin": self.plugin,
        }


class VllmMetaxQwen2MlpRfHook:
    """RuntimeFusion MLP hook for vLLM-metax Qwen2 (vllm + MetaX host)."""

    def __init__(
        self,
        vllm_decoder_layer,
        *,
        layer_id: Optional[int] = None,
        maca_spec: Optional[MacaServingRfSpec] = None,
    ):
        require_vllm_metax()
        view = extract_qwen2_mlp_weights(vllm_decoder_layer, layer_id=layer_id)
        self.view = view
        self.maca_spec = maca_spec or MacaServingRfSpec()
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
        meta = rf_step_meta_for_vllm_metax(
            spec=self.maca_spec,
            base=rf_meta if isinstance(rf_meta, Mapping) else None,
        )
        if rf_meta is not None and not isinstance(rf_meta, Mapping):
            step = StepMeta.from_mapping(rf_meta)
            if step.enabled is not None:
                meta["enabled"] = set(step.enabled)
            if step.sm_budget is not None:
                meta["sm_budget"] = step.sm_budget
        enabled = meta.get("enabled")
        if enabled is None:
            meta["enabled"] = {self.override.capsule_name}
        return self.override.forward_mlp_only(hidden_after_attn, rf_meta=meta)

    def inspect(self) -> Dict[str, Any]:
        return {
            "plugin": "VllmMetaxQwen2MlpRfHook",
            "vllm_metax_tier": True,
            "metax_torch": is_metax_torch(),
            "maca_spec": self.maca_spec.maca_serving_payload(),
            "weight_view": self.view.inspect(),
            "override": self.override.inspect(),
        }


def build_vllm_metax_qwen2_mlp_rf_hook(
    vllm_decoder_layer,
    *,
    layer_id: Optional[int] = None,
    maca_spec: Optional[MacaServingRfSpec] = None,
) -> VllmMetaxQwen2MlpRfHook:
    return VllmMetaxQwen2MlpRfHook(
        vllm_decoder_layer,
        layer_id=layer_id,
        maca_spec=maca_spec,
    )


__all__ = [
    "VLLM_QWEN2_MLP_ATTACH",
    "VllmMetaxMlpRfHookReport",
    "VllmMetaxQwen2MlpRfHook",
    "build_vllm_metax_qwen2_mlp_rf_hook",
    "is_metax_torch",
    "is_vllm_metax_available",
    "require_vllm_metax",
    "rf_step_meta_for_vllm_metax",
]
