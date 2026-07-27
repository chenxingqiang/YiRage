# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S16: SGLang-metax Qwen2 MLP RuntimeFusion plugin tier.

Requires installed ``sglang`` on a MetaX mcPytorch host (SGLang-metax fork).
Measured torch validation without the fork:
:func:`build_sglang_metax_batch_torch_mlp_rf_hook` +
:mod:`yirage.serving.sglang_metax_e2e`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from .exec_backend import BACKEND_TORCH
from .layer_override import (
    LayerForwardResult,
    RuntimeFusionMlpLayerOverride,
    build_layer_mlp_capsule,
    capsule_name_for_layer,
)
from .maca_serving_meta import MacaServingRfSpec, maca_serving_present
from .runtime_fusion import RuntimeFusion, StepMeta
from .sglang_plugin import (
    SGLANG_QWEN2_MLP_ATTACH,
    SglangBatchTorchMlpRfHook,
    _SglangMlpLayerAdapter,
    extract_qwen2_mlp_weights,
    is_sglang_available,
    require_sglang,
    rf_step_meta_from_forward_batch,
)
from .torch_engine import TorchDecoderLayer
from .vllm_metax_plugin import is_metax_torch


def is_sglang_metax_available() -> bool:
    """True when sglang is installed on a MetaX-capable host (SGLang-metax tier)."""
    if not is_sglang_available():
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


def require_sglang_metax() -> None:
    if not is_sglang_metax_available():
        raise RuntimeError(
            "SGLang-metax plugin requires sglang on a MetaX mcPytorch host "
            "(or YIRAGE_MACA_INTEGRATION=1 for contract smoke). "
            "Use build_sglang_metax_batch_torch_mlp_rf_hook + sglang_metax_e2e on CPU."
        )


def rf_step_meta_for_sglang_metax(
    forward_batch: Any = None,
    *,
    spec: Optional[MacaServingRfSpec] = None,
    block_tables: Any = None,
    seq_lens: Optional[Sequence[int]] = None,
    extend_lens: Optional[Sequence[int]] = None,
    page_size: int = 16,
    enabled: Optional[Sequence[str]] = None,
    sm_budget: Optional[int] = None,
    base: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Merge SGLang ForwardBatch meta with MACA serving constraints."""
    from .radix_meta import build_sglang_rf_step_meta

    spec = spec or MacaServingRfSpec()
    if forward_batch is not None:
        sglang_meta = rf_step_meta_from_forward_batch(
            forward_batch,
            base=base,
            enabled=enabled,
            sm_budget=sm_budget,
        )
    else:
        sglang_meta = build_sglang_rf_step_meta(
            block_tables=block_tables,
            seq_lens=list(seq_lens or []),
            extend_lens=list(extend_lens or []),
            page_size=int(page_size),
            enabled=enabled,
            sm_budget=sm_budget,
        )
        if base:
            for key, val in base.items():
                if key not in sglang_meta:
                    sglang_meta[key] = val

    maca_meta = spec.as_rf_meta(sm_budget=sm_budget)
    out = dict(sglang_meta)
    out_extras = dict(out.get("extras") or {})
    maca_extras = dict(maca_meta.get("extras") or {})
    out_extras.update(maca_extras)
    out["extras"] = out_extras
    if "sm_budget" in maca_meta and "sm_budget" not in out:
        out["sm_budget"] = maca_meta["sm_budget"]
    return out


class SglangMetaxBatchTorchMlpRfHook:
    """Torch MLP hook + SGLang ForwardBatch + MACA serving meta (no ``sglang`` required)."""

    def __init__(
        self,
        layer: TorchDecoderLayer,
        *,
        maca_spec: Optional[MacaServingRfSpec] = None,
    ):
        self.layer = layer
        self.maca_spec = maca_spec or MacaServingRfSpec()
        self._inner = SglangBatchTorchMlpRfHook(layer)

    def forward_mlp(
        self,
        hidden_after_attn,
        *,
        forward_batch: Any = None,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
    ) -> LayerForwardResult:
        meta = rf_meta
        if forward_batch is not None:
            enabled = {self._inner._inner.override.capsule_name}
            base = dict(rf_meta) if isinstance(rf_meta, Mapping) else None
            meta = rf_step_meta_for_sglang_metax(
                forward_batch,
                spec=self.maca_spec,
                base=base,
                enabled=enabled,
            )
        elif isinstance(rf_meta, Mapping) and not maca_serving_present(rf_meta):
            meta = rf_step_meta_for_sglang_metax(
                spec=self.maca_spec,
                base=rf_meta,
                enabled=list(rf_meta.get("enabled") or []),
            )
        return self._inner.forward_mlp(hidden_after_attn, forward_batch=None, rf_meta=meta)

    def inspect(self) -> Dict[str, Any]:
        return {
            "plugin": "SglangMetaxBatchTorchMlpRfHook",
            "maca_spec": self.maca_spec.maca_serving_payload(),
            "inner": self._inner.inspect(),
        }


class SglangMetaxQwen2MlpRfHook:
    """RuntimeFusion MLP hook for SGLang-metax Qwen2 (sglang + MetaX host)."""

    def __init__(
        self,
        sglang_decoder_layer,
        *,
        layer_id: Optional[int] = None,
        maca_spec: Optional[MacaServingRfSpec] = None,
    ):
        require_sglang_metax()
        view = extract_qwen2_mlp_weights(sglang_decoder_layer, layer_id=layer_id)
        self.view = view
        self.maca_spec = maca_spec or MacaServingRfSpec()
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
        enabled = {self.override.capsule_name}
        base = dict(rf_meta) if isinstance(rf_meta, Mapping) else None
        if forward_batch is not None:
            meta = rf_step_meta_for_sglang_metax(
                forward_batch,
                spec=self.maca_spec,
                base=base,
                enabled=enabled,
            )
        else:
            meta = rf_step_meta_for_sglang_metax(
                spec=self.maca_spec,
                base=base,
                enabled=enabled,
            )
        return self.override.forward_mlp_only(hidden_after_attn, rf_meta=meta)

    def inspect(self) -> Dict[str, Any]:
        return {
            "plugin": "SglangMetaxQwen2MlpRfHook",
            "sglang_metax_tier": True,
            "metax_torch": is_metax_torch(),
            "maca_spec": self.maca_spec.maca_serving_payload(),
            "weight_view": self.view.inspect(),
            "override": self.override.inspect(),
        }


def build_sglang_metax_batch_torch_mlp_rf_hook(
    layer: TorchDecoderLayer,
    *,
    maca_spec: Optional[MacaServingRfSpec] = None,
) -> SglangMetaxBatchTorchMlpRfHook:
    return SglangMetaxBatchTorchMlpRfHook(layer, maca_spec=maca_spec)


def build_sglang_metax_qwen2_mlp_rf_hook(
    sglang_decoder_layer,
    *,
    layer_id: Optional[int] = None,
    maca_spec: Optional[MacaServingRfSpec] = None,
) -> SglangMetaxQwen2MlpRfHook:
    return SglangMetaxQwen2MlpRfHook(
        sglang_decoder_layer,
        layer_id=layer_id,
        maca_spec=maca_spec,
    )


__all__ = [
    "SGLANG_QWEN2_MLP_ATTACH",
    "SglangMetaxBatchTorchMlpRfHook",
    "SglangMetaxQwen2MlpRfHook",
    "build_sglang_metax_batch_torch_mlp_rf_hook",
    "build_sglang_metax_qwen2_mlp_rf_hook",
    "is_sglang_metax_available",
    "require_sglang_metax",
    "rf_step_meta_for_sglang_metax",
]
