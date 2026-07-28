# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S2: model-layer MLP Override — Attention stays on engine; MLP via RF.step.

This is a **plugin wrapper** around a duck-typed decoder layer (vLLM Qwen2-shaped).
It does not vendor vLLM; a real deployment would wrap
``vllm.model_executor.models.qwen2.Qwen2DecoderLayer`` the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np

from .engine_stub import EngineAttentionMeta, EngineDecoderLayerStub
from .mlp_capsule import MlpFusionCapsule
from .runtime_fusion import RuntimeFusion, StepMeta, StepResult
from .torch_exec import to_numpy


def capsule_name_for_layer(layer_id: int) -> str:
    return f"mlp_layer_{layer_id}"


def build_layer_mlp_capsule(
    layer,
    *,
    backend: Optional[str] = None,
) -> MlpFusionCapsule:
    """Build an MLP FusionCapsule sharing the engine layer's MLP weights."""
    from .exec_backend import BACKEND_NUMPY_REF, default_serving_backend
    from .plan import FusionPlan

    be = backend or (
        BACKEND_NUMPY_REF
        if isinstance(layer, EngineDecoderLayerStub)
        else default_serving_backend()
    )
    plan = FusionPlan.mlp(
        name=capsule_name_for_layer(layer.layer_id),
        hidden_size=layer.hidden_size,
        intermediate_size=layer.intermediate_size,
        dtype="float32",
        backend=be,
    )
    return MlpFusionCapsule(
        plan,
        rms_weight=layer.rms_weight,
        w_gate=layer.w_gate,
        w_up=layer.w_up,
        w_down=layer.w_down,
        device=getattr(layer, "device", None),
    )


@dataclass
class LayerForwardResult:
    hidden: np.ndarray
    rf: Optional[StepResult]
    used_rf_mlp: bool
    layer_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "used_rf_mlp": self.used_rf_mlp,
            "rf": None if self.rf is None else self.rf.to_dict(),
            "hidden_shape": list(self.hidden.shape),
        }


class RuntimeFusionMlpLayerOverride:
    """Override one decoder layer: engine Attention + RF MLP Capsule.

    When RF skips the capsule, falls back to ``layer.mlp_forward`` (engine owns MLP).
    """

    def __init__(
        self,
        layer: EngineDecoderLayerStub,
        rf: RuntimeFusion,
        *,
        capsule_name: Optional[str] = None,
    ):
        self.layer = layer
        self.rf = rf
        self.capsule_name = capsule_name or capsule_name_for_layer(layer.layer_id)
        names = {c.name for c in rf.capsules}
        if self.capsule_name not in names:
            raise KeyError(
                f"RF has no capsule {self.capsule_name!r}; registered={sorted(names)}"
            )

    def forward(
        self,
        hidden: np.ndarray,
        *,
        attn_meta: Optional[EngineAttentionMeta] = None,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
    ) -> LayerForwardResult:
        # 1) Attention / PagedAttention stays on the engine.
        h = self.layer.attention_forward(hidden, attn_meta)

        # 2) MLP: RF.step select, else engine MLP fallback (incl. SM budget skip).
        meta = StepMeta.from_mapping(rf_meta)
        if meta.should_run(self.capsule_name):
            step_meta = {
                "enabled": {self.capsule_name},
                "block_tables": meta.block_tables,
                "seq_lens": meta.seq_lens,
                "page_size": meta.page_size,
                "radix_hit_mask": meta.radix_hit_mask,
                "sm_budget": meta.sm_budget,
                "extras": meta.extras,
            }
            result = self.rf.step({"hidden": h}, meta=step_meta)
            if self.capsule_name in result.ran:
                out = result.outputs["hidden"]
                return LayerForwardResult(
                    hidden=out, rf=result, used_rf_mlp=True, layer_id=self.layer.layer_id
                )
            if self.capsule_name in result.skipped_radix:
                # S6: Radix all-hit — cache owns MLP; pass post-attention hidden through.
                return LayerForwardResult(
                    hidden=h, rf=result, used_rf_mlp=False, layer_id=self.layer.layer_id
                )
            # RF skipped (SM budget / internal policy) → engine owns MLP.
            out = self.layer.mlp_forward(h)
            return LayerForwardResult(
                hidden=out, rf=result, used_rf_mlp=False, layer_id=self.layer.layer_id
            )

        out = self.layer.mlp_forward(h)
        skip_result = StepResult(
            outputs={"hidden": out},
            ran=[],
            skipped=[self.capsule_name],
            meta=meta,
        )
        return LayerForwardResult(
            hidden=out, rf=skip_result, used_rf_mlp=False, layer_id=self.layer.layer_id
        )

    def forward_mlp_only(
        self,
        hidden_after_attn,
        *,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
    ) -> LayerForwardResult:
        """Run only the MLP RF path (vLLM hook: Attention already executed)."""
        meta = StepMeta.from_mapping(rf_meta)
        h = hidden_after_attn
        if meta.should_run(self.capsule_name):
            step_meta = {
                "enabled": {self.capsule_name},
                "block_tables": meta.block_tables,
                "seq_lens": meta.seq_lens,
                "page_size": meta.page_size,
                "radix_hit_mask": meta.radix_hit_mask,
                "sm_budget": meta.sm_budget,
                "extras": meta.extras,
            }
            result = self.rf.step({"hidden": h}, meta=step_meta)
            if self.capsule_name in result.ran:
                return LayerForwardResult(
                    hidden=result.outputs["hidden"],
                    rf=result,
                    used_rf_mlp=True,
                    layer_id=self.layer.layer_id,
                )
            if self.capsule_name in result.skipped_radix:
                return LayerForwardResult(
                    hidden=h, rf=result, used_rf_mlp=False, layer_id=self.layer.layer_id
                )
            out = self.layer.mlp_forward(h)
            return LayerForwardResult(
                hidden=out, rf=result, used_rf_mlp=False, layer_id=self.layer.layer_id
            )
        out = self.layer.mlp_forward(h)
        skip_result = StepResult(
            outputs={"hidden": out},
            ran=[],
            skipped=[self.capsule_name],
            meta=meta,
        )
        return LayerForwardResult(
            hidden=out, rf=skip_result, used_rf_mlp=False, layer_id=self.layer.layer_id
        )

    def inspect(self) -> Dict[str, Any]:
        return {
            "override": "RuntimeFusionMlpLayerOverride",
            "layer_id": self.layer.layer_id,
            "capsule_name": self.capsule_name,
            "hf_attach": dict(self.layer.hf_attach),
            "attention": "engine",
            "mlp": "RuntimeFusion.step | engine_fallback",
        }
