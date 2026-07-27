# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S7: decoder segment override with multi-Capsule RF pipelines per layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Union

import numpy as np

from .capsule_orchestration import build_split_mlp_runtime_fusion, pipeline_meta_for_layer
from .engine_stub import EngineAttentionMeta, EngineDecoderLayerStub, EngineModelStub
from .layer_override import LayerForwardResult
from .runtime_fusion import RuntimeFusion, StepMeta, StepResult


def resolve_segment_layer_ids(
    num_layers: int,
    *,
    layer_start: Optional[int] = None,
    layer_end: Optional[int] = None,
    layer_ids: Optional[Sequence[int]] = None,
) -> List[int]:
    if layer_ids is not None:
        ids = sorted({int(i) for i in layer_ids})
        for i in ids:
            if i < 0 or i >= num_layers:
                raise ValueError(f"layer_ids out of range: {i} not in [0,{num_layers})")
        return ids
    if layer_start is None or layer_end is None:
        raise ValueError("provide layer_ids or (layer_start, layer_end)")
    start, end = int(layer_start), int(layer_end)
    if start < 0 or end > num_layers or start >= end:
        raise ValueError(f"invalid segment [{start},{end}) for num_layers={num_layers}")
    return list(range(start, end))


@dataclass
class SegmentForwardResult:
    hidden: np.ndarray
    layer_results: List[LayerForwardResult] = field(default_factory=list)
    segment_layer_ids: List[int] = field(default_factory=list)
    capsules_per_step: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_layer_ids": list(self.segment_layer_ids),
            "capsules_per_step": self.capsules_per_step,
            "layers": [r.to_dict() for r in self.layer_results],
            "hidden_shape": list(self.hidden.shape),
        }


class RuntimeFusionSplitMlpLayerOverride:
    """One decoder layer: engine Attention + 2-Capsule RF MLP pipeline (S7)."""

    def __init__(
        self,
        layer: EngineDecoderLayerStub,
        rf: Optional[RuntimeFusion] = None,
        *,
        backend: Optional[str] = None,
    ):
        self.layer = layer
        self.rf = rf or build_split_mlp_runtime_fusion(layer, backend=backend)
        self.pipeline_names = [c.name for c in self.rf.capsules]

    def forward(
        self,
        hidden: np.ndarray,
        *,
        attn_meta: Optional[EngineAttentionMeta] = None,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
    ) -> LayerForwardResult:
        h = self.layer.attention_forward(hidden, attn_meta)
        meta = StepMeta.from_mapping(rf_meta)
        if not meta.should_run(self.pipeline_names[0]) and not meta.should_run(
            self.pipeline_names[1]
        ):
            out = self.layer.mlp_forward(h)
            skip = StepResult(outputs={"hidden": out}, ran=[], skipped=self.pipeline_names, meta=meta)
            return LayerForwardResult(
                hidden=out, rf=skip, used_rf_mlp=False, layer_id=self.layer.layer_id
            )

        step_meta = pipeline_meta_for_layer(self.layer.layer_id, base=dict(rf_meta or {}))
        if meta.block_tables is not None:
            step_meta["block_tables"] = meta.block_tables
        if meta.seq_lens is not None:
            step_meta["seq_lens"] = meta.seq_lens
        step_meta["page_size"] = meta.page_size
        if meta.radix_hit_mask is not None:
            step_meta["radix_hit_mask"] = meta.radix_hit_mask
        if meta.sm_budget is not None:
            step_meta["sm_budget"] = meta.sm_budget
        step_meta["extras"] = dict(meta.extras)

        result = self.rf.step({"hidden": h}, meta=step_meta)
        if len(result.ran) == 2:
            return LayerForwardResult(
                hidden=result.outputs["hidden"],
                rf=result,
                used_rf_mlp=True,
                layer_id=self.layer.layer_id,
            )
        if result.skipped_radix and len(result.ran) == 0:
            return LayerForwardResult(
                hidden=h, rf=result, used_rf_mlp=False, layer_id=self.layer.layer_id
            )
        out = self.layer.mlp_forward(h)
        return LayerForwardResult(
            hidden=out, rf=result, used_rf_mlp=False, layer_id=self.layer.layer_id
        )

    def inspect(self) -> Dict[str, Any]:
        return {
            "override": "RuntimeFusionSplitMlpLayerOverride",
            "layer_id": self.layer.layer_id,
            "pipeline": self.pipeline_names,
            "rf": self.rf.inspect(),
        }


class DecoderSegmentOverride:
    """Override a contiguous decoder segment with multi-Capsule RF MLP per layer (S7).

    Attention + KV stay on the engine; each layer's MLP runs as gate_up → down in one
    ``RF.step``. Does **not** monopolize the full model graph.
    """

    def __init__(
        self,
        model: EngineModelStub,
        *,
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        layer_ids: Optional[Sequence[int]] = None,
        backend: Optional[str] = None,
    ):
        self.model = model
        self.segment_layer_ids = resolve_segment_layer_ids(
            len(model.layers),
            layer_start=layer_start,
            layer_end=layer_end,
            layer_ids=layer_ids,
        )
        self.overrides: Dict[int, RuntimeFusionSplitMlpLayerOverride] = {
            lid: RuntimeFusionSplitMlpLayerOverride(model.layers[lid], backend=backend)
            for lid in self.segment_layer_ids
        }

    def inspect(self) -> Dict[str, Any]:
        return {
            "segment": "DecoderSegmentOverride",
            "segment_layer_ids": self.segment_layer_ids,
            "capsules_per_step": 2,
            "layers": {str(i): o.inspect() for i, o in self.overrides.items()},
        }

    def forward_segment(
        self,
        hidden: np.ndarray,
        *,
        attn_meta: Optional[EngineAttentionMeta] = None,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
    ) -> SegmentForwardResult:
        h = hidden
        results: List[LayerForwardResult] = []
        for lid in self.segment_layer_ids:
            result = self.overrides[lid].forward(h, attn_meta=attn_meta, rf_meta=rf_meta)
            h = result.hidden
            results.append(result)
        return SegmentForwardResult(
            hidden=h,
            layer_results=results,
            segment_layer_ids=list(self.segment_layer_ids),
        )


class SegmentHybridModelOverride:
    """Full model: segment uses split MLP pipeline; other layers engine or single-capsule RF."""

    def __init__(
        self,
        model: EngineModelStub,
        *,
        segment_layer_ids: Sequence[int],
        rf_mlp_layer_ids: Optional[Set[int]] = None,
        backend: Optional[str] = None,
    ):
        from .hybrid_model import resolve_rf_mlp_layer_ids
        from .layer_override import RuntimeFusionMlpLayerOverride, build_layer_mlp_capsule

        self.model = model
        self.segment_ids = resolve_segment_layer_ids(
            len(model.layers), layer_ids=segment_layer_ids
        )
        self.segment = DecoderSegmentOverride(
            model, layer_ids=self.segment_ids, backend=backend
        )
        seg_set = set(self.segment_ids)
        extra_rf = resolve_rf_mlp_layer_ids(
            len(model.layers),
            rf_mlp_layer_ids=sorted(rf_mlp_layer_ids) if rf_mlp_layer_ids else None,
            max_rf_mlp_layers=None if rf_mlp_layer_ids else 0,
        )
        self.single_rf_ids = extra_rf - seg_set
        self.single_overrides: Dict[int, RuntimeFusionMlpLayerOverride] = {}
        for lid in sorted(self.single_rf_ids):
            cap = build_layer_mlp_capsule(model.layers[lid], backend=backend)
            rf = RuntimeFusion([cap])
            self.single_overrides[lid] = RuntimeFusionMlpLayerOverride(model.layers[lid], rf)

    def forward(
        self,
        hidden: np.ndarray,
        *,
        attn_meta: Optional[EngineAttentionMeta] = None,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
    ) -> SegmentForwardResult:
        h = hidden
        results: List[LayerForwardResult] = []
        for layer in self.model.layers:
            lid = layer.layer_id
            if lid in self.segment.overrides:
                r = self.segment.overrides[lid].forward(h, attn_meta=attn_meta, rf_meta=rf_meta)
            elif lid in self.single_overrides:
                r = self.single_overrides[lid].forward(h, attn_meta=attn_meta, rf_meta=rf_meta)
            else:
                h = layer.forward_engine_full(h, attn_meta)
                r = LayerForwardResult(hidden=h, rf=None, used_rf_mlp=False, layer_id=lid)
            h = r.hidden
            results.append(r)
        rf_segment = [r.layer_id for r in results if r.used_rf_mlp]
        return SegmentForwardResult(
            hidden=h,
            layer_results=results,
            segment_layer_ids=rf_segment,
            capsules_per_step=2,
        )
