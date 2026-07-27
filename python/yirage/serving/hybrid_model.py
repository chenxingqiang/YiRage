# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S3: hybrid model loop — first K layers use RF MLP Capsules; rest stay engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Union

import numpy as np

from .engine_stub import EngineAttentionMeta, EngineModelStub
from .layer_override import (
    LayerForwardResult,
    RuntimeFusionMlpLayerOverride,
    build_layer_mlp_capsule,
    capsule_name_for_layer,
)
from .runtime_fusion import RuntimeFusion, StepMeta


@dataclass
class HybridForwardResult:
    hidden: np.ndarray
    layer_results: List[LayerForwardResult] = field(default_factory=list)
    rf_layer_ids: List[int] = field(default_factory=list)
    engine_mlp_layer_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rf_layer_ids": list(self.rf_layer_ids),
            "engine_mlp_layer_ids": list(self.engine_mlp_layer_ids),
            "layers": [r.to_dict() for r in self.layer_results],
            "hidden_shape": list(self.hidden.shape),
        }


def resolve_rf_mlp_layer_ids(
    num_layers: int,
    *,
    max_rf_mlp_layers: Optional[int] = None,
    rf_mlp_layer_ids: Optional[Sequence[int]] = None,
) -> Set[int]:
    """Select which decoder layers route MLP through RuntimeFusion.

    Precedence: explicit ``rf_mlp_layer_ids`` wins; else first ``max_rf_mlp_layers``.
    """
    if rf_mlp_layer_ids is not None:
        ids = {int(i) for i in rf_mlp_layer_ids}
        for i in ids:
            if i < 0 or i >= num_layers:
                raise ValueError(f"rf_mlp_layer_ids out of range: {i} not in [0,{num_layers})")
        return ids
    if max_rf_mlp_layers is None:
        return set()
    k = int(max_rf_mlp_layers)
    if k < 0 or k > num_layers:
        raise ValueError(f"max_rf_mlp_layers={k} invalid for num_layers={num_layers}")
    return set(range(k))


class HybridModelOverride:
    """vLLM-style ``for layer in layers`` with selective RF MLP Override (S3)."""

    def __init__(
        self,
        model: EngineModelStub,
        *,
        max_rf_mlp_layers: Optional[int] = None,
        rf_mlp_layer_ids: Optional[Sequence[int]] = None,
    ):
        self.model = model
        self.rf_layer_ids = resolve_rf_mlp_layer_ids(
            len(model.layers),
            max_rf_mlp_layers=max_rf_mlp_layers,
            rf_mlp_layer_ids=rf_mlp_layer_ids,
        )
        capsules = [
            build_layer_mlp_capsule(model.layers[i]) for i in sorted(self.rf_layer_ids)
        ]
        self.rf = RuntimeFusion(capsules)
        self.overrides: Dict[int, RuntimeFusionMlpLayerOverride] = {
            i: RuntimeFusionMlpLayerOverride(
                model.layers[i],
                self.rf,
                capsule_name=capsule_name_for_layer(i),
            )
            for i in self.rf_layer_ids
        }

    def inspect(self) -> Dict[str, Any]:
        return {
            "hybrid": "HybridModelOverride",
            "num_layers": len(self.model.layers),
            "rf_mlp_layer_ids": sorted(self.rf_layer_ids),
            "rf": self.rf.inspect(),
            "overrides": {str(i): o.inspect() for i, o in self.overrides.items()},
        }

    def forward(
        self,
        hidden: np.ndarray,
        *,
        attn_meta: Optional[EngineAttentionMeta] = None,
        rf_meta: Optional[Union[StepMeta, Mapping[str, Any]]] = None,
        force_engine_mlp: bool = False,
    ) -> HybridForwardResult:
        h = hidden
        layer_results: List[LayerForwardResult] = []
        rf_ids: List[int] = []
        eng_ids: List[int] = []

        for layer in self.model.layers:
            lid = layer.layer_id
            if (not force_engine_mlp) and lid in self.overrides:
                # Per-layer enable unless caller force-skips all.
                meta = StepMeta.from_mapping(rf_meta)
                if meta.force_skip_all:
                    step_meta: Mapping[str, Any] = {"force_skip_all": True}
                else:
                    step_meta = {
                        "enabled": {capsule_name_for_layer(lid)},
                        "block_tables": meta.block_tables,
                        "radix_hit_mask": meta.radix_hit_mask,
                        "sm_budget": meta.sm_budget,
                        "extras": meta.extras,
                    }
                result = self.overrides[lid].forward(
                    h, attn_meta=attn_meta, rf_meta=step_meta
                )
                h = result.hidden
                layer_results.append(result)
                if result.used_rf_mlp:
                    rf_ids.append(lid)
                else:
                    eng_ids.append(lid)
            else:
                h = layer.forward_engine_full(h, attn_meta)
                eng_ids.append(lid)
                layer_results.append(
                    LayerForwardResult(
                        hidden=h, rf=None, used_rf_mlp=False, layer_id=lid
                    )
                )

        return HybridForwardResult(
            hidden=h,
            layer_results=layer_results,
            rf_layer_ids=rf_ids,
            engine_mlp_layer_ids=eng_ids,
        )
