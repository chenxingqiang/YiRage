# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S7: multi-Capsule pipeline orchestration for RuntimeFusion.step."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

from .capsule import FusionCapsule
from .runtime_fusion import RuntimeFusion, StepMeta


def split_mlp_gate_up_name(layer_id: int) -> str:
    return f"mlp_layer_{layer_id}_gate_up"


def split_mlp_down_name(layer_id: int) -> str:
    return f"mlp_layer_{layer_id}_down"


def split_mlp_pipeline_names(layer_id: int) -> Tuple[str, str]:
    return split_mlp_gate_up_name(layer_id), split_mlp_down_name(layer_id)


def resolve_capsule_pipeline(
    capsules: Sequence[FusionCapsule],
    meta: StepMeta,
) -> List[FusionCapsule]:
    """Return capsules in execution order for this step.

    Precedence:
    1. ``meta.pipeline`` explicit name list
    2. ``meta.extras['capsule_pipeline']``
    3. registration order on ``RuntimeFusion``
    """
    by_name = {c.name: c for c in capsules}
    pipeline = meta.pipeline
    if pipeline is None:
        extra = meta.extras.get("capsule_pipeline")
        if extra is not None:
            pipeline = tuple(extra)
    if pipeline is None:
        return list(capsules)
    ordered: List[FusionCapsule] = []
    for name in pipeline:
        cap = by_name.get(name)
        if cap is None:
            raise KeyError(f"capsule_pipeline references unknown capsule {name!r}")
        ordered.append(cap)
    return ordered


def build_split_mlp_runtime_fusion(
    layer,
    *,
    backend: Optional[str] = None,
) -> RuntimeFusion:
    """Build a 2-capsule RF pipeline (gate_up → down) for one decoder layer."""
    from .split_mlp_capsule import build_layer_split_mlp_capsules

    gate_up, down = build_layer_split_mlp_capsules(layer, backend=backend)
    return RuntimeFusion([gate_up, down])


def pipeline_meta_for_layer(layer_id: int, base: Optional[Mapping[str, Any]] = None) -> dict:
    """Step meta enabling a layer's split MLP pipeline in order."""
    gate, down = split_mlp_pipeline_names(layer_id)
    out = dict(base or {})
    out["enabled"] = {gate, down}
    out["pipeline"] = [gate, down]
    return out
