# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""RuntimeFusion: per-step capsule selection from engine meta."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set

from .capsule import FusionCapsule


@dataclass
class StepMeta:
    """Engine-provided (or test) meta for one RF.step.

    Selection rules (S1):
    - If ``enabled`` is non-empty, only those capsule names may run.
    - Else if ``disabled`` is non-empty, all registered except those may run.
    - Else all registered capsules run (default select-all for standalone smoke).
    - ``force_skip_all``: run nothing (engine owns the layer this step).
    """

    enabled: Optional[Set[str]] = None
    disabled: Optional[Set[str]] = None
    force_skip_all: bool = False
    # Forward-compatible slots for S4+ (not consumed in S1 execute path).
    block_tables: Any = None
    radix_hit_mask: Any = None
    sm_budget: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "StepMeta":
        if data is None:
            return cls()
        if isinstance(data, StepMeta):
            return data
        enabled = data.get("enabled")
        disabled = data.get("disabled")
        return cls(
            enabled=set(enabled) if enabled is not None else None,
            disabled=set(disabled) if disabled is not None else None,
            force_skip_all=bool(data.get("force_skip_all", False)),
            block_tables=data.get("block_tables"),
            radix_hit_mask=data.get("radix_hit_mask"),
            sm_budget=data.get("sm_budget"),
            extras=dict(data.get("extras") or {}),
        )

    def should_run(self, capsule_name: str) -> bool:
        if self.force_skip_all:
            return False
        if self.enabled is not None:
            return capsule_name in self.enabled
        if self.disabled is not None:
            return capsule_name not in self.disabled
        return True


@dataclass
class StepResult:
    """Outcome of :meth:`RuntimeFusion.step`."""

    outputs: Dict[str, Any]
    ran: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    meta: Optional[StepMeta] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ran": list(self.ran),
            "skipped": list(self.skipped),
            "output_keys": sorted(self.outputs.keys()),
            "force_skip_all": bool(self.meta.force_skip_all) if self.meta else False,
        }


class RuntimeFusion:
    """Dynamic fusion runtime: select/orchestrate FusionCapsules per step.

    Replaces legacy Mirage-style megakernel monopoly as the *product* identity.
    Persistence/worker backends remain optional capsule executors (not S1 scope).
    """

    def __init__(self, capsules: Optional[Sequence[FusionCapsule]] = None):
        self._capsules: List[FusionCapsule] = list(capsules or [])
        self._by_name: Dict[str, FusionCapsule] = {c.name: c for c in self._capsules}
        if len(self._by_name) != len(self._capsules):
            raise ValueError("FusionCapsule names must be unique within RuntimeFusion")

    def register(self, capsule: FusionCapsule) -> None:
        if capsule.name in self._by_name:
            raise ValueError(f"duplicate FusionCapsule name: {capsule.name!r}")
        self._capsules.append(capsule)
        self._by_name[capsule.name] = capsule

    @property
    def capsules(self) -> Sequence[FusionCapsule]:
        return tuple(self._capsules)

    def inspect(self) -> Dict[str, Any]:
        return {
            "runtime": "RuntimeFusion",
            "version": "s1",
            "capsules": [c.inspect() for c in self._capsules],
        }

    def step(
        self,
        inputs: Mapping[str, Any],
        meta: Optional[Mapping[str, Any]] = None,
    ) -> StepResult:
        """Run one serving step: select capsules from meta, execute in order.

        When a capsule is skipped, its outputs are not applied (engine retains
        responsibility for that fragment — identity on ``hidden`` for S1 MLP).
        """
        step_meta = StepMeta.from_mapping(meta)
        # Shallow copy so capsule outputs do not mutate caller unexpectedly.
        state: Dict[str, Any] = dict(inputs)
        ran: List[str] = []
        skipped: List[str] = []

        for cap in self._capsules:
            if step_meta.should_run(cap.name):
                out = cap.execute(state, meta=step_meta.extras)
                state.update(out)
                ran.append(cap.name)
            else:
                skipped.append(cap.name)

        return StepResult(outputs=state, ran=ran, skipped=skipped, meta=step_meta)
