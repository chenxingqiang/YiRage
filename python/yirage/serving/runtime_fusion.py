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

    Selection rules:
    - If ``enabled`` is non-empty, only those capsule names may run.
    - Else if ``disabled`` is non-empty, all registered except those may run.
    - Else all registered capsules run (default select-all for standalone smoke).
    - ``force_skip_all``: run nothing (engine owns the layer this step).

    S4 KV bridge: when ``block_tables`` + ``seq_lens`` are set, RuntimeFusion.step
    converts them into ``extras['paged_kv_*']`` via :mod:`yirage.serving.kv_meta`.
    """

    enabled: Optional[Set[str]] = None
    disabled: Optional[Set[str]] = None
    force_skip_all: bool = False
    block_tables: Any = None
    seq_lens: Any = None
    page_size: int = 16
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
            seq_lens=data.get("seq_lens"),
            page_size=int(data.get("page_size", 16)),
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

    def with_paged_kv_bridge(self) -> "StepMeta":
        """Return a copy with ``paged_kv_*`` filled from ``block_tables`` when possible."""
        if self.block_tables is None or self.seq_lens is None:
            return self
        from .kv_meta import block_tables_to_paged_kv

        paged = block_tables_to_paged_kv(
            self.block_tables,
            self.seq_lens,
            page_size=self.page_size,
            slot_mapping=self.extras.get("slot_mapping"),
        )
        extras = dict(self.extras)
        extras.update(paged.as_rf_extras())
        return StepMeta(
            enabled=set(self.enabled) if self.enabled is not None else None,
            disabled=set(self.disabled) if self.disabled is not None else None,
            force_skip_all=self.force_skip_all,
            block_tables=paged.block_tables,
            seq_lens=paged.seq_lens,
            page_size=paged.page_size,
            radix_hit_mask=self.radix_hit_mask,
            sm_budget=self.sm_budget,
            extras=extras,
        )


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
            "version": "s4",
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
        step_meta = StepMeta.from_mapping(meta).with_paged_kv_bridge()
        # Shallow copy so capsule outputs do not mutate caller unexpectedly.
        state: Dict[str, Any] = dict(inputs)
        ran: List[str] = []
        skipped: List[str] = []

        for cap in self._capsules:
            if step_meta.should_run(cap.name):
                # Pass full step meta extras (includes S4 paged_kv_* when bridged).
                out = cap.execute(state, meta=step_meta.extras)
                state.update(out)
                ran.append(cap.name)
            else:
                skipped.append(cap.name)

        return StepResult(outputs=state, ran=ran, skipped=skipped, meta=step_meta)
