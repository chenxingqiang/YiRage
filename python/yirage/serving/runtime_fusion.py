# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""RuntimeFusion: per-step capsule selection from engine meta."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from .capsule import FusionCapsule
from .sm_budget import SmStepAllocation, capsule_sm_cost, resolve_sm_worker_quota


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

    S5 SM budget: ``sm_budget`` + ``extras['total_sms']`` / ``reserved_aux_sms`` cap
    capsule launches; over-budget capsules are skipped (engine owns that fragment).

    S6 Radix: ``radix_hit_mask`` (bool [batch]) from SGLang RadixAttention — all-hit
    skips the capsule; partial hit shrinks MLP to miss rows only (hits pass-through).

    S7 Pipeline: ``pipeline`` explicit capsule execution order (gate_up → down).
    """

    enabled: Optional[Set[str]] = None
    disabled: Optional[Set[str]] = None
    pipeline: Optional[Tuple[str, ...]] = None
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
        pipeline = data.get("pipeline")
        return cls(
            enabled=set(enabled) if enabled is not None else None,
            disabled=set(disabled) if disabled is not None else None,
            pipeline=tuple(pipeline) if pipeline is not None else None,
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
            pipeline=self.pipeline,
            force_skip_all=self.force_skip_all,
            block_tables=paged.block_tables,
            seq_lens=paged.seq_lens,
            page_size=paged.page_size,
            radix_hit_mask=self.radix_hit_mask,
            sm_budget=self.sm_budget,
            extras=extras,
        )

    def with_radix_bridge(self, *, batch_size: Optional[int] = None) -> "StepMeta":
        """Return a copy with normalized ``radix_hit`` in extras when mask is set."""
        if self.radix_hit_mask is None:
            return self
        from .radix_meta import parse_radix_hit_mask

        bs = batch_size
        if bs is None and self.seq_lens is not None:
            import numpy as np

            bs = int(np.asarray(self.seq_lens).reshape(-1).shape[0])
        radix = parse_radix_hit_mask(self.radix_hit_mask, batch_size=bs)
        if radix is None:
            return self
        extras = dict(self.extras)
        extras.update(radix.as_rf_extras())
        return StepMeta(
            enabled=set(self.enabled) if self.enabled is not None else None,
            disabled=set(self.disabled) if self.disabled is not None else None,
            pipeline=self.pipeline,
            force_skip_all=self.force_skip_all,
            block_tables=self.block_tables,
            seq_lens=self.seq_lens,
            page_size=self.page_size,
            radix_hit_mask=radix.hit_mask,
            sm_budget=self.sm_budget,
            extras=extras,
        )


@dataclass
class StepResult:
    """Outcome of :meth:`RuntimeFusion.step`."""

    outputs: Dict[str, Any]
    ran: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    skipped_radix: List[str] = field(default_factory=list)
    meta: Optional[StepMeta] = None
    sm_allocation: Optional[SmStepAllocation] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "ran": list(self.ran),
            "skipped": list(self.skipped),
            "skipped_radix": list(self.skipped_radix),
            "output_keys": sorted(self.outputs.keys()),
            "force_skip_all": bool(self.meta.force_skip_all) if self.meta else False,
        }
        if self.sm_allocation is not None:
            d["sm_allocation"] = self.sm_allocation.to_dict()
        return d


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
            "version": "s23",
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
        radix = None
        if step_meta.radix_hit_mask is not None:
            from .radix_meta import infer_batch_size_from_hidden, parse_radix_hit_mask

            bs = infer_batch_size_from_hidden(inputs.get("hidden"))
            step_meta = step_meta.with_radix_bridge(batch_size=bs)
            radix = parse_radix_hit_mask(step_meta.radix_hit_mask, batch_size=bs)
        quota = resolve_sm_worker_quota(
            sm_budget=step_meta.sm_budget,
            extras=step_meta.extras,
        )
        remaining_sms = quota.capsule_budget_sms
        sm_alloc = SmStepAllocation(quota=quota, remaining_sms=remaining_sms)

        # Shallow copy so capsule outputs do not mutate caller unexpectedly.
        state: Dict[str, Any] = dict(inputs)
        ran: List[str] = []
        skipped: List[str] = []
        skipped_radix: List[str] = []

        from .capsule_orchestration import resolve_capsule_pipeline

        ordered_caps = resolve_capsule_pipeline(self._capsules, step_meta)
        for cap in ordered_caps:
            if not step_meta.should_run(cap.name):
                skipped.append(cap.name)
                continue
            if radix is not None and radix.skip_capsule_entirely():
                skipped.append(cap.name)
                skipped_radix.append(cap.name)
                continue
            cost = capsule_sm_cost(cap)
            if cost > remaining_sms:
                skipped.append(cap.name)
                sm_alloc.skipped_budget.append(cap.name)
                continue
            exec_meta = dict(step_meta.extras)
            if radix is not None and radix.needs_shrink():
                exec_meta.update(radix.as_rf_extras())
            out = cap.execute(state, meta=exec_meta)
            state.update(out)
            ran.append(cap.name)
            sm_alloc.ran.append((cap.name, cost))
            remaining_sms -= cost

        sm_alloc.remaining_sms = remaining_sms
        return StepResult(
            outputs=state,
            ran=ran,
            skipped=skipped,
            skipped_radix=skipped_radix,
            meta=step_meta,
            sm_allocation=sm_alloc,
        )
