# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S5: SM worker quota for RuntimeFusion co-residence with Sampler/NCCL.

Capsule launches must not monopolize all SMs. Engines reserve an aux budget for
sampling, NCCL, and multimodal side streams; RF only schedules within the
remaining capsule budget. When a capsule's ``sm_cost`` would exceed the remaining
budget, RF skips it and the engine owns that fragment (no hang).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Generic defaults (engines override via StepMeta / extras). Not tied to a vendor.
DEFAULT_TOTAL_SMS = 108
DEFAULT_RESERVED_AUX_SMS = 8
DEFAULT_CAPSULE_SM_COST = 1


@dataclass(frozen=True)
class SmWorkerQuota:
    """Per-worker SM split: RF capsules vs aux (Sampler/NCCL/multimodal)."""

    total_sms: int
    reserved_aux_sms: int
    capsule_budget_sms: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_sms": self.total_sms,
            "reserved_aux_sms": self.reserved_aux_sms,
            "capsule_budget_sms": self.capsule_budget_sms,
        }


@dataclass
class SmStepAllocation:
    """Record of SM use for one :meth:`RuntimeFusion.step`."""

    quota: SmWorkerQuota
    remaining_sms: int
    ran: List[Tuple[str, int]] = field(default_factory=list)
    skipped_budget: List[str] = field(default_factory=list)

    @property
    def used_sms(self) -> int:
        return sum(cost for _, cost in self.ran)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quota": self.quota.to_dict(),
            "remaining_sms": self.remaining_sms,
            "used_sms": self.used_sms,
            "ran": [{"name": n, "sm_cost": c} for n, c in self.ran],
            "skipped_budget": list(self.skipped_budget),
        }


def resolve_sm_worker_quota(
    *,
    total_sms: Optional[int] = None,
    reserved_aux_sms: Optional[int] = None,
    sm_budget: Optional[int] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> SmWorkerQuota:
    """Build worker SM quota.

    Precedence for numbers:
    - Explicit kwargs
    - ``extras['total_sms']`` / ``extras['reserved_aux_sms']``
    - Module defaults

    ``sm_budget`` (also ``StepMeta.sm_budget``) caps the capsule budget but must
    leave ``reserved_aux_sms`` untouched (never schedule capsules into aux SMs).
    """
    ex = extras or {}
    total = int(total_sms if total_sms is not None else ex.get("total_sms", DEFAULT_TOTAL_SMS))
    reserved = int(
        reserved_aux_sms
        if reserved_aux_sms is not None
        else ex.get("reserved_aux_sms", DEFAULT_RESERVED_AUX_SMS)
    )
    if total <= 0:
        raise ValueError(f"total_sms must be > 0, got {total}")
    if reserved < 0 or reserved >= total:
        raise ValueError(f"reserved_aux_sms={reserved} invalid for total_sms={total}")

    max_capsule = total - reserved
    if sm_budget is None and "sm_budget" in ex:
        sm_budget = ex.get("sm_budget")
    if sm_budget is None:
        capsule_budget = max_capsule
    else:
        capsule_budget = int(sm_budget)
        if capsule_budget < 0:
            raise ValueError(f"sm_budget must be >= 0, got {capsule_budget}")
        if capsule_budget > max_capsule:
            raise ValueError(
                f"sm_budget={capsule_budget} would eat reserved aux "
                f"(max capsule budget={max_capsule} = total_sms={total} - reserved_aux_sms={reserved})"
            )
    return SmWorkerQuota(
        total_sms=total,
        reserved_aux_sms=reserved,
        capsule_budget_sms=capsule_budget,
    )


def capsule_sm_cost(capsule: Any, default: int = DEFAULT_CAPSULE_SM_COST) -> int:
    """Read per-capsule SM cost from ``plan.extras['sm_cost']`` or ``sm_cost`` attr."""
    if hasattr(capsule, "sm_cost"):
        try:
            attr = getattr(capsule, "sm_cost")
            if callable(attr):
                attr = attr()
            if attr is not None:
                return max(0, int(attr))
        except Exception:
            pass
    plan = getattr(capsule, "plan", None)
    if plan is not None:
        extras = getattr(plan, "extras", None) or {}
        if "sm_cost" in extras:
            return max(0, int(extras["sm_cost"]))
    return max(0, int(default))


def assert_aux_coresidence(allocation: SmStepAllocation) -> None:
    """Contract: capsule SMs never consume reserved aux (Sampler/NCCL room)."""
    q = allocation.quota
    if q.reserved_aux_sms <= 0:
        raise AssertionError("aux co-residence requires reserved_aux_sms > 0")
    if allocation.used_sms > q.capsule_budget_sms:
        raise AssertionError(
            f"used_sms={allocation.used_sms} exceeds capsule_budget_sms={q.capsule_budget_sms}"
        )
    if allocation.used_sms + allocation.remaining_sms != q.capsule_budget_sms:
        raise AssertionError(
            "used+remaining must equal capsule_budget_sms "
            f"(used={allocation.used_sms}, remaining={allocation.remaining_sms}, "
            f"budget={q.capsule_budget_sms})"
        )
    # Aux slice is definitionally outside capsule_budget; never scheduled here.
    if q.capsule_budget_sms + q.reserved_aux_sms > q.total_sms:
        raise AssertionError("quota invariant broken: capsule+aux > total")
