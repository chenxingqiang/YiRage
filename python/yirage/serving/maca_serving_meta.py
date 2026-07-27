# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S15: MACA serving RF meta bridge (64-warp, C500 SM budget, block dims).

Bridges MetaX MACA hardware constraints into :class:`~yirage.serving.runtime_fusion.StepMeta`
``extras`` so RuntimeFusion hooks align with vLLM-metax / MACA superoptimize paths.

Cloud CPU pytest validates meta shape + full-layer torch hybrid; MetaX VM exercises
``backend=yirage_maca`` tier separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

# C500 defaults (aligned with ``python/yirage/backends/maca/config.py``).
MACA_SERVING_WARP_SIZE = 64
MACA_SERVING_SM_COUNT_C500 = 104
MACA_SERVING_SHARED_MEM_PER_BLOCK = 65536
MACA_SERVING_DEFAULT_BLOCK_DIM: Tuple[int, int, int] = (256, 1, 1)
MACA_SERVING_DEFAULT_RESERVED_AUX_SMS = 8


def validate_maca_block_dim(block_dim: Sequence[int]) -> None:
    """Block x-dimension must be a multiple of MACA warp size (64)."""
    if len(block_dim) < 1:
        raise ValueError(f"block_dim must be non-empty, got {block_dim!r}")
    bx = int(block_dim[0])
    if bx % MACA_SERVING_WARP_SIZE != 0:
        raise ValueError(
            f"MACA block_dim[0]={bx} must be a multiple of warp_size="
            f"{MACA_SERVING_WARP_SIZE}"
        )


@dataclass(frozen=True)
class MacaServingRfSpec:
    """MACA hardware + search constraints for one RF ``step`` (no MACA SDK import)."""

    warp_size: int = MACA_SERVING_WARP_SIZE
    sm_count: int = MACA_SERVING_SM_COUNT_C500
    shared_mem_per_block: int = MACA_SERVING_SHARED_MEM_PER_BLOCK
    block_dim: Tuple[int, int, int] = MACA_SERVING_DEFAULT_BLOCK_DIM
    reserved_aux_sms: int = MACA_SERVING_DEFAULT_RESERVED_AUX_SMS

    def __post_init__(self) -> None:
        validate_maca_block_dim(self.block_dim)

    def maca_serving_payload(self) -> Dict[str, Any]:
        return {
            "warp_size": int(self.warp_size),
            "sm_count": int(self.sm_count),
            "shared_mem_per_block": int(self.shared_mem_per_block),
            "block_dim": list(self.block_dim),
            "backend_hint": "maca",
        }

    def as_rf_meta(
        self,
        *,
        sm_budget: Optional[int] = None,
        extras: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build StepMeta mapping with ``maca_serving`` + SM quota extras."""
        merged_extras: Dict[str, Any] = {
            "total_sms": int(self.sm_count),
            "reserved_aux_sms": int(self.reserved_aux_sms),
            "maca_serving": self.maca_serving_payload(),
        }
        if extras:
            merged_extras.update(dict(extras))
        out: Dict[str, Any] = {"extras": merged_extras}
        if sm_budget is not None:
            out["sm_budget"] = int(sm_budget)
        return out


def attach_maca_serving_to_step_meta(
    meta: Mapping[str, Any],
    *,
    spec: Optional[MacaServingRfSpec] = None,
) -> Dict[str, Any]:
    """Merge ``maca_serving`` into an existing RF meta dict (returns new dict)."""
    spec = spec or MacaServingRfSpec()
    out = dict(meta)
    extras = dict(out.get("extras") or {})
    payload = spec.maca_serving_payload()
    extras.setdefault("total_sms", spec.sm_count)
    extras.setdefault("reserved_aux_sms", spec.reserved_aux_sms)
    extras["maca_serving"] = payload
    out["extras"] = extras
    return out


def maca_serving_present(meta: Optional[Mapping[str, Any]]) -> bool:
    if not meta:
        return False
    extras = dict(meta.get("extras") or {})
    return "maca_serving" in extras


def inspect_maca_serving_meta(meta: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not meta:
        return None
    extras = dict(meta.get("extras") or {})
    payload = extras.get("maca_serving")
    return dict(payload) if isinstance(payload, Mapping) else None
