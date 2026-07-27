# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S6: SGLang RadixAttention hit meta → skip/shrink FusionCapsule work.

When the engine reports prefix cache hits via ``radix_hit_mask`` (bool per batch
row), RuntimeFusion may:

- **Skip** the capsule entirely when every row hit (engine/cache owns MLP output).
- **Shrink** execution to miss rows only when some rows hit (pass-through on hits).

Hit rows keep ``hidden`` unchanged through the MLP stage (identity).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TypeVar, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[bool], Sequence[int]]


@dataclass(frozen=True)
class RadixHitMeta:
    """Normalized per-batch Radix prefix hit view."""

    hit_mask: np.ndarray  # bool [batch]; True = skip MLP for this row
    batch_size: int

    @property
    def all_hit(self) -> bool:
        return bool(self.hit_mask.size > 0 and np.all(self.hit_mask))

    @property
    def any_hit(self) -> bool:
        return bool(np.any(self.hit_mask))

    @property
    def any_miss(self) -> bool:
        return bool(np.any(~self.hit_mask))

    def active_row_indices(self) -> np.ndarray:
        return np.where(~self.hit_mask)[0].astype(np.int64)

    def skip_capsule_entirely(self) -> bool:
        return self.all_hit

    def needs_shrink(self) -> bool:
        return self.any_hit and self.any_miss

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "radix_hit_mask": self.hit_mask.tolist(),
            "all_hit": self.all_hit,
            "any_hit": self.any_hit,
        }

    def as_rf_extras(self) -> Dict[str, Any]:
        return {
            "radix_hit": self.to_dict(),
            "radix_hit_mask": self.hit_mask,
        }


def parse_radix_hit_mask(
    mask: Optional[ArrayLike],
    *,
    batch_size: Optional[int] = None,
) -> Optional[RadixHitMeta]:
    """Parse engine ``radix_hit_mask`` into :class:`RadixHitMeta`.

    Accepts bool/int 1d arrays or nested lists. ``None`` → no Radix info.
    """
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"radix_hit_mask must be rank-1 [batch], got shape={arr.shape}")
    hit = arr.astype(bool, copy=False)
    bs = int(batch_size if batch_size is not None else hit.shape[0])
    if hit.shape[0] != bs:
        raise ValueError(f"radix_hit_mask length {hit.shape[0]} != batch_size {bs}")
    return RadixHitMeta(hit_mask=hit, batch_size=bs)


def infer_batch_size_from_hidden(hidden: Any) -> int:
    if hasattr(hidden, "shape"):
        shape = tuple(hidden.shape)
        if len(shape) == 1:
            return 1
        if len(shape) >= 2:
            return int(shape[0])
    arr = np.asarray(hidden)
    if arr.ndim == 1:
        return 1
    return int(arr.shape[0])


def attach_radix_to_step_meta(
    meta: Mapping[str, Any],
    *,
    radix_hit_mask: ArrayLike,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Merge Radix hit fields into a step-meta mapping (for engine plugins)."""
    out = dict(meta)
    out["radix_hit_mask"] = radix_hit_mask
    bs = batch_size
    if bs is None and "seq_lens" in out:
        bs = len(np.asarray(out["seq_lens"]).reshape(-1))
    radix = parse_radix_hit_mask(radix_hit_mask, batch_size=bs)
    if radix is not None:
        extras = dict(out.get("extras") or {})
        extras.update(radix.as_rf_extras())
        out["extras"] = extras
    return out


T = TypeVar("T")


def apply_radix_shrink(
    hidden: T,
    radix: RadixHitMeta,
    compute_active: Callable[[T], T],
) -> T:
    """Run ``compute_active`` on miss rows; pass-through hit rows unchanged."""
    if radix.all_hit:
        return hidden
    if not radix.any_hit:
        return compute_active(hidden)

    active = radix.active_row_indices()
    if hasattr(hidden, "shape") and hasattr(hidden, "index_copy"):
        import torch

        if isinstance(hidden, torch.Tensor):
            out = hidden.clone()
            active_t = torch.as_tensor(active, device=hidden.device)
            out.index_copy_(0, active_t, compute_active(hidden.index_select(0, active_t)))
            return out

    hidden_np = np.asarray(hidden)
    out = np.array(hidden_np, copy=True)
    out[active] = np.asarray(compute_active(hidden_np[active]))
    return out  # type: ignore[return-value]
