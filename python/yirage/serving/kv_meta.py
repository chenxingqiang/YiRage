# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S4: bridge vLLM ``block_tables`` / seq lens → YiRage ``paged_kv_*`` meta.

Engine owns the KV pool; RuntimeFusion consumes a stable FlashInfer-style
indptr/indices view for Capsules that need non-contiguous KV addressing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[Sequence[int]], Sequence[int]]


@dataclass(frozen=True)
class PagedKvMeta:
    """FlashInfer-style paged KV addressing (matches PK meta buffer roles)."""

    paged_kv_indptr: np.ndarray  # int32 [batch + 1]
    paged_kv_indices: np.ndarray  # int32 [num_blocks_total]
    paged_kv_last_page_len: np.ndarray  # int32 [batch]
    page_size: int
    batch_size: int
    block_tables: np.ndarray  # int32 [batch, max_blocks] (normalized, -1 pad)
    seq_lens: np.ndarray  # int32 [batch]
    slot_mapping: Optional[np.ndarray] = None  # optional engine map

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "page_size": self.page_size,
            "batch_size": self.batch_size,
            "paged_kv_indptr": self.paged_kv_indptr.tolist(),
            "paged_kv_indices": self.paged_kv_indices.tolist(),
            "paged_kv_last_page_len": self.paged_kv_last_page_len.tolist(),
            "block_tables": self.block_tables.tolist(),
            "seq_lens": self.seq_lens.tolist(),
        }
        if self.slot_mapping is not None:
            d["slot_mapping"] = self.slot_mapping.tolist()
        return d

    def as_rf_extras(self) -> Dict[str, Any]:
        """Payload to merge into ``StepMeta.extras`` / capsule meta."""
        return {
            "paged_kv": self.to_dict(),
            "paged_kv_indptr": self.paged_kv_indptr,
            "paged_kv_indices": self.paged_kv_indices,
            "paged_kv_last_page_len": self.paged_kv_last_page_len,
            "page_size": self.page_size,
        }


def _as_int32_2d(block_tables: ArrayLike) -> np.ndarray:
    arr = np.asarray(block_tables, dtype=np.int32)
    if arr.ndim != 2:
        raise ValueError(f"block_tables must be rank-2 [B, max_blocks], got {arr.shape}")
    return arr


def _as_int32_1d(seq_lens: ArrayLike, batch: int) -> np.ndarray:
    arr = np.asarray(seq_lens, dtype=np.int32).reshape(-1)
    if arr.shape[0] != batch:
        raise ValueError(f"seq_lens length {arr.shape[0]} != batch {batch}")
    if np.any(arr < 0):
        raise ValueError("seq_lens must be non-negative")
    return arr


def _num_valid_blocks(row: np.ndarray) -> int:
    """Count leading valid physical block ids (padding = -1)."""
    n = 0
    for v in row.tolist():
        if int(v) < 0:
            break
        n += 1
    return n


def last_page_len_from_seq(seq_len: int, page_size: int) -> int:
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if seq_len <= 0:
        return 0
    rem = seq_len % page_size
    return page_size if rem == 0 else rem


def block_tables_to_paged_kv(
    block_tables: ArrayLike,
    seq_lens: ArrayLike,
    *,
    page_size: int = 16,
    slot_mapping: Optional[ArrayLike] = None,
) -> PagedKvMeta:
    """Convert vLLM-style block tables into ``paged_kv_indptr/indices/last_page_len``.

    ``block_tables[b, :]`` lists physical page ids for request ``b``, padded with ``-1``.
    ``seq_lens[b]`` is the logical KV length used to derive ``last_page_len``.
    """
    tables = _as_int32_2d(block_tables)
    batch, max_blocks = tables.shape
    lens = _as_int32_1d(seq_lens, batch)

    indptr = np.zeros(batch + 1, dtype=np.int32)
    indices_list = []
    last_page = np.zeros(batch, dtype=np.int32)

    for b in range(batch):
        nblocks = _num_valid_blocks(tables[b])
        expected = int((int(lens[b]) + page_size - 1) // page_size) if lens[b] > 0 else 0
        if nblocks < expected:
            raise ValueError(
                f"request {b}: block_tables has {nblocks} blocks but seq_len={lens[b]} "
                f"needs >= {expected} pages (page_size={page_size})"
            )
        # Use exactly the pages required by seq_len when present; else all valid.
        use = expected if lens[b] > 0 else nblocks
        chunk = tables[b, :use].astype(np.int32, copy=False)
        indices_list.append(chunk)
        indptr[b + 1] = indptr[b] + use
        last_page[b] = last_page_len_from_seq(int(lens[b]), page_size)

    indices = (
        np.concatenate(indices_list).astype(np.int32)
        if indices_list and indptr[-1] > 0
        else np.zeros(0, dtype=np.int32)
    )
    slot = None
    if slot_mapping is not None:
        slot = np.asarray(slot_mapping, dtype=np.int32)

    return PagedKvMeta(
        paged_kv_indptr=indptr,
        paged_kv_indices=indices,
        paged_kv_last_page_len=last_page,
        page_size=int(page_size),
        batch_size=batch,
        block_tables=tables,
        seq_lens=lens,
        slot_mapping=slot,
    )


def attach_paged_kv_to_step_meta(
    step_meta: Mapping[str, Any],
    *,
    block_tables: ArrayLike,
    seq_lens: ArrayLike,
    page_size: int = 16,
    slot_mapping: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """Return a new step-meta dict with ``block_tables`` + converted ``paged_kv`` extras."""
    paged = block_tables_to_paged_kv(
        block_tables,
        seq_lens,
        page_size=page_size,
        slot_mapping=slot_mapping,
    )
    out = dict(step_meta)
    out["block_tables"] = paged.block_tables
    extras = dict(out.get("extras") or {})
    extras.update(paged.as_rf_extras())
    out["extras"] = extras
    return out
