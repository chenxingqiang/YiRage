# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Hardware-to-search-space derivation helpers.

This module is intentionally dependency-free (no torch, no C++ core, no
relative imports beyond ``yirage.hardware.chip_arch``) so it can be imported
anywhere — including test environments that do not have the full stack
installed.

The key exported symbol is :func:`chip_arch_to_search_config`, which converts
a :class:`~yirage.hardware.chip_arch.ChipArchitecture` object into the
``griddims`` / ``blockdims`` / ``fmaps`` / ``franges`` dict that
:py:meth:`~yirage.kernel.graph.KNGraph.superoptimize` consumes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Compile-mode constants (mirrors CompileMode enum in unified.py so this
#: module needs no import from that file).
MODE_FAST = "FAST"
MODE_SUPEROPTIMIZE = "SUPEROPTIMIZE"
MODE_AGGRESSIVE = "AGGRESSIVE"
MODE_RL_GUIDED = "RL_GUIDED"
MODE_MLIR_ONLY = "MLIR_ONLY"


def chip_arch_to_search_config(
    chip_arch: Optional[Any],
    mode: str = MODE_SUPEROPTIMIZE,
) -> Dict[str, Any]:
    """
    Derive a muGraph search-space configuration from a chip architecture.

    Priority for every dimension:

    1. ``chip_arch.search_config_overrides[key]`` — explicit per-chip tuning.
    2. Values *derived* from ``chip_arch.compute`` (SM count, thread budget,
       shared-memory size, warp size).
    3. Hard-coded fallback values (identical to the original hard-coded defaults
       so existing behaviour is preserved when no chip is detected).

    Args:
        chip_arch:
            A :class:`~yirage.hardware.chip_arch.ChipArchitecture` instance,
            or *None* to use the hard-coded fallbacks.
        mode:
            One of ``"FAST"``, ``"SUPEROPTIMIZE"``, ``"AGGRESSIVE"``,
            ``"RL_GUIDED"``, or ``"MLIR_ONLY"``.  Determines how aggressively
            the search space is explored.

    Returns:
        A ``dict`` with keys ``"griddims"``, ``"blockdims"``, and optionally
        ``"fmaps"`` and ``"franges"``.
    """

    def _override(key: str, fallback: Any) -> Any:
        if chip_arch is not None:
            ov = getattr(chip_arch, "search_config_overrides", {}) or {}
            if key in ov:
                return ov[key]
        return fallback

    # ----------------------------------------------------------------- helpers

    def _derive_griddims() -> Optional[List[Tuple[int, int, int]]]:
        if chip_arch is None:
            return None
        compute = getattr(chip_arch, "compute", None)
        sms = getattr(compute, "num_compute_units", 0) if compute else 0
        if sms <= 0:
            return None
        dims = sorted({1, max(1, sms // 8), max(1, sms // 4), max(1, sms // 2)})
        return [(d, 1, 1) for d in dims]

    def _derive_blockdims() -> Optional[List[Tuple[int, int, int]]]:
        if chip_arch is None:
            return None
        compute = getattr(chip_arch, "compute", None)
        if compute is None:
            return None
        max_t = getattr(compute, "max_threads_per_block", 0)
        warp = getattr(compute, "warp_size", 32) or 32
        if max_t <= 0:
            return None
        candidates = [warp, warp * 2, warp * 4, warp * 8]
        return [(t, 1, 1) for t in candidates if t <= max_t]

    def _derive_franges() -> Optional[List[int]]:
        if chip_arch is None:
            return None
        compute = getattr(chip_arch, "compute", None)
        smem_kb = getattr(compute, "shared_mem_per_block_kb", 0) if compute else 0
        if smem_kb <= 0:
            return None
        if smem_kb >= 128:
            return [4, 8, 16, 32, 64]
        elif smem_kb >= 64:
            return [4, 8, 16, 32]
        else:
            return [4, 8, 16]

    # ----------------------------------------------------------------- FAST

    if mode == MODE_FAST:
        return {
            "griddims": _override("griddims", [(1, 1, 1)]),
            "blockdims": _override("blockdims", [(128, 1, 1)]),
        }

    # ----------------------------------------------------------------- AGGRESSIVE

    if mode == MODE_AGGRESSIVE:
        return {
            "griddims": _override(
                "griddims",
                _derive_griddims() or [(1, 1, 1), (2, 1, 1), (4, 1, 1), (8, 1, 1)],
            ),
            "blockdims": _override(
                "blockdims",
                _derive_blockdims()
                or [(64, 1, 1), (128, 1, 1), (256, 1, 1), (512, 1, 1)],
            ),
            "fmaps": _override("fmaps", [1, 2, 4]),
            "franges": _override("franges", _derive_franges() or [4, 8, 16, 32]),
        }

    # ----------------------------------------------------------------- SUPEROPTIMIZE / RL_GUIDED / default

    return {
        "griddims": _override(
            "griddims",
            _derive_griddims() or [(1, 1, 1), (2, 1, 1), (4, 1, 1)],
        ),
        "blockdims": _override(
            "blockdims",
            _derive_blockdims() or [(128, 1, 1), (256, 1, 1)],
        ),
        "fmaps": _override("fmaps", [1, 2]),
        "franges": _override("franges", _derive_franges() or [4, 8, 16]),
    }


__all__ = [
    "chip_arch_to_search_config",
    "MODE_FAST",
    "MODE_SUPEROPTIMIZE",
    "MODE_AGGRESSIVE",
    "MODE_RL_GUIDED",
    "MODE_MLIR_ONLY",
]
