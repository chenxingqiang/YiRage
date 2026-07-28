# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""MACA yirage.core execution helpers for RuntimeFusion serving (S15/S16).

Full ``backend=yirage_maca`` MLP capsules require MetaX VM build
(``YIRAGE_BACKEND=maca pip install -e .``). Cloud CPU cert uses
:mod:`yirage.serving.maca_serving_e2e` torch meta bridge instead.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .torch_exec import require_torch, to_torch
from .yirage_exec import (
    YirageMlpCompileArtifacts,
    YirageServingMlpRunner,
    build_down_matmul_seed_graph,
    build_gate_up_seed_graph,
    is_yirage_core_available,
    require_yirage_core,
    superoptimize_kwargs,
)


def is_yirage_maca_available() -> bool:
    """True when ``yirage.core`` is built with MACA backend enabled."""
    if os.environ.get("YIRAGE_SKIP_NATIVE") == "1":
        return False
    if not is_yirage_core_available():
        return False
    if os.environ.get("YIRAGE_BACKEND", "").lower() == "maca":
        return True
    try:
        from yirage.backends.api import is_backend_available

        return bool(is_backend_available("maca"))
    except Exception:
        return False


def require_yirage_maca() -> None:
    if not is_yirage_maca_available():
        raise RuntimeError(
            "yirage_maca serving tier requires YIRAGE_BACKEND=maca and built yirage.core "
            "on a MetaX GPU host. Use maca_serving_e2e torch meta bridge on CPU CI."
        )


def apply_serving_maca_search_tractability() -> None:
    """Set env for tractable MACA superoptimize during serving smoke."""
    os.environ.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    os.environ.setdefault("YIRAGE_MACA_USE_RAY", "0")


def maca_superoptimize_kwargs(*, quick: bool = True) -> Dict[str, Any]:
    """Superoptimize kwargs aligned with MACA 64-warp search (MetaX VM)."""
    require_yirage_maca()
    try:
        from yirage.backends.maca.config import resolve_maca_search_config

        cfg = resolve_maca_search_config(quick=quick)
        grid = cfg.get("grid_dims_to_explore") or [(4, 1, 1)]
        block = cfg.get("block_dims_to_explore") or [(256, 1, 1)]
        franges = cfg.get("franges_to_explore") or [8]
    except Exception:
        grid = [(4, 1, 1)]
        block = [(256, 1, 1)]
        franges = [8]
    base = superoptimize_kwargs(quick=quick)
    return {
        **base,
        "backend": "maca",
        "griddims": [grid[0]],
        "blockdims": [block[0]],
        "franges": [franges[0]],
    }


def superoptimize_down_matmul_maca(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
    quick: bool = True,
):
    """Superoptimize down-projection matmul on MACA backend."""
    require_yirage_maca()
    apply_serving_maca_search_tractability()
    graph = build_down_matmul_seed_graph(
        hidden_size,
        intermediate_size,
        dtype_name=dtype_name,
    )
    optimized = graph.superoptimize(**maca_superoptimize_kwargs(quick=quick))
    if optimized is None:
        raise RuntimeError(
            f"MACA superoptimize found 0 valid µGraphs for down matmul "
            f"(H={hidden_size}, I={intermediate_size})"
        )
    return optimized


@dataclass
class MacaSuperoptimizeTiming:
    elapsed_s: float
    hidden_size: int
    intermediate_size: int
    backend: str = "maca"


def bench_superoptimize_down_matmul_maca(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
    quick: bool = True,
) -> Tuple[Any, MacaSuperoptimizeTiming]:
    t0 = time.perf_counter()
    opt = superoptimize_down_matmul_maca(
        hidden_size,
        intermediate_size,
        dtype_name=dtype_name,
        quick=quick,
    )
    elapsed = time.perf_counter() - t0
    return opt, MacaSuperoptimizeTiming(
        elapsed_s=elapsed,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


class YirageMacaServingMlpRunner(YirageServingMlpRunner):
    """Hybrid MLP: yirage.core gate_up + MACA superoptimized down matmul."""

    def __init__(
        self,
        *,
        rms_weight: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        eps: float = 1e-6,
        device: Optional[str] = None,
        dtype_name: str = "float32",
        quick_superopt: bool = True,
    ):
        require_yirage_maca()
        require_torch()
        import torch

        self.eps = float(eps)
        self.dtype_name = dtype_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.rms_weight = to_torch(rms_weight, device=self._device, dtype=torch.float32)
        self.w_gate = to_torch(w_gate, device=self._device, dtype=torch.float32)
        self.w_up = to_torch(w_up, device=self._device, dtype=torch.float32)
        self.w_down = to_torch(w_down, device=self._device, dtype=torch.float32)

        h = self.w_gate.shape[0]
        i = self.w_gate.shape[1]
        self.hidden_size = h
        self.intermediate_size = i

        self._w_gate_up = torch.cat([self.w_gate, self.w_up], dim=1)
        self._gate_up_graph = build_gate_up_seed_graph(h, i, dtype_name=dtype_name)
        self._down_optimized, timing = bench_superoptimize_down_matmul_maca(
            h,
            i,
            dtype_name=dtype_name,
            quick=quick_superopt,
        )
        self.superopt_elapsed_s = timing.elapsed_s
        from .yirage_exec import _yr_dtype

        self._yr_dtype = _yr_dtype(dtype_name)

    @property
    def artifacts(self) -> YirageMlpCompileArtifacts:
        return YirageMlpCompileArtifacts(
            gate_up_graph=self._gate_up_graph,
            down_optimized=self._down_optimized,
            superopt_elapsed_s=self.superopt_elapsed_s,
        )


def inspect_maca_serving_yirage_tier() -> Dict[str, Any]:
    return {
        "yirage_maca_available": is_yirage_maca_available(),
        "yirage_backend_env": os.environ.get("YIRAGE_BACKEND"),
    }
