"""Shared helpers for MACA demos and benchmarks."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from yirage.maca_config import MACA_WARP_SIZE, resolve_maca_search_config


def apply_maca_demo_env() -> None:
    """Default tractable MACA search for smoke demos (override with env)."""
    os.environ.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    os.environ.setdefault("YIRAGE_MACA_SKIP_PROFILE", "1")
    os.environ.setdefault("MACA_PATH", "/opt/maca")


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def maca_search_kwargs(*, quick: Optional[bool] = None) -> Dict[str, Any]:
    cfg = resolve_maca_search_config(quick=quick)
    return {
        "griddims": cfg.get("grid_dims_to_explore"),
        "blockdims": cfg.get("block_dims_to_explore"),
        "fmaps": cfg.get("fmaps_to_explore"),
        "franges": cfg.get("franges_to_explore"),
    }


def benchmark_callable(
    fn,
    *,
    warmup: int = 10,
    iters: int = 50,
    device: Optional[torch.device] = None,
) -> float:
    for _ in range(warmup):
        fn()
    if device is not None:
        sync_device(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device is not None:
        sync_device(device)
    return (time.perf_counter() - start) / iters


def benchmark_mugraph(
    graph,
    inputs: Sequence[torch.Tensor],
    *,
    warmup: int = 10,
    iters: int = 50,
) -> Optional[float]:
    device = inputs[0].device

    def _run():
        out = graph(inputs=list(inputs))
        if out is None:
            raise RuntimeError("graph execution returned None (compile failed?)")
        return out

    try:
        return benchmark_callable(_run, warmup=warmup, iters=iters, device=device)
    except Exception:
        return None


def superoptimize_matmul(
    m: int,
    n: int,
    k: int,
    *,
    dtype=torch.float16,
    backend: str = "maca",
    quick: Optional[bool] = None,
    verbose: bool = False,
):
    import yirage

    graph = yirage.new_kernel_graph()
    a = graph.new_input(dims=(m, k), dtype=yirage.float16)
    b = graph.new_input(dims=(k, n), dtype=yirage.float16)
    c = graph.matmul(a, b)
    graph.mark_output(c)
    search = maca_search_kwargs(quick=quick)
    return graph.superoptimize(
        backend=backend,
        use_ray=False,
        verbose=verbose,
        **search,
    )


def describe_search_config(cfg: Dict[str, Any]) -> Tuple[int, int]:
    return len(cfg.get("grid_dims_to_explore", [])), len(cfg.get("block_dims_to_explore", []))


__all__ = [
    "MACA_WARP_SIZE",
    "apply_maca_demo_env",
    "benchmark_callable",
    "benchmark_mugraph",
    "describe_search_config",
    "maca_search_kwargs",
    "resolve_maca_search_config",
    "superoptimize_matmul",
    "sync_device",
]
