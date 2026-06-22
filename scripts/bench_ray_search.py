#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Fair Ray vs sequential µGraph search benchmark (cache isolated per run).

Run:
  export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
  PYTHONPATH=. python3 scripts/bench_ray_search.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


@contextmanager
def isolated_mugraph_store():
    """Use a fresh HOME so persistent MuGraph cache cannot cross-contaminate runs."""
    prev_home = os.environ.get("HOME")
    with tempfile.TemporaryDirectory(prefix="yirage_bench_home_") as tmp:
        os.environ["HOME"] = tmp
        try:
            import yirage.storage.mugraph_store as ms

            ms._default_store = None
            yield tmp
        finally:
            if prev_home is not None:
                os.environ["HOME"] = prev_home
            else:
                os.environ.pop("HOME", None)


def _build_matmul_graph(yr, m: int = 8, k: int = 32, n: int = 64):
    g = yr.new_kernel_graph()
    a = g.new_input(dims=(m, k), dtype=yr.float16)
    b = g.new_input(dims=(k, n), dtype=yr.float16)
    g.mark_output(g.matmul(a, b))
    return g


def _run_search(
    yr,
    griddims: List[Tuple[int, int, int]],
    use_ray: bool,
    num_workers: int = 2,
) -> float:
    import ray

    if ray.is_initialized():
        ray.shutdown()

    with isolated_mugraph_store():
        g = _build_matmul_graph(yr)
        t0 = time.perf_counter()
        g.superoptimize(
            backend="cpu",
            griddims=griddims,
            blockdims=[(128, 1, 1)],
            use_graph_dataset=False,
            use_cached_graphs=False,
            use_persistent_cache=False,
            use_ray=use_ray,
            num_workers=num_workers,
            verbose=False,
        )
        return time.perf_counter() - t0


def _griddim_sets() -> Dict[str, List[Tuple[int, int, int]]]:
    return {
        "small_3": [(1, 1, 1), (2, 1, 1), (4, 1, 1)],
        "medium_6": [(1, 1, 1), (2, 1, 1), (4, 1, 1), (8, 1, 1), (16, 1, 1), (32, 1, 1)],
        "large_9": [
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
            (16, 1, 1),
            (32, 1, 1),
            (64, 1, 1),
            (128, 1, 1),
            (256, 1, 1),
        ],
    }


def main() -> int:
    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    import yirage as yr

    print("Ray vs sequential µGraph search (isolated cache, use_persistent_cache=False)")
    print(f"backend=cpu  workers=2  blockdim=(128,1,1)\n")
    print(f"{'suite':<12} {'griddims':>8} {'sequential_s':>14} {'ray_s':>10} {'speedup':>10}")
    print("-" * 60)

    rows: List[Dict[str, Any]] = []
    for name, griddims in _griddim_sets().items():
        seq_s = _run_search(yr, griddims, use_ray=False)
        ray_s = _run_search(yr, griddims, use_ray=True, num_workers=2)
        speedup = seq_s / max(ray_s, 1e-6)
        rows.append(
            {
                "suite": name,
                "num_griddims": len(griddims),
                "sequential_s": seq_s,
                "ray_s": ray_s,
                "speedup": speedup,
            }
        )
        print(
            f"{name:<12} {len(griddims):>8} {seq_s:>14.2f} {ray_s:>10.2f} {speedup:>10.2f}x"
        )

    print("-" * 60)
    best = max(rows, key=lambda r: r["speedup"])
    print(
        f"Best speedup: {best['suite']} ({best['speedup']:.2f}x) "
        f"— Ray pays off when griddim partitions amortize startup."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
