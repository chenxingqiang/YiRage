#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Fair Ray vs sequential µGraph search benchmark (cache isolated per run).

Run:
  export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:$LD_LIBRARY_PATH
  PYTHONPATH=. python3 scripts/bench_ray_search.py
  PYTHONPATH=. python3 scripts/bench_ray_search.py --backend maca --quick
"""

from __future__ import annotations

import argparse
import json
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


def resolve_bench_backend(explicit: str | None = None) -> str:
    """Return bench backend (cpu or maca)."""
    backend = (explicit or os.environ.get("YIRAGE_BACKEND") or "cpu").strip().lower()
    if backend not in ("cpu", "maca"):
        raise ValueError(f"bench_ray_search supports backend=cpu|maca, got {backend!r}")
    return backend


def _load_maca_config_module():
    import importlib.util

    path = os.path.join(_REPO, "python", "yirage", "backends", "maca", "config.py")
    spec = importlib.util.spec_from_file_location("maca_config_bench", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def bench_search_kwargs(backend: str, *, quick: bool) -> Dict[str, Any]:
    """Superoptimize kwargs besides griddims / use_ray (blockdims, franges, …)."""
    if backend == "maca":
        os.environ.setdefault("MACA_PATH", "/opt/maca")
        if quick:
            os.environ["YIRAGE_MACA_SEARCH_QUICK"] = "1"
        maca_cfg = _load_maca_config_module()
        cfg = maca_cfg.resolve_maca_search_config(quick=quick)
        kwargs = {
            "blockdims": cfg.get("block_dims_to_explore"),
            "fmaps": cfg.get("fmaps_to_explore"),
            "franges": cfg.get("franges_to_explore"),
        }
        return kwargs
    kwargs: Dict[str, Any] = {"blockdims": [(128, 1, 1)]}
    if quick:
        from scripts.cpu_cert_utils import apply_plain_matmul_search_tractability

        apply_plain_matmul_search_tractability()
        kwargs["franges"] = [1]
    return kwargs


def griddim_sets(backend: str, *, quick: bool) -> Dict[str, List[Tuple[int, int, int]]]:
    """Named griddim suites for seq vs Ray comparison."""
    if backend == "maca":
        if quick:
            return {
                "maca_quick_1": [(4, 1, 1)],
                "maca_quick_2": [(4, 1, 1), (8, 1, 1)],
            }
        return {
            "maca_small_3": [(4, 1, 1), (8, 1, 1), (16, 1, 1)],
            "maca_medium_5": [(4, 1, 1), (8, 1, 1), (16, 1, 1), (26, 1, 1), (52, 1, 1)],
        }
    if quick:
        return {
            "cpu_quick_1": [(1, 1, 1)],
            "cpu_quick_2": [(1, 1, 1), (2, 1, 1)],
        }
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
    *,
    backend: str,
    search_kwargs: Dict[str, Any],
    num_workers: int = 2,
) -> float:
    import ray

    if ray.is_initialized():
        ray.shutdown()

    with isolated_mugraph_store():
        g = _build_matmul_graph(yr)
        t0 = time.perf_counter()
        g.superoptimize(
            backend=backend,
            griddims=griddims,
            use_graph_dataset=False,
            use_cached_graphs=False,
            use_persistent_cache=False,
            use_ray=use_ray,
            num_workers=num_workers,
            verbose=False,
            **search_kwargs,
        )
        return time.perf_counter() - t0


def run_bench(
    *,
    backend: str | None = None,
    quick: bool = False,
    num_workers: int = 2,
    emit_json: bool = False,
) -> List[Dict[str, Any]]:
    """Run seq vs Ray suites; return row dicts."""
    backend = resolve_bench_backend(backend)
    os.environ["YIRAGE_BACKEND"] = backend
    import yirage as yr

    search_kwargs = bench_search_kwargs(backend, quick=quick)
    suites = griddim_sets(backend, quick=quick)
    rows: List[Dict[str, Any]] = []

    if not emit_json:
        print("Ray vs sequential µGraph search (isolated cache, use_persistent_cache=False)")
        blockdim = search_kwargs.get("blockdims", [(128, 1, 1)])
        print(f"backend={backend}  workers={num_workers}  blockdim={blockdim[0]}\n")
        print(f"{'suite':<16} {'griddims':>8} {'sequential_s':>14} {'ray_s':>10} {'speedup':>10}")
        print("-" * 64)

    for name, griddims in suites.items():
        seq_s = _run_search(
            yr,
            griddims,
            use_ray=False,
            backend=backend,
            search_kwargs=search_kwargs,
            num_workers=num_workers,
        )
        ray_s = _run_search(
            yr,
            griddims,
            use_ray=True,
            backend=backend,
            search_kwargs=search_kwargs,
            num_workers=num_workers,
        )
        speedup = seq_s / max(ray_s, 1e-6)
        row = {
            "backend": backend,
            "quick": quick,
            "suite": name,
            "num_griddims": len(griddims),
            "sequential_s": seq_s,
            "ray_s": ray_s,
            "speedup": speedup,
            "blockdims": search_kwargs.get("blockdims"),
        }
        rows.append(row)
        if not emit_json:
            print(
                f"{name:<16} {len(griddims):>8} {seq_s:>14.2f} {ray_s:>10.2f} {speedup:>10.2f}x"
            )

    if not emit_json:
        print("-" * 64)
        best = max(rows, key=lambda r: r["speedup"])
        print(
            f"Best speedup: {best['suite']} ({best['speedup']:.2f}x) "
            f"— Ray pays off when griddim partitions amortize startup."
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("cpu", "maca"),
        default=None,
        help="Search backend (default: YIRAGE_BACKEND or cpu)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tractable griddim suites + MACA/CPU search caps",
    )
    parser.add_argument("--workers", type=int, default=2, help="Ray worker count")
    parser.add_argument("--json", action="store_true", help="Emit JSON rows to stdout")
    args = parser.parse_args()

    rows = run_bench(
        backend=args.backend,
        quick=args.quick,
        num_workers=args.workers,
        emit_json=args.json,
    )
    if args.json:
        print(json.dumps({"rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
