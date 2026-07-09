#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end MACA capability walkthrough (CUDA-aligned Ray/RL stack on MetaX GPU):
  YiRage µGraph search on ``backend=maca`` → profile/select on MACA
  → Ray parallelism → AccelForge prescreen → RL reward loop.

Run on MetaX VM:
  export MACA_PATH=/opt/maca
  export LD_LIBRARY_PATH=build/abstract_subexpr/release:build/formal_verifier/release:${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib
  export YIRAGE_BACKEND=maca
  PYTHONPATH=. python3 scripts/maca_capability_walkthrough.py --quick
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.business_capability_walkthrough import (  # noqa: E402
    WalkthroughReport,
    _isolated_mugraph_store,
    _timed,
    compute_business_scores,
    stage_accelforge,
    stage_rl_accelforge_loop,
    walkthrough_report_to_dict,
)


def _walkthrough_quick_enabled() -> bool:
    return os.environ.get("YIRAGE_MACA_WALKTHROUGH_QUICK", "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _maca_search_kwargs() -> Dict[str, Any]:
    from yirage.backends.maca.config import resolve_maca_search_config

    cfg = resolve_maca_search_config(quick=_walkthrough_quick_enabled())
    return {
        "griddims": cfg.get("grid_dims_to_explore"),
        "blockdims": cfg.get("block_dims_to_explore"),
        "fmaps": cfg.get("fmaps_to_explore"),
        "franges": cfg.get("franges_to_explore"),
    }


def stage_maca_graph():
    import yirage as yr
    from yirage.backends.maca.config import (
        MACA_WARP_SIZE,
        get_maca_device_info,
        get_maca_memory_config,
    )

    g = yr.new_kernel_graph()
    a = g.new_input(dims=(64, 32), dtype=yr.float16)
    b = g.new_input(dims=(32, 64), dtype=yr.float16)
    out = g.matmul(a, b)
    g.mark_output(out)
    mem = get_maca_memory_config()
    device = get_maca_device_info() or {}
    return {
        "backend": os.environ.get("YIRAGE_BACKEND", "maca"),
        "design": "same-backend MACA (search/profile/execute on MetaX GPU)",
        "input_shapes": "64×32 @ 32×64",
        "kn_ops_before_search": len(g.cygraph.get_graph_structure()),
        "maca_warp_size": MACA_WARP_SIZE,
        "maca_device": device.get("device_type", "unknown"),
        "maca_hbm_gb": mem.get("hbm_gb"),
        "maca_shared_mem_kb": mem.get("shared_mem_kb"),
    }, g


def stage_maca_homomorphic_value(graph) -> Dict[str, Any]:
    """Measure correctness and latency on MACA after superoptimize (same backend)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("MetaX GPU (mcPytorch cuda) required for MACA walkthrough")

    search = _maca_search_kwargs()
    with _isolated_mugraph_store():
        optimized = graph.superoptimize(
            backend="maca",
            use_ray=False,
            use_graph_dataset=False,
            use_cached_graphs=False,
            use_persistent_cache=False,
            warmup_iters=1 if _walkthrough_quick_enabled() else 4,
            profile_iters=5 if _walkthrough_quick_enabled() else 40,
            verbose=False,
            **search,
        )

    assert optimized is not None
    assert optimized.backend == "maca"

    device = torch.device("cuda")
    ref_a = torch.randn(64, 32, dtype=torch.float16, device=device)
    ref_b = torch.randn(32, 64, dtype=torch.float16, device=device)
    ref_out = torch.matmul(ref_a, ref_b)

    out = optimized(inputs=[ref_a, ref_b])
    max_err = (out[0].float() - ref_out.float()).abs().max().item()
    correct = max_err < 0.1

    iters = 10 if _walkthrough_quick_enabled() else 40
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        torch.matmul(ref_a, ref_b)
    torch.cuda.synchronize()
    baseline_ms = (time.perf_counter() - t0) / iters * 1000

    for _ in range(3):
        optimized(inputs=[ref_a, ref_b])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        optimized(inputs=[ref_a, ref_b])
    torch.cuda.synchronize()
    optimized_ms = (time.perf_counter() - t0) / iters * 1000

    return {
        "optimized_backend": optimized.backend,
        "numerically_correct": correct,
        "max_abs_error": max_err,
        "baseline_mcPytorch_matmul_ms": baseline_ms,
        "optimized_mugraph_ms": optimized_ms,
        "mugraph_vs_mcPytorch_ratio": optimized_ms / max(baseline_ms, 1e-6),
        "grids_profiled": len(search.get("griddims", [])),
        "blocks_profiled": len(search.get("blockdims", [])),
    }


def _search_once_maca(yr, use_ray: bool) -> float:
    import ray

    if ray.is_initialized():
        ray.shutdown()
    search = _maca_search_kwargs()
    with _isolated_mugraph_store():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(64, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 64), dtype=yr.float16)
        g.mark_output(g.matmul(a, b))
        t0 = time.perf_counter()
        g.superoptimize(
            backend="maca",
            use_ray=use_ray,
            num_workers=2,
            use_graph_dataset=False,
            use_cached_graphs=False,
            use_persistent_cache=False,
            verbose=False,
            **search,
        )
        return time.perf_counter() - t0


def stage_maca_ray_search(graph) -> tuple[Dict[str, Any], Any, Optional[str]]:
    import yirage as yr
    from yirage.storage.graph_serde import serialize_optimized_graph

    quick = _walkthrough_quick_enabled()
    search = _maca_search_kwargs()

    if quick:
        seq_elapsed = None
        ray_elapsed = None
        speedup = None
    else:
        seq_elapsed = _search_once_maca(yr, use_ray=False)
        ray_elapsed = _search_once_maca(yr, use_ray=True)
        speedup = seq_elapsed / max(ray_elapsed, 1e-6)

    with _isolated_mugraph_store():
        t_ray = time.perf_counter()
        optimized = graph.superoptimize(
            backend="maca",
            use_ray=True,
            num_workers=2,
            use_graph_dataset=False,
            use_cached_graphs=False,
            use_persistent_cache=False,
            verbose=False,
            **search,
        )
        prod_ray_s = time.perf_counter() - t_ray

    payload = serialize_optimized_graph(optimized)
    note = (
        "quick mode: seq/ray benchmark skipped; single-grid Ray search only"
        if quick
        else "speedup>1 means Ray finished faster (isolated cache)"
    )
    return {
        "griddims_benchmarked": len(search.get("griddims", [])),
        "sequential_search_s": seq_elapsed,
        "ray_search_s": ray_elapsed,
        "speedup_seq_over_ray": speedup,
        "demo_ray_search_s": prod_ray_s,
        "graph_json_bytes": len(payload or ""),
        "optimized_ops": len(optimized.cygraph.get_graph_structure()),
        "note": note,
    }, optimized, payload


def build_maca_walkthrough_report(*, quick: bool = True) -> WalkthroughReport:
    """Run MACA walkthrough stages and return structured report."""
    if quick:
        os.environ["YIRAGE_MACA_WALKTHROUGH_QUICK"] = "1"
    else:
        os.environ["YIRAGE_MACA_WALKTHROUGH_QUICK"] = "0"

    report = WalkthroughReport()
    optimized = None
    payload: Optional[str] = None

    def s1():
        nonlocal optimized, payload
        details, optimized = stage_maca_graph()
        return details

    report.add(_timed("1. YiRage MACA graph build", s1))

    def s1b():
        if optimized is None:
            raise RuntimeError("graph not built")
        return stage_maca_homomorphic_value(optimized)

    report.add(_timed("1b. MACA same-backend value", s1b))

    def s2():
        nonlocal optimized, payload
        if optimized is None:
            raise RuntimeError("graph not built")
        details, optimized, payload = stage_maca_ray_search(optimized)
        return details

    report.add(_timed("2. Ray µGraph search (maca)", s2))

    def s3():
        if not payload:
            raise RuntimeError("no graph json")
        return stage_accelforge(payload)

    report.add(_timed("3. AccelForge prescreen", s3))

    def s4():
        if not payload:
            raise RuntimeError("no graph json")
        return stage_rl_accelforge_loop(payload)

    report.add(_timed("4. RL × AccelForge env loop", s4))

    compute_business_scores(report)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON report")
    parser.add_argument(
        "--quick",
        action="store_true",
        default=True,
        help="Tractable MACA search grids (default on)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Disable quick tractability (longer Ray seq vs Ray benchmark)",
    )
    args = parser.parse_args()
    quick = not args.full

    report = build_maca_walkthrough_report(quick=quick)
    if args.json:
        print("YIRAGE_MACA_WALKTHROUGH_JSON_BEGIN")
        print(json.dumps(walkthrough_report_to_dict(report), indent=2))
        print("YIRAGE_MACA_WALKTHROUGH_JSON_END", flush=True)
    else:
        report.print_report()
    return 0 if all(s.ok for s in report.stages) else 1


if __name__ == "__main__":
    os.environ.setdefault("YIRAGE_BACKEND", "maca")
    os.environ.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    raise SystemExit(main())
