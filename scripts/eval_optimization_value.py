#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Evaluate whether same-backend CPU superoptimize delivers measurable value."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import yirage as yr
from yirage.backends.cpu.config import get_cpu_info, get_cpu_runtime_config
from yirage.kernel.graph import KNGraph


def bench_ms(fn, warmup: int = 20, iters: int = 200) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000


def profile_candidates(
    g,
    griddims,
    blockdims,
    ref_a,
    ref_b,
    warmup=8,
    profile_iters=50,
    *,
    verifier_cfg=None,
):
    """Search with explicit arch-aware space (mirrors superoptimize defaults)."""
    del ref_a, ref_b
    from yirage.backends.cpu.config import apply_cpu_search_env, resolve_cpu_search_space
    from yirage.search.verifier_config import resolve_verifier_config

    if verifier_cfg is None:
        verifier_cfg = resolve_verifier_config()
    cpu_space = resolve_cpu_search_space(g.cygraph)
    apply_cpu_search_env(cpu_space)
    with tempfile.TemporaryDirectory(prefix="yirage_eval_") as tmp:
        os.environ["HOME"] = tmp
        import yirage.storage.mugraph_store as ms

        ms._default_store = None
        t0 = time.perf_counter()
        opt = g.superoptimize(
            backend="cpu",
            griddims=griddims,
            blockdims=blockdims,
            franges=cpu_space.get("franges_to_explore"),
            use_ray=False,
            use_graph_dataset=False,
            use_cached_graphs=False,
            use_persistent_cache=False,
            warmup_iters=warmup,
            profile_iters=profile_iters,
            verbose=False,
            is_formal_verified=verifier_cfg.is_formal_verified,
        )
        return opt, time.perf_counter() - t0, verifier_cfg


def main() -> int:
    from yirage.search.verifier_config import resolve_verifier_config

    cpu_info = get_cpu_info()
    rt = get_cpu_runtime_config()
    verifier_cfg = resolve_verifier_config()
    print("=== Environment (same-backend CPU) ===")
    print(
        json.dumps(
            {
                "cpu_info": cpu_info,
                "runtime": rt,
                "verification": verifier_cfg.to_dict(),
            },
            indent=2,
        )
    )

    g = yr.new_kernel_graph()
    a = g.new_input(dims=(8, 32), dtype=yr.float16)
    b = g.new_input(dims=(32, 64), dtype=yr.float16)
    g.mark_output(g.matmul(a, b))

    ref_a = torch.randn(8, 32, dtype=torch.float16)
    ref_b = torch.randn(32, 64, dtype=torch.float16)
    ref_out = torch.matmul(ref_a, ref_b)

    # Use architecture + graph-shaped search space (no hand-picked griddims)
    from yirage.backends.cpu.config import resolve_cpu_search_space

    cpu_space = resolve_cpu_search_space(g.cygraph)
    print("\n=== Architecture-aware search space ===")
    print(json.dumps(
        {
            k: cpu_space[k]
            for k in (
                "num_cores",
                "simd_type",
                "vector_width",
                "problem_mnk",
                "grid_dims_to_explore",
                "block_dims_to_explore",
                "franges_to_explore",
                "search_thread",
            )
        },
        indent=2,
    ))

    opt, search_s, verifier_cfg = profile_candidates(
        g,
        cpu_space["grid_dims_to_explore"],
        cpu_space["block_dims_to_explore"],
        ref_a,
        ref_b,
        verifier_cfg=verifier_cfg,
    )
    if opt is None:
        print("FAIL: superoptimize returned None")
        return 1

    out = opt(inputs=[ref_a, ref_b])
    err = (out[0].float() - ref_out.float()).abs().max().item()

    ops = opt.cygraph.get_graph_structure()
    op_types = [o["op_type"] for o in ops]

    g2 = yr.new_kernel_graph()
    a2 = g2.new_input(dims=(8, 32), dtype=yr.float16)
    b2 = g2.new_input(dims=(32, 64), dtype=yr.float16)
    g2.mark_output(g2.matmul(a2, b2))
    unopt = KNGraph(g2.cygraph, backend="cpu")

    baseline_torch = bench_ms(lambda: torch.matmul(ref_a, ref_b))
    unopt_ms = bench_ms(lambda: unopt(inputs=[ref_a, ref_b]))
    optimized_ms = bench_ms(lambda: opt(inputs=[ref_a, ref_b]))
    unopt_err = (unopt(inputs=[ref_a, ref_b])[0].float() - ref_out.float()).abs().max().item()

    print("\n=== Verification ===")
    print(f"  search verifier: {verifier_cfg.verifier_type}")
    print(f"  search_verified: {opt is not None} (fingerprint/formal during search)")
    print(f"  runtime_verified: {err < 0.05 and unopt_err < 0.05} (torch reference)")

    print("\n=== Correctness (vs torch.matmul reference) ===")
    print(f"  unoptimized muGraph max_abs_error: {unopt_err:.6f}")
    print(f"  superoptimize selected max_abs_error: {err:.6f}")
    correctness_ok = err < 0.05 and unopt_err < 0.05
    print(f"  verdict: {'PASS' if correctness_ok else 'FAIL'}")

    print("\n=== Latency (ms, same CPU device) ===")
    print(f"  torch.matmul (PyTorch native):  {baseline_torch:.4f}")
    print(f"  unoptimized KNGraph (cpu_call): {unopt_ms:.4f}")
    print(f"  superoptimize winner (cpu_call): {optimized_ms:.4f}")

    opt_vs_unopt = unopt_ms / max(optimized_ms, 1e-9)
    opt_vs_torch = baseline_torch / max(optimized_ms, 1e-9)
    print(f"\n  speedup winner vs unoptimized interpreter: {opt_vs_unopt:.3f}x")
    print(f"  speedup winner vs torch.matmul:            {opt_vs_torch:.3f}x")

    print("\n=== Search outcome ===")
    print(f"  search_time_s: {search_s:.2f}")
    print(f"  search_space: {len(cpu_space['grid_dims_to_explore'])} grids × {len(cpu_space['block_dims_to_explore'])} blocks (arch-aware)")
    print(f"  selected kn ops: {op_types}")
    print(f"  backend tag: {opt.backend}")

  # Value judgment
    print("\n=== Optimization value assessment ===")
    if not correctness_ok:
        value = "NO — numerical correctness failed"
    elif opt_vs_unopt >= 1.15:
        value = "YES — measurable gain vs unoptimized same-backend path"
    elif opt_vs_unopt >= 1.03:
        value = "MARGINAL — small gain vs unoptimized interpreter"
    elif abs(opt_vs_unopt - 1.0) < 0.03:
        value = "WEAK — search ran but winner ≈ unoptimized (often plain kn_matmul_op)"
    else:
        value = "NEGATIVE — selected graph slower than unoptimized baseline"

    if opt_vs_torch < 0.5:
        value += "; NOT competitive with PyTorch native on this workload"

    print(f"  {value}")

    # Amortization hint
    if search_s > 0 and optimized_ms > 0:
        runs_to_break_even = search_s * 1000 / max(unopt_ms - optimized_ms, 0.001)
        print(f"  search amortization (vs unopt): ~{runs_to_break_even:.0f} executions to recover search cost")

    return 0


if __name__ == "__main__":
    os.environ.setdefault("YIRAGE_BACKEND", "cpu")
    raise SystemExit(main())
