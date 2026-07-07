#!/usr/bin/env python3
"""MACA superoptimization smoke: tractable search + optional GPU execution."""

import argparse
import os
import sys
import time

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yirage
from demo._maca_utils import (
    apply_maca_demo_env,
    benchmark_mugraph,
    maca_search_kwargs,
    sync_device,
)
from yirage.maca_config import MACA_WARP_SIZE, resolve_maca_search_config


def main():
    parser = argparse.ArgumentParser(description="MACA kernel superoptimization smoke")
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--full-search", action="store_true", help="Use full MACA search grid")
    parser.add_argument("--skip-exec", action="store_true", help="Search only, skip GPU execution")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    apply_maca_demo_env()
    if args.full_search:
        os.environ["YIRAGE_MACA_SEARCH_QUICK"] = "0"

    print("=" * 50)
    print("  MACA Kernel Superoptimization Test")
    print("=" * 50)

    cfg = resolve_maca_search_config()
    print(f"\nSearch: {len(cfg.get('grid_dims_to_explore', []))} grids × "
          f"{len(cfg.get('block_dims_to_explore', []))} blocks, "
          f"franges={cfg.get('franges_to_explore')}")

    graph = yirage.new_kernel_graph()
    a = graph.new_input(dims=(args.m, args.k), dtype=yirage.float16)
    b = graph.new_input(dims=(args.k, args.n), dtype=yirage.float16)
    c = graph.matmul(a, b)
    graph.mark_output(c)
    print(f"\nMatmul: ({args.m}x{args.k}) @ ({args.k}x{args.n}) = ({args.m}x{args.n})")

    search = maca_search_kwargs()
    print("\nRunning superoptimize(backend='maca')...")
    start = time.time()
    try:
        opt = graph.superoptimize(
            backend="maca",
            use_ray=False,
            verbose=args.verbose,
            **search,
        )
        elapsed = time.time() - start
        print(f"Search completed in {elapsed:.2f}s")
    except Exception as exc:
        print(f"Search failed after {time.time() - start:.2f}s: {exc}")
        return 1

    if opt is None:
        print("No optimized µGraph returned")
        return 1

    print("Optimized µGraph found")

    if args.skip_exec or not torch.cuda.is_available():
        if not torch.cuda.is_available():
            print("GPU not available — skipping execution benchmark")
        return 0

    device = torch.device("cuda")
    a_t = torch.randn(args.m, args.k, dtype=torch.float16, device=device)
    b_t = torch.randn(args.k, args.n, dtype=torch.float16, device=device)
    opt.backend = "maca"

    def pytorch_ref():
        return torch.matmul(a_t, b_t)

    from demo._maca_utils import benchmark_callable

    pt_s = benchmark_callable(pytorch_ref, device=device)
    yr_s = benchmark_mugraph(opt, [a_t, b_t])
    if yr_s is None:
        print("YiRage execution failed (compile or kernel error)")
        return 1

    speedup = pt_s / yr_s if yr_s > 0 else 0.0
    print(f"\nExecution (GPU):")
    print(f"  PyTorch: {pt_s * 1000:.3f} ms")
    print(f"  YiRage:  {yr_s * 1000:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")
    sync_device(device)

    print(f"\nWarp size: {MACA_WARP_SIZE}, compiler: mxcc when nvcc absent")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
