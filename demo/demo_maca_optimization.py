#!/usr/bin/env python3
"""
MACA Backend Optimization Demo for YiRage

Demonstrates device detection, MACA search config, superoptimize on MACA,
and optional fused matmul+GELU GPU execution timing vs PyTorch.
"""

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
    benchmark_callable,
    benchmark_mugraph,
    maca_search_kwargs,
    sync_device,
)
from yirage.maca_config import (
    MACA_MAX_THREADS_PER_BLOCK,
    MACA_SHARED_MEM_PER_BLOCK,
    MACA_WARP_SIZE,
    get_maca_device_info,
    get_maca_memory_config,
    get_maca_search_config,
    get_optimal_block_size,
    is_maca_available,
    resolve_maca_search_config,
    validate_block_size,
)


def print_header(title):
    print()
    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)


def demo_device_info():
    print_header("MACA Device Information")
    if not is_maca_available():
        print("  MACA SDK not found (set MACA_PATH=/opt/maca)")
    info = get_maca_device_info()
    if info:
        print(f"  Available: {info.get('available', False)}")
        print(f"  Device: {info.get('device_type', 'Unknown')}")
        print(f"  HBM Memory: {info.get('hbm_gb', 0)} GB")
        print(f"  SM Count: {info.get('sm_count', 0)}")
        print(f"  Warp Size: {info.get('warp_size', MACA_WARP_SIZE)}")
        return bool(info.get("available"))
    return torch.cuda.is_available()


def demo_maca_config():
    print_header("MACA Hardware Configuration")
    print(f"  Warp Size: {MACA_WARP_SIZE} threads (NVIDIA uses 32)")
    print(f"  Max Threads/Block: {MACA_MAX_THREADS_PER_BLOCK}")
    print(f"  Shared Memory: {MACA_SHARED_MEM_PER_BLOCK // 1024} KB/block")
    full = get_maca_search_config()
    quick = resolve_maca_search_config(quick=True)
    print(f"  Full search: {len(full.get('grid_dims_to_explore', []))} grids × "
          f"{len(full.get('block_dims_to_explore', []))} blocks")
    print(f"  Quick search (demo default): {len(quick.get('grid_dims_to_explore', []))} grid × "
          f"{len(quick.get('block_dims_to_explore', []))} block")


def demo_block_optimization():
    print_header("Block Size Optimization")
    for size in (256, 512, 1024, 4096, 16384):
        optimal = get_optimal_block_size(size)
        warps = optimal // MACA_WARP_SIZE
        print(f"    {size:>6} -> {optimal:>4} ({warps} warps, valid: {validate_block_size(optimal)})")


def demo_superoptimize_matmul(m, n, k, *, verbose=False):
    print_header("MatMul Superoptimize + GPU Execution")
    print(f"  Shape: ({m}x{k}) @ ({k}x{n})")

    graph = yirage.new_kernel_graph()
    a = graph.new_input(dims=(m, k), dtype=yirage.float16)
    b = graph.new_input(dims=(k, n), dtype=yirage.float16)
    c = graph.matmul(a, b)
    graph.mark_output(c)

    search = maca_search_kwargs()
    t0 = time.time()
    opt = graph.superoptimize(backend="maca", use_ray=False, verbose=verbose, **search)
    print(f"  Superoptimize: {time.time() - t0:.2f}s")
    if opt is None:
        print("  No valid µGraph")
        return None

    if not torch.cuda.is_available():
        print("  CUDA/mcPytorch not available — search-only")
        return opt

    device = torch.device("cuda")
    a_t = torch.randn(m, k, dtype=torch.float16, device=device)
    b_t = torch.randn(k, n, dtype=torch.float16, device=device)
    opt.backend = "maca"

    pt_s = benchmark_callable(lambda: torch.matmul(a_t, b_t), device=device)
    yr_s = benchmark_mugraph(opt, [a_t, b_t])
    if yr_s is not None:
        print(f"  PyTorch: {pt_s * 1000:.3f} ms | YiRage: {yr_s * 1000:.3f} ms | "
              f"speedup {pt_s / yr_s:.2f}x")
    else:
        print("  YiRage GPU execution failed")
    sync_device(device)
    return opt


def demo_memory_config():
    print_header("MACA Memory Configuration")
    config = get_maca_memory_config()
    print(f"  Shared Memory/Block: {config.get('shared_mem_kb', config.get('shared_memory_per_block', 0) // 1024)} KB")
    print(f"  L2 Cache: {config.get('l2_cache_mb', config.get('l2_cache_size', 0) // (1024 * 1024))} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", default=True, help="Use quick MACA search (default)")
    parser.add_argument("--full-search", action="store_true", help="Use full MACA search grid")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    apply_maca_demo_env()
    if args.full_search:
        os.environ["YIRAGE_MACA_SEARCH_QUICK"] = "0"

    print_header("MACA Backend Optimization Demo")
    has_device = demo_device_info()
    demo_maca_config()
    demo_block_optimization()
    demo_memory_config()
    demo_superoptimize_matmul(args.m, args.n, args.k, verbose=args.verbose)

    print_header("Summary")
    print(f"  Warp size {MACA_WARP_SIZE}; block dims must be multiples of 64")
    print(f"  GPU ready: {has_device}")
    print("=" * 50)


if __name__ == "__main__":
    main()
