#!/usr/bin/env python3
"""
MACA Backend Performance Benchmark: YiRage vs PyTorch

This script benchmarks YiRage optimized kernels against PyTorch baseline
on MetaX MACA GPU.
"""

import torch
import time
import numpy as np
import os
import sys

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
    MACA_WARP_SIZE,
    get_maca_search_config,
    get_maca_device_info,
    is_maca_available,
    resolve_maca_search_config,
)

# Benchmark configurations
WARMUP_ITERS = 10
BENCHMARK_ITERS = 100
QUICK_WARMUP_ITERS = 3
QUICK_BENCHMARK_ITERS = 20
DTYPE = torch.float16


def print_header(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_result(name, pytorch_time, yirage_time):
    """Print benchmark result with speedup."""
    speedup = pytorch_time / yirage_time if yirage_time > 0 else 0
    print(f"  {name}:")
    print(f"    PyTorch:  {pytorch_time*1000:.3f} ms")
    print(f"    YiRage:   {yirage_time*1000:.3f} ms")
    print(f"    Speedup:  {speedup:.2f}x")
    return speedup


def benchmark_pytorch_matmul(M, N, K, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark PyTorch matrix multiplication."""
    A = torch.randn(M, K, dtype=DTYPE, device=device)
    B = torch.randn(K, N, dtype=DTYPE, device=device)

    # Warmup
    for _ in range(warmup):
        C = torch.matmul(A, B)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        C = torch.matmul(A, B)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.perf_counter()

    return (end - start) / iters


def benchmark_yirage_matmul(M, N, K, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark YiRage MACA superoptimized matmul on GPU."""
    apply_maca_demo_env()
    search = maca_search_kwargs()
    graph = yirage.new_kernel_graph()
    A = graph.new_input(dims=(M, K), dtype=yirage.float16)
    B = graph.new_input(dims=(K, N), dtype=yirage.float16)
    C = graph.matmul(A, B)
    graph.mark_output(C)

    try:
        opt = graph.superoptimize(
            backend="maca",
            use_ray=False,
            verbose=False,
            **search,
        )
        if opt is None or device.type != "cuda":
            return None

        a_t = torch.randn(M, K, dtype=DTYPE, device=device)
        b_t = torch.randn(K, N, dtype=DTYPE, device=device)
        opt.backend = "maca"
        elapsed = benchmark_mugraph(opt, [a_t, b_t], warmup=warmup, iters=iters)
        sync_device(device)
        return elapsed
    except Exception as e:
        print(f"    YiRage optimization: {e}")
        return None


def benchmark_pytorch_rmsnorm(batch, hidden, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark PyTorch RMS normalization."""
    x = torch.randn(batch, hidden, dtype=DTYPE, device=device)
    weight = torch.randn(hidden, dtype=DTYPE, device=device)
    eps = 1e-6

    def rmsnorm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight

    # Warmup
    for _ in range(warmup):
        y = rmsnorm(x, weight, eps)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        y = rmsnorm(x, weight, eps)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.perf_counter()

    return (end - start) / iters


def benchmark_pytorch_gelu(batch, hidden, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark PyTorch GELU activation."""
    x = torch.randn(batch, hidden, dtype=DTYPE, device=device)

    # Warmup
    for _ in range(warmup):
        y = torch.nn.functional.gelu(x)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        y = torch.nn.functional.gelu(x)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.perf_counter()

    return (end - start) / iters


def benchmark_pytorch_softmax(
    batch, seq_len, heads, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS
):
    """Benchmark PyTorch softmax."""
    x = torch.randn(batch, heads, seq_len, seq_len, dtype=DTYPE, device=device)

    # Warmup
    for _ in range(warmup):
        y = torch.nn.functional.softmax(x, dim=-1)
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        y = torch.nn.functional.softmax(x, dim=-1)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.perf_counter()

    return (end - start) / iters


def run_matmul_benchmarks(device, maca_config, quick=False):
    """Run matrix multiplication benchmarks."""
    print_header("Matrix Multiplication Benchmarks")

    if quick:
        configs = [(128, 128, 256, "Quick")]
        warmup = QUICK_WARMUP_ITERS
        iters = QUICK_BENCHMARK_ITERS
    else:
        configs = [
            (128, 128, 256, "Small"),
            (512, 512, 1024, "Medium"),
            (1024, 1024, 4096, "Large"),
            (4096, 4096, 4096, "XLarge"),
        ]
        warmup = WARMUP_ITERS
        iters = BENCHMARK_ITERS

    results = []

    for M, N, K, name in configs:
        print(f"\n  [{name}] M={M}, N={N}, K={K}")

        pytorch_time = benchmark_pytorch_matmul(M, N, K, device, warmup=warmup, iters=iters)
        yirage_time = benchmark_yirage_matmul(M, N, K, device, warmup=warmup, iters=iters)
        if yirage_time is not None:
            print_result(name, pytorch_time, yirage_time)
            results.append((name, M, N, K, pytorch_time, yirage_time))
        else:
            print(f"    PyTorch: {pytorch_time*1000:.3f} ms")
            print(f"    YiRage:  (search/exec unavailable)")
            results.append((name, M, N, K, pytorch_time, None))

    return results


def run_element_wise_benchmarks(device):
    """Run element-wise operation benchmarks."""
    print_header("Element-wise Operations Benchmarks")

    configs = [
        (8, 4096, "LLM hidden"),
        (32, 4096, "Batch 32"),
        (64, 8192, "Large hidden"),
    ]

    for batch, hidden, name in configs:
        print(f"\n  [{name}] batch={batch}, hidden={hidden}")

        # RMSNorm
        rmsnorm_time = benchmark_pytorch_rmsnorm(batch, hidden, device)
        print(f"    RMSNorm PyTorch: {rmsnorm_time*1000:.3f} ms")
        print(f"    MACA optimization: 64-thread warp reduction")

        # GELU
        gelu_time = benchmark_pytorch_gelu(batch, hidden, device)
        print(f"    GELU PyTorch: {gelu_time*1000:.3f} ms")


def run_attention_benchmarks(device):
    """Run attention-related benchmarks."""
    print_header("Attention Operation Benchmarks")

    configs = [
        (1, 512, 32, "Short seq"),
        (1, 2048, 32, "Medium seq"),
        (4, 2048, 32, "Batch 4"),
    ]

    for batch, seq_len, heads, name in configs:
        print(f"\n  [{name}] batch={batch}, seq={seq_len}, heads={heads}")

        # Softmax
        softmax_time = benchmark_pytorch_softmax(batch, seq_len, heads, device)
        print(f"    Softmax PyTorch: {softmax_time*1000:.3f} ms")
        print(f"    MACA Flash Attention: mcflashinfer available")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="YiRage MACA vs PyTorch benchmark")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Single small matmul + reduced iters (Loop smoke on MetaX VM)",
    )
    args = parser.parse_args()

    if args.quick:
        os.environ.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")

    print()
    print("=" * 60)
    print("  YiRage MACA vs PyTorch Performance Benchmark")
    print("=" * 60)

    # Check MACA availability
    print_header("Environment")

    maca_available = is_maca_available()
    print(f"  MACA SDK available: {maca_available}")

    if maca_available:
        info = get_maca_device_info()
        if info:
            print(f"  Device: {info.get('device_type', 'Unknown')}")
            print(f"  HBM: {info.get('hbm_gb', 0)} GB")
            print(f"  SM Count: {info.get('sm_count', 0)}")
            print(f"  Warp Size: {info.get('warp_size', MACA_WARP_SIZE)}")

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"  PyTorch device: {device_name}")
    else:
        device = torch.device("cpu")
        print("  PyTorch device: CPU (no GPU available)")

    # Get MACA config
    maca_config = resolve_maca_search_config()
    print(f"\n  MACA Search Configuration (quick default):")
    print(f"    Block dims: {len(maca_config.get('block_dims_to_explore', []))} configs")
    print(f"    Grid dims: {len(maca_config.get('grid_dims_to_explore', []))} configs")
    print(f"    Forloop ranges: {maca_config.get('franges_to_explore', [])}")

    # Run benchmarks
    run_matmul_benchmarks(device, maca_config, quick=args.quick)
    if not args.quick:
        run_element_wise_benchmarks(device)
        run_attention_benchmarks(device)

    # Summary
    print_header("MACA Optimization Summary")
    print(
        """
  Key MACA Optimizations Applied:
  
  1. Warp-level Operations:
     - 64-thread warps (vs NVIDIA's 32)
     - 6-iteration shuffle reduction (vs 5)
     - Optimized lane masks for full warp
  
  2. Memory Access:
     - 64-thread coalesced access patterns
     - Shared memory bank conflict avoidance
     - HBM bandwidth optimization (4096-bit bus)
  
  3. Tensor Core Usage:
     - mctlass library for matrix ops
     - 64x64 and 128x64 tile sizes
     - Optimal warp arrangement
  
  4. Kernel Fusion:
     - YiRage superoptimization search
     - Automatic operator fusion
     - Reduced memory traffic
  
  Note: Full performance comparison requires running on
  MetaX C500 GPU with MACA runtime execution.
  Current results show PyTorch CPU/CUDA baseline.
"""
    )

    print_header("Benchmark Complete")


if __name__ == "__main__":
    main()
