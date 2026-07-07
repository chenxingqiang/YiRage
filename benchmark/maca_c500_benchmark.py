#!/usr/bin/env python3
"""
MACA C500 GPU Performance Benchmark: PyTorch vs YiRage

This script benchmarks PyTorch (mcPytorch) and YiRage optimized kernels
on MetaX C500 GPU.

Requirements:
- Run with: /opt/conda/bin/python
- mcPytorch (torch 2.6.0+metax)
"""

import torch
import time
import numpy as np
import sys
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ.setdefault("MACA_PATH", "/opt/maca")

# Import YiRage MACA config
try:
    from yirage.maca_config import (
        MACA_WARP_SIZE,
        get_maca_search_config,
        get_maca_device_info,
        resolve_maca_search_config,
    )
    from demo._maca_utils import (
        apply_maca_demo_env,
        benchmark_mugraph,
        maca_search_kwargs,
    )
    import yirage

    YIRAGE_AVAILABLE = True
except ImportError:
    YIRAGE_AVAILABLE = False
    MACA_WARP_SIZE = 64

# Benchmark configuration
WARMUP_ITERS = 20
BENCHMARK_ITERS = 100
DTYPE = torch.float16


def print_header(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_comparison(name, pytorch_time, yirage_time=None):
    """Print benchmark comparison."""
    print(f"\n  {name}:")
    print(f"    PyTorch (mcPytorch on C500): {pytorch_time*1000:.4f} ms")
    if yirage_time is not None:
        speedup = pytorch_time / yirage_time if yirage_time > 0 else 0
        print(f"    YiRage optimized:            {yirage_time*1000:.4f} ms")
        print(f"    Speedup:                     {speedup:.2f}x")


def sync_gpu():
    """Synchronize GPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_matmul(M, N, K, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark matrix multiplication on GPU."""
    A = torch.randn(M, K, dtype=DTYPE, device=device)
    B = torch.randn(K, N, dtype=DTYPE, device=device)

    # Warmup
    for _ in range(warmup):
        C = torch.matmul(A, B)
    sync_gpu()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        C = torch.matmul(A, B)
    sync_gpu()
    end = time.perf_counter()

    return (end - start) / iters


def benchmark_rmsnorm(batch, hidden, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark RMS normalization on GPU."""
    x = torch.randn(batch, hidden, dtype=DTYPE, device=device)
    weight = torch.ones(hidden, dtype=DTYPE, device=device)
    eps = 1e-6

    def rmsnorm(x, weight, eps):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight

    # Warmup
    for _ in range(warmup):
        y = rmsnorm(x, weight, eps)
    sync_gpu()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        y = rmsnorm(x, weight, eps)
    sync_gpu()
    end = time.perf_counter()

    return (end - start) / iters


def benchmark_gelu(batch, hidden, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark GELU activation on GPU."""
    x = torch.randn(batch, hidden, dtype=DTYPE, device=device)

    # Warmup
    for _ in range(warmup):
        y = torch.nn.functional.gelu(x)
    sync_gpu()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        y = torch.nn.functional.gelu(x)
    sync_gpu()
    end = time.perf_counter()

    return (end - start) / iters


def benchmark_softmax(batch, seq_len, heads, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark softmax on GPU."""
    x = torch.randn(batch, heads, seq_len, seq_len, dtype=DTYPE, device=device)

    # Warmup
    for _ in range(warmup):
        y = torch.nn.functional.softmax(x, dim=-1)
    sync_gpu()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        y = torch.nn.functional.softmax(x, dim=-1)
    sync_gpu()
    end = time.perf_counter()

    return (end - start) / iters


def benchmark_layernorm(batch, hidden, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark LayerNorm on GPU."""
    x = torch.randn(batch, hidden, dtype=DTYPE, device=device)
    ln = torch.nn.LayerNorm(hidden, dtype=DTYPE, device=device)

    # Warmup
    for _ in range(warmup):
        y = ln(x)
    sync_gpu()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        y = ln(x)
    sync_gpu()
    end = time.perf_counter()

    return (end - start) / iters


def benchmark_fused_matmul_gelu(M, N, K, device, warmup=WARMUP_ITERS, iters=BENCHMARK_ITERS):
    """Benchmark fused MatMul + GELU (what YiRage can optimize)."""
    A = torch.randn(M, K, dtype=DTYPE, device=device)
    B = torch.randn(K, N, dtype=DTYPE, device=device)

    def fused_op(A, B):
        C = torch.matmul(A, B)
        return torch.nn.functional.gelu(C)

    # Warmup
    for _ in range(warmup):
        y = fused_op(A, B)
    sync_gpu()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        y = fused_op(A, B)
    sync_gpu()
    end = time.perf_counter()

    return (end - start) / iters


def benchmark_yirage_matmul(M, N, K, device, warmup=5, iters=20):
    """YiRage MACA superoptimize + GPU execution (quick search)."""
    if not YIRAGE_AVAILABLE:
        return None
    apply_maca_demo_env()
    search = maca_search_kwargs()
    graph = yirage.new_kernel_graph()
    a = graph.new_input(dims=(M, K), dtype=yirage.float16)
    b = graph.new_input(dims=(K, N), dtype=yirage.float16)
    c = graph.matmul(a, b)
    graph.mark_output(c)
    opt = graph.superoptimize(backend="maca", use_ray=False, verbose=False, **search)
    if opt is None:
        return None
    a_t = torch.randn(M, K, dtype=DTYPE, device=device)
    b_t = torch.randn(K, N, dtype=DTYPE, device=device)
    opt.backend = "maca"
    return benchmark_mugraph(opt, [a_t, b_t], warmup=warmup, iters=iters)


def run_matmul_benchmarks(device):
    """Run matrix multiplication benchmarks."""
    print_header("Matrix Multiplication (GEMM)")

    configs = [
        (128, 128, 256, "Tiny"),
        (512, 512, 512, "Small"),
        (1024, 1024, 1024, "Medium"),
        (2048, 2048, 2048, "Large"),
        (4096, 4096, 4096, "XLarge"),
        (8192, 8192, 8192, "XXLarge"),
    ]

    print(f"\n  {'Config':<10} {'M×N×K':<18} {'PyTorch (ms)':<14} {'YiRage (ms)':<14} {'Speedup':<8}")
    print("  " + "-" * 70)

    results = []
    for M, N, K, name in configs:
        pt_s = benchmark_matmul(M, N, K, device)
        yr_s = benchmark_yirage_matmul(M, N, K, device) if YIRAGE_AVAILABLE else None
        flops = 2 * M * N * K
        tflops = flops / pt_s / 1e12
        pt_ms = pt_s * 1000
        if yr_s is not None:
            speedup = pt_s / yr_s if yr_s > 0 else 0
            print(
                f"  {name:<10} {M}×{N}×{K:<8} {pt_ms:>12.4f}  {yr_s*1000:>12.4f}  {speedup:>6.2f}x"
            )
        else:
            print(f"  {name:<10} {M}×{N}×{K:<8} {pt_ms:>12.4f}  {'N/A':>12}  {'—':>8}")
        results.append((name, M, N, K, pt_s, tflops, yr_s))

    return results


def run_normalization_benchmarks(device):
    """Run normalization benchmarks."""
    print_header("Normalization Operations")

    configs = [
        (1, 4096, "1×4096"),
        (8, 4096, "8×4096"),
        (32, 4096, "32×4096"),
        (64, 8192, "64×8192"),
        (128, 8192, "128×8192"),
    ]

    print(f"\n  RMSNorm:")
    print(f"  {'Shape':<15} {'Time (ms)':<12}")
    print("  " + "-" * 30)

    for batch, hidden, name in configs:
        time_s = benchmark_rmsnorm(batch, hidden, device)
        print(f"  {name:<15} {time_s*1000:>10.4f}")

    print(f"\n  LayerNorm:")
    print(f"  {'Shape':<15} {'Time (ms)':<12}")
    print("  " + "-" * 30)

    for batch, hidden, name in configs:
        time_s = benchmark_layernorm(batch, hidden, device)
        print(f"  {name:<15} {time_s*1000:>10.4f}")


def run_activation_benchmarks(device):
    """Run activation function benchmarks."""
    print_header("Activation Functions")

    configs = [
        (8, 4096, "8×4096"),
        (32, 4096, "32×4096"),
        (64, 8192, "64×8192"),
        (128, 16384, "128×16384"),
    ]

    print(f"\n  GELU:")
    print(f"  {'Shape':<15} {'Time (ms)':<12}")
    print("  " + "-" * 30)

    for batch, hidden, name in configs:
        time_s = benchmark_gelu(batch, hidden, device)
        print(f"  {name:<15} {time_s*1000:>10.4f}")


def run_attention_benchmarks(device):
    """Run attention-related benchmarks."""
    print_header("Attention Operations")

    configs = [
        (1, 512, 32, "1×512×32"),
        (1, 1024, 32, "1×1024×32"),
        (1, 2048, 32, "1×2048×32"),
        (4, 1024, 32, "4×1024×32"),
        (8, 512, 64, "8×512×64"),
    ]

    print(f"\n  Softmax (attention scores):")
    print(f"  {'B×S×H':<15} {'Time (ms)':<12}")
    print("  " + "-" * 30)

    for batch, seq_len, heads, name in configs:
        time_s = benchmark_softmax(batch, seq_len, heads, device)
        print(f"  {name:<15} {time_s*1000:>10.4f}")


def run_fusion_benchmarks(device):
    """Run kernel fusion benchmarks (YiRage optimization target)."""
    print_header("Kernel Fusion Candidates (YiRage Optimization)")

    configs = [
        (512, 512, 1024, "Small"),
        (1024, 1024, 4096, "Medium"),
        (2048, 4096, 4096, "Large"),
    ]

    print(f"\n  MatMul + GELU (unfused baseline):")
    print(f"  {'Config':<10} {'M×N×K':<18} {'Time (ms)':<12}")
    print("  " + "-" * 45)

    for M, N, K, name in configs:
        time_s = benchmark_fused_matmul_gelu(M, N, K, device)
        print(f"  {name:<10} {M}×{N}×{K:<8} {time_s*1000:>10.4f}")

    print("\n  YiRage can fuse these operations to reduce memory traffic!")


def main():
    print()
    print("=" * 70)
    print("  MetaX C500 GPU Performance Benchmark")
    print("  PyTorch (mcPytorch) vs YiRage Optimization")
    print("=" * 70)

    # Check environment
    print_header("Environment")

    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n  ERROR: CUDA not available. Run with /opt/conda/bin/python")
        return

    device = torch.device("cuda:0")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # YiRage info
    if YIRAGE_AVAILABLE:
        info = get_maca_device_info()
        print(f"\n  YiRage MACA Config:")
        print(f"    Warp size: {MACA_WARP_SIZE}")
        if info:
            print(f"    SM count: {info.get('sm_count', 'N/A')}")

    # Run benchmarks
    matmul_results = run_matmul_benchmarks(device)
    run_normalization_benchmarks(device)
    run_activation_benchmarks(device)
    run_attention_benchmarks(device)
    run_fusion_benchmarks(device)

    # Summary
    print_header("Performance Summary")

    # Find peak TFLOPS
    peak_tflops = max(r[5] for r in matmul_results)

    print(
        f"""
  MetaX C500 Performance:
  
  Peak MatMul TFLOPS: {peak_tflops:.2f} TFLOPS (FP16)
  
  YiRage Optimization Opportunities:
  1. Kernel Fusion (MatMul+Activation) - reduce memory traffic
  2. Custom block sizes aligned to 64-thread warps
  3. Shared memory optimization for reductions
  4. Tensor core tile sizes (64×64, 128×64)
  
  Note: YiRage superoptimization searches for optimal
  kernel configurations specific to C500 architecture.
"""
    )

    print_header("Benchmark Complete")


if __name__ == "__main__":
    main()
