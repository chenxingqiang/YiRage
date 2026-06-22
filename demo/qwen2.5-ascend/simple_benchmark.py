#!/usr/bin/env python3
"""
Simple benchmark for YiRage on Ascend NPU
Tests verified kernel patterns without model dependencies
"""

import time
import torch
import torch_npu
import yirage as yr


def benchmark_matmul(hidden_size, output_size, warmup=3, iterations=20):
    """Benchmark MatMul with YiRage optimization"""
    device = "npu:0"

    # Create test tensors
    X = torch.randn(1, hidden_size, dtype=torch.bfloat16, device=device)
    W = torch.randn(hidden_size, output_size, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(X, W)
    torch.npu.synchronize()

    # Benchmark PyTorch
    start = time.time()
    for _ in range(iterations):
        _ = torch.matmul(X, W)
    torch.npu.synchronize()
    pytorch_time = (time.time() - start) / iterations * 1000

    return pytorch_time


def benchmark_yirage_search(hidden_size, output_size):
    """Benchmark YiRage kernel search"""
    print(f"\n=== YiRage Search: MatMul ({hidden_size}x{output_size}) ===")

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(1, hidden_size), dtype=yr.bfloat16)
    W = graph.new_input(
        dims=(hidden_size, output_size), strides=(1, hidden_size), dtype=yr.bfloat16
    )
    O = graph.matmul(X, W)
    graph.mark_output(O)

    start = time.time()
    result = graph.superoptimize(
        imaps=[(-1, -1, -1)],
        omaps=[(-1, -1, -1)],
        griddims=[(1, 1, 1)],
        blockdims=[(1, 1, 1)],
        fmaps=[-1],
        franges=[4],
        backend="ascend",
        verbose=False,
    )
    search_time = time.time() - start

    return result, search_time


def benchmark_yirage_rms_matmul(hidden_size, output_size):
    """Benchmark YiRage RMSNorm + MatMul search"""
    print(f"\n=== YiRage Search: RMSNorm + MatMul ({hidden_size}x{output_size}) ===")

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(1, hidden_size), dtype=yr.bfloat16)
    W = graph.new_input(
        dims=(hidden_size, output_size), strides=(1, hidden_size), dtype=yr.bfloat16
    )
    D = graph.rms_norm(X, normalized_shape=(hidden_size,))
    O = graph.matmul(D, W)
    graph.mark_output(O)

    start = time.time()
    result = graph.superoptimize(
        imaps=[(-1, -1, -1)],
        omaps=[(-1, -1, -1)],
        griddims=[(1, 1, 1)],
        blockdims=[(1, 1, 1)],
        fmaps=[-1],
        franges=[4],
        backend="ascend",
        verbose=False,
    )
    search_time = time.time() - start

    return result, search_time


def main():
    print("=" * 60)
    print("YiRage Ascend NPU Benchmark")
    print("=" * 60)

    # Check NPU availability
    if not torch.npu.is_available():
        print("ERROR: Ascend NPU not available")
        return

    device_name = torch.npu.get_device_name(0)
    print(f"Device: {device_name}")

    # Qwen2.5-0.5B dimensions
    hidden_size = 896
    intermediate_size = 4864

    print(f"\nModel dimensions (Qwen2.5-0.5B):")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")

    # Benchmark PyTorch MatMul
    print("\n--- PyTorch Baseline ---")
    pytorch_time = benchmark_matmul(hidden_size, intermediate_size)
    print(f"PyTorch MatMul: {pytorch_time:.4f} ms")

    # Benchmark YiRage MatMul search
    result1, search_time1 = benchmark_yirage_search(hidden_size, intermediate_size)
    if result1:
        print(f"✓ YiRage found optimized kernel (search: {search_time1:.1f}s)")
    else:
        print(f"✗ No kernel found (search: {search_time1:.1f}s)")

    # Benchmark YiRage RMSNorm + MatMul search
    result2, search_time2 = benchmark_yirage_rms_matmul(hidden_size, intermediate_size)
    if result2:
        print(f"✓ YiRage found optimized kernel (search: {search_time2:.1f}s)")
    else:
        print(f"✗ No kernel found (search: {search_time2:.1f}s)")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  PyTorch MatMul baseline: {pytorch_time:.4f} ms")
    print(f"  YiRage MatMul kernel: {'Found' if result1 else 'Not found'}")
    print(f"  YiRage RMSNorm+MatMul kernel: {'Found' if result2 else 'Not found'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
