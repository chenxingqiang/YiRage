#!/usr/bin/env python3
"""
Test script for V100 (Volta, sm_70) GPU support with PyTorch comparison.

Usage:
    cd /root/YiRage
    export PATH=/usr/local/cuda-12.1/bin:$PATH
    export PYTHONPATH=/root/YiRage/python:$PYTHONPATH
    export LD_LIBRARY_PATH=/root/YiRage/build/abstract_subexpr/release:/root/YiRage/build/formal_verifier/release:/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
    python3 -u tests/runtime_python/test_v100_matmul.py
"""

import time
import torch
import yirage as yr


def log(msg):
    """Print with immediate flush."""
    print(msg, flush=True)


def benchmark_pytorch(X_t, W_t, warmup=10, iters=100):
    """Benchmark PyTorch matmul."""
    # Warmup
    for _ in range(warmup):
        _ = torch.matmul(X_t, W_t)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(X_t, W_t)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iters * 1000  # ms


def benchmark_yirage(result, X_t, W_t, warmup=10, iters=100):
    """Benchmark YiRage kernel."""
    # Warmup
    for _ in range(warmup):
        _ = result(inputs=[X_t, W_t])
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = result(inputs=[X_t, W_t])
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iters * 1000  # ms


def test_small_matmul():
    """Test small matrix multiplication on V100."""
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    
    cc = torch.cuda.get_device_capability()
    log(f"Compute Capability: {cc[0]}.{cc[1]}")
    
    # Test dimensions
    M, K, N = 8, 64, 64
    log(f"\nMatrix dimensions: ({M}, {K}) x ({K}, {N})")
    
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(M, K), dtype=yr.float16)
    W = graph.new_input(dims=(K, N), dtype=yr.float16)
    O = graph.matmul(X, W)
    graph.mark_output(O)

    log("\nStarting superoptimize...")
    result = graph.superoptimize(backend="cuda")
    log("Superoptimize SUCCESS!")

    # Create test tensors
    X_t = torch.randn(M, K, dtype=torch.float16, device="cuda")
    W_t = torch.randn(K, N, dtype=torch.float16, device="cuda")
    
    # Test execution
    outputs = result(inputs=[X_t, W_t])
    log("Execution SUCCESS!")

    # Verify correctness
    expected = torch.matmul(X_t, W_t)
    diff = torch.abs(outputs[0] - expected).max().item()
    log(f"Max diff from torch.matmul: {diff}")
    
    # Benchmark comparison
    log("\n=== Performance Comparison ===")
    pytorch_time = benchmark_pytorch(X_t, W_t)
    yirage_time = benchmark_yirage(result, X_t, W_t)
    
    log(f"PyTorch matmul:  {pytorch_time:.4f} ms")
    log(f"YiRage kernel:   {yirage_time:.4f} ms")
    log(f"Speedup:         {pytorch_time / yirage_time:.2f}x")
    
    assert diff < 0.1, f"Result differs too much: {diff}"
    log("\nTest PASSED!")


if __name__ == "__main__":
    test_small_matmul()
