#!/usr/bin/env python3
"""
YiRage MPS MatMul Demo

This demo shows how to use YiRage on Apple Silicon MPS:
1. Create a kernel graph for MatMul
2. Search for optimal muGraphs
3. Execute and profile the optimized kernel

Usage:
    python demo/mps/demo_mps_matmul.py

Author: YiRage Team
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo._device_utils import ensure_native_ld_library_path, ensure_repo_on_path, require_mps

ensure_repo_on_path()
ensure_native_ld_library_path()

import torch
import yirage as yr
from yirage.core import search
from yirage.kernel import KNGraph


def main():
    require_mps("YiRage MPS MatMul demo requires Apple Silicon MPS.")

    print("=" * 60)
    print("  YiRage MPS MatMul Demo")
    print("=" * 60)

    print(f"\n  PyTorch: {torch.__version__}")
    print(f"  YiRage: {yr.__version__}")
    print(f"  MPS: {torch.backends.mps.is_available()}")
    print(f"  Backends: {yr.get_available_backends()}")

    # Problem configuration (small for quick demo)
    M, K, N = 8, 256, 256
    print(f"\n  Problem: ({M}, {K}) @ ({K}, {N}) FP16 MatMul")

    # Step 1: Create kernel graph
    print("\n[Step 1] Create Kernel Graph")
    print("-" * 40)

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(M, K), dtype=yr.float16)
    W = graph.new_input(dims=(K, N), dtype=yr.float16)
    O = graph.matmul(X, W)
    graph.mark_output(O)
    print("  ✅ Kernel graph created")

    # Step 2: Search for optimal muGraphs
    print("\n[Step 2] Search for muGraphs")
    print("-" * 40)

    print("  Searching...")
    start_search = time.perf_counter()

    cygraphs = search(
        graph.cygraph,
        backend="mps",
        griddims=[(1, 1, 1)],
        blockdims=[(32, 1, 1)],
        fmaps=[-1],
        franges=[4],
        verbose=False,
        is_formal_verified=False,
    )

    search_time = time.perf_counter() - start_search
    print(f"  ✅ Found {len(cygraphs)} muGraphs in {search_time:.2f}s")

    if len(cygraphs) == 0:
        print("  ❌ No muGraph found!")
        return 1

    # Step 3: Profile and select best
    print("\n[Step 3] Profile muGraphs")
    print("-" * 40)

    input_x = torch.randn(M, K, dtype=torch.float16, device="mps")
    input_w = torch.randn(K, N, dtype=torch.float16, device="mps")

    # PyTorch baseline
    torch.mps.synchronize()
    ref = torch.matmul(input_x, input_w)

    for _ in range(10):
        _ = torch.matmul(input_x, input_w)
    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = torch.matmul(input_x, input_w)
    torch.mps.synchronize()
    pytorch_time = (time.perf_counter() - start) / 100 * 1000
    print(f"  PyTorch: {pytorch_time:.4f} ms")

    # Profile top muGraphs
    best_graph = None
    best_time = float("inf")

    num_to_profile = min(5, len(cygraphs))
    print(f"\n  Profiling top {num_to_profile} muGraphs...")

    for idx, cygraph in enumerate(cygraphs[:num_to_profile]):
        try:
            g = KNGraph(cygraph, backend="mps")

            # Warmup
            for _ in range(10):
                outputs = g(inputs=[input_x, input_w])
            torch.mps.synchronize()

            # Profile
            start = time.perf_counter()
            for _ in range(100):
                outputs = g(inputs=[input_x, input_w])
            torch.mps.synchronize()
            elapsed = (time.perf_counter() - start) / 100 * 1000

            # Verify
            max_diff = (outputs[0].cpu() - ref.cpu()).abs().max().item()

            if elapsed < best_time:
                best_time = elapsed
                best_graph = g

            speedup = pytorch_time / elapsed
            status = "✓" if max_diff < 0.1 else "✗"
            print(
                f"    muGraph[{idx}]: {elapsed:.4f} ms ({speedup:.2f}x), diff={max_diff:.6f} {status}"
            )

        except Exception as e:
            print(f"    muGraph[{idx}]: Error - {e}")

    # Step 4: Results
    print("\n[Step 4] Results")
    print("-" * 40)

    if best_graph:
        speedup = pytorch_time / best_time
        print(f"  Best muGraph: {best_time:.4f} ms")
        print(f"  PyTorch:      {pytorch_time:.4f} ms")
        print(f"  Speedup:      {speedup:.2f}x")

        # Calculate FLOPS
        flops = 2 * M * K * N
        tflops = flops / (best_time / 1000) / 1e12
        print(f"\n  FLOPs: {flops / 1e6:.2f} MFLOPs")
        print(f"  Throughput: {tflops:.4f} TFLOPS")

        print("\n  🎉 Demo completed successfully!")
        return 0
    else:
        print("  ❌ No valid muGraph executed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
