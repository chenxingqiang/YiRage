#!/usr/bin/env python3
"""
YiRage MPS Qwen2.5-0.5B Complete Demo

This demo runs all Qwen2.5-0.5B layer kernels on MPS with YiRage optimization.
Search takes ~60s per kernel, total ~4 minutes.

Usage:
    python demo/mps/qwen05b/run_demo.py
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from demo._device_utils import ensure_native_ld_library_path, ensure_repo_on_path, require_mps

ensure_repo_on_path()
ensure_native_ld_library_path()

import torch
import torch.nn.functional as F
import yirage as yr
from yirage.core import search
from yirage.kernel import KNGraph


def main():
    require_mps("Qwen2.5-0.5B MPS demo requires Apple Silicon MPS.")

    print("=" * 70)
    print("  YiRage MPS Qwen2.5-0.5B Complete Demo")
    print("=" * 70)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  YiRage: {yr.__version__}")
    print(f"  MPS: {torch.backends.mps.is_available()}")

    # Qwen2.5-0.5B config
    hidden_size = 896
    num_layers = 24
    num_heads = 14
    num_kv_heads = 2
    head_dim = hidden_size // num_heads
    qkv_dim = (num_heads + 2 * num_kv_heads) * head_dim
    intermediate_size = 4864
    batch_size = 1

    print(f"\nQwen2.5-0.5B Architecture:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  qkv_dim: {qkv_dim}")

    results = []
    total_start = time.perf_counter()

    # Test 1: QKV Projection
    print("\n" + "-" * 70)
    print("  [1/4] QKV Projection MatMul")
    print("-" * 70)
    print(f"  Shape: ({batch_size}, {hidden_size}) @ ({hidden_size}, {qkv_dim})")

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size, hidden_size), dtype=yr.float16)
    W = graph.new_input(dims=(hidden_size, qkv_dim), dtype=yr.float16)
    O = graph.matmul(X, W)
    graph.mark_output(O)

    print("  Searching...")
    start = time.perf_counter()
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
    search_time = time.perf_counter() - start
    print(f"  Found {len(cygraphs)} muGraphs in {search_time:.1f}s")

    if len(cygraphs) > 0:
        input_x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="mps")
        input_w = torch.randn(hidden_size, qkv_dim, dtype=torch.float16, device="mps")

        torch.mps.synchronize()
        for _ in range(10):
            ref = torch.matmul(input_x, input_w)
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.matmul(input_x, input_w)
        torch.mps.synchronize()
        pt = (time.perf_counter() - start) / 100 * 1000

        g = KNGraph(cygraphs[0], backend="mps")
        for _ in range(10):
            outputs = g(inputs=[input_x, input_w])
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            outputs = g(inputs=[input_x, input_w])
        torch.mps.synchronize()
        yr_t = (time.perf_counter() - start) / 100 * 1000

        speedup = pt / yr_t
        print(f"  PyTorch: {pt:.4f}ms, YiRage: {yr_t:.4f}ms, Speedup: {speedup:.2f}x")
        results.append(("QKV Proj", speedup, pt, yr_t))

    # Test 2: Output Projection
    print("\n" + "-" * 70)
    print("  [2/4] Output Projection")
    print("-" * 70)
    o_dim = num_heads * head_dim
    print(f"  Shape: ({batch_size}, {o_dim}) @ ({o_dim}, {hidden_size})")

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size, o_dim), dtype=yr.float16)
    W = graph.new_input(dims=(o_dim, hidden_size), dtype=yr.float16)
    O = graph.matmul(X, W)
    graph.mark_output(O)

    print("  Searching...")
    start = time.perf_counter()
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
    search_time = time.perf_counter() - start
    print(f"  Found {len(cygraphs)} muGraphs in {search_time:.1f}s")

    if len(cygraphs) > 0:
        input_x = torch.randn(batch_size, o_dim, dtype=torch.float16, device="mps")
        input_w = torch.randn(o_dim, hidden_size, dtype=torch.float16, device="mps")

        torch.mps.synchronize()
        for _ in range(10):
            ref = torch.matmul(input_x, input_w)
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.matmul(input_x, input_w)
        torch.mps.synchronize()
        pt = (time.perf_counter() - start) / 100 * 1000

        g = KNGraph(cygraphs[0], backend="mps")
        for _ in range(10):
            outputs = g(inputs=[input_x, input_w])
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            outputs = g(inputs=[input_x, input_w])
        torch.mps.synchronize()
        yr_t = (time.perf_counter() - start) / 100 * 1000

        speedup = pt / yr_t
        print(f"  PyTorch: {pt:.4f}ms, YiRage: {yr_t:.4f}ms, Speedup: {speedup:.2f}x")
        results.append(("O Proj", speedup, pt, yr_t))

    # Test 3: Gate+Up Projection (for MLP)
    print("\n" + "-" * 70)
    print("  [3/4] Gate+Up Projection (MLP)")
    print("-" * 70)
    print(f"  Shape: ({batch_size}, {hidden_size}) @ ({hidden_size}, {intermediate_size}) x2")

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size, hidden_size), dtype=yr.float16)
    W_gate = graph.new_input(dims=(hidden_size, intermediate_size), dtype=yr.float16)
    W_up = graph.new_input(dims=(hidden_size, intermediate_size), dtype=yr.float16)
    gate = graph.matmul(X, W_gate)
    up = graph.matmul(X, W_up)
    gate_silu = graph.silu(gate)
    O = graph.mul(gate_silu, up)
    graph.mark_output(O)

    print("  Searching...")
    start = time.perf_counter()
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
    search_time = time.perf_counter() - start
    print(f"  Found {len(cygraphs)} muGraphs in {search_time:.1f}s")

    if len(cygraphs) > 0:
        input_x = torch.randn(batch_size, hidden_size, dtype=torch.float16, device="mps")
        w_gate = torch.randn(hidden_size, intermediate_size, dtype=torch.float16, device="mps")
        w_up = torch.randn(hidden_size, intermediate_size, dtype=torch.float16, device="mps")

        torch.mps.synchronize()
        for _ in range(10):
            g_out = torch.matmul(input_x, w_gate)
            u_out = torch.matmul(input_x, w_up)
            ref = F.silu(g_out) * u_out
        start = time.perf_counter()
        for _ in range(100):
            g_out = torch.matmul(input_x, w_gate)
            u_out = torch.matmul(input_x, w_up)
            _ = F.silu(g_out) * u_out
        torch.mps.synchronize()
        pt = (time.perf_counter() - start) / 100 * 1000

        g = KNGraph(cygraphs[0], backend="mps")
        for _ in range(10):
            outputs = g(inputs=[input_x, w_gate, w_up])
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            outputs = g(inputs=[input_x, w_gate, w_up])
        torch.mps.synchronize()
        yr_t = (time.perf_counter() - start) / 100 * 1000

        speedup = pt / yr_t
        print(f"  PyTorch: {pt:.4f}ms, YiRage: {yr_t:.4f}ms, Speedup: {speedup:.2f}x")
        results.append(("Gate+Up MLP", speedup, pt, yr_t))

    # Test 4: Down Projection
    print("\n" + "-" * 70)
    print("  [4/4] Down Projection (MLP)")
    print("-" * 70)
    print(f"  Shape: ({batch_size}, {intermediate_size}) @ ({intermediate_size}, {hidden_size})")

    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(batch_size, intermediate_size), dtype=yr.float16)
    W = graph.new_input(dims=(intermediate_size, hidden_size), dtype=yr.float16)
    O = graph.matmul(X, W)
    graph.mark_output(O)

    print("  Searching...")
    start = time.perf_counter()
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
    search_time = time.perf_counter() - start
    print(f"  Found {len(cygraphs)} muGraphs in {search_time:.1f}s")

    if len(cygraphs) > 0:
        input_x = torch.randn(batch_size, intermediate_size, dtype=torch.float16, device="mps")
        input_w = torch.randn(intermediate_size, hidden_size, dtype=torch.float16, device="mps")

        torch.mps.synchronize()
        for _ in range(10):
            ref = torch.matmul(input_x, input_w)
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.matmul(input_x, input_w)
        torch.mps.synchronize()
        pt = (time.perf_counter() - start) / 100 * 1000

        g = KNGraph(cygraphs[0], backend="mps")
        for _ in range(10):
            outputs = g(inputs=[input_x, input_w])
        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(100):
            outputs = g(inputs=[input_x, input_w])
        torch.mps.synchronize()
        yr_t = (time.perf_counter() - start) / 100 * 1000

        speedup = pt / yr_t
        print(f"  PyTorch: {pt:.4f}ms, YiRage: {yr_t:.4f}ms, Speedup: {speedup:.2f}x")
        results.append(("Down Proj", speedup, pt, yr_t))

    total_time = time.perf_counter() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)

    total_pt = 0
    total_yr = 0
    for name, speedup, pt, yr_t in results:
        print(f"  {name:15s}: PyTorch={pt:.4f}ms, YiRage={yr_t:.4f}ms, Speedup={speedup:.2f}x")
        total_pt += pt
        total_yr += yr_t

    if results:
        avg = sum(s for _, s, _, _ in results) / len(results)
        overall_speedup = total_pt / total_yr

        print(f"\n  Per-kernel average speedup: {avg:.2f}x")
        print(f"  Overall layer speedup: {overall_speedup:.2f}x")

        # Estimate full model
        print(f"\n  Qwen2.5-0.5B Model Estimate:")
        print(f"    Layers: {num_layers}")
        print(f"    PyTorch per layer: {total_pt:.4f}ms")
        print(f"    YiRage per layer: {total_yr:.4f}ms")
        print(f"    PyTorch full model: {total_pt * num_layers:.2f}ms")
        print(f"    YiRage full model: {total_yr * num_layers:.2f}ms")
        print(f"    Estimated speedup: {overall_speedup:.2f}x")

    print(f"\n  Total demo time: {total_time:.1f}s")
    print("\n" + "=" * 70)
    print("  🎉 Qwen2.5-0.5B MPS Demo Complete!")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
