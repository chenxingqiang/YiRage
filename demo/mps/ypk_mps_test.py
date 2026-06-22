#!/usr/bin/env python3
"""
YPK MPS Full Test - Ray Distributed Search with Auto-Save

Features:
- Ray distributed search enabled by default
- Automatic muGraph storage to persistent cache
- Full MPS search configuration for Apple Silicon
- Qwen2.5-0.5B model layer optimization
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from demo._device_utils import ensure_native_ld_library_path, ensure_repo_on_path, require_mps

ensure_repo_on_path()
ensure_native_ld_library_path()

import torch
import yirage as yr
from yirage.mps_config import get_mps_search_config, detect_apple_silicon


def main():
    require_mps("YPK MPS test requires Apple Silicon MPS.")

    print("=" * 70)
    print("  YiRage YPK MPS Test - Ray Mode + Auto-Save")
    print("=" * 70)

    # Detect Apple Silicon
    chip_family, specs = detect_apple_silicon()
    print(f"\nDevice: {chip_family.value} - {specs.chip_name}")
    print(f"GPU Cores: {specs.gpu_cores}")
    print(f"Memory BW: {specs.memory_bandwidth_gbps} GB/s")

    # Get MPS search config
    mps_config = get_mps_search_config()
    print(f"\nSearch Configuration:")
    print(f"  Grid dims: {len(mps_config['grid_dims_to_explore'])} configs")
    print(f"  Block dims: {len(mps_config['block_dims_to_explore'])} configs")
    print(f"  Fmaps: {mps_config['fmaps_to_explore']}")
    print(f"  Franges: {mps_config['franges_to_explore']}")

    # Qwen2.5-0.5B dimensions
    hidden_size = 896
    intermediate_size = 4864
    batch_size = 1
    seq_len = 256

    print(f"\nModel: Qwen2.5-0.5B")
    print(f"  hidden_size: {hidden_size}")
    print(f"  intermediate_size: {intermediate_size}")
    print(f"  batch: {batch_size}, seq: {seq_len}")

    # Test kernels - MatMul only for now
    kernels = [
        ("qkv_proj", (batch_size, seq_len, hidden_size)),
        ("output_proj", (batch_size, seq_len, hidden_size)),
    ]

    results = []

    for kernel_name, input_shape in kernels:
        print(f"\n{'='*70}")
        print(f"  Optimizing: {kernel_name}")
        print(f"  Input: {input_shape}")
        print(f"{'='*70}")

        try:
            # Create MatMul graph
            graph = yr.new_kernel_graph()

            if kernel_name == "qkv_proj":
                # QKV Projection: X @ W_qkv (256x896 @ 896x2688)
                M = input_shape[0] * input_shape[1]
                K = input_shape[2]
                N = hidden_size * 3  # 2688
                X = graph.new_input(dims=(M, K), dtype=yr.float16)
                W = graph.new_input(dims=(K, N), dtype=yr.float16)
                output = graph.matmul(X, W)

            elif kernel_name == "output_proj":
                # Output Projection: X @ W_o (256x896 @ 896x896)
                M = input_shape[0] * input_shape[1]
                K = input_shape[2]
                N = hidden_size
                X = graph.new_input(dims=(M, K), dtype=yr.float16)
                W = graph.new_input(dims=(K, N), dtype=yr.float16)
                output = graph.matmul(X, W)

            graph.mark_output(output)

            # Run superoptimize with Ray mode (default)
            # use_ray=True, use_persistent_cache=True are defaults
            optimized = graph.superoptimize(
                backend="mps",
                verbose=True,
                # use_ray=True,  # Already default
                # use_persistent_cache=True,  # Already default
                num_workers=8,
            )

            if optimized:
                print(f"\n✓ {kernel_name}: Optimization successful")
                results.append((kernel_name, "SUCCESS", optimized))
            else:
                print(f"\n⚠ {kernel_name}: No valid muGraph found")
                results.append((kernel_name, "NO_MUGRAPH", None))

        except Exception as e:
            print(f"\n✗ {kernel_name}: Error - {e}")
            results.append((kernel_name, "ERROR", str(e)))

    # Summary
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")

    for name, status, _ in results:
        print(f"  {name}: {status}")

    # Check storage
    print(f"\n{'='*70}")
    print("  Stored MuGraphs")
    print(f"{'='*70}")

    from yirage.mugraph_store import get_mugraph_store

    store = get_mugraph_store()
    stats = store.get_stats()
    print(f"\nTotal entries: {stats['total_entries']}")
    print(f"MPS entries: {stats['by_backend'].get('mps', {}).get('count', 0)}")

    entries = store.list_all(backend="mps", limit=10)
    for e in entries:
        print(f"  - {e.metadata.graph_hash}: {e.metadata.latency_ms:.4f}ms")

    return 0


if __name__ == "__main__":
    sys.exit(main())
