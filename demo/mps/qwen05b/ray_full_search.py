#!/usr/bin/env python3
"""
YiRage Ray Distributed Search with Full MPS Config

Demonstrates parallel kernel search using Ray with complete MPS configuration:
- 15 grid dimensions (SIMD-aligned)
- 12 block dimensions (threadgroup-optimized)  
- 4 fmaps (forloop dimension mappings)
- 3 franges (forloop ranges)

Total: 2160 configuration combinations
"""

import ray
import time
import torch
import itertools
from typing import List, Tuple, Dict

# Check Ray
try:
    ray.init(ignore_reinit_error=True, logging_level="WARNING")
except Exception as e:
    print(f"Ray init failed: {e}")
    exit(1)

print("=" * 70)
print("  YiRage Ray Distributed Search - Full MPS Config")
print("=" * 70)
print(f"Ray CPUs: {ray.cluster_resources().get('CPU', 0):.0f}")


@ray.remote
def search_partition(
    partition_id: int,
    hidden_size: int,
    output_size: int,
    grid_dims: List[Tuple],
    block_dims: List[Tuple],
    fmaps: List[int],
    franges: List[int],
) -> Dict:
    """
    Worker function to search a partition of the configuration space.

    Each worker gets a subset of grid_dims to explore with all block/fmap/frange combinations.
    """
    import yirage as yr
    from yirage.core import search
    import time as _time

    start = _time.perf_counter()

    # Create graph inside worker
    graph = yr.new_kernel_graph()
    X = graph.new_input(dims=(1, hidden_size), dtype=yr.float16)
    W = graph.new_input(dims=(hidden_size, output_size), dtype=yr.float16)
    O = graph.matmul(X, W)
    graph.mark_output(O)

    # Search with assigned partition
    try:
        cygraphs = search(
            graph.cygraph,
            backend="mps",
            griddims=grid_dims,
            blockdims=block_dims,
            fmaps=fmaps,
            franges=franges,
            verbose=False,
            is_formal_verified=False,
        )
        num_graphs = len(cygraphs)
        success = True
    except Exception as e:
        num_graphs = 0
        success = False

    elapsed = _time.perf_counter() - start

    return {
        "partition_id": partition_id,
        "num_graphs": num_graphs,
        "grid_dims": len(grid_dims),
        "block_dims": len(block_dims),
        "fmaps": fmaps,
        "franges": franges,
        "elapsed": elapsed,
        "success": success,
    }


def main():
    # Qwen2.5-0.5B QKV projection
    hidden_size = 896
    qkv_dim = 1152  # (14 + 2*2) * 64

    print(f"\nQwen2.5-0.5B QKV Projection: ({hidden_size} -> {qkv_dim})")

    # Full MPS config from mps_config.py
    all_grid_dims = [
        (32, 1, 1),
        (64, 1, 1),
        (96, 1, 1),
        (128, 1, 1),
        (160, 1, 1),
        (192, 1, 1),
        (224, 1, 1),
        (256, 1, 1),
        (320, 1, 1),
        (384, 1, 1),
        (512, 1, 1),
        (32, 2, 1),
        (64, 2, 1),
        (32, 4, 1),
        (64, 4, 1),
    ]

    all_block_dims = [
        (32, 1, 1),
        (64, 1, 1),
        (96, 1, 1),
        (128, 1, 1),
        (160, 1, 1),
        (192, 1, 1),
        (224, 1, 1),
        (256, 1, 1),
        (320, 1, 1),
        (384, 1, 1),
        (448, 1, 1),
        (512, 1, 1),
    ]

    all_fmaps = [-1, 0, 1, 2]
    all_franges = [4, 8, 16]

    # Calculate total combinations
    total_combos = len(all_grid_dims) * len(all_block_dims) * len(all_fmaps) * len(all_franges)
    print(f"\nFull MPS Config:")
    print(f"  Grid dims: {len(all_grid_dims)}")
    print(f"  Block dims: {len(all_block_dims)}")
    print(f"  Fmaps: {all_fmaps}")
    print(f"  Franges: {all_franges}")
    print(f"  Total combinations: {total_combos}")

    # Partition by grid dims (each worker gets subset of grids)
    num_workers = min(8, len(all_grid_dims))
    grids_per_worker = (len(all_grid_dims) + num_workers - 1) // num_workers

    print(f"\nDistributed Search:")
    print(f"  Workers: {num_workers}")
    print(f"  Grids per worker: ~{grids_per_worker}")

    # Create partitions
    partitions = []
    for i in range(num_workers):
        start_idx = i * grids_per_worker
        end_idx = min(start_idx + grids_per_worker, len(all_grid_dims))
        if start_idx >= len(all_grid_dims):
            break
        partitions.append(all_grid_dims[start_idx:end_idx])

    print(f"  Actual partitions: {len(partitions)}")

    # Launch parallel searches
    print("\nStarting parallel search...")
    start_time = time.perf_counter()

    futures = [
        search_partition.remote(
            partition_id=i,
            hidden_size=hidden_size,
            output_size=qkv_dim,
            grid_dims=partitions[i],
            block_dims=all_block_dims,
            fmaps=all_fmaps,
            franges=all_franges,
        )
        for i in range(len(partitions))
    ]

    # Wait for results
    results = ray.get(futures)
    total_time = time.perf_counter() - start_time

    # Aggregate results
    total_graphs = sum(r["num_graphs"] for r in results)
    max_worker_time = max(r["elapsed"] for r in results)

    print("\n" + "=" * 70)
    print("  Results")
    print("=" * 70)

    for r in results:
        status = "✓" if r["success"] else "✗"
        print(
            f"  Worker {r['partition_id']}: {r['num_graphs']:3d} muGraphs, "
            f"{r['grid_dims']} grids, {r['elapsed']:.1f}s {status}"
        )

    print("\n" + "-" * 70)
    print(f"  Total muGraphs: {total_graphs}")
    print(f"  Parallel time: {total_time:.1f}s")
    print(f"  Max worker time: {max_worker_time:.1f}s")
    print(f"  Estimated sequential: {sum(r['elapsed'] for r in results):.1f}s")
    print(f"  Speedup: {sum(r['elapsed'] for r in results) / total_time:.1f}x")

    ray.shutdown()
    print("\n" + "=" * 70)
    print("  ✅ Ray Full MPS Search Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
