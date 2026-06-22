#!/usr/bin/env python3
"""
MuGraph Persistent Storage Demo

Demonstrates the MuGraphStore functionality for saving and loading
optimized muGraphs across sessions.

Storage Directory Structure:
    ~/.yirage/mugraphs/
    ├── mps/           # Apple Silicon MPS graphs
    ├── cuda/          # NVIDIA CUDA graphs
    ├── cpu/           # CPU graphs
    └── index.json     # Global index
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from demo._device_utils import ensure_native_ld_library_path, ensure_repo_on_path

ensure_repo_on_path()
ensure_native_ld_library_path()

import time
from pathlib import Path


def test_mugraph_store():
    """Test MuGraphStore functionality."""
    print("=" * 70)
    print("  MuGraph Persistent Storage Demo")
    print("=" * 70)

    from yirage.mugraph_store import (
        MuGraphStore,
        MuGraphMetadata,
        MuGraphEntry,
        get_mugraph_store,
        save_mugraph,
        find_mugraph,
        find_best_mugraph,
    )

    # Initialize store
    store = get_mugraph_store()
    print(f"\n✓ MuGraphStore initialized")
    print(f"  Root path: {store.root_path}")

    # Check directory structure
    print("\n--- Directory Structure ---")
    for backend_dir in store.root_path.iterdir():
        if backend_dir.is_dir():
            count = len(list(backend_dir.glob("*.json")))
            print(f"  {backend_dir.name}/: {count} entries")

    # Get stats
    print("\n--- Storage Statistics ---")
    stats = store.get_stats()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total size: {stats['total_size_bytes'] / 1024:.2f} KB")

    for backend, info in stats["by_backend"].items():
        if info["count"] > 0:
            print(f"  {backend}: {info['count']} entries ({info['size_bytes']/1024:.2f} KB)")

    if stats["avg_latency_ms"] > 0:
        print(f"\n  Latency Stats:")
        print(f"    Average: {stats['avg_latency_ms']:.4f} ms")
        print(f"    Best:    {stats['best_latency_ms']:.4f} ms")
        print(f"    Worst:   {stats['worst_latency_ms']:.4f} ms")

    # Test save functionality with mock data
    print("\n--- Testing Save Functionality ---")

    # Create a mock muGraph object
    class MockGraph:
        def __init__(self):
            self.backend = "mps"
            self.cygraph = None

    mock_graph = MockGraph()
    test_hash = f"test_{int(time.time())}"

    path = store.save(
        graph_hash=test_hash,
        optimized_graph=mock_graph,
        backend="mps",
        imaps=[[0, 1], [1, 0]],
        omaps=[[0]],
        griddims=[(4, 1, 1), (8, 1, 1)],
        blockdims=[(64, 1, 1), (128, 1, 1)],
        fmaps=[0, 1],
        franges=[1, 2, 4],
        latency_ms=1.234,
        memory_bytes=1024 * 1024,
        num_candidates_searched=42,
        search_time_s=5.5,
        input_shapes=[[32, 128], [128, 64]],
        output_shapes=[[32, 64]],
        device_name="Apple M1 Pro",
        device_info={"metal_family": "Apple7"},
    )
    print(f"  ✓ Saved test entry to: {path}")

    # Test find functionality
    print("\n--- Testing Find Functionality ---")
    entry = store.find(
        graph_hash=test_hash,
        backend="mps",
        imaps=[[0, 1], [1, 0]],
        omaps=[[0]],
        griddims=[(4, 1, 1), (8, 1, 1)],
        blockdims=[(64, 1, 1), (128, 1, 1)],
        fmaps=[0, 1],
        franges=[1, 2, 4],
    )

    if entry:
        print(f"  ✓ Found cached entry!")
        print(f"    - Graph hash: {entry.metadata.graph_hash}")
        print(f"    - Backend: {entry.metadata.backend}")
        print(f"    - Latency: {entry.metadata.latency_ms:.4f} ms")
        print(f"    - Device: {entry.metadata.device_name}")
        print(f"    - Created: {entry.metadata.created_at}")
    else:
        print("  ✗ Entry not found")

    # Test find_best
    print("\n--- Testing Find Best Functionality ---")
    best = store.find_best(test_hash, "mps")
    if best:
        print(f"  ✓ Best entry for hash {test_hash[:16]}...")
        print(f"    - Latency: {best.metadata.latency_ms:.4f} ms")

    # List all entries
    print("\n--- Listing All Entries ---")
    all_entries = store.list_all(limit=10)
    print(f"  Total entries (limit 10): {len(all_entries)}")

    for i, meta in enumerate(all_entries[:5]):
        print(
            f"  [{i+1}] {meta.backend}/{meta.graph_hash[:16]}... "
            f"latency={meta.latency_ms:.4f}ms"
        )

    # Delete test entry
    print("\n--- Cleanup ---")
    deleted = store.delete(
        graph_hash=test_hash,
        config_hash=entry.metadata.config_hash if entry else "",
        backend="mps",
    )
    print(f"  Test entry deleted: {deleted}")

    # Final stats
    print("\n--- Final Statistics ---")
    final_stats = store.get_stats()
    print(f"  Total entries: {final_stats['total_entries']}")

    print("\n" + "=" * 70)
    print("  ✅ MuGraph Store Demo Complete!")
    print("=" * 70)


def demo_with_superoptimize():
    """Demo MuGraphStore integration with superoptimize."""
    print("\n" + "=" * 70)
    print("  MuGraphStore Integration with superoptimize")
    print("=" * 70)

    import torch

    if not torch.backends.mps.is_available():
        print("  ⚠️  MPS not available, skipping superoptimize demo")
        return

    try:
        import yirage

        print("\n--- Creating Simple Kernel Graph ---")

        # Create a simple matmul graph
        graph = yirage.new_kernel_graph()

        # Small test dimensions
        M, K, N = 32, 64, 32

        A = graph.new_input(dims=(M, K), dtype=yirage.float16)
        B = graph.new_input(dims=(K, N), dtype=yirage.float16)
        C = graph.matmul(A, B)
        graph.mark_output(C)

        print(f"  Created matmul graph: ({M}, {K}) x ({K}, {N}) -> ({M}, {N})")

        # Get graph hash
        graph_hash = hex(graph.cygraph.get_owner_independent_hash())[2:]
        print(f"  Graph hash: {graph_hash}")

        # Check if we have cached results
        from yirage.mugraph_store import get_mugraph_store

        store = get_mugraph_store()

        cached = store.find_best(graph_hash, "mps")
        if cached:
            print(f"\n  ✓ Found cached muGraph!")
            print(f"    Latency: {cached.metadata.latency_ms:.4f} ms")
            print(f"    Stored at: {cached.metadata.created_at}")
        else:
            print(f"\n  No cached muGraph found, running superoptimize...")

            # Run superoptimize (this will save to persistent storage)
            optimized = graph.superoptimize(
                backend="mps",
                warmup_iters=2,
                profile_iters=10,
                use_persistent_cache=True,
            )

            if optimized:
                print(f"\n  ✓ Optimization complete!")

                # Verify it was saved
                cached = store.find_best(graph_hash, "mps")
                if cached:
                    print(f"  ✓ Saved to persistent storage")
                    print(f"    Path: {store._get_backend_dir('mps')}")

    except Exception as e:
        print(f"  ⚠️  Error during superoptimize demo: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    print("\n")

    # Test basic store functionality
    test_mugraph_store()

    # Test integration with superoptimize
    demo_with_superoptimize()

    # Show final storage location
    print("\n📁 Storage Location:")
    home = Path.home()
    mugraph_dir = home / ".yirage" / "mugraphs"
    print(f"   {mugraph_dir}")

    if mugraph_dir.exists():
        for backend_dir in sorted(mugraph_dir.iterdir()):
            if backend_dir.is_dir():
                files = list(backend_dir.glob("*.json"))
                if files:
                    print(f"   ├── {backend_dir.name}/ ({len(files)} files)")
                    for f in files[:3]:
                        print(f"   │   └── {f.name}")
                    if len(files) > 3:
                        print(f"   │   └── ... ({len(files)-3} more)")


if __name__ == "__main__":
    main()
