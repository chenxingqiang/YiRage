#!/usr/bin/env python3
"""
Ray Distributed Optimization Demo

Demonstrates the full Ray integration for distributed kernel search:
1. Object store integration for large data
2. Placement groups for GPU affinity
3. Fault tolerance with checkpoints
4. Collective operations for result aggregation
5. Integration with C++ search core (when available)

Usage:
    python examples/ray_distributed_demo.py

Requirements:
    pip install ray
"""

import os
import sys
import time
import json
from typing import Dict, List, Any

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

# Setup module loading for development
import types
import importlib.util


def load_module_by_path(module_name: str, file_path: str):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def setup_modules():
    """Setup yirage module hierarchy."""
    base_path = os.path.join(os.path.dirname(__file__), "..", "python", "yirage")

    if "yirage" not in sys.modules:
        yirage_pkg = types.ModuleType("yirage")
        yirage_pkg.__path__ = [base_path]
        sys.modules["yirage"] = yirage_pkg

    if "yirage.distributed" not in sys.modules:
        dist_path = os.path.join(base_path, "distributed")
        dist_pkg = types.ModuleType("yirage.distributed")
        dist_pkg.__path__ = [dist_path]
        sys.modules["yirage.distributed"] = dist_pkg
        sys.modules["yirage"].distributed = dist_pkg

        # Load modules
        ray_engine = load_module_by_path(
            "yirage.distributed.ray_engine", os.path.join(dist_path, "ray_engine.py")
        )
        collectives = load_module_by_path(
            "yirage.distributed.collectives", os.path.join(dist_path, "collectives.py")
        )

        # Export to package
        for name in [
            "RayDistributedEngine",
            "RayEngineConfig",
            "DistributedSearchResult",
            "SearchCheckpoint",
            "PartitionStrategy",
            "create_distributed_engine",
        ]:
            if hasattr(ray_engine, name):
                setattr(dist_pkg, name, getattr(ray_engine, name))

        for name in ["CollectiveOperations", "sum_reduce", "mean_reduce", "min_reduce"]:
            if hasattr(collectives, name):
                setattr(dist_pkg, name, getattr(collectives, name))


# Setup modules
setup_modules()

# Check Ray availability
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not installed. Install with: pip install ray")


def demo_basic_distributed_search():
    """Demo: Basic distributed kernel search."""
    print("\n" + "=" * 80)
    print("Demo 1: Basic Distributed Kernel Search")
    print("=" * 80)

    if not RAY_AVAILABLE:
        print("\nRay not available. Skipping.")
        return

    from yirage.distributed import create_distributed_engine

    # Create engine
    engine = create_distributed_engine(
        num_workers=4,
        backend="cuda",
        use_placement_groups=False,  # Simpler for demo
    )

    print(f"\nCreated distributed engine with {engine.config.num_workers} workers")

    # Create a sample computation graph
    graph = {
        "name": "attention_block",
        "inputs": [
            {"name": "Q", "dims": [32, 2048, 4096], "dtype": "fp16"},
            {"name": "K", "dims": [32, 2048, 4096], "dtype": "fp16"},
            {"name": "V", "dims": [32, 2048, 4096], "dtype": "fp16"},
        ],
        "operators": [
            {"type": "matmul", "inputs": ["Q", "K_T"], "output": "scores"},
            {"type": "softmax", "inputs": ["scores"], "output": "attn"},
            {"type": "matmul", "inputs": ["attn", "V"], "output": "output"},
        ],
        "estimated_flops": 4 * 32 * 2048 * 4096 * 4096,  # Approximate
    }

    # Define search space
    search_space = {
        "grid_dims": [
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
            (1, 2, 1),
            (2, 2, 1),
            (4, 2, 1),
            (1, 4, 1),
            (2, 4, 1),
        ],
        "block_dims": [
            (128, 1, 1),
            (256, 1, 1),
            (128, 2, 1),
            (64, 4, 1),
        ],
    }

    print(
        f"\nSearch space: {len(search_space['grid_dims'])} grid dims × {len(search_space['block_dims'])} block dims"
    )
    print(
        f"Total configurations: {len(search_space['grid_dims']) * len(search_space['block_dims'])}"
    )

    # Run distributed optimization
    print("\nRunning distributed optimization...")
    start_time = time.time()

    result = engine.optimize(graph, search_space)

    elapsed = time.time() - start_time

    # Print results
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Search time: {result.search_time_s:.3f}s")
    print(f"Candidates searched: {result.total_candidates_searched}")
    print(f"Valid graphs found: {result.total_valid_graphs}")

    if result.best_config:
        print(f"\nBest configuration:")
        print(f"  Grid dim: {result.best_config.get('grid_dim')}")
        print(f"  Block dim: {result.best_config.get('block_dim')}")
        print(f"  Estimated latency: {result.best_latency_ms:.3f} ms")

    # Worker statistics
    print(f"\nPer-worker statistics:")
    for stat in result.worker_stats:
        print(
            f"  Worker {stat['worker_id']}: {stat['num_candidates']} candidates, {stat['num_valid']} valid"
        )

    # Clean up
    engine.shutdown()

    return result


def demo_object_store_efficiency():
    """Demo: Object store for efficient data sharing."""
    print("\n" + "=" * 80)
    print("Demo 2: Object Store Efficiency")
    print("=" * 80)

    if not RAY_AVAILABLE:
        print("\nRay not available. Skipping.")
        return

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR")

    # Create a large computation graph
    large_graph = {
        "name": "large_transformer",
        "layers": [
            {"type": "attention", "weights": [f"weight_{i}_{j}" for j in range(100)]}
            for i in range(24)
        ],
        "config": {
            "hidden_dim": 4096,
            "num_heads": 32,
            "seq_len": 8192,
        },
    }

    graph_size = len(json.dumps(large_graph))
    print(f"\nGraph size: {graph_size / 1024:.1f} KB")

    # Without object store: each worker gets a copy
    @ray.remote
    def process_without_store(graph, worker_id):
        return {"worker_id": worker_id, "processed": True}

    # With object store: single copy, multiple references
    @ray.remote
    def process_with_store(graph, worker_id):
        # graph is auto-resolved from ObjectRef
        return {"worker_id": worker_id, "processed": True}

    num_workers = 8

    # Test without object store (multiple copies)
    print(f"\nWithout object store ({num_workers} copies)...")
    start = time.time()
    futures = [process_without_store.remote(large_graph, i) for i in range(num_workers)]
    ray.get(futures)
    time_without = time.time() - start
    print(f"  Time: {time_without:.4f}s")

    # Test with object store (single copy)
    print(f"\nWith object store (1 copy, {num_workers} references)...")
    start = time.time()
    graph_ref = ray.put(large_graph)
    futures = [process_with_store.remote(graph_ref, i) for i in range(num_workers)]
    ray.get(futures)
    time_with = time.time() - start
    print(f"  Time: {time_with:.4f}s")

    print(f"\nSpeedup: {time_without / time_with:.2f}x")


def demo_fault_tolerance():
    """Demo: Fault tolerance with checkpoints."""
    print("\n" + "=" * 80)
    print("Demo 3: Fault Tolerance with Checkpoints")
    print("=" * 80)

    if not RAY_AVAILABLE:
        print("\nRay not available. Skipping.")
        return

    from yirage.distributed import SearchCheckpoint
    import tempfile

    # Simulate a long-running search with checkpoints
    checkpoint_dir = tempfile.mkdtemp()
    print(f"\nCheckpoint directory: {checkpoint_dir}")

    # Simulate search progress
    checkpoint = SearchCheckpoint(
        partition_id=0,
        completed_configs=0,
        best_latency_ms=float("inf"),
    )

    print("\nSimulating search with periodic checkpoints...")

    for i in range(5):
        # Simulate processing
        checkpoint.completed_configs += 100

        # Simulate finding a better configuration
        latency = 10.0 / (i + 1)
        if latency < checkpoint.best_latency_ms:
            checkpoint.best_latency_ms = latency
            checkpoint.best_config = {"grid_dim": (i + 1, 1, 1)}

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.json")
        checkpoint.save(checkpoint_path)

        print(
            f"  Step {i+1}: {checkpoint.completed_configs} configs, "
            f"best latency: {checkpoint.best_latency_ms:.2f}ms"
        )

    # Simulate crash and recovery
    print("\n[Simulating crash...]")

    # Load checkpoint
    recovered = SearchCheckpoint.load(checkpoint_path)
    print(f"\n[Recovered from checkpoint]")
    print(f"  Completed configs: {recovered.completed_configs}")
    print(f"  Best latency: {recovered.best_latency_ms:.2f}ms")
    print(f"  Best config: {recovered.best_config}")

    # Clean up
    import shutil

    shutil.rmtree(checkpoint_dir)


def demo_collective_operations():
    """Demo: Collective operations for distributed training."""
    print("\n" + "=" * 80)
    print("Demo 4: Collective Operations")
    print("=" * 80)

    if not RAY_AVAILABLE:
        print("\nRay not available. Skipping.")
        return

    from yirage.distributed import sum_reduce, mean_reduce, min_reduce

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR")

    @ray.remote
    class TrainingWorker:
        def __init__(self, worker_id: int):
            self.worker_id = worker_id
            self.local_loss = 0.0
            self.local_count = 0

        def train_step(self, data):
            # Simulate training
            import random

            self.local_loss += random.random()
            self.local_count += 1
            return {
                "worker_id": self.worker_id,
                "loss": self.local_loss / self.local_count,
                "count": self.local_count,
            }

        def get_metrics(self):
            return {
                "loss": self.local_loss / max(self.local_count, 1),
                "count": self.local_count,
            }

    print("\nCreating distributed training workers...")
    workers = [TrainingWorker.remote(i) for i in range(4)]

    print("\nRunning distributed training steps...")
    for step in range(3):
        # Each worker does a training step
        futures = [w.train_step.remote(f"batch_{step}") for w in workers]
        results = ray.get(futures)

        # Aggregate metrics using collective reduce
        aggregated = mean_reduce(results)

        print(
            f"  Step {step+1}: avg_loss={aggregated['loss']:.4f}, total_count={sum(r['count'] for r in results)}"
        )

    # Final reduction
    final_metrics = ray.get([w.get_metrics.remote() for w in workers])

    print("\nFinal metrics per worker:")
    for m in final_metrics:
        print(f"  Worker: loss={m['loss']:.4f}, count={m['count']}")

    global_avg = mean_reduce(final_metrics)
    print(f"\nGlobal average loss: {global_avg['loss']:.4f}")


def demo_end_to_end_pipeline():
    """Demo: Complete end-to-end optimization pipeline."""
    print("\n" + "=" * 80)
    print("Demo 5: End-to-End Optimization Pipeline")
    print("=" * 80)

    if not RAY_AVAILABLE:
        print("\nRay not available. Skipping.")
        return

    from yirage.distributed import (
        create_distributed_engine,
        RayEngineConfig,
        PartitionStrategy,
    )

    print("\nPipeline steps:")
    print("  1. Create computation graph from PyTorch module")
    print("  2. Configure distributed search")
    print("  3. Execute parallel optimization")
    print("  4. Collect and aggregate results")
    print("  5. Return best kernel configuration")

    # Step 1: Create computation graph (simulated)
    print("\n[Step 1] Creating computation graph...")
    graph = {
        "name": "gpt2_block",
        "batch_size": 32,
        "seq_len": 2048,
        "hidden_dim": 4096,
        "num_heads": 32,
        "operators": [
            {"type": "layer_norm", "inputs": ["x"]},
            {"type": "attention", "inputs": ["x_norm", "x_norm", "x_norm"]},
            {"type": "add", "inputs": ["x", "attn_out"]},
            {"type": "layer_norm", "inputs": ["x_attn"]},
            {"type": "mlp", "inputs": ["x_attn_norm"]},
            {"type": "add", "inputs": ["x_attn", "mlp_out"]},
        ],
        "estimated_flops": 2e12,
    }
    print(f"  Graph: {graph['name']}")
    print(f"  Operators: {len(graph['operators'])}")

    # Step 2: Configure search
    print("\n[Step 2] Configuring distributed search...")
    config = RayEngineConfig(
        num_workers=4,
        backend="cuda",
        use_placement_groups=False,
        max_search_time_s=60.0,
        checkpoint_interval_s=10.0,
    )
    print(f"  Workers: {config.num_workers}")
    print(f"  Backend: {config.backend}")
    print(f"  Max search time: {config.max_search_time_s}s")

    engine = create_distributed_engine(
        num_workers=config.num_workers,
        backend=config.backend,
    )

    # Step 3: Define search space
    print("\n[Step 3] Defining search space...")
    search_space = {
        "grid_dims": [(2**i, 1, 1) for i in range(5)],  # 1, 2, 4, 8, 16
        "block_dims": [(64, 1, 1), (128, 1, 1), (256, 1, 1)],
    }
    print(f"  Grid dimensions: {len(search_space['grid_dims'])}")
    print(f"  Block dimensions: {len(search_space['block_dims'])}")

    # Step 4: Execute optimization
    print("\n[Step 4] Executing distributed optimization...")
    start_time = time.time()
    result = engine.optimize(graph, search_space)
    elapsed = time.time() - start_time

    # Step 5: Report results
    print("\n[Step 5] Optimization complete!")
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Search time: {result.search_time_s:.3f}s")
    print(f"Configurations evaluated: {result.total_candidates_searched}")
    print(f"Valid kernels found: {result.total_valid_graphs}")

    if result.best_config:
        print(f"\nOptimal Configuration:")
        print(f"  Grid dimensions: {result.best_config.get('grid_dim')}")
        print(f"  Block dimensions: {result.best_config.get('block_dim')}")
        print(f"  Estimated latency: {result.best_latency_ms:.4f} ms")

        # Calculate theoretical speedup
        single_gpu_latency = graph["estimated_flops"] / (312e12) * 1000  # A100
        speedup = single_gpu_latency / result.best_latency_ms
        print(f"  Theoretical speedup: {speedup:.2f}x")

    # Clean up
    engine.shutdown()


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("Ray Distributed Optimization Demo")
    print("=" * 80)

    if RAY_AVAILABLE:
        print(f"\nRay version: {ray.__version__}")
        print("Ray is available for distributed execution.")

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, logging_level="ERROR")
            print(f"Ray initialized with resources: {ray.cluster_resources()}")
    else:
        print("\nRay is not installed. Install with: pip install ray")
        return

    try:
        # Run demos
        demo_basic_distributed_search()
        demo_object_store_efficiency()
        demo_fault_tolerance()
        demo_collective_operations()
        demo_end_to_end_pipeline()

    finally:
        # Clean up
        if ray.is_initialized():
            ray.shutdown()

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Distributed kernel search with Ray workers")
    print("  ✓ Object store for efficient large data sharing")
    print("  ✓ Fault tolerance with checkpoint save/restore")
    print("  ✓ Collective operations (reduce, broadcast)")
    print("  ✓ End-to-end optimization pipeline")


if __name__ == "__main__":
    main()
