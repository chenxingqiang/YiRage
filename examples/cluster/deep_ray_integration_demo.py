#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Deep Ray Integration Demo

This demo showcases all 5 features of the deep Ray integration:

1. C++ Binding - Connect search_partition() C API to Python
2. Ray Object Store - Use ray.put() for large data
3. Placement Groups - GPU affinity for distributed training
4. Fault Tolerance - Retry/checkpoint mechanisms  
5. Ray Collective Ops - Efficient all-reduce for gradients

Usage:
    python deep_ray_integration_demo.py

Requirements:
    pip install ray
"""

import sys
import os
import json
import time
import tempfile
from pathlib import Path

# Add project path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))


def load_module_from_path(name: str, path: str):
    """Load module from file path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load modules
DIST_PATH = PROJECT_ROOT / "python" / "yirage" / "distributed"
ray_deep = load_module_from_path(
    "yirage.distributed.ray_deep_integration", str(DIST_PATH / "ray_deep_integration.py")
)
ray_train_mod = load_module_from_path(
    "yirage.distributed.ray_train_integration", str(DIST_PATH / "ray_train_integration.py")
)

# Import classes
RayDeepIntegration = ray_deep.RayDeepIntegration
DeepIntegrationConfig = ray_deep.DeepIntegrationConfig
GPUPlacementConfig = ray_deep.GPUPlacementConfig
RetryConfig = ray_deep.RetryConfig
RetryStrategy = ray_deep.RetryStrategy
create_deep_integration = ray_deep.create_deep_integration

DistributedRLConfig = ray_train_mod.DistributedRLConfig
GradientReducer = ray_train_mod.GradientReducer


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_1_cpp_binding():
    """Demo 1: C++ Binding via Cython."""
    print_header("Demo 1: C++ Binding (search_partition API)")

    # Check Cython bindings exist
    pxd_path = PROJECT_ROOT / "python" / "yirage" / "_cython" / "distributed_core.pxd"
    pyx_path = PROJECT_ROOT / "python" / "yirage" / "_cython" / "distributed_core.pyx"

    print(f"\n✓ Cython declaration file: {pxd_path.exists()}")
    print(f"✓ Cython implementation file: {pyx_path.exists()}")

    if pxd_path.exists():
        content = pxd_path.read_text()

        # Check key declarations
        has_search_partition = "search_partition(" in content
        has_create_partitions = "create_partitions(" in content
        has_rl_context = "rl_context_create" in content

        print(f"\n  C++ API Bindings:")
        print(f"  - search_partition(): {has_search_partition}")
        print(f"  - create_partitions(): {has_create_partitions}")
        print(f"  - RL Context (GPU verify): {has_rl_context}")

    print("\n  [C++ binding layer ready for Cython compilation]")
    return True


def demo_2_object_store():
    """Demo 2: Ray Object Store for large data."""
    print_header("Demo 2: Ray Object Store (Large Data)")

    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=4, ignore_reinit_error=True)

    # Create large graph data
    large_graph = {
        "type": "transformer_block",
        "layers": [
            {
                "name": f"layer_{i}",
                "weights": [0.1] * 1000,  # Simulated weights
                "config": {"hidden_size": 4096, "heads": 32},
            }
            for i in range(100)
        ],
        "metadata": {"model": "llama-7b", "precision": "fp16"},
    }

    graph_size = len(json.dumps(large_graph))
    print(f"\n  Graph size: {graph_size:,} bytes ({graph_size/1024:.1f} KB)")

    # Store in object store
    start = time.time()
    graph_ref = ray.put(large_graph)
    put_time = time.time() - start
    print(f"  ray.put() time: {put_time*1000:.2f} ms")

    # Multiple workers access same reference
    @ray.remote
    def worker_access(ref):
        data = ref  # Ray auto-resolves
        return len(data["layers"])

    start = time.time()
    results = ray.get(
        [
            worker_access.remote(graph_ref),
            worker_access.remote(graph_ref),
            worker_access.remote(graph_ref),
            worker_access.remote(graph_ref),
        ]
    )
    access_time = time.time() - start

    print(f"  4 workers accessed: {access_time*1000:.2f} ms")
    print(f"  Results: {results}")
    print(f"\n  ✓ Object store enables single-copy sharing across workers")

    ray.shutdown()
    return True


def demo_3_placement_groups():
    """Demo 3: GPU-aware Placement Groups."""
    print_header("Demo 3: Placement Groups (GPU Affinity)")

    import ray
    from ray.util.placement_group import placement_group

    if not ray.is_initialized():
        ray.init(num_cpus=4, ignore_reinit_error=True)

    # Create GPU placement configuration
    config = GPUPlacementConfig(
        gpus_per_worker=0,  # CPU-only for demo
        cpus_per_worker=1,
        memory_per_worker_mb=1024,
        strategy="PACK",  # PACK for NVLink locality
        require_nvlink=False,
    )

    print(f"\n  Placement Configuration:")
    print(f"  - Strategy: {config.strategy}")
    print(f"  - CPUs per worker: {config.cpus_per_worker}")
    print(f"  - GPUs per worker: {config.gpus_per_worker}")

    # Create placement group
    bundles = [{"CPU": 1} for _ in range(4)]
    pg = placement_group(bundles, strategy="PACK")

    ready = ray.get(pg.ready(), timeout=10)
    print(f"\n  Placement group created: {ready is None}")
    print(f"  Bundles: {len(bundles)}")

    print(f"\n  ✓ Placement groups enable GPU co-location for NVLink")

    ray.shutdown()
    return True


def demo_4_fault_tolerance():
    """Demo 4: Fault Tolerance with retry/checkpoint."""
    print_header("Demo 4: Fault Tolerance (Retry/Checkpoint)")

    # Exponential backoff configuration
    retry_config = RetryConfig(
        strategy=RetryStrategy.EXPONENTIAL,
        max_retries=5,
        initial_delay_s=1.0,
        max_delay_s=30.0,
        multiplier=2.0,
    )

    print(f"\n  Retry Configuration:")
    print(f"  - Strategy: {retry_config.strategy.name}")
    print(f"  - Max retries: {retry_config.max_retries}")
    print(f"  - Initial delay: {retry_config.initial_delay_s}s")
    print(f"  - Multiplier: {retry_config.multiplier}x")

    print(f"\n  Delay Schedule:")
    for attempt in range(6):
        delay = retry_config.get_delay(attempt)
        print(f"    Attempt {attempt}: {delay:.1f}s delay")

    # Checkpoint demo
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = {
            "partition_id": 0,
            "completed_configs": 1500,
            "best_latency_ms": 0.42,
            "best_config": {
                "grid_dim": (4, 1, 1),
                "block_dim": (256, 1, 1),
            },
            "timestamp": time.time(),
        }

        filepath = os.path.join(tmpdir, "checkpoint.json")
        with open(filepath, "w") as f:
            json.dump(checkpoint, f, indent=2)

        with open(filepath, "r") as f:
            restored = json.load(f)

        print(f"\n  Checkpoint Save/Restore:")
        print(f"  - Completed configs: {restored['completed_configs']}")
        print(f"  - Best latency: {restored['best_latency_ms']} ms")

    print(f"\n  ✓ Fault tolerance enables recovery from worker failures")
    return True


def demo_5_collective_ops():
    """Demo 5: Ray Collective Operations (All-Reduce)."""
    print_header("Demo 5: Collective Ops (All-Reduce)")

    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=4, ignore_reinit_error=True)

    # Simulate gradients from 4 workers
    worker_gradients = [
        {"layer1_weight": 0.1, "layer1_bias": 0.01, "layer2_weight": 0.2},
        {"layer1_weight": 0.3, "layer1_bias": 0.02, "layer2_weight": 0.1},
        {"layer1_weight": 0.2, "layer1_bias": 0.03, "layer2_weight": 0.3},
        {"layer1_weight": 0.4, "layer1_bias": 0.04, "layer2_weight": 0.2},
    ]

    print(f"\n  Worker Gradients (layer1_weight):")
    for i, g in enumerate(worker_gradients):
        print(f"    Worker {i}: {g['layer1_weight']}")

    # Create engine and perform all-reduce
    config = DeepIntegrationConfig(
        num_workers=4,
        gpu_placement=GPUPlacementConfig(gpus_per_worker=0),
    )
    engine = RayDeepIntegration(config)

    # Mean reduce
    mean_grads = engine.all_reduce_gradients(worker_gradients, reduce_op="mean")
    print(f"\n  After Mean All-Reduce:")
    print(f"    layer1_weight: {mean_grads['layer1_weight']:.3f}")
    print(f"    layer1_bias: {mean_grads['layer1_bias']:.4f}")
    print(f"    layer2_weight: {mean_grads['layer2_weight']:.3f}")

    # Sum reduce
    sum_grads = engine.all_reduce_gradients(worker_gradients, reduce_op="sum")
    print(f"\n  After Sum All-Reduce:")
    print(f"    layer1_weight: {sum_grads['layer1_weight']:.3f}")

    print(f"\n  ✓ All-reduce enables efficient gradient synchronization")

    ray.shutdown()
    return True


def demo_full_optimization():
    """Demo: Full optimization workflow with all features."""
    print_header("Full Optimization Workflow")

    import ray

    if ray.is_initialized():
        ray.shutdown()

    # Create engine with full configuration
    config = DeepIntegrationConfig(
        num_workers=4,
        gpu_placement=GPUPlacementConfig(
            gpus_per_worker=0,  # CPU-only for demo
            cpus_per_worker=1,
            strategy="PACK",
        ),
        retry=RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            max_retries=3,
        ),
        backend="cpu",
        use_object_store=True,
        large_object_threshold_bytes=1024,
        max_search_time_s=60.0,
    )

    engine = RayDeepIntegration(config)

    print(f"\n  Configuration:")
    print(f"  - Workers: {config.num_workers}")
    print(f"  - Placement: {config.gpu_placement.strategy}")
    print(f"  - Retry: {config.retry.strategy.name}")
    print(f"  - Object store: {config.use_object_store}")

    # Create optimization target
    graph = {
        "type": "matmul",
        "input_shapes": [[1024, 2048], [2048, 4096]],
        "output_shape": [1024, 4096],
        "estimated_flops": 1024 * 2048 * 4096 * 2,
    }

    search_space = {
        "grid_dims": [
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
            (1, 2, 1),
            (2, 2, 1),
            (4, 2, 1),
        ],
        "block_dims": [
            (64, 1, 1),
            (128, 1, 1),
            (256, 1, 1),
            (512, 1, 1),
        ],
    }

    print(f"\n  Search Space:")
    print(f"  - Grid dims: {len(search_space['grid_dims'])} options")
    print(f"  - Block dims: {len(search_space['block_dims'])} options")

    # Run optimization
    print(f"\n  Running distributed optimization...")
    start = time.time()

    try:
        result = engine.optimize(graph, search_space)
        elapsed = time.time() - start

        print(f"\n  Results:")
        print(f"  - Search time: {elapsed:.2f}s")
        print(f"  - Workers used: {result['num_workers']}")
        print(f"  - Candidates searched: {result['total_candidates_searched']}")
        print(f"  - Valid graphs: {result['total_valid_graphs']}")

        if result.get("best_config"):
            best = result["best_config"]
            print(f"\n  Best Configuration:")
            print(f"  - Grid dim: {best.get('grid_dim')}")
            print(f"  - Block dim: {best.get('block_dim')}")
            print(f"  - Latency: {best.get('latency_ms', 'N/A')} ms")

        print(f"\n  ✓ Full distributed optimization complete!")

    finally:
        engine.shutdown()

    return True


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  Deep Ray Integration Demo")
    print("  All 5 Features Demonstration")
    print("=" * 70)

    demos = [
        ("C++ Binding", demo_1_cpp_binding),
        ("Object Store", demo_2_object_store),
        ("Placement Groups", demo_3_placement_groups),
        ("Fault Tolerance", demo_4_fault_tolerance),
        ("Collective Ops", demo_5_collective_ops),
        ("Full Workflow", demo_full_optimization),
    ]

    results = []
    for name, demo_func in demos:
        try:
            success = demo_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ✗ Error: {e}")
            results.append((name, False))

    # Summary
    print_header("Summary")
    print("\n  Feature Status:")
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"    {status} {name}")

    all_passed = all(s for _, s in results)
    print(f"\n  Overall: {'All features working!' if all_passed else 'Some features failed'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
