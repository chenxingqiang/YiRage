#!/usr/bin/env python3
"""
Ray Integration Demo

Demonstrates integration of the Universal Optimizer with Ray for:
1. Distributed task optimization
2. Parallel strategy search
3. Simulated cluster execution with Ray actors

Note: This demo uses self-contained Ray actors that don't rely on
module imports, making it work without package installation.
"""

import sys
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

# Setup module loading
import types
import importlib.util


def load_module_by_path(module_name: str, file_path: str):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Setup fake yirage package
yirage_pkg = types.ModuleType("yirage")
yirage_pkg.__path__ = [os.path.join(os.path.dirname(__file__), "..", "..", "python", "yirage")]
sys.modules["yirage"] = yirage_pkg

rl_pkg = types.ModuleType("yirage.rl")
rl_pkg.__path__ = [os.path.join(yirage_pkg.__path__[0], "rl")]
sys.modules["yirage.rl"] = rl_pkg
yirage_pkg.rl = rl_pkg

cluster_path = os.path.join(rl_pkg.__path__[0], "cluster")
cluster_pkg = types.ModuleType("yirage.rl.cluster")
cluster_pkg.__path__ = [cluster_path]
sys.modules["yirage.rl.cluster"] = cluster_pkg
rl_pkg.cluster = cluster_pkg

# Load cluster modules
topology = load_module_by_path(
    "yirage.rl.cluster.topology", os.path.join(cluster_path, "topology.py")
)
task = load_module_by_path("yirage.rl.cluster.task", os.path.join(cluster_path, "task.py"))
simulator = load_module_by_path(
    "yirage.rl.cluster.simulator", os.path.join(cluster_path, "simulator.py")
)
placer = load_module_by_path("yirage.rl.cluster.placer", os.path.join(cluster_path, "placer.py"))
executor = load_module_by_path(
    "yirage.rl.cluster.executor", os.path.join(cluster_path, "executor.py")
)
auto_optimizer = load_module_by_path(
    "yirage.rl.cluster.auto_optimizer", os.path.join(cluster_path, "auto_optimizer.py")
)
e2e_optimizer = load_module_by_path(
    "yirage.rl.cluster.e2e_optimizer", os.path.join(cluster_path, "e2e_optimizer.py")
)

# Check Ray availability
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Warning: Ray not installed. Running in simulation mode.")

# ============================================================================
# Demo Functions
# ============================================================================


def demo_ray_basic():
    """Demo: Basic Ray parallelization for optimization."""
    print("\n" + "=" * 80)
    print("Demo 1: Ray Basic Parallelization")
    print("=" * 80)

    if not RAY_AVAILABLE:
        print("\nRay not available. Skipping.")
        return

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR")

    print("\nRay initialized.")

    # Define remote function (self-contained, no module imports)
    @ray.remote
    def optimize_config(config_dict: Dict) -> Dict:
        """Remote function to optimize a configuration."""
        import numpy as np

        # Simulate optimization (self-contained logic)
        batch = config_dict.get("batch", 1)
        seq_len = config_dict.get("seq_len", 2048)
        hidden = config_dict.get("hidden_dim", 4096)
        num_gpus = config_dict.get("num_gpus", 8)

        # Simple performance model
        flops = 4 * batch * seq_len * hidden * hidden  # Approximate transformer FLOPs
        peak_tflops = 312.0 * num_gpus  # A100

        compute_time_ms = (flops / (peak_tflops * 1e12)) * 1000
        comm_overhead = 0.1 * (num_gpus - 1)  # Simple comm model
        total_time = compute_time_ms * (1 + comm_overhead)

        return {
            "config": config_dict,
            "estimated_time_ms": total_time,
            "throughput": batch / total_time * 1000 if total_time > 0 else 0,
            "strategy": f"tensor_parallel_{num_gpus}" if num_gpus > 1 else "single",
        }

    # Create configurations to test
    configs = [
        {"batch": 1, "seq_len": 2048, "hidden_dim": 4096, "num_gpus": 1},
        {"batch": 1, "seq_len": 2048, "hidden_dim": 4096, "num_gpus": 2},
        {"batch": 1, "seq_len": 2048, "hidden_dim": 4096, "num_gpus": 4},
        {"batch": 1, "seq_len": 2048, "hidden_dim": 4096, "num_gpus": 8},
        {"batch": 8, "seq_len": 2048, "hidden_dim": 4096, "num_gpus": 8},
        {"batch": 32, "seq_len": 2048, "hidden_dim": 4096, "num_gpus": 8},
    ]

    print(f"\nOptimizing {len(configs)} configurations in parallel...")

    # Submit all tasks
    start_time = time.time()
    futures = [optimize_config.remote(cfg) for cfg in configs]
    results = ray.get(futures)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.3f}s")
    print(f"\n{'Batch':<8} {'GPUs':<8} {'Time (ms)':<12} {'Throughput':<12} {'Strategy':<20}")
    print("-" * 60)

    for result in results:
        cfg = result["config"]
        print(
            f"{cfg['batch']:<8} {cfg['num_gpus']:<8} {result['estimated_time_ms']:<12.3f} {result['throughput']:<12.1f} {result['strategy']:<20}"
        )


def demo_ray_parallel_search():
    """Demo: Parallel strategy search with Ray."""
    print("\n" + "=" * 80)
    print("Demo 2: Ray Parallel Strategy Search")
    print("=" * 80)

    if not RAY_AVAILABLE:
        print("\nRay not available. Skipping.")
        return

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR")

    @ray.remote
    def search_strategy(strategy: str, batch: int, seq_len: int, num_gpus: int) -> Dict:
        """Search a specific parallelism strategy."""
        import numpy as np

        hidden = 4096
        flops = 4 * batch * seq_len * hidden * hidden
        peak_tflops = 312.0  # A100 per GPU

        if strategy == "data_parallel":
            # Data parallel: batch split across GPUs
            per_gpu_batch = batch // num_gpus
            compute_time = (flops / num_gpus) / (peak_tflops * 1e12) * 1000
            # AllReduce communication
            grad_size = 2 * hidden * hidden  # Approximate gradient size
            comm_time = grad_size * 8 / (300e9) * 1000 * 2 * (num_gpus - 1) / num_gpus
            total = compute_time + comm_time

        elif strategy == "tensor_parallel":
            # Tensor parallel: model split across GPUs
            compute_time = (flops / num_gpus) / (peak_tflops * 1e12) * 1000
            # AllGather communication after each matmul
            activation_size = batch * seq_len * hidden / num_gpus
            comm_time = activation_size * 2 * 8 / (300e9) * 1000 * (num_gpus - 1)
            total = compute_time + comm_time

        elif strategy == "pipeline_parallel":
            # Pipeline parallel: layers split across GPUs
            stages = min(num_gpus, 12)  # Assume 12 layers
            micro_batches = 4
            compute_time = (flops / stages) / (peak_tflops * 1e12) * 1000
            # Pipeline bubble
            bubble = (stages - 1) / (micro_batches + stages - 1)
            total = compute_time * stages * (1 + bubble)

        else:
            total = flops / (peak_tflops * 1e12) * 1000

        return {
            "strategy": strategy,
            "num_gpus": num_gpus,
            "total_time_ms": total,
            "throughput": batch / total * 1000 if total > 0 else 0,
        }

    print("\nSearching strategies in parallel...")

    strategies = ["data_parallel", "tensor_parallel", "pipeline_parallel"]
    gpu_counts = [2, 4, 8]
    batch = 32
    seq_len = 2048

    # Submit all searches
    start_time = time.time()
    futures = []
    for strategy in strategies:
        for num_gpus in gpu_counts:
            futures.append(search_strategy.remote(strategy, batch, seq_len, num_gpus))

    results = ray.get(futures)
    elapsed = time.time() - start_time

    print(f"\nSearched {len(results)} configurations in {elapsed:.3f}s")
    print(f"\n{'Strategy':<20} {'GPUs':<8} {'Time (ms)':<12} {'Throughput':<12}")
    print("-" * 52)

    for result in sorted(results, key=lambda x: x["total_time_ms"]):
        print(
            f"{result['strategy']:<20} {result['num_gpus']:<8} {result['total_time_ms']:<12.3f} {result['throughput']:<12.1f}"
        )

    # Find best
    best = min(results, key=lambda x: x["total_time_ms"])
    print(
        f"\nBest: {best['strategy']} with {best['num_gpus']} GPUs ({best['total_time_ms']:.3f} ms)"
    )


def demo_ray_actor_cluster():
    """Demo: Ray actors simulating a GPU cluster."""
    print("\n" + "=" * 80)
    print("Demo 3: Ray Actor Simulated Cluster")
    print("=" * 80)

    if not RAY_AVAILABLE:
        print("\nRay not available. Skipping.")
        return

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR")

    @ray.remote
    class GPUSimulator:
        """Simulated GPU device."""

        def __init__(self, gpu_id: int, peak_tflops: float = 312.0):
            self.gpu_id = gpu_id
            self.peak_tflops = peak_tflops
            self.tasks_completed = 0
            self.total_time = 0.0

        def execute(self, task: Dict) -> Dict:
            """Execute a task on this GPU."""
            import time

            flops = task.get("flops", 1e12)
            name = task.get("name", "unknown")

            # Compute time
            compute_time_ms = (flops / (self.peak_tflops * 1e12)) * 1000

            # Simulate (scaled down)
            time.sleep(compute_time_ms / 1000 * 0.001)

            self.tasks_completed += 1
            self.total_time += compute_time_ms

            return {
                "gpu_id": self.gpu_id,
                "task": name,
                "time_ms": compute_time_ms,
                "status": "completed",
            }

        def get_stats(self) -> Dict:
            """Get GPU statistics."""
            return {
                "gpu_id": self.gpu_id,
                "tasks_completed": self.tasks_completed,
                "total_time_ms": self.total_time,
            }

    print("\nCreating simulated 8-GPU cluster...")

    # Create GPU actors
    gpus = [GPUSimulator.remote(i, 312.0) for i in range(8)]

    # Create tasks
    tasks = [{"name": f"matmul_{i}", "flops": 1e12 + i * 0.5e12} for i in range(16)]

    print(f"Distributing {len(tasks)} tasks across {len(gpus)} GPUs...")

    # Distribute tasks round-robin
    start_time = time.time()
    futures = []
    for i, task in enumerate(tasks):
        gpu = gpus[i % len(gpus)]
        futures.append(gpu.execute.remote(task))

    results = ray.get(futures)
    elapsed = time.time() - start_time

    print(f"\nExecution completed in {elapsed:.4f}s")

    # Get stats
    stats_futures = [gpu.get_stats.remote() for gpu in gpus]
    stats = ray.get(stats_futures)

    print(f"\n{'GPU':<8} {'Tasks':<10} {'Total Time (ms)':<15}")
    print("-" * 33)
    for s in stats:
        print(f"GPU {s['gpu_id']:<4} {s['tasks_completed']:<10} {s['total_time_ms']:<15.2f}")


def demo_ray_distributed_optimization():
    """Demo: Full distributed optimization with Ray."""
    print("\n" + "=" * 80)
    print("Demo 4: Distributed Optimization Pipeline")
    print("=" * 80)

    if not RAY_AVAILABLE:
        print("\nRay not available. Skipping.")
        return

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR")

    @ray.remote
    class OptimizationWorker:
        """Worker for distributed optimization."""

        def __init__(self, worker_id: int):
            self.worker_id = worker_id
            self.results = []

        def optimize_batch(self, task_specs: List[Dict]) -> List[Dict]:
            """Optimize a batch of tasks."""
            import numpy as np

            results = []
            for spec in task_specs:
                # Simple optimization logic
                batch = spec.get("batch", 1)
                seq = spec.get("seq_len", 2048)
                hidden = spec.get("hidden_dim", 4096)

                # Try different strategies
                best_time = float("inf")
                best_strategy = "single"

                for num_gpus in [1, 2, 4, 8]:
                    flops = 4 * batch * seq * hidden * hidden
                    compute = (flops / num_gpus) / (312e12) * 1000
                    comm = 0.05 * (num_gpus - 1) * compute
                    total = compute + comm

                    if total < best_time:
                        best_time = total
                        best_strategy = f"tp_{num_gpus}" if num_gpus > 1 else "single"

                results.append(
                    {
                        "spec": spec,
                        "strategy": best_strategy,
                        "latency_ms": best_time,
                        "throughput": batch / best_time * 1000,
                        "worker_id": self.worker_id,
                    }
                )

            self.results.extend(results)
            return results

        def get_results(self) -> List[Dict]:
            return self.results

    print("\nCreating distributed optimization workers...")

    # Create workers
    num_workers = 4
    workers = [OptimizationWorker.remote(i) for i in range(num_workers)]

    # Create task specifications to optimize
    task_specs = [
        {"name": f"config_{i}", "batch": 2**i, "seq_len": 2048, "hidden_dim": 4096}
        for i in range(8)
    ]

    # Distribute specs across workers
    specs_per_worker = len(task_specs) // num_workers

    print(f"Distributing {len(task_specs)} tasks across {num_workers} workers...")

    start_time = time.time()
    futures = []
    for i, worker in enumerate(workers):
        start_idx = i * specs_per_worker
        end_idx = start_idx + specs_per_worker if i < num_workers - 1 else len(task_specs)
        worker_specs = task_specs[start_idx:end_idx]
        futures.append(worker.optimize_batch.remote(worker_specs))

    all_results = ray.get(futures)
    elapsed = time.time() - start_time

    # Flatten results
    results = [r for batch in all_results for r in batch]

    print(f"\nOptimized {len(results)} configurations in {elapsed:.3f}s")
    print(f"\n{'Config':<12} {'Batch':<8} {'Strategy':<12} {'Latency (ms)':<15} {'Worker':<8}")
    print("-" * 55)

    for r in results:
        spec = r["spec"]
        print(
            f"{spec['name']:<12} {spec['batch']:<8} {r['strategy']:<12} {r['latency_ms']:<15.3f} {r['worker_id']:<8}"
        )


def demo_local_optimization():
    """Demo: Local optimization without Ray."""
    print("\n" + "=" * 80)
    print("Demo 5: Local Optimization (No Ray)")
    print("=" * 80)

    print("\nRunning optimization locally...")

    # Use the loaded modules directly
    cluster = topology.ClusterTopology.create_single_node(8, "A100", nvlink=True)
    optimizer = auto_optimizer.UniversalOptimizer(cluster)

    print(f"Cluster: {cluster.name}")
    print(f"Devices: {cluster.num_devices()}")
    print(f"Total compute: {cluster.total_compute_tflops('fp16'):.0f} TFLOPS")

    # Test different tasks
    tasks = [
        ("Attention B32", task.ComputeTask.create_attention(32, 2048, 32, 128)),
        ("MLP B32", task.ComputeTask.create_mlp(32, 2048, 4096, 16384)),
        ("Transformer B8", task.ComputeTask.create_transformer_block(8, 2048, 4096, 32, 16384)),
    ]

    print(f"\n{'Task':<20} {'Strategy':<20} {'Latency (ms)':<15} {'Throughput':<12}")
    print("-" * 67)

    for name, compute_task in tasks:
        batch = int(name.split("B")[1]) if "B" in name else 1
        result = optimizer.optimize(compute_task, batch_size=batch)
        print(
            f"{name:<20} {result.parallelism_strategy:<20} {result.estimated_latency_ms:<15.3f} {result.estimated_throughput_tps:<12.1f}"
        )


def demo_ray_with_local_modules():
    """Demo: Ray with local module integration."""
    print("\n" + "=" * 80)
    print("Demo 6: Ray + Local Modules Integration")
    print("=" * 80)

    if not RAY_AVAILABLE:
        print("\nRay not available. Skipping.")
        return

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, logging_level="ERROR")

    # Use Ray for parallel execution, local modules for optimization
    @ray.remote
    def simulate_strategy(strategy: str, batch: int, num_gpus: int) -> Dict:
        """Simulate a strategy (self-contained)."""
        import numpy as np

        seq_len = 2048
        hidden = 4096
        flops = 4 * batch * seq_len * hidden * hidden
        peak = 312.0 * num_gpus

        base_time = (flops / (peak * 1e12)) * 1000

        # Strategy-specific overhead
        if strategy == "tensor_parallel":
            overhead = 0.1 * (num_gpus - 1)
        elif strategy == "data_parallel":
            overhead = 0.15 * (num_gpus - 1)
        else:
            overhead = 0

        total = base_time * (1 + overhead)

        return {
            "strategy": f"{strategy}_{num_gpus}",
            "time_ms": total,
            "efficiency": base_time / total if total > 0 else 0,
        }

    print("\nUsing Ray for parallel search, local modules for detailed simulation...")

    # Step 1: Quick parallel search with Ray
    strategies = ["tensor_parallel", "data_parallel"]
    gpu_counts = [2, 4, 8]
    batch = 32

    futures = []
    for strategy in strategies:
        for num_gpus in gpu_counts:
            futures.append(simulate_strategy.remote(strategy, batch, num_gpus))

    ray_results = ray.get(futures)

    # Find best from Ray search
    best_ray = min(ray_results, key=lambda x: x["time_ms"])
    print(f"\nRay quick search found: {best_ray['strategy']} ({best_ray['time_ms']:.3f} ms)")

    # Step 2: Detailed local simulation for the best strategy
    print("\nRunning detailed local simulation...")

    cluster = topology.ClusterTopology.create_single_node(8, "A100", nvlink=True)
    sim = simulator.ClusterSimulator(cluster)
    compute_task = task.ComputeTask.create_attention(batch, 2048, 32, 128)

    # Simulate with local module
    tp_result = sim.simulate_tensor_parallel(compute_task, 8)
    dp_result = sim.simulate_data_parallel(compute_task, 8, batch)

    print(f"\nDetailed simulation results:")
    print(
        f"  Tensor Parallel 8: {tp_result.total_time_ms:.3f} ms, efficiency: {tp_result.compute_efficiency()*100:.1f}%"
    )
    print(
        f"  Data Parallel 8: {dp_result.total_time_ms:.3f} ms, efficiency: {dp_result.compute_efficiency()*100:.1f}%"
    )


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("Ray Integration Demo")
    print("=" * 80)

    if RAY_AVAILABLE:
        print(f"\nRay version: {ray.__version__}")
        print("Ray is available for distributed execution.")
    else:
        print("\nRay is not installed. Running local demos only.")
        print("To enable Ray features: pip install ray")

    # Always run local demo first
    demo_local_optimization()

    # Run Ray demos if available
    if RAY_AVAILABLE:
        demo_ray_basic()
        demo_ray_parallel_search()
        demo_ray_actor_cluster()
        demo_ray_distributed_optimization()
        demo_ray_with_local_modules()

        # Shutdown Ray
        ray.shutdown()

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nKey Points:")
    print("  1. Ray enables parallel optimization across configurations")
    print("  2. Ray actors can simulate distributed GPU clusters")
    print("  3. Local modules provide accurate performance modeling")
    print("  4. Combined approach: Ray for parallelism, local for accuracy")


if __name__ == "__main__":
    main()
