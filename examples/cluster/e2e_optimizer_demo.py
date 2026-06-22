#!/usr/bin/env python3
"""
E2E Optimizer Demo

Shows how to optimize any compute task with a single function call.
"""

import sys
import os

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

# Import
E2EOptimizer = e2e_optimizer.E2EOptimizer
OptimizationRequest = e2e_optimizer.OptimizationRequest
optimize_any_task = e2e_optimizer.optimize_any_task
ComputeTask = task.ComputeTask
ClusterTopology = topology.ClusterTopology


def demo_quick_optimize():
    """Demo: One-line optimization."""
    print("\n" + "=" * 80)
    print("Demo 1: One-Line Optimization")
    print("=" * 80)

    # Optimize attention with single function call
    result = optimize_any_task(
        {"type": "attention", "batch": 32, "seq_len": 2048, "num_heads": 32, "head_dim": 128}
    )

    print("\nOptimization Result:")
    print(result.summary())


def demo_custom_cluster():
    """Demo: Optimize on custom cluster."""
    print("\n" + "=" * 80)
    print("Demo 2: Custom Cluster Configuration")
    print("=" * 80)

    # Define custom cluster
    result = optimize_any_task(
        {"type": "transformer", "batch": 8, "seq_len": 4096, "hidden_dim": 4096},
        cluster_spec={
            "type": "multi_node",
            "num_nodes": 4,
            "gpus_per_node": 8,
            "gpu_type": "H100",
            "inter_node_bandwidth_gbps": 400.0,
        },
    )

    print("\nCluster Info:")
    for k, v in result.cluster_info.items():
        print(f"  {k}: {v}")

    print("\nOptimization Result:")
    print(result.summary())


def demo_full_workflow():
    """Demo: Full optimization workflow with µGraph configs."""
    print("\n" + "=" * 80)
    print("Demo 3: Full Workflow with µGraph Configs")
    print("=" * 80)

    # Create optimizer with specific cluster
    cluster = ClusterTopology.create_single_node(8, "A100", nvlink=True)
    optimizer = E2EOptimizer(cluster)

    # Create request
    request = OptimizationRequest(
        operation_spec={
            "type": "transformer",
            "batch": 16,
            "seq_len": 2048,
            "hidden_dim": 4096,
            "num_heads": 32,
            "intermediate_dim": 16384,
        },
        max_latency_ms=100.0,
        strategy="latency",
        enable_fusion=True,
    )

    # Optimize
    output = optimizer.optimize(request)

    print("\nPredictions:")
    for k, v in output.predictions.items():
        print(f"  {k}: {v:.4f}")

    print(f"\nKernel Configs: {len(output.kernel_configs)} operators")
    for op_id, config in list(output.kernel_configs.items())[:3]:
        print(f"  {op_id}: {config}")

    print(f"\nµGraph Configs: {len(output.mugraph_configs)} operators")
    if output.mugraph_configs:
        print(f"  First config search space: {output.mugraph_configs[0].get('search_space', {})}")

    print("\nRecommendations:")
    for rec in output.recommendations:
        print(f"  - {rec}")


def demo_benchmark():
    """Demo: Benchmark across batch sizes."""
    print("\n" + "=" * 80)
    print("Demo 4: Batch Size Benchmark")
    print("=" * 80)

    cluster = ClusterTopology.create_single_node(4, "A100")
    optimizer = E2EOptimizer(cluster)

    request = OptimizationRequest(
        operation_spec={"type": "attention", "seq_len": 2048, "num_heads": 32, "head_dim": 128},
        batch_sizes=[1, 4, 8, 16, 32, 64],
    )

    results = optimizer.benchmark(request)

    print("\nBatch Size Performance:")
    print(f"{'Batch':<8} {'Latency (ms)':<15} {'Throughput':<15} {'Strategy':<20}")
    print("-" * 58)

    for r in results["results"]:
        print(
            f"{r['batch_size']:<8} {r['latency_ms']:<15.3f} {r['throughput_tps']:<15.1f} {r['strategy']:<20}"
        )

    if results["optimal"]:
        opt = results["optimal"]
        print(
            f"\nOptimal: batch_size={opt['batch_size']}, throughput={opt['throughput_tps']:.1f} samples/sec"
        )


def demo_heterogeneous():
    """Demo: Heterogeneous cluster optimization."""
    print("\n" + "=" * 80)
    print("Demo 5: Heterogeneous Cluster")
    print("=" * 80)

    result = optimize_any_task(
        {
            "type": "mlp",
            "batch": 32,
            "seq_len": 1024,
            "hidden_dim": 4096,
            "intermediate_dim": 16384,
        },
        cluster_spec={
            "type": "heterogeneous",
            "devices": [
                {
                    "device_type": "cuda",
                    "count": 4,
                    "specs": {
                        "compute_units": 108,
                        "clock_mhz": 1410,
                        "peak_tflops_fp16": 312.0,
                        "peak_tflops_fp32": 19.5,
                        "memory_gb": 80.0,
                        "memory_bandwidth_gbps": 2039.0,
                    },
                },
                {
                    "device_type": "maca",
                    "count": 2,
                    "specs": {
                        "compute_units": 64,
                        "clock_mhz": 1800,
                        "peak_tflops_fp16": 200.0,
                        "peak_tflops_fp32": 40.0,
                        "memory_gb": 64.0,
                        "memory_bandwidth_gbps": 1600.0,
                    },
                },
            ],
        },
    )

    print("\nCluster Info:")
    for k, v in result.cluster_info.items():
        print(f"  {k}: {v}")

    print("\n" + result.summary())


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("E2E Optimizer Demo - Universal Compute Optimization")
    print("=" * 80)
    print("\nOptimize ANY compute task on ANY hardware with minimal code.")

    demo_quick_optimize()
    demo_custom_cluster()
    demo_full_workflow()
    demo_benchmark()
    demo_heterogeneous()

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
