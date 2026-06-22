#!/usr/bin/env python3
"""
Universal Optimizer Demo

Demonstrates automatic optimization of any compute task on any cluster.
All simulation happens on a single device - no real cluster needed.
"""

import sys
import os

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

# Setup module loading without z3 dependency
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

# Setup rl package
rl_pkg = types.ModuleType("yirage.rl")
rl_pkg.__path__ = [os.path.join(yirage_pkg.__path__[0], "rl")]
sys.modules["yirage.rl"] = rl_pkg
yirage_pkg.rl = rl_pkg

# Setup cluster package
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

# Import what we need
ClusterTopology = topology.ClusterTopology
ComputeTask = task.ComputeTask
DataType = task.DataType
UniversalOptimizer = auto_optimizer.UniversalOptimizer
OptimizationConfig = auto_optimizer.OptimizationConfig
OptimizationStrategy = auto_optimizer.OptimizationStrategy
SimulatedExecutor = executor.SimulatedExecutor
ExecutionPlan = executor.ExecutionPlan


def demo_single_node_8gpu():
    """Demo: Optimize on single 8xA100 node."""
    print("\n" + "=" * 80)
    print("Demo 1: Single Node 8xA100 Optimization")
    print("=" * 80)

    # Create cluster topology
    cluster = ClusterTopology.create_single_node(
        num_gpus=8,
        gpu_type="A100",
        nvlink=True,
    )

    print(f"\nCluster: {cluster.name}")
    print(f"  Total devices: {cluster.num_devices()}")
    print(f"  Total memory: {cluster.total_memory_gb():.1f} GB")
    print(f"  Total compute: {cluster.total_compute_tflops('fp16'):.1f} TFLOPS (FP16)")

    # Create optimizer
    config = OptimizationConfig(
        strategy=OptimizationStrategy.BALANCED,
        enable_fusion=True,
    )
    optimizer = UniversalOptimizer(cluster, config)

    # Test different workloads
    workloads = [
        ("MatMul 4K", ComputeTask.create_matmul(4096, 4096, 4096)),
        ("Attention B32 S2048", ComputeTask.create_attention(32, 2048, 32, 128)),
        ("MLP B32 S2048", ComputeTask.create_mlp(32, 2048, 4096, 16384)),
        ("Transformer Block", ComputeTask.create_transformer_block(8, 2048, 4096, 32, 16384)),
    ]

    for name, task in workloads:
        print(f"\n--- {name} ---")
        print(f"  Total FLOPs: {task.total_flops() / 1e12:.2f} TFLOPS")
        print(f"  Total Memory: {task.total_memory_bytes() / 1e9:.2f} GB")

        result = optimizer.optimize(task, batch_size=32)

        print(f"\n  Optimal Strategy: {result.parallelism_strategy}")
        print(f"  Estimated Latency: {result.estimated_latency_ms:.2f} ms")
        print(f"  Throughput: {result.estimated_throughput_tps:.1f} samples/sec")
        print(f"  Compute Efficiency: {result.compute_efficiency*100:.1f}%")


def demo_multi_node_cluster():
    """Demo: Optimize on 4-node cluster."""
    print("\n" + "=" * 80)
    print("Demo 2: Multi-Node Cluster (4x8 H100)")
    print("=" * 80)

    # Create cluster topology
    cluster = ClusterTopology.create_multi_node(
        num_nodes=4,
        gpus_per_node=8,
        gpu_type="H100",
        inter_node_bandwidth_gbps=400.0,  # NDR InfiniBand
        inter_node_latency_us=1.5,
    )

    print(f"\nCluster: {cluster.name}")
    print(f"  Total devices: {cluster.num_devices()}")
    print(f"  Total memory: {cluster.total_memory_gb():.1f} GB")
    print(f"  Total compute: {cluster.total_compute_tflops('fp16'):.1f} TFLOPS (FP16)")

    # Create optimizer
    optimizer = UniversalOptimizer(cluster)

    # Large transformer block (LLM scale)
    task = ComputeTask.create_transformer_block(
        batch=64,
        seq_len=4096,
        hidden_dim=8192,
        num_heads=64,
        intermediate_dim=32768,
    )

    print(f"\nTask: Large Transformer Block")
    print(f"  Total FLOPs: {task.total_flops() / 1e12:.2f} TFLOPS")
    print(f"  Total Memory: {task.total_memory_bytes() / 1e9:.2f} GB")

    result = optimizer.optimize(task, batch_size=64)

    print(f"\n  Optimal Strategy: {result.parallelism_strategy}")
    print(f"  Device Placement: {len(result.device_placement)} devices")
    print(f"  Estimated Latency: {result.estimated_latency_ms:.2f} ms")
    print(f"  Throughput: {result.estimated_throughput_tps:.1f} samples/sec")
    print(f"  Compute Efficiency: {result.compute_efficiency*100:.1f}%")


def demo_heterogeneous_cluster():
    """Demo: Heterogeneous cluster with mixed hardware."""
    print("\n" + "=" * 80)
    print("Demo 3: Heterogeneous Cluster")
    print("=" * 80)

    # Create heterogeneous cluster
    cluster = ClusterTopology.create_heterogeneous(
        [
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
                    "tensor_cores": True,
                    "supports_bf16": True,
                },
            },
            {
                "device_type": "maca",
                "count": 2,
                "specs": {
                    "compute_units": 64,
                    "clock_mhz": 1800,
                    "peak_tflops_fp16": 150.0,
                    "peak_tflops_fp32": 30.0,
                    "memory_gb": 64.0,
                    "memory_bandwidth_gbps": 1600.0,
                },
            },
        ]
    )

    print(f"\nCluster: {cluster.name}")
    print(f"  Total devices: {cluster.num_devices()}")

    devices = cluster.all_devices()
    for dev_id, dev_spec in devices:
        print(
            f"  - {dev_id}: {dev_spec.device_type.value}, {dev_spec.peak_compute('fp16'):.0f} TFLOPS"
        )

    # Create optimizer
    optimizer = UniversalOptimizer(cluster)

    # Attention workload
    task = ComputeTask.create_attention(16, 2048, 32, 128)

    print(f"\nTask: Attention")
    print(f"  Total FLOPs: {task.total_flops() / 1e12:.2f} TFLOPS")

    result = optimizer.optimize(task, batch_size=16)

    print(f"\n  Optimal Strategy: {result.parallelism_strategy}")
    print(f"  Estimated Latency: {result.estimated_latency_ms:.2f} ms")


def demo_compare_strategies():
    """Demo: Compare different parallelism strategies."""
    print("\n" + "=" * 80)
    print("Demo 4: Strategy Comparison")
    print("=" * 80)

    cluster = ClusterTopology.create_single_node(num_gpus=8, gpu_type="A100")

    # Large matmul
    task = ComputeTask.create_matmul(8192, 8192, 8192, batch=32)

    print(f"\nTask: Large Batched MatMul")
    print(f"  Shape: [32, 8192, 8192] x [32, 8192, 8192]")
    print(f"  Total FLOPs: {task.total_flops() / 1e12:.2f} TFLOPS")

    # Simulate different strategies
    simulator = auto_optimizer.ClusterSimulator(cluster)

    strategies = [
        ("1 GPU", lambda: simulator.simulate_data_parallel(task, 1, 32)),
        ("2 GPU DP", lambda: simulator.simulate_data_parallel(task, 2, 32)),
        ("4 GPU DP", lambda: simulator.simulate_data_parallel(task, 4, 32)),
        ("8 GPU DP", lambda: simulator.simulate_data_parallel(task, 8, 32)),
        ("2 GPU TP", lambda: simulator.simulate_tensor_parallel(task, 2)),
        ("4 GPU TP", lambda: simulator.simulate_tensor_parallel(task, 4)),
        ("8 GPU TP", lambda: simulator.simulate_tensor_parallel(task, 8)),
    ]

    print(f"\n{'Strategy':<15} {'Time (ms)':<12} {'Compute %':<12} {'Comm (ms)':<12}")
    print("-" * 51)

    for name, sim_fn in strategies:
        try:
            sim = sim_fn()
            eff = sim.compute_efficiency() * 100
            print(f"{name:<15} {sim.total_time_ms:<12.3f} {eff:<12.1f} {sim.comm_time_ms:<12.3f}")
        except Exception as e:
            print(f"{name:<15} {'Error':<12}")


def demo_find_optimal_batch():
    """Demo: Find optimal batch size."""
    print("\n" + "=" * 80)
    print("Demo 5: Optimal Batch Size Search")
    print("=" * 80)

    cluster = ClusterTopology.create_single_node(num_gpus=4, gpu_type="A100")
    executor = SimulatedExecutor(cluster)

    def task_factory(batch_size):
        return ComputeTask.create_attention(batch_size, 2048, 32, 128)

    def plan_factory(task):
        return ExecutionPlan(
            task_name=task.name,
            parallelism_strategy="data_parallel_4",
            device_placement={
                "replica_0": "node0/gpu0",
                "replica_1": "node0/gpu1",
                "replica_2": "node0/gpu2",
                "replica_3": "node0/gpu3",
            },
        )

    batch_sizes = [4, 8, 16, 32, 64, 128]

    result = executor.find_optimal_batch_size(
        task_factory,
        plan_factory,
        batch_sizes,
        target_latency_ms=50.0,
    )

    print(f"\nBatch size analysis (target latency: 50ms):")
    print(f"{'Batch':<8} {'Latency (ms)':<15} {'Throughput':<15} {'Memory (GB)':<12}")
    print("-" * 50)

    for r in result["all_results"]:
        print(
            f"{r['batch_size']:<8} {r['latency_ms']:<15.2f} {r['throughput_tps']:<15.1f} {r['memory_gb']:<12.2f}"
        )

    optimal = result["optimal"]
    print(f"\nOptimal batch size: {optimal['batch_size']}")
    print(f"  Latency: {optimal['latency_ms']:.2f} ms")
    print(f"  Throughput: {optimal['throughput_tps']:.1f} samples/sec")


def demo_llm_optimization():
    """Demo: LLM-scale workload optimization."""
    print("\n" + "=" * 80)
    print("Demo 6: LLM Inference Optimization (7B Model Scale)")
    print("=" * 80)

    # 4-node cluster for LLM serving
    cluster = ClusterTopology.create_multi_node(
        num_nodes=2,
        gpus_per_node=4,
        gpu_type="A100",
    )

    print(f"\nCluster: {cluster.name}")
    print(f"  Total GPUs: {cluster.num_devices()}")
    print(f"  Total Memory: {cluster.total_memory_gb():.0f} GB")

    optimizer = UniversalOptimizer(cluster)

    # LLM-7B single layer (scaled up)
    hidden_dim = 4096
    num_heads = 32
    intermediate = 11008  # LLaMA 7B FFN

    # Different scenarios
    scenarios = [
        ("Prefill B1 S2048", 1, 2048),
        ("Prefill B8 S2048", 8, 2048),
        ("Decode B32 S1", 32, 1),
        ("Decode B128 S1", 128, 1),
    ]

    print(f"\nOptimizing LLM-7B inference (per-layer):")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Heads: {num_heads}")
    print(f"  Intermediate: {intermediate}")

    print(f"\n{'Scenario':<20} {'Strategy':<25} {'Latency (ms)':<15} {'TPS':<10}")
    print("-" * 70)

    for name, batch, seq_len in scenarios:
        task = ComputeTask.create_transformer_block(
            batch=batch,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            intermediate_dim=intermediate,
        )

        result = optimizer.optimize(task, batch_size=batch)

        print(
            f"{name:<20} {result.parallelism_strategy:<25} {result.estimated_latency_ms:<15.3f} {result.estimated_throughput_tps:<10.1f}"
        )


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("Universal Compute Optimizer Demo")
    print("=" * 80)
    print("\nThis demo shows automatic optimization of any compute task")
    print("on any cluster configuration, all simulated on a single device.")

    demo_single_node_8gpu()
    demo_multi_node_cluster()
    demo_heterogeneous_cluster()
    demo_compare_strategies()
    demo_find_optimal_batch()
    demo_llm_optimization()

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Any compute task can be represented and optimized")
    print("  2. Any cluster topology can be simulated")
    print("  3. Communication costs are accurately modeled")
    print("  4. Optimal parallelism strategy is automatically selected")
    print("  5. All simulation runs on single device - no real cluster needed")


if __name__ == "__main__":
    main()
