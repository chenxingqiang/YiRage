"""
Ray-Integrated Cluster Optimizer

Provides distributed optimization capabilities using Ray for:
1. Parallel strategy search across configurations
2. Distributed task optimization
3. Simulated multi-GPU cluster execution
4. Async verification and profiling
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable, Tuple
import time
import json

from .topology import ClusterTopology, DeviceSpec, DeviceType
from .task import ComputeTask, TaskGraph, SubTask, Operator, OperatorType, TensorSpec, DataType
from .simulator import ClusterSimulator, SimulatedExecution
from .auto_optimizer import UniversalOptimizer, OptimizationConfig, OptimizationResult
from .executor import SimulatedExecutor, ExecutionPlan, ExecutionResult

# Check Ray availability
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@dataclass
class RayClusterConfig:
    """Configuration for Ray-based distributed optimization."""

    # Ray settings
    num_workers: int = 4
    num_gpus_per_worker: int = 0  # 0 = CPU only

    # Optimization settings
    parallel_strategies: List[str] = field(
        default_factory=lambda: ["data_parallel", "tensor_parallel", "pipeline_parallel"]
    )
    gpu_counts_to_try: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # Search settings
    max_search_time_s: float = 60.0
    early_stop: bool = True

    # Verification settings
    verify_on_gpu: bool = False  # If True, use GPU for verification


@dataclass
class DistributedSearchResult:
    """Result from distributed optimization search."""

    best_strategy: str
    best_config: Dict[str, Any]
    best_latency_ms: float
    best_throughput_tps: float

    all_results: List[Dict] = field(default_factory=list)
    search_time_s: float = 0.0
    num_configurations_searched: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "best_strategy": self.best_strategy,
            "best_config": self.best_config,
            "best_latency_ms": self.best_latency_ms,
            "best_throughput_tps": self.best_throughput_tps,
            "num_configurations_searched": self.num_configurations_searched,
            "search_time_s": self.search_time_s,
        }


class RayClusterOptimizer:
    """
    Ray-integrated cluster optimizer.

    Uses Ray for distributed optimization:
    - Parallel strategy search
    - Distributed task optimization
    - Simulated cluster execution
    """

    def __init__(
        self,
        cluster: ClusterTopology,
        config: Optional[RayClusterConfig] = None,
    ):
        """
        Initialize Ray cluster optimizer.

        Args:
            cluster: Target cluster topology
            config: Ray cluster configuration
        """
        self.cluster = cluster
        self.config = config or RayClusterConfig()
        self.simulator = ClusterSimulator(cluster)
        self._ray_initialized = False

    def _ensure_ray(self):
        """Ensure Ray is initialized."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not installed. Install with: pip install ray")

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, logging_level="WARNING")
            self._ray_initialized = True

    def shutdown(self):
        """Shutdown Ray if we initialized it."""
        if self._ray_initialized and ray.is_initialized():
            ray.shutdown()
            self._ray_initialized = False

    def parallel_strategy_search(
        self,
        task: ComputeTask,
        batch_size: int = 1,
    ) -> DistributedSearchResult:
        """
        Search for optimal strategy using parallel Ray workers.

        Args:
            task: Compute task to optimize
            batch_size: Batch size for the task

        Returns:
            DistributedSearchResult with best configuration
        """
        self._ensure_ray()

        start_time = time.time()

        # Create search configurations
        search_configs = []
        for strategy in self.config.parallel_strategies:
            for num_gpus in self.config.gpu_counts_to_try:
                if num_gpus <= self.cluster.num_devices():
                    search_configs.append(
                        {
                            "strategy": strategy,
                            "num_gpus": num_gpus,
                            "batch_size": batch_size,
                        }
                    )

        # Define remote search function
        @ray.remote
        def search_single_config(
            config: Dict,
            task_dict: Dict,
            cluster_dict: Dict,
        ) -> Dict:
            """Search a single configuration."""
            try:
                # Recreate objects from dicts (serialization-friendly)
                strategy = config["strategy"]
                num_gpus = config["num_gpus"]
                batch = config["batch_size"]

                # Simple performance model
                task_flops = task_dict.get("flops", 1e12)
                peak_tflops = cluster_dict.get("peak_tflops_per_gpu", 312.0) * num_gpus

                # Base compute time
                compute_time_ms = (task_flops / (peak_tflops * 1e12)) * 1000

                # Strategy-specific overhead
                if strategy == "data_parallel":
                    # AllReduce overhead
                    comm_factor = 0.15 * (num_gpus - 1) / num_gpus
                    total_time = compute_time_ms * (1 + comm_factor)
                elif strategy == "tensor_parallel":
                    # AllGather overhead
                    comm_factor = 0.1 * (num_gpus - 1)
                    total_time = compute_time_ms * (1 + comm_factor)
                elif strategy == "pipeline_parallel":
                    # Pipeline bubble
                    bubble = (num_gpus - 1) / (4 + num_gpus - 1)  # 4 micro-batches
                    total_time = compute_time_ms * num_gpus * (1 + bubble)
                else:
                    total_time = compute_time_ms

                throughput = batch / total_time * 1000 if total_time > 0 else 0

                return {
                    "config": config,
                    "latency_ms": total_time,
                    "throughput_tps": throughput,
                    "strategy": f"{strategy}_{num_gpus}",
                    "success": True,
                }
            except Exception as e:
                return {
                    "config": config,
                    "success": False,
                    "error": str(e),
                }

        # Prepare task and cluster info for serialization
        task_dict = {
            "flops": task.total_flops(),
            "memory_bytes": task.total_memory_bytes(),
        }

        devices = self.cluster.all_devices()
        cluster_dict = {
            "num_devices": len(devices),
            "peak_tflops_per_gpu": devices[0][1].peak_compute("fp16") if devices else 312.0,
        }

        # Submit all searches
        futures = [
            search_single_config.remote(cfg, task_dict, cluster_dict) for cfg in search_configs
        ]

        # Gather results
        results = ray.get(futures)

        # Filter successful results
        valid_results = [r for r in results if r.get("success", False)]

        if not valid_results:
            raise RuntimeError("All search configurations failed")

        # Find best
        best = min(valid_results, key=lambda x: x["latency_ms"])

        search_time = time.time() - start_time

        return DistributedSearchResult(
            best_strategy=best["strategy"],
            best_config=best["config"],
            best_latency_ms=best["latency_ms"],
            best_throughput_tps=best["throughput_tps"],
            all_results=valid_results,
            search_time_s=search_time,
            num_configurations_searched=len(search_configs),
        )

    def distributed_optimize(
        self,
        tasks: List[ComputeTask],
        batch_sizes: Optional[List[int]] = None,
    ) -> List[DistributedSearchResult]:
        """
        Optimize multiple tasks in parallel.

        Args:
            tasks: List of compute tasks
            batch_sizes: Batch sizes for each task (default: [1] * len(tasks))

        Returns:
            List of optimization results
        """
        self._ensure_ray()

        if batch_sizes is None:
            batch_sizes = [1] * len(tasks)

        @ray.remote
        def optimize_task(
            task_dict: Dict,
            batch_size: int,
            cluster_dict: Dict,
            strategies: List[str],
            gpu_counts: List[int],
        ) -> Dict:
            """Optimize a single task."""
            best_time = float("inf")
            best_result = None

            for strategy in strategies:
                for num_gpus in gpu_counts:
                    if num_gpus > cluster_dict["num_devices"]:
                        continue

                    # Performance model
                    task_flops = task_dict.get("flops", 1e12)
                    peak_tflops = cluster_dict.get("peak_tflops", 312.0) * num_gpus

                    compute_time = (task_flops / (peak_tflops * 1e12)) * 1000

                    if strategy == "data_parallel":
                        comm_factor = 0.15 * (num_gpus - 1) / num_gpus
                    elif strategy == "tensor_parallel":
                        comm_factor = 0.1 * (num_gpus - 1)
                    else:
                        comm_factor = 0.2 * (num_gpus - 1)

                    total_time = compute_time * (1 + comm_factor)

                    if total_time < best_time:
                        best_time = total_time
                        best_result = {
                            "strategy": f"{strategy}_{num_gpus}",
                            "latency_ms": total_time,
                            "throughput_tps": batch_size / total_time * 1000,
                            "config": {"strategy": strategy, "num_gpus": num_gpus},
                        }

            return best_result or {"error": "No valid configuration found"}

        # Prepare cluster info
        devices = self.cluster.all_devices()
        cluster_dict = {
            "num_devices": len(devices),
            "peak_tflops": devices[0][1].peak_compute("fp16") if devices else 312.0,
        }

        # Submit all optimizations
        futures = []
        for task, batch in zip(tasks, batch_sizes):
            task_dict = {
                "name": task.name,
                "flops": task.total_flops(),
                "memory_bytes": task.total_memory_bytes(),
            }
            futures.append(
                optimize_task.remote(
                    task_dict,
                    batch,
                    cluster_dict,
                    self.config.parallel_strategies,
                    self.config.gpu_counts_to_try,
                )
            )

        # Gather results
        results = ray.get(futures)

        return [
            DistributedSearchResult(
                best_strategy=r.get("strategy", "unknown"),
                best_config=r.get("config", {}),
                best_latency_ms=r.get("latency_ms", 0),
                best_throughput_tps=r.get("throughput_tps", 0),
            )
            for r in results
        ]

    def simulate_distributed_execution(
        self,
        task: ComputeTask,
        strategy: str,
        num_devices: int,
    ) -> SimulatedExecution:
        """
        Simulate distributed execution with detailed timing.

        Args:
            task: Compute task
            strategy: Parallelism strategy
            num_devices: Number of devices to use

        Returns:
            SimulatedExecution with detailed metrics
        """
        if strategy == "data_parallel":
            # Assume batch is in task's first dimension
            return self.simulator.simulate_data_parallel(task, num_devices, batch_size=32)
        elif strategy == "tensor_parallel":
            return self.simulator.simulate_tensor_parallel(task, num_devices)
        elif strategy == "pipeline_parallel":
            num_stages = min(num_devices, len(task.operators))
            return self.simulator.simulate_pipeline_parallel(task, num_stages, num_micro_batches=8)
        else:
            # Single device
            task_graph = TaskGraph(
                original_task=task,
                subtasks=[SubTask("main", task.name, [op.op_id for op in task.operators])],
            )
            return self.simulator.simulate_execution(task_graph, {"main": "node0/gpu0"})


def create_ray_optimizer(
    num_gpus: int = 8,
    gpu_type: str = "A100",
    num_workers: int = 4,
) -> RayClusterOptimizer:
    """
    Convenience function to create a Ray cluster optimizer.

    Args:
        num_gpus: Number of GPUs in simulated cluster
        gpu_type: GPU type (A100, H100, V100)
        num_workers: Number of Ray workers

    Returns:
        Configured RayClusterOptimizer
    """
    cluster = ClusterTopology.create_single_node(num_gpus, gpu_type, nvlink=True)
    config = RayClusterConfig(num_workers=num_workers)
    return RayClusterOptimizer(cluster, config)


# Export helper for checking Ray availability
def is_ray_available() -> bool:
    """Check if Ray is available."""
    return RAY_AVAILABLE
