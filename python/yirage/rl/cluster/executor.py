"""
Simulated Executor

Executes optimized task graphs on simulated or real clusters.
Provides unified interface for both simulation and actual execution.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
import time
import json

from .topology import ClusterTopology
from .task import ComputeTask, TaskGraph
from .simulator import ClusterSimulator, SimulatedExecution, ComputeEvent, CommunicationEvent


class ExecutionMode(Enum):
    """Execution modes."""

    SIMULATE = "simulate"  # Pure simulation
    PROFILE = "profile"  # Run and profile
    EXECUTE = "execute"  # Full execution


@dataclass
class ExecutionPlan:
    """
    A complete execution plan for a task on a cluster.
    """

    # Task info
    task_name: str

    # Parallelism configuration
    parallelism_strategy: str
    device_placement: Dict[str, str]

    # Execution schedule
    schedule: List[str] = field(default_factory=list)

    # Kernel configurations
    kernel_configs: Dict[str, Dict] = field(default_factory=dict)

    # Communication plan
    comm_schedule: List[Dict] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "task_name": self.task_name,
            "parallelism_strategy": self.parallelism_strategy,
            "device_placement": self.device_placement,
            "schedule": self.schedule,
            "kernel_configs": self.kernel_configs,
            "comm_schedule": self.comm_schedule,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "ExecutionPlan":
        """Create from dictionary."""
        return cls(
            task_name=d["task_name"],
            parallelism_strategy=d["parallelism_strategy"],
            device_placement=d["device_placement"],
            schedule=d.get("schedule", []),
            kernel_configs=d.get("kernel_configs", {}),
            comm_schedule=d.get("comm_schedule", []),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ExecutionResult:
    """
    Result of task execution.
    """

    # Success status
    success: bool = True
    error_message: str = ""

    # Timing
    total_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    comm_time_ms: float = 0.0

    # Output data (for verification)
    outputs: Dict[str, Any] = field(default_factory=dict)

    # Detailed timing
    operator_times: Dict[str, float] = field(default_factory=dict)
    comm_times: Dict[str, float] = field(default_factory=dict)

    # Resource usage
    peak_memory_gb: float = 0.0
    device_utilization: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "total_time_ms": self.total_time_ms,
            "compute_time_ms": self.compute_time_ms,
            "comm_time_ms": self.comm_time_ms,
            "operator_times": self.operator_times,
            "comm_times": self.comm_times,
            "peak_memory_gb": self.peak_memory_gb,
            "device_utilization": self.device_utilization,
        }


@dataclass
class SimulatedExecutor:
    """
    Executor that uses simulation for development and optimization.
    Can be extended to real execution.
    """

    cluster: ClusterTopology
    simulator: ClusterSimulator = field(init=False)

    # Execution callbacks (for real execution)
    compute_callback: Optional[Callable] = None
    comm_callback: Optional[Callable] = None

    # History
    execution_history: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize simulator."""
        self.simulator = ClusterSimulator(self.cluster)

    def execute(
        self,
        task: ComputeTask,
        plan: ExecutionPlan,
        mode: ExecutionMode = ExecutionMode.SIMULATE,
    ) -> ExecutionResult:
        """
        Execute a task according to plan.

        Args:
            task: The compute task
            plan: Execution plan
            mode: Execution mode

        Returns:
            ExecutionResult with timing and outputs
        """

        if mode == ExecutionMode.SIMULATE:
            return self._execute_simulated(task, plan)
        elif mode == ExecutionMode.PROFILE:
            return self._execute_with_profile(task, plan)
        else:
            return self._execute_real(task, plan)

    def _execute_simulated(
        self,
        task: ComputeTask,
        plan: ExecutionPlan,
    ) -> ExecutionResult:
        """Execute in simulation mode."""

        # Create task graph from plan
        task_graph = self._plan_to_task_graph(task, plan)

        # Run simulation
        sim = self.simulator.simulate_execution(
            task_graph,
            plan.device_placement,
            plan.schedule if plan.schedule else None,
        )

        # Convert to result
        result = ExecutionResult(
            success=True,
            total_time_ms=sim.total_time_ms,
            compute_time_ms=sim.compute_time_ms,
            comm_time_ms=sim.comm_time_ms,
            device_utilization=sim.device_utilization,
        )

        # Operator times
        for event in sim.compute_events:
            result.operator_times[event.operator_id] = event.end_time_ms - event.start_time_ms

        # Communication times
        for event in sim.comm_events:
            result.comm_times[event.tensor_name] = event.end_time_ms - event.start_time_ms

        # Estimate memory
        result.peak_memory_gb = task.total_memory_bytes() / (1024**3)

        # Store in history
        self.execution_history.append(
            {
                "task": task.name,
                "plan": plan.to_dict(),
                "result": result.to_dict(),
            }
        )

        return result

    def _execute_with_profile(
        self,
        task: ComputeTask,
        plan: ExecutionPlan,
    ) -> ExecutionResult:
        """Execute with profiling on real hardware."""

        # For now, simulate with more accurate timing
        result = self._execute_simulated(task, plan)

        # If we had real execution callbacks, we would use them here
        if self.compute_callback:
            # Profile actual kernels
            pass

        return result

    def _execute_real(
        self,
        task: ComputeTask,
        plan: ExecutionPlan,
    ) -> ExecutionResult:
        """Real execution on hardware."""

        # This would be implemented with actual GPU/device calls
        # For now, fall back to simulation
        return self._execute_simulated(task, plan)

    def _plan_to_task_graph(
        self,
        task: ComputeTask,
        plan: ExecutionPlan,
    ) -> TaskGraph:
        """Convert execution plan to task graph."""

        from .task import SubTask

        # Create subtasks based on placement
        subtasks = []
        device_ops = {}  # device_id -> list of op_ids

        for subtask_id, device_id in plan.device_placement.items():
            if device_id not in device_ops:
                device_ops[device_id] = []

        # Assign operators to devices (simple round-robin if not specified)
        ops = [op.op_id for op in task.operators]
        devices = list(plan.device_placement.values())

        for i, op_id in enumerate(ops):
            device = devices[i % len(devices)] if devices else "gpu0"
            if device in device_ops:
                device_ops[device].append(op_id)

        for i, (device_id, ops_list) in enumerate(device_ops.items()):
            subtask = SubTask(
                subtask_id=f"subtask_{i}",
                original_task=task.name,
                operators=ops_list,
                device_id=device_id,
            )
            subtasks.append(subtask)

        return TaskGraph(
            original_task=task,
            subtasks=subtasks,
        )

    def benchmark(
        self,
        task: ComputeTask,
        plan: ExecutionPlan,
        num_iterations: int = 10,
        warmup: int = 3,
    ) -> Dict[str, float]:
        """
        Benchmark execution with multiple iterations.

        Returns:
            Statistics (mean, std, min, max) of execution time
        """

        times = []

        # Warmup
        for _ in range(warmup):
            self.execute(task, plan, ExecutionMode.SIMULATE)

        # Benchmark
        for _ in range(num_iterations):
            result = self.execute(task, plan, ExecutionMode.SIMULATE)
            times.append(result.total_time_ms)

        import numpy as np

        times = np.array(times)

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p99_ms": float(np.percentile(times, 99)),
        }

    def compare_plans(
        self,
        task: ComputeTask,
        plans: List[ExecutionPlan],
    ) -> Dict[str, Dict]:
        """
        Compare multiple execution plans.

        Returns:
            Performance comparison for each plan
        """

        results = {}

        for plan in plans:
            result = self.execute(task, plan, ExecutionMode.SIMULATE)

            results[plan.parallelism_strategy] = {
                "latency_ms": result.total_time_ms,
                "compute_time_ms": result.compute_time_ms,
                "comm_time_ms": result.comm_time_ms,
                "efficiency": (
                    result.compute_time_ms / result.total_time_ms if result.total_time_ms > 0 else 0
                ),
            }

        return results

    def find_optimal_batch_size(
        self,
        task_factory: Callable[[int], ComputeTask],
        plan_factory: Callable[[ComputeTask], ExecutionPlan],
        batch_sizes: List[int],
        target_latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Find optimal batch size for a task.

        Args:
            task_factory: Function to create task with batch size
            plan_factory: Function to create plan for task
            batch_sizes: Batch sizes to try
            target_latency_ms: Optional latency constraint

        Returns:
            Analysis results with optimal batch size
        """

        results = []

        for batch_size in batch_sizes:
            task = task_factory(batch_size)
            plan = plan_factory(task)

            result = self.execute(task, plan, ExecutionMode.SIMULATE)

            throughput = batch_size / result.total_time_ms * 1000 if result.total_time_ms > 0 else 0

            results.append(
                {
                    "batch_size": batch_size,
                    "latency_ms": result.total_time_ms,
                    "throughput_tps": throughput,
                    "memory_gb": task.total_memory_bytes() / (1024**3),
                }
            )

        # Find optimal
        if target_latency_ms:
            # Maximize throughput within latency constraint
            valid = [r for r in results if r["latency_ms"] <= target_latency_ms]
            if valid:
                optimal = max(valid, key=lambda r: r["throughput_tps"])
            else:
                optimal = min(results, key=lambda r: r["latency_ms"])
        else:
            # Maximize throughput
            optimal = max(results, key=lambda r: r["throughput_tps"])

        return {
            "optimal": optimal,
            "all_results": results,
        }
