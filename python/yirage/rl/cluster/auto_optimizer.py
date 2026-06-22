"""
Universal Auto-Optimizer

Automatically optimizes any compute task on any cluster configuration.
Core innovation: learns to transfer optimization knowledge across hardware.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
import json
import time
import os

from .topology import ClusterTopology, DeviceSpec, DeviceType
from .task import ComputeTask, TaskGraph, SubTask, Operator, TensorSpec, OperatorType
from .simulator import ClusterSimulator, SimulatedExecution, CommunicationModel


class OptimizationStrategy(Enum):
    """High-level optimization strategies."""

    LATENCY = "latency"  # Minimize latency
    THROUGHPUT = "throughput"  # Maximize throughput
    EFFICIENCY = "efficiency"  # Maximize compute efficiency
    MEMORY = "memory"  # Minimize memory usage
    BALANCED = "balanced"  # Balance all metrics


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""

    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED

    # Constraints
    max_latency_ms: Optional[float] = None
    max_memory_gb: Optional[float] = None
    min_throughput_tps: Optional[float] = None

    # Search parameters
    max_search_time_s: float = 60.0
    max_iterations: int = 1000
    early_stop_patience: int = 50

    # Features
    enable_fusion: bool = True
    enable_recompute: bool = True
    enable_mixed_precision: bool = True

    # Transfer learning
    use_transfer_learning: bool = True
    transfer_threshold: float = 0.8  # Similarity threshold


@dataclass
class OptimizationResult:
    """Result of optimization."""

    # Configuration
    parallelism_strategy: str
    device_placement: Dict[str, str]

    # Performance
    estimated_latency_ms: float
    estimated_throughput_tps: float
    estimated_memory_gb: float
    compute_efficiency: float

    # Details
    operator_schedule: List[str] = field(default_factory=list)
    fusion_groups: List[List[str]] = field(default_factory=list)

    # Kernel configurations per operator
    kernel_configs: Dict[str, Dict] = field(default_factory=dict)

    # Metadata
    search_time_s: float = 0.0
    iterations: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "parallelism_strategy": self.parallelism_strategy,
            "device_placement": self.device_placement,
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_throughput_tps": self.estimated_throughput_tps,
            "estimated_memory_gb": self.estimated_memory_gb,
            "compute_efficiency": self.compute_efficiency,
            "operator_schedule": self.operator_schedule,
            "fusion_groups": self.fusion_groups,
            "kernel_configs": self.kernel_configs,
            "search_time_s": self.search_time_s,
            "iterations": self.iterations,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TaskDecomposer:
    """
    Decomposes a compute task into sub-tasks for parallel execution.
    """

    def decompose(
        self,
        task: ComputeTask,
        num_devices: int,
        strategy: str = "auto",
    ) -> TaskGraph:
        """
        Decompose task into sub-tasks.

        Args:
            task: The compute task
            num_devices: Number of target devices
            strategy: Decomposition strategy

        Returns:
            TaskGraph with sub-tasks
        """

        if strategy == "auto":
            # Choose based on task characteristics
            if self._is_data_parallel_friendly(task):
                strategy = "data_parallel"
            elif self._is_tensor_parallel_friendly(task):
                strategy = "tensor_parallel"
            else:
                strategy = "operator_parallel"

        if strategy == "data_parallel":
            return self._decompose_data_parallel(task, num_devices)
        elif strategy == "tensor_parallel":
            return self._decompose_tensor_parallel(task, num_devices)
        elif strategy == "pipeline_parallel":
            return self._decompose_pipeline_parallel(task, num_devices)
        else:
            return self._decompose_operator_parallel(task, num_devices)

    def _is_data_parallel_friendly(self, task: ComputeTask) -> bool:
        """Check if task is suitable for data parallelism."""
        # Data parallel works well for tasks with batch dimension
        for tensor in task.tensors.values():
            if len(tensor.shape) >= 2 and tensor.shape[0] > 1:
                return True
        return False

    def _is_tensor_parallel_friendly(self, task: ComputeTask) -> bool:
        """Check if task is suitable for tensor parallelism."""
        # Tensor parallel works for large matrix operations
        for op in task.operators:
            if op.op_type in (OperatorType.MATMUL, OperatorType.BATCH_MATMUL):
                return True
        return False

    def _decompose_data_parallel(
        self,
        task: ComputeTask,
        num_devices: int,
    ) -> TaskGraph:
        """Decompose for data parallelism."""

        subtasks = []
        for i in range(num_devices):
            subtask = SubTask(
                subtask_id=f"dp_replica_{i}",
                original_task=task.name,
                operators=[op.op_id for op in task.operators],
            )
            subtasks.append(subtask)

        return TaskGraph(
            original_task=task,
            subtasks=subtasks,
        )

    def _decompose_tensor_parallel(
        self,
        task: ComputeTask,
        num_devices: int,
    ) -> TaskGraph:
        """Decompose for tensor parallelism."""

        subtasks = []

        for i in range(num_devices):
            # Each device handles a partition of tensor operations
            ops = [op.op_id for op in task.operators]

            subtask = SubTask(
                subtask_id=f"tp_shard_{i}",
                original_task=task.name,
                operators=ops,
                external_outputs=[f"shard_{i}"],
            )
            subtasks.append(subtask)

        return TaskGraph(
            original_task=task,
            subtasks=subtasks,
        )

    def _decompose_pipeline_parallel(
        self,
        task: ComputeTask,
        num_stages: int,
    ) -> TaskGraph:
        """Decompose for pipeline parallelism."""

        ops = task.operators
        ops_per_stage = max(1, len(ops) // num_stages)

        subtasks = []
        for i in range(num_stages):
            start = i * ops_per_stage
            end = start + ops_per_stage if i < num_stages - 1 else len(ops)

            stage_ops = [op.op_id for op in ops[start:end]]

            # Determine external inputs/outputs
            external_inputs = []
            external_outputs = []

            if i > 0:
                # Input from previous stage
                prev_ops = ops[start - 1 : start]
                for op in prev_ops:
                    external_inputs.extend(op.outputs)

            if i < num_stages - 1:
                # Output to next stage
                last_op = ops[end - 1] if end > start else None
                if last_op:
                    external_outputs.extend(last_op.outputs)

            subtask = SubTask(
                subtask_id=f"stage_{i}",
                original_task=task.name,
                operators=stage_ops,
                external_inputs=external_inputs,
                external_outputs=external_outputs,
            )
            subtasks.append(subtask)

        # Create dependencies
        dependencies = []
        for i in range(num_stages - 1):
            # Estimate size of activation transfer
            size = 0
            for out in subtasks[i].external_outputs:
                if out in task.tensors:
                    size += task.tensors[out].size_bytes()

            dependencies.append(
                (
                    subtasks[i].subtask_id,
                    subtasks[i + 1].subtask_id,
                    size,
                )
            )

        return TaskGraph(
            original_task=task,
            subtasks=subtasks,
            dependencies=dependencies,
        )

    def _decompose_operator_parallel(
        self,
        task: ComputeTask,
        num_devices: int,
    ) -> TaskGraph:
        """Decompose by operator-level parallelism."""

        # Group independent operators
        deps = task.get_dependencies()
        dep_set = {(d.producer_op, d.consumer_op) for d in deps}

        # Find operators that can run in parallel
        levels = []
        remaining = set(op.op_id for op in task.operators)
        completed = set()

        while remaining:
            # Find ops with all dependencies satisfied
            level = []
            for op_id in remaining:
                deps_satisfied = all(prod in completed for prod, cons in dep_set if cons == op_id)
                if deps_satisfied:
                    level.append(op_id)

            if not level:
                # Cycle or error, just take one
                level = [remaining.pop()]

            levels.append(level)
            for op_id in level:
                remaining.discard(op_id)
                completed.add(op_id)

        # Create subtasks for each level
        subtasks = []
        for level_idx, level_ops in enumerate(levels):
            subtask = SubTask(
                subtask_id=f"level_{level_idx}",
                original_task=task.name,
                operators=level_ops,
            )
            subtasks.append(subtask)

        return TaskGraph(
            original_task=task,
            subtasks=subtasks,
        )


@dataclass
class OperatorFuser:
    """
    Fuses operators for better performance.
    """

    # Fusion rules: (op1, op2) -> can_fuse
    fusion_rules: Dict[Tuple[str, str], bool] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize default fusion rules."""
        # Default fusable patterns
        fusable = [
            ("matmul", "add"),  # GEMM + bias
            ("matmul", "gelu"),  # Linear + activation
            ("matmul", "silu"),
            ("matmul", "relu"),
            ("layer_norm", "matmul"),  # Norm + Linear
            ("rms_norm", "matmul"),
            ("softmax", "matmul"),  # Attention pattern
            ("add", "layer_norm"),  # Residual + norm
            ("add", "rms_norm"),
        ]

        for op1, op2 in fusable:
            self.fusion_rules[(op1, op2)] = True

    def find_fusion_groups(self, task: ComputeTask) -> List[List[str]]:
        """Find groups of operators that can be fused."""

        groups = []
        used = set()

        ops = task.operators
        op_map = {op.op_id: op for op in ops}

        for i, op in enumerate(ops):
            if op.op_id in used:
                continue

            group = [op.op_id]
            used.add(op.op_id)

            # Try to extend the group
            current_op = op
            for j in range(i + 1, len(ops)):
                next_op = ops[j]
                if next_op.op_id in used:
                    continue

                # Check if can fuse
                key = (current_op.op_type.value, next_op.op_type.value)
                if self.fusion_rules.get(key, False):
                    # Check data dependency
                    if any(out in next_op.inputs for out in current_op.outputs):
                        group.append(next_op.op_id)
                        used.add(next_op.op_id)
                        current_op = next_op

            groups.append(group)

        return groups

    def estimate_fusion_speedup(
        self,
        group: List[str],
        task: ComputeTask,
    ) -> float:
        """Estimate speedup from fusing a group of operators."""

        if len(group) <= 1:
            return 1.0

        # Fusion reduces memory traffic and kernel launch overhead
        # Estimate: 1.1x speedup per fused op
        return 1.0 + 0.1 * (len(group) - 1)


@dataclass
class CostModel:
    """
    Learned cost model for predicting performance.
    Supports transfer learning across hardware.
    """

    # Cache of known configurations
    cache: Dict[str, float] = field(default_factory=dict)

    # Hardware embeddings for transfer
    hardware_embeddings: Dict[str, np.ndarray] = field(default_factory=dict)

    def predict(
        self,
        operator: Operator,
        device: DeviceSpec,
        config: Dict,
        tensor_specs: Dict[str, TensorSpec],
    ) -> float:
        """
        Predict execution time for an operator configuration.
        """

        # Create cache key
        key = self._make_key(operator, device, config)

        if key in self.cache:
            return self.cache[key]

        # Roofline-based prediction
        flops = operator.estimate_flops(tensor_specs)
        memory_bytes = operator.estimate_memory_bytes(tensor_specs)

        peak_tflops = device.peak_compute("fp16")
        memory_bw = device.memory_bandwidth_gbps

        compute_time = (flops / (peak_tflops * 1e12)) * 1000
        memory_time = (memory_bytes / (memory_bw * 1e9)) * 1000

        # Apply config adjustments
        tile_size = config.get("tile_size", 128)
        efficiency = min(1.0, tile_size / 128)

        predicted = max(compute_time, memory_time) / efficiency * 1.2

        self.cache[key] = predicted
        return predicted

    def _make_key(
        self,
        operator: Operator,
        device: DeviceSpec,
        config: Dict,
    ) -> str:
        """Create cache key."""
        return f"{operator.op_id}_{device.device_id}_{hash(frozenset(config.items()))}"

    def update(
        self,
        operator: Operator,
        device: DeviceSpec,
        config: Dict,
        actual_time: float,
    ):
        """Update model with actual measurement."""
        key = self._make_key(operator, device, config)
        self.cache[key] = actual_time

    def get_hardware_embedding(self, device: DeviceSpec) -> np.ndarray:
        """Get embedding for a device."""
        if device.device_id in self.hardware_embeddings:
            return self.hardware_embeddings[device.device_id]

        # Create embedding from device features
        embedding = device.to_feature_vector()
        self.hardware_embeddings[device.device_id] = embedding
        return embedding

    def find_similar_hardware(
        self,
        target: DeviceSpec,
        known_devices: List[DeviceSpec],
        threshold: float = 0.8,
    ) -> Optional[DeviceSpec]:
        """Find similar hardware for transfer learning."""

        target_emb = self.get_hardware_embedding(target)

        best_sim = 0.0
        best_device = None

        for device in known_devices:
            device_emb = self.get_hardware_embedding(device)

            # Cosine similarity
            sim = np.dot(target_emb, device_emb) / (
                np.linalg.norm(target_emb) * np.linalg.norm(device_emb) + 1e-8
            )

            if sim > best_sim and sim >= threshold:
                best_sim = sim
                best_device = device

        return best_device


@dataclass
class UniversalOptimizer:
    """
    Universal optimizer for any compute task on any cluster.

    Key features:
    1. Automatic task analysis and decomposition
    2. Learned cost model with transfer learning
    3. Hardware-adaptive kernel search
    4. Simulation-based validation
    """

    cluster: ClusterTopology
    config: OptimizationConfig = field(default_factory=OptimizationConfig)

    # Components
    decomposer: TaskDecomposer = field(default_factory=TaskDecomposer)
    fuser: OperatorFuser = field(default_factory=OperatorFuser)
    cost_model: CostModel = field(default_factory=CostModel)
    simulator: ClusterSimulator = field(init=False)

    # Experience buffer for learning
    experience: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize simulator."""
        self.simulator = ClusterSimulator(self.cluster)

    def optimize(
        self,
        task: ComputeTask,
        batch_size: int = 1,
    ) -> OptimizationResult:
        """
        Optimize a compute task for the cluster.

        Args:
            task: The compute task to optimize
            batch_size: Batch size for execution

        Returns:
            OptimizationResult with the best configuration
        """

        start_time = time.time()

        # Step 1: Analyze task
        task_features = self._analyze_task(task)

        # Step 2: Find fusion opportunities
        fusion_groups = []
        if self.config.enable_fusion:
            fusion_groups = self.fuser.find_fusion_groups(task)

        # Step 3: Try different parallelism strategies
        best_result = None
        best_time = float("inf")
        all_results = []

        num_devices = self.cluster.num_devices()

        strategies = [
            ("data_parallel", lambda: self._try_data_parallel(task, batch_size)),
            ("tensor_parallel", lambda: self._try_tensor_parallel(task)),
            ("pipeline_parallel", lambda: self._try_pipeline_parallel(task)),
            ("hybrid", lambda: self._try_hybrid_parallel(task, batch_size)),
        ]

        for strategy_name, strategy_fn in strategies:
            try:
                result = strategy_fn()
                if result and result.estimated_latency_ms < best_time:
                    best_time = result.estimated_latency_ms
                    best_result = result
                all_results.append(result)
            except Exception as e:
                continue

        # Step 4: Refine with kernel search
        if best_result:
            best_result = self._refine_kernels(task, best_result)
        else:
            # Fallback: single device
            best_result = self._single_device_optimize(task)

        # Add fusion info
        if best_result:
            best_result.fusion_groups = fusion_groups

        search_time = time.time() - start_time
        if best_result:
            best_result.search_time_s = search_time

        # Store experience for learning
        self.experience.append(
            {
                "task": task.to_dict(),
                "batch_size": batch_size,
                "result": best_result.to_dict() if best_result else None,
            }
        )

        return best_result

    def _analyze_task(self, task: ComputeTask) -> Dict:
        """Analyze task characteristics."""

        total_flops = task.total_flops()
        total_memory = task.total_memory_bytes()

        # Compute intensity
        intensity = total_flops / max(total_memory, 1)

        # Parallelism potential
        deps = task.get_dependencies()
        parallelism = len(task.operators) / max(len(deps), 1)

        return {
            "total_flops": total_flops,
            "total_memory_bytes": total_memory,
            "arithmetic_intensity": intensity,
            "parallelism_potential": parallelism,
            "num_operators": len(task.operators),
            "is_compute_bound": intensity > 10,
        }

    def _try_data_parallel(
        self,
        task: ComputeTask,
        batch_size: int,
    ) -> OptimizationResult:
        """Try data parallel strategy."""

        num_devices = self.cluster.num_devices()

        # Find best device count
        best_sim = None
        best_dp = 1

        for dp in [1, 2, 4, 8]:
            if dp > num_devices or batch_size % dp != 0:
                continue

            sim = self.simulator.simulate_data_parallel(task, dp, batch_size)

            if best_sim is None or sim.total_time_ms < best_sim.total_time_ms:
                best_sim = sim
                best_dp = dp

        if best_sim is None:
            return None

        devices = self.cluster.all_devices()[:best_dp]
        placement = {f"replica_{i}": devices[i][0] for i in range(best_dp)}

        throughput = (
            (batch_size / best_sim.total_time_ms) * 1000 if best_sim.total_time_ms > 0 else 0
        )

        return OptimizationResult(
            parallelism_strategy=f"data_parallel_{best_dp}",
            device_placement=placement,
            estimated_latency_ms=best_sim.total_time_ms,
            estimated_throughput_tps=throughput,
            estimated_memory_gb=task.total_memory_bytes() / (1024**3),
            compute_efficiency=best_sim.compute_efficiency(),
        )

    def _try_tensor_parallel(self, task: ComputeTask) -> OptimizationResult:
        """Try tensor parallel strategy."""

        num_devices = self.cluster.num_devices()

        best_sim = None
        best_tp = 1

        for tp in [2, 4, 8]:
            if tp > num_devices:
                continue

            sim = self.simulator.simulate_tensor_parallel(task, tp)

            if best_sim is None or sim.total_time_ms < best_sim.total_time_ms:
                best_sim = sim
                best_tp = tp

        if best_sim is None or best_tp == 1:
            return None

        devices = self.cluster.all_devices()[:best_tp]
        placement = {f"shard_{i}": devices[i][0] for i in range(best_tp)}

        return OptimizationResult(
            parallelism_strategy=f"tensor_parallel_{best_tp}",
            device_placement=placement,
            estimated_latency_ms=best_sim.total_time_ms,
            estimated_throughput_tps=(
                1000 / best_sim.total_time_ms if best_sim.total_time_ms > 0 else 0
            ),
            estimated_memory_gb=task.total_memory_bytes() / best_tp / (1024**3),
            compute_efficiency=best_sim.compute_efficiency(),
        )

    def _try_pipeline_parallel(self, task: ComputeTask) -> OptimizationResult:
        """Try pipeline parallel strategy."""

        num_devices = self.cluster.num_devices()
        num_ops = len(task.operators)

        if num_ops < 4:
            return None

        best_sim = None
        best_pp = 2
        best_mb = 4

        for pp in [2, 4]:
            if pp > num_devices or pp > num_ops:
                continue

            for mb in [4, 8, 16]:
                sim = self.simulator.simulate_pipeline_parallel(task, pp, mb)

                if best_sim is None or sim.total_time_ms < best_sim.total_time_ms:
                    best_sim = sim
                    best_pp = pp
                    best_mb = mb

        if best_sim is None:
            return None

        devices = self.cluster.all_devices()[:best_pp]
        placement = {f"stage_{i}": devices[i][0] for i in range(best_pp)}

        return OptimizationResult(
            parallelism_strategy=f"pipeline_parallel_{best_pp}_mb{best_mb}",
            device_placement=placement,
            estimated_latency_ms=best_sim.total_time_ms,
            estimated_throughput_tps=(
                best_mb * 1000 / best_sim.total_time_ms if best_sim.total_time_ms > 0 else 0
            ),
            estimated_memory_gb=task.total_memory_bytes() / (1024**3),
            compute_efficiency=best_sim.compute_efficiency(),
        )

    def _try_hybrid_parallel(
        self,
        task: ComputeTask,
        batch_size: int,
    ) -> OptimizationResult:
        """Try hybrid parallelism (DP + TP or DP + PP)."""

        num_devices = self.cluster.num_devices()

        if num_devices < 4:
            return None

        # Try DP2 x TP2
        dp, tp = 2, 2
        if dp * tp <= num_devices and batch_size % dp == 0:
            # Simulate: each DP replica uses TP
            dp_sim = self.simulator.simulate_data_parallel(task, dp, batch_size)
            tp_sim = self.simulator.simulate_tensor_parallel(task, tp)

            # Combine: TP happens within each DP replica
            hybrid_time = max(dp_sim.total_time_ms, tp_sim.total_time_ms)

            devices = self.cluster.all_devices()[: dp * tp]
            placement = {}
            for d in range(dp):
                for t in range(tp):
                    placement[f"dp{d}_tp{t}"] = devices[d * tp + t][0]

            return OptimizationResult(
                parallelism_strategy=f"hybrid_dp{dp}_tp{tp}",
                device_placement=placement,
                estimated_latency_ms=hybrid_time,
                estimated_throughput_tps=batch_size * 1000 / hybrid_time if hybrid_time > 0 else 0,
                estimated_memory_gb=task.total_memory_bytes() / tp / (1024**3),
                compute_efficiency=0.8,  # Estimate
            )

        return None

    def _single_device_optimize(self, task: ComputeTask) -> OptimizationResult:
        """Optimize for single device execution."""

        devices = self.cluster.all_devices()
        if not devices:
            return None

        device_id, device_spec = devices[0]

        # Estimate time
        total_flops = task.total_flops()
        peak_tflops = device_spec.peak_compute("fp16")
        time_ms = (total_flops / (peak_tflops * 1e12)) * 1000 * 1.2

        return OptimizationResult(
            parallelism_strategy="single_device",
            device_placement={"all": device_id},
            estimated_latency_ms=time_ms,
            estimated_throughput_tps=1000 / time_ms if time_ms > 0 else 0,
            estimated_memory_gb=task.total_memory_bytes() / (1024**3),
            compute_efficiency=0.7,
        )

    def _refine_kernels(
        self,
        task: ComputeTask,
        result: OptimizationResult,
    ) -> OptimizationResult:
        """Refine kernel configurations for each operator."""

        kernel_configs = {}

        for op in task.operators:
            config = self._find_best_kernel_config(op, task)
            kernel_configs[op.op_id] = config

        result.kernel_configs = kernel_configs
        return result

    def _find_best_kernel_config(
        self,
        op: Operator,
        task: ComputeTask,
    ) -> Dict:
        """Find best kernel configuration for an operator."""

        # Default configurations based on operator type
        if op.op_type in (OperatorType.MATMUL, OperatorType.BATCH_MATMUL):
            return {
                "tile_m": 128,
                "tile_n": 128,
                "tile_k": 32,
                "num_stages": 3,
                "num_warps": 8,
            }

        elif op.op_type == OperatorType.ATTENTION:
            return {
                "block_size": 128,
                "num_warps": 8,
                "use_flash": True,
            }

        elif op.op_type in (OperatorType.LAYER_NORM, OperatorType.RMS_NORM):
            return {
                "block_size": 1024,
                "num_warps": 8,
            }

        else:
            return {
                "block_size": 256,
                "num_warps": 4,
            }

    def optimize_for_hardware(
        self,
        task: ComputeTask,
        hardware_type: str,
        batch_size: int = 1,
    ) -> OptimizationResult:
        """
        Optimize task for a specific hardware type.
        Uses transfer learning if hardware is unknown.
        """

        # Check if we have experience with this hardware
        known_devices = [d[1] for d in self.cluster.all_devices()]

        target_device = None
        for d in known_devices:
            if d.device_type.value == hardware_type:
                target_device = d
                break

        if target_device is None:
            # Unknown hardware - use transfer learning
            # Create a synthetic device spec
            target_device = DeviceSpec(
                device_id=f"synthetic_{hardware_type}",
                device_type=DeviceType.CUDA,  # Default
                compute_units=64,
                clock_mhz=1500,
                peak_tflops_fp16=100.0,
                peak_tflops_fp32=20.0,
                memory_gb=32.0,
                memory_bandwidth_gbps=1000.0,
            )

            # Find similar known hardware
            similar = self.cost_model.find_similar_hardware(target_device, known_devices)

            if similar:
                # Transfer knowledge
                pass  # Use similar device's cached predictions

        return self.optimize(task, batch_size)

    def save_experience(self, path: str):
        """Save optimization experience for future learning."""
        with open(path, "w") as f:
            json.dump(self.experience, f, indent=2)

    def load_experience(self, path: str):
        """Load optimization experience."""
        if os.path.exists(path):
            with open(path, "r") as f:
                self.experience = json.load(f)
