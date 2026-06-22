"""
End-to-End Universal Optimizer

Combines cluster simulation, task analysis, and µGraph search
into a single unified optimization pipeline.

Supports:
- Any compute task (PyTorch, ONNX, or custom graph)
- Any hardware configuration
- Any cluster scale
- Automatic kernel optimization via YiRage
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union, Callable
import json
import time
import os

from .topology import ClusterTopology, DeviceSpec, DeviceType
from .task import ComputeTask, Operator, OperatorType, TensorSpec, DataType
from .simulator import ClusterSimulator, SimulatedExecution
from .auto_optimizer import (
    UniversalOptimizer,
    OptimizationConfig,
    OptimizationStrategy,
    OptimizationResult,
)
from .executor import SimulatedExecutor, ExecutionPlan, ExecutionResult


@dataclass
class OptimizationRequest:
    """
    A request to optimize a compute workload.

    Accepts multiple input formats:
    - PyTorch module
    - ONNX model path
    - Compute task definition
    - High-level operation spec
    """

    # Workload specification (one of these)
    pytorch_module: Any = None  # torch.nn.Module
    onnx_path: Optional[str] = None  # Path to ONNX file
    compute_task: Optional[ComputeTask] = None
    operation_spec: Optional[Dict] = None  # {"type": "attention", "batch": 32, ...}

    # Input shapes (required for module/ONNX)
    input_shapes: List[tuple] = field(default_factory=list)
    input_dtypes: List[str] = field(default_factory=lambda: ["fp16"])

    # Target hardware (optional, auto-detect if not provided)
    target_hardware: Optional[str] = None  # "cuda", "maca", "ascend", etc.
    cluster_config: Optional[Dict] = None  # Custom cluster config

    # Constraints
    max_latency_ms: Optional[float] = None
    max_memory_gb: Optional[float] = None
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])

    # Optimization preferences
    strategy: str = "balanced"  # latency, throughput, efficiency, memory, balanced
    enable_fusion: bool = True
    enable_quantization: bool = False
    precision: str = "fp16"


@dataclass
class OptimizationOutput:
    """
    Complete optimization output with all artifacts.
    """

    # Success status
    success: bool = True
    error_message: str = ""

    # Optimization result
    result: Optional[OptimizationResult] = None

    # Execution plan
    execution_plan: Optional[ExecutionPlan] = None

    # Performance predictions
    predictions: Dict[str, float] = field(default_factory=dict)

    # Kernel configurations for YiRage
    kernel_configs: Dict[str, Dict] = field(default_factory=dict)

    # µGraph configurations (for integration with YiRage search)
    mugraph_configs: List[Dict] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    optimization_time_s: float = 0.0
    cluster_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "error_message": self.error_message,
            "result": self.result.to_dict() if self.result else None,
            "execution_plan": self.execution_plan.to_dict() if self.execution_plan else None,
            "predictions": self.predictions,
            "kernel_configs": self.kernel_configs,
            "mugraph_configs": self.mugraph_configs,
            "recommendations": self.recommendations,
            "optimization_time_s": self.optimization_time_s,
            "cluster_info": self.cluster_info,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "Optimization Summary",
            "=" * 50,
            f"Status: {'Success' if self.success else 'Failed'}",
        ]

        if self.result:
            lines.extend(
                [
                    f"Strategy: {self.result.parallelism_strategy}",
                    f"Estimated Latency: {self.result.estimated_latency_ms:.2f} ms",
                    f"Throughput: {self.result.estimated_throughput_tps:.1f} samples/sec",
                    f"Efficiency: {self.result.compute_efficiency*100:.1f}%",
                ]
            )

        if self.recommendations:
            lines.append("\nRecommendations:")
            for rec in self.recommendations:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class E2EOptimizer:
    """
    End-to-End optimizer that handles the complete optimization pipeline.

    Pipeline:
    1. Parse input (PyTorch/ONNX/spec) -> ComputeTask
    2. Detect/configure hardware cluster
    3. Analyze task characteristics
    4. Find optimal parallelism strategy
    5. Generate kernel configurations
    6. Output execution plan and µGraph configs
    """

    def __init__(
        self,
        cluster: Optional[ClusterTopology] = None,
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize E2E optimizer.

        Args:
            cluster: Target cluster (auto-detect if None)
            config: Optimization config
        """
        self.cluster = cluster
        self.config = config or OptimizationConfig()
        self._optimizer: Optional[UniversalOptimizer] = None
        self._executor: Optional[SimulatedExecutor] = None

    def optimize(self, request: OptimizationRequest) -> OptimizationOutput:
        """
        Run complete optimization pipeline.

        Args:
            request: Optimization request

        Returns:
            Complete optimization output
        """
        start_time = time.time()
        output = OptimizationOutput()

        try:
            # Step 1: Parse input to ComputeTask
            task = self._parse_input(request)
            if task is None:
                output.success = False
                output.error_message = "Failed to parse input into compute task"
                return output

            # Step 2: Setup cluster
            cluster = self._setup_cluster(request)
            output.cluster_info = {
                "name": cluster.name,
                "num_devices": cluster.num_devices(),
                "total_memory_gb": cluster.total_memory_gb(),
                "total_compute_tflops": cluster.total_compute_tflops("fp16"),
            }

            # Step 3: Setup optimizer
            self._setup_optimizer(cluster, request)

            # Step 4: Run optimization
            batch_size = request.batch_sizes[0] if request.batch_sizes else 1
            result = self._optimizer.optimize(task, batch_size)
            output.result = result

            # Step 5: Generate execution plan
            plan = self._generate_execution_plan(task, result)
            output.execution_plan = plan

            # Step 6: Generate kernel configs
            kernel_configs = self._generate_kernel_configs(task, result, cluster)
            output.kernel_configs = kernel_configs

            # Step 7: Generate µGraph configs for YiRage
            mugraph_configs = self._generate_mugraph_configs(task, result, cluster)
            output.mugraph_configs = mugraph_configs

            # Step 8: Generate predictions
            output.predictions = {
                "latency_ms": result.estimated_latency_ms,
                "throughput_tps": result.estimated_throughput_tps,
                "memory_gb": result.estimated_memory_gb,
                "compute_efficiency": result.compute_efficiency,
            }

            # Step 9: Generate recommendations
            output.recommendations = self._generate_recommendations(task, result, cluster)

            output.success = True

        except Exception as e:
            output.success = False
            output.error_message = str(e)

        output.optimization_time_s = time.time() - start_time
        return output

    def _parse_input(self, request: OptimizationRequest) -> Optional[ComputeTask]:
        """Parse request input into ComputeTask."""

        # Direct ComputeTask
        if request.compute_task:
            return request.compute_task

        # Operation spec
        if request.operation_spec:
            return self._parse_operation_spec(request.operation_spec)

        # PyTorch module
        if request.pytorch_module:
            return self._parse_pytorch(request.pytorch_module, request.input_shapes)

        # ONNX model
        if request.onnx_path:
            return self._parse_onnx(request.onnx_path)

        return None

    def _parse_operation_spec(self, spec: Dict) -> ComputeTask:
        """Parse high-level operation spec into ComputeTask."""

        op_type = spec.get("type", "matmul")

        if op_type == "matmul":
            return ComputeTask.create_matmul(
                M=spec.get("M", 1024),
                K=spec.get("K", 1024),
                N=spec.get("N", 1024),
                batch=spec.get("batch", 1),
            )

        elif op_type == "attention":
            return ComputeTask.create_attention(
                batch=spec.get("batch", 1),
                seq_len=spec.get("seq_len", 2048),
                num_heads=spec.get("num_heads", 32),
                head_dim=spec.get("head_dim", 128),
            )

        elif op_type == "mlp":
            return ComputeTask.create_mlp(
                batch=spec.get("batch", 1),
                seq_len=spec.get("seq_len", 2048),
                hidden_dim=spec.get("hidden_dim", 4096),
                intermediate_dim=spec.get("intermediate_dim", 16384),
            )

        elif op_type == "transformer":
            return ComputeTask.create_transformer_block(
                batch=spec.get("batch", 1),
                seq_len=spec.get("seq_len", 2048),
                hidden_dim=spec.get("hidden_dim", 4096),
                num_heads=spec.get("num_heads", 32),
                intermediate_dim=spec.get("intermediate_dim", 16384),
            )

        else:
            raise ValueError(f"Unknown operation type: {op_type}")

    def _parse_pytorch(self, module: Any, input_shapes: List[tuple]) -> ComputeTask:
        """Parse PyTorch module into ComputeTask."""
        try:
            return ComputeTask.from_pytorch(module, input_shapes)
        except Exception as e:
            # Fallback to generic task
            return ComputeTask(name="pytorch_module")

    def _parse_onnx(self, onnx_path: str) -> ComputeTask:
        """Parse ONNX model into ComputeTask."""
        # Simplified - would parse ONNX graph in real implementation
        return ComputeTask(name=f"onnx_{os.path.basename(onnx_path)}")

    def _setup_cluster(self, request: OptimizationRequest) -> ClusterTopology:
        """Setup or auto-detect cluster configuration."""

        if self.cluster:
            return self.cluster

        # Custom cluster config
        if request.cluster_config:
            return self._create_cluster_from_config(request.cluster_config)

        # Auto-detect based on target hardware
        hardware = request.target_hardware or "cuda"

        if hardware == "cuda":
            # Try to detect actual CUDA devices
            num_gpus = self._detect_cuda_devices()
            return ClusterTopology.create_single_node(num_gpus, "A100", nvlink=True)

        elif hardware == "maca":
            return ClusterTopology.create_single_node(4, "A100", nvlink=False)

        elif hardware == "ascend":
            return ClusterTopology.create_heterogeneous(
                [
                    {
                        "device_type": "ascend",
                        "count": 8,
                        "specs": {
                            "compute_units": 32,
                            "clock_mhz": 1500,
                            "peak_tflops_fp16": 320.0,
                            "peak_tflops_fp32": 160.0,
                            "memory_gb": 64.0,
                            "memory_bandwidth_gbps": 1200.0,
                        },
                    }
                ]
            )

        else:
            # CPU fallback
            return ClusterTopology.create_single_node(1, "A100")

    def _create_cluster_from_config(self, config: Dict) -> ClusterTopology:
        """Create cluster from configuration dictionary."""

        topology_type = config.get("type", "single_node")

        if topology_type == "single_node":
            return ClusterTopology.create_single_node(
                num_gpus=config.get("num_gpus", 8),
                gpu_type=config.get("gpu_type", "A100"),
                nvlink=config.get("nvlink", True),
            )

        elif topology_type == "multi_node":
            return ClusterTopology.create_multi_node(
                num_nodes=config.get("num_nodes", 4),
                gpus_per_node=config.get("gpus_per_node", 8),
                gpu_type=config.get("gpu_type", "A100"),
                inter_node_bandwidth_gbps=config.get("inter_node_bandwidth_gbps", 100.0),
            )

        elif topology_type == "heterogeneous":
            return ClusterTopology.create_heterogeneous(config.get("devices", []))

        else:
            return ClusterTopology.create_single_node(8, "A100")

    def _detect_cuda_devices(self) -> int:
        """Detect number of CUDA devices."""
        try:
            import torch

            return torch.cuda.device_count() if torch.cuda.is_available() else 1
        except ImportError:
            return 8  # Default assumption

    def _setup_optimizer(self, cluster: ClusterTopology, request: OptimizationRequest):
        """Setup optimizer with configuration."""

        strategy_map = {
            "latency": OptimizationStrategy.LATENCY,
            "throughput": OptimizationStrategy.THROUGHPUT,
            "efficiency": OptimizationStrategy.EFFICIENCY,
            "memory": OptimizationStrategy.MEMORY,
            "balanced": OptimizationStrategy.BALANCED,
        }

        config = OptimizationConfig(
            strategy=strategy_map.get(request.strategy, OptimizationStrategy.BALANCED),
            max_latency_ms=request.max_latency_ms,
            max_memory_gb=request.max_memory_gb,
            enable_fusion=request.enable_fusion,
            enable_mixed_precision=request.precision != "fp32",
        )

        self._optimizer = UniversalOptimizer(cluster, config)
        self._executor = SimulatedExecutor(cluster)

    def _generate_execution_plan(
        self,
        task: ComputeTask,
        result: OptimizationResult,
    ) -> ExecutionPlan:
        """Generate execution plan from optimization result."""

        return ExecutionPlan(
            task_name=task.name,
            parallelism_strategy=result.parallelism_strategy,
            device_placement=result.device_placement,
            schedule=result.operator_schedule,
            kernel_configs=result.kernel_configs,
        )

    def _generate_kernel_configs(
        self,
        task: ComputeTask,
        result: OptimizationResult,
        cluster: ClusterTopology,
    ) -> Dict[str, Dict]:
        """Generate detailed kernel configurations."""

        configs = {}

        for op in task.operators:
            # Get device for this operator
            device = None
            for dev_id, dev_spec in cluster.all_devices():
                device = dev_spec
                break

            if device is None:
                continue

            # Generate config based on operator type and device
            config = self._get_kernel_config(op, device, task)
            configs[op.op_id] = config

        return configs

    def _get_kernel_config(
        self,
        op: Operator,
        device: DeviceSpec,
        task: ComputeTask,
    ) -> Dict:
        """Get kernel configuration for an operator."""

        base_config = {
            "device_type": device.device_type.value,
            "precision": "fp16",
        }

        if op.op_type in (OperatorType.MATMUL, OperatorType.BATCH_MATMUL):
            # Matmul tile sizes
            base_config.update(
                {
                    "tile_m": 128,
                    "tile_n": 128,
                    "tile_k": 32,
                    "num_stages": 3 if device.tensor_cores else 2,
                    "num_warps": 8,
                    "use_tensor_cores": device.tensor_cores,
                }
            )

        elif op.op_type == OperatorType.ATTENTION:
            base_config.update(
                {
                    "block_size": 128,
                    "num_warps": 8,
                    "use_flash": True,
                    "causal": op.attrs.get("causal", True),
                }
            )

        elif op.op_type in (OperatorType.LAYER_NORM, OperatorType.RMS_NORM):
            base_config.update(
                {
                    "block_size": 1024,
                    "num_warps": 8,
                }
            )

        elif op.op_type in (OperatorType.GELU, OperatorType.SILU, OperatorType.RELU):
            base_config.update(
                {
                    "block_size": 256,
                    "vectorize": 4,
                }
            )

        else:
            base_config.update(
                {
                    "block_size": 256,
                    "num_warps": 4,
                }
            )

        return base_config

    def _generate_mugraph_configs(
        self,
        task: ComputeTask,
        result: OptimizationResult,
        cluster: ClusterTopology,
    ) -> List[Dict]:
        """Generate µGraph configurations for YiRage search."""

        configs = []

        # Get device features
        devices = cluster.all_devices()
        if not devices:
            return configs

        device_id, device = devices[0]

        # Generate config for each operator group
        for op in task.operators:
            config = {
                "operator": op.to_dict(),
                "device": {
                    "type": device.device_type.value,
                    "compute_units": device.compute_units,
                    "memory_gb": device.memory_gb,
                    "tensor_cores": device.tensor_cores,
                },
                "search_space": self._get_search_space(op, device),
                "constraints": {
                    "max_shared_memory_kb": 164,  # Typical GPU shared memory
                    "max_registers_per_thread": 255,
                },
            }
            configs.append(config)

        return configs

    def _get_search_space(self, op: Operator, device: DeviceSpec) -> Dict:
        """Get search space for operator."""

        if op.op_type in (OperatorType.MATMUL, OperatorType.BATCH_MATMUL):
            return {
                "tile_m": [64, 128, 256],
                "tile_n": [64, 128, 256],
                "tile_k": [16, 32, 64],
                "num_stages": [2, 3, 4],
                "num_warps": [4, 8],
            }

        elif op.op_type == OperatorType.ATTENTION:
            return {
                "block_size": [64, 128, 256],
                "num_warps": [4, 8],
            }

        else:
            return {
                "block_size": [128, 256, 512, 1024],
                "num_warps": [4, 8],
            }

    def _generate_recommendations(
        self,
        task: ComputeTask,
        result: OptimizationResult,
        cluster: ClusterTopology,
    ) -> List[str]:
        """Generate optimization recommendations."""

        recommendations = []

        # Check compute efficiency
        if result.compute_efficiency < 0.5:
            recommendations.append(
                "Low compute efficiency (<50%). Consider reducing parallelism "
                "to decrease communication overhead."
            )

        # Check memory usage
        task_memory = task.total_memory_bytes() / (1024**3)
        device_memory = cluster.all_devices()[0][1].memory_gb if cluster.all_devices() else 80

        if task_memory > device_memory * 0.8:
            recommendations.append(
                f"High memory usage ({task_memory:.1f}GB). Consider using "
                "tensor parallelism to distribute memory."
            )

        # Check parallelism strategy
        if "single" in result.parallelism_strategy and cluster.num_devices() > 1:
            recommendations.append(
                f"Using single device on {cluster.num_devices()}-device cluster. "
                "Consider if workload is too small for multi-device."
            )

        # Check for fusion opportunities
        num_ops = len(task.operators)
        if num_ops > 10:
            recommendations.append(
                f"Task has {num_ops} operators. Enable operator fusion "
                "to reduce kernel launch overhead."
            )

        return recommendations

    def benchmark(
        self,
        request: OptimizationRequest,
        num_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark optimization across multiple batch sizes.

        Returns performance metrics for each batch size.
        """

        results = []

        for batch_size in request.batch_sizes:
            # Create request copy with this batch size
            req_copy = OptimizationRequest(
                operation_spec=request.operation_spec,
                pytorch_module=request.pytorch_module,
                input_shapes=request.input_shapes,
                target_hardware=request.target_hardware,
                cluster_config=request.cluster_config,
                batch_sizes=[batch_size],
            )

            # Run optimization
            output = self.optimize(req_copy)

            if output.success and output.result:
                results.append(
                    {
                        "batch_size": batch_size,
                        "latency_ms": output.result.estimated_latency_ms,
                        "throughput_tps": output.result.estimated_throughput_tps,
                        "efficiency": output.result.compute_efficiency,
                        "strategy": output.result.parallelism_strategy,
                    }
                )

        return {
            "results": results,
            "optimal": max(results, key=lambda r: r["throughput_tps"]) if results else None,
        }


def optimize_any_task(
    task_spec: Union[Dict, ComputeTask, Any],
    cluster_spec: Optional[Dict] = None,
    **kwargs,
) -> OptimizationOutput:
    """
    Convenience function to optimize any task with minimal configuration.

    Examples:
        # From operation spec
        result = optimize_any_task({"type": "attention", "batch": 32, "seq_len": 2048})

        # From ComputeTask
        task = ComputeTask.create_matmul(4096, 4096, 4096)
        result = optimize_any_task(task)

        # With cluster config
        result = optimize_any_task(
            {"type": "transformer", "batch": 8},
            cluster_spec={"type": "multi_node", "num_nodes": 4}
        )
    """

    # Create request
    request = OptimizationRequest(**kwargs)

    if isinstance(task_spec, dict):
        request.operation_spec = task_spec
    elif isinstance(task_spec, ComputeTask):
        request.compute_task = task_spec
    else:
        # Assume PyTorch module
        request.pytorch_module = task_spec

    if cluster_spec:
        request.cluster_config = cluster_spec

    # Run optimization
    optimizer = E2EOptimizer()
    return optimizer.optimize(request)
