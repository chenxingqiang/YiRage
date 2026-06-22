"""
Cluster Simulation Module

Simulates multi-device/multi-node cluster execution on a single GPU
by modeling communication costs and execution patterns.

Key Features:
- Universal task representation for any compute workload
- Accurate communication modeling (NVLink, PCIe, InfiniBand)
- Automatic parallelism strategy selection
- Simulation-based optimization before real execution
"""

from .topology import (
    ClusterTopology,
    ComputeNode,
    DeviceSpec,
    DeviceType,
    NetworkLink,
    TopologyType,
)

from .task import (
    ComputeTask,
    TensorSpec,
    TaskGraph,
    SubTask,
    DataDependency,
    Operator,
    OperatorType,
    DataType,
    TorchOp,
)

from .simulator import (
    ClusterSimulator,
    SimulatedExecution,
    CommunicationModel,
    NVLinkModel,
    PCIeModel,
    InfiniBandModel,
    EthernetModel,
    CommunicationType,
)

from .placer import (
    DevicePlacer,
    PlacementStrategy,
    GreedyPlacer,
    DPPlacer,
    LearnedPlacer,
)

from .executor import (
    SimulatedExecutor,
    ExecutionPlan,
    ExecutionResult,
    ExecutionMode,
    CommunicationEvent,
    ComputeEvent,
)

from .auto_optimizer import (
    UniversalOptimizer,
    OptimizationConfig,
    OptimizationStrategy,
    OptimizationResult,
    TaskDecomposer,
    OperatorFuser,
    CostModel,
)

from .e2e_optimizer import (
    E2EOptimizer,
    OptimizationRequest,
    OptimizationOutput,
    optimize_any_task,
)

from .ray_optimizer import (
    RayClusterOptimizer,
    RayClusterConfig,
    DistributedSearchResult,
    create_ray_optimizer,
    is_ray_available,
)

from .device_registry import (
    DeviceRegistry,
    DEVICE_SPECS,
    get_device_spec,
    register_custom_device,
    list_supported_devices,
)

from .kernel_coverage import (
    KernelOpType,
    SupportLevel,
    KernelSupport,
    KernelCoverageAnalyzer,
    KERNEL_COVERAGE,
    check_kernel_support,
    get_best_device_for_ops,
)

from .kernel_generator import (
    KernelDataType,
    TensorSpec,
    KernelSpec,
    GeneratedKernel,
    KernelGenerator,
    generate_kernel,
    generate_kernels_for_all_targets,
)

from .kernel_templates import (
    CUDA_TEMPLATES,
    ROCM_TEMPLATES,
    TRITON_TEMPLATES,
    ASCEND_TEMPLATES,
    TPU_TEMPLATES,
    CPU_TEMPLATES,
    ALL_TEMPLATES,
    get_template,
    list_available_templates,
)

__all__ = [
    # Topology
    "ClusterTopology",
    "ComputeNode",
    "DeviceSpec",
    "DeviceType",
    "NetworkLink",
    "TopologyType",
    # Task
    "ComputeTask",
    "TensorSpec",
    "TaskGraph",
    "SubTask",
    "DataDependency",
    "Operator",
    "OperatorType",
    "DataType",
    "TorchOp",
    # Simulator
    "ClusterSimulator",
    "SimulatedExecution",
    "CommunicationModel",
    "CommunicationType",
    "NVLinkModel",
    "PCIeModel",
    "InfiniBandModel",
    "EthernetModel",
    # Placer
    "DevicePlacer",
    "PlacementStrategy",
    "GreedyPlacer",
    "DPPlacer",
    "LearnedPlacer",
    # Executor
    "SimulatedExecutor",
    "ExecutionPlan",
    "ExecutionResult",
    "ExecutionMode",
    "CommunicationEvent",
    "ComputeEvent",
    # Auto Optimizer
    "UniversalOptimizer",
    "OptimizationConfig",
    "OptimizationStrategy",
    "OptimizationResult",
    "TaskDecomposer",
    "OperatorFuser",
    "CostModel",
    # E2E Optimizer
    "E2EOptimizer",
    "OptimizationRequest",
    "OptimizationOutput",
    "optimize_any_task",
    # Ray Optimizer
    "RayClusterOptimizer",
    "RayClusterConfig",
    "DistributedSearchResult",
    "create_ray_optimizer",
    "is_ray_available",
    # Device Registry
    "DeviceRegistry",
    "DEVICE_SPECS",
    "get_device_spec",
    "register_custom_device",
    "list_supported_devices",
    # Kernel Coverage
    "KernelOpType",
    "SupportLevel",
    "KernelSupport",
    "KernelCoverageAnalyzer",
    "KERNEL_COVERAGE",
    "check_kernel_support",
    "get_best_device_for_ops",
    # Kernel Generator
    "KernelDataType",
    "TensorSpec",
    "KernelSpec",
    "GeneratedKernel",
    "KernelGenerator",
    "generate_kernel",
    "generate_kernels_for_all_targets",
    # Kernel Templates
    "CUDA_TEMPLATES",
    "ROCM_TEMPLATES",
    "TRITON_TEMPLATES",
    "ASCEND_TEMPLATES",
    "TPU_TEMPLATES",
    "CPU_TEMPLATES",
    "ALL_TEMPLATES",
    "get_template",
    "list_available_templates",
]
