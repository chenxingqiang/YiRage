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
YiRage RL-Guided Kernel Search Module

Complete closed-loop integration:

    ┌─────────────────────────────────────────────────────────────┐
    │                     Closed Loop                              │
    │                                                              │
    │  RL Policy ──action──> YiRage Env ──verify(GPU)──> reward   │
    │      │                      │                        │       │
    │      └──────── obs <────────┴───── feedback <────────┘       │
    │                                                              │
    │                    policy update                             │
    └─────────────────────────────────────────────────────────────┘

Key Components:
- YiRageSearchEnv: Gymnasium environment bridging RL and YiRage
- GPUVerifier: Ray Actor for GPU-based fingerprint verification
- VerifierPool: Manages GPU resources for async verification
- RewardComputer: Multi-objective reward computation
"""

from .env import (
    YiRageSearchEnv,
    EnvConfig,
    ActionSpace,
    ObservationSpace,
)

from .verifier import (
    GPUVerifier,
    VerifierPool,
    VerifyResult,
    ProfileResult,
    AccelForgeVerifier,
)

from .training import (
    train_rl_search,
    TrainingConfig,
    YiRageCallbacks,
    GRPOConfig,
    GRPOTrainer,
)

# Hardware detection and config coupling
from .hardware import (
    HardwareProfile,
    WorkloadSpec,
    PerformanceEstimate,
    HardwareRegistry,
    detect_hardware,
    get_hardware_features,
    ConfigGenerator,
    HardwareSearchCoupling,
    get_optimal_config,
    AccelForgeBridge,
    AccelForgeDesignPoint,
    AccelForgeMetrics,
    AccelForgeDetector,
    get_accelforge_availability,
    is_accelforge_available,
    # Surrogate Model (Problem 5)
    SurrogateModel,
    CalibrationPoint,
)

# Feature Extraction and Processing
from .features import (
    MuGraphFeature,
    OperatorFeature,
    TensorFeature,
    FeatureProcessor,
    FeatureNormalizer,
    GraphFeatureExtractor,
    # Dynamic Features (Problem 4)
    DynamicFeatureDict,
    DynamicFeatureProcessor,
    FeatureSpec,
    FEATURE_REGISTRY,
)

# Neural Network Models
from .models import (
    GraphEncoder,
    SimpleGraphEncoder,
)

# ActionMaskingModel requires RLlib
try:
    from .models import ActionMaskingModel
except ImportError:
    ActionMaskingModel = None

# Hierarchical Search (Level 0: Accelerator, Level 1: Config, Level 2: Graph)
from .search import (
    # Level 0: Accelerator Design Space (AccelForge)
    AcceleratorEnv,
    AcceleratorActionSpace,
    AcceleratorObservationSpace,
    AcceleratorDesignConstraints,
    ParetoFrontTracker,
    ParetoPoint,
    # Bottom-up Feedback (Problem 1)
    KernelCharacteristics,
    # Level 1: Config Space
    HardwareConfig,
    SearchSpaceConstraints,
    ConfigActionSpace,
    ConfigObservationSpace,
    # Level 2: Graph Space
    ConstrainedGraphActionSpace,
    GraphObservationSpace,
    GraphAction,
    # Hierarchical Environments
    HierarchicalSearchEnv,
    ConfigEnv,
    ConstrainedGraphEnv,
    # Training
    HierarchicalTrainer,
    HierarchicalSearchCoordinator,
    # Batch Search (Problem 3)
    BatchSearchAPI,
    BatchSearchConfig,
    KernelSearchResult,
    # Cross-Backend Migration (Problem 6c)
    KernelMigrationEngine,
    MigrationResult,
    BackendCapability,
    BACKEND_CAPABILITIES,
    # Persistent Kernel (Problem 6d)
    PersistentKernelConfig,
    PersistentKernelSearchSpace,
)

# Cluster Simulation and Universal Optimization
from .cluster import (
    # Topology
    ClusterTopology,
    ComputeNode,
    DeviceSpec,
    DeviceType,
    NetworkLink,
    TopologyType,
    # Task Representation
    ComputeTask,
    TensorSpec,
    TaskGraph,
    SubTask,
    Operator,
    OperatorType,
    DataType,
    # Simulator
    ClusterSimulator,
    SimulatedExecution,
    CommunicationModel,
    # Auto Optimizer
    UniversalOptimizer,
    OptimizationConfig,
    OptimizationStrategy,
    OptimizationResult,
    # E2E Optimizer
    E2EOptimizer,
    OptimizationRequest,
    OptimizationOutput,
    optimize_any_task,
    # Executor
    SimulatedExecutor,
    ExecutionPlan,
    ExecutionResult,
    # Ray Optimizer
    RayClusterOptimizer,
    RayClusterConfig,
    DistributedSearchResult,
    create_ray_optimizer,
    is_ray_available,
)

__all__ = [
    # Hardware Detection
    "HardwareProfile",
    "WorkloadSpec",
    "PerformanceEstimate",
    "HardwareRegistry",
    "detect_hardware",
    "get_hardware_features",
    "ConfigGenerator",
    "HardwareSearchCoupling",
    "get_optimal_config",
    # AccelForge Integration
    "AccelForgeBridge",
    "AccelForgeDesignPoint",
    "AccelForgeMetrics",
    "AccelForgeDetector",
    "get_accelforge_availability",
    "is_accelforge_available",
    # Surrogate Model (Problem 5)
    "SurrogateModel",
    "CalibrationPoint",
    # Feature Extraction
    "MuGraphFeature",
    "OperatorFeature",
    "TensorFeature",
    "FeatureProcessor",
    "FeatureNormalizer",
    "GraphFeatureExtractor",
    # Dynamic Features (Problem 4)
    "DynamicFeatureDict",
    "DynamicFeatureProcessor",
    "FeatureSpec",
    "FEATURE_REGISTRY",
    # Neural Network Models
    "GraphEncoder",
    "SimpleGraphEncoder",
    "ActionMaskingModel",
    # Environment (Flat)
    "YiRageSearchEnv",
    "EnvConfig",
    "ActionSpace",
    "ObservationSpace",
    # Verifier
    "GPUVerifier",
    "VerifierPool",
    "VerifyResult",
    "ProfileResult",
    "AccelForgeVerifier",
    # Training (Flat)
    "train_rl_search",
    "TrainingConfig",
    "YiRageCallbacks",
    # GRPO Training
    "GRPOConfig",
    "GRPOTrainer",
    # Hierarchical Search - Level 0 (Accelerator Design)
    "AcceleratorEnv",
    "AcceleratorActionSpace",
    "AcceleratorObservationSpace",
    "AcceleratorDesignConstraints",
    "ParetoFrontTracker",
    "ParetoPoint",
    # Bottom-up Feedback (Problem 1)
    "KernelCharacteristics",
    # Hierarchical Search - Level 1 (Config)
    "HardwareConfig",
    "SearchSpaceConstraints",
    "ConfigActionSpace",
    "ConfigObservationSpace",
    # Hierarchical Search - Level 2 (Graph)
    "ConstrainedGraphActionSpace",
    "GraphObservationSpace",
    "GraphAction",
    # Hierarchical Environments
    "HierarchicalSearchEnv",
    "ConfigEnv",
    "ConstrainedGraphEnv",
    # Hierarchical Training
    "HierarchicalTrainer",
    "HierarchicalSearchCoordinator",
    # Batch Search (Problem 3)
    "BatchSearchAPI",
    "BatchSearchConfig",
    "KernelSearchResult",
    # Cross-Backend Migration (Problem 6c)
    "KernelMigrationEngine",
    "MigrationResult",
    "BackendCapability",
    "BACKEND_CAPABILITIES",
    # Persistent Kernel (Problem 6d)
    "PersistentKernelConfig",
    "PersistentKernelSearchSpace",
    # Cluster Simulation
    "ClusterTopology",
    "ComputeNode",
    "DeviceSpec",
    "DeviceType",
    "NetworkLink",
    "TopologyType",
    # Task Representation
    "ComputeTask",
    "TensorSpec",
    "TaskGraph",
    "SubTask",
    "Operator",
    "OperatorType",
    "DataType",
    # Simulator
    "ClusterSimulator",
    "SimulatedExecution",
    "CommunicationModel",
    # Auto Optimizer
    "UniversalOptimizer",
    "OptimizationConfig",
    "OptimizationStrategy",
    "OptimizationResult",
    # E2E Optimizer
    "E2EOptimizer",
    "OptimizationRequest",
    "OptimizationOutput",
    "optimize_any_task",
    # Executor
    "SimulatedExecutor",
    "ExecutionPlan",
    "ExecutionResult",
    # Ray Optimizer
    "RayClusterOptimizer",
    "RayClusterConfig",
    "DistributedSearchResult",
    "create_ray_optimizer",
    "is_ray_available",
]


def is_available() -> bool:
    """Check if RL training is available (requires ray and rllib)."""
    try:
        import ray
        from ray import rllib

        return True
    except ImportError:
        return False
