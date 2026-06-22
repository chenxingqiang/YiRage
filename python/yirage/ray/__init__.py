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
YiRage Distributed Search Module

Provides CPU-based distributed search using Ray for coordination.
This module enables scaling kernel search across multiple CPU cores
and cluster nodes without requiring GPU resources.

Architecture:
- Ray is used for distributed coordination (CPU-only)
- Each worker calls C++ search core independently
- Results are aggregated and best kernel selected
- Feedback data is collected for future RL training

Enhanced Features:
- Object store integration for large data transfer
- Placement groups for GPU affinity
- Fault tolerance with checkpoints
- Collective operations for distributed training
"""

from .coordinator import DistributedSearchCoordinator
from .worker import SearchWorker, create_workers
from .feedback import SearchFeedback, CandidateInfo, TrainingSample
from .partition import SearchPartition, create_partitions

# Unified Ray distributed module
from .ray_distributed import (
    # Core
    RayDistributedEngine,
    DistributedConfig,
    DistributedResult,
    GPUPlacementConfig,
    RetryConfig,
    RetryStrategy,
    PartitionStrategy,
    # Factory
    create_engine,
    is_ray_available as _is_ray_available,
    # RL Training
    RLTrainConfig,
    RLGradientAllReduce,
    run_distributed_training,
    create_train_loop,
)

# Collective operations
from .collectives import (
    CollectiveConfig,
    CollectiveOperations,
    DistributedSearchPattern,
    DistributedTrainingPattern,
    sum_reduce,
    mean_reduce,
    min_reduce,
    max_reduce,
    concat_reduce,
)

# Backwards compatibility aliases
RayDeepIntegration = RayDistributedEngine
DeepIntegrationConfig = DistributedConfig
DistributedSearchResult = DistributedResult
RayEngineConfig = DistributedConfig
DistributedRLConfig = RLTrainConfig
create_distributed_engine = create_engine
create_deep_integration = create_engine
run_distributed_rl_training = run_distributed_training
create_ray_train_loop = create_train_loop


class GradientReducer:
    """Stub for backwards compatibility."""

    def __init__(self, world_size: int, backend: str = "gloo"):
        self.world_size = world_size

    def all_reduce_dict(self, tensors, op="mean"):
        return tensors


# Benchmarking utilities
from .benchmark import (
    BenchmarkResult,
    benchmark_object_store,
    benchmark_worker_scaling,
    benchmark_all_reduce,
    run_all_benchmarks,
    save_benchmark_results,
)

# YPK (Persistent Kernel) integration
from .ypk_integration import (
    YPKBackend,
    YPKMode,
    YPKConfig,
    KernelSearchSpace,
    YPKOptimizationResult,
    YPKRayOptimizer,
    optimize_ypk_kernel,
    BACKEND_CAPABILITIES,
    get_backend_capabilities,
    is_mode_supported,
    get_default_mode,
    get_compile_flags,
)

__all__ = [
    # Core classes
    "DistributedSearchCoordinator",
    "SearchWorker",
    # Unified Ray distributed
    "RayDistributedEngine",
    "DistributedConfig",
    "DistributedResult",
    "GPUPlacementConfig",
    "RetryConfig",
    "RetryStrategy",
    "PartitionStrategy",
    "create_engine",
    # Backwards compatibility
    "RayDeepIntegration",
    "DeepIntegrationConfig",
    "RayEngineConfig",
    "DistributedSearchResult",
    "create_distributed_engine",
    "create_deep_integration",
    # Data structures
    "SearchFeedback",
    "CandidateInfo",
    "TrainingSample",
    "SearchPartition",
    # Collective operations
    "CollectiveConfig",
    "CollectiveOperations",
    "DistributedSearchPattern",
    "DistributedTrainingPattern",
    "sum_reduce",
    "mean_reduce",
    "min_reduce",
    "max_reduce",
    "concat_reduce",
    # RL Training
    "RLTrainConfig",
    "DistributedRLConfig",
    "GradientReducer",
    "RLGradientAllReduce",
    "create_train_loop",
    "create_ray_train_loop",
    "run_distributed_training",
    "run_distributed_rl_training",
    # Benchmarking
    "BenchmarkResult",
    "benchmark_object_store",
    "benchmark_worker_scaling",
    "benchmark_all_reduce",
    "run_all_benchmarks",
    "save_benchmark_results",
    # YPK Integration
    "YPKBackend",
    "YPKMode",
    "YPKConfig",
    "KernelSearchSpace",
    "YPKOptimizationResult",
    "YPKRayOptimizer",
    "optimize_ypk_kernel",
    "BACKEND_CAPABILITIES",
    "get_backend_capabilities",
    "is_mode_supported",
    "get_default_mode",
    "get_compile_flags",
    # Utilities
    "create_workers",
    "create_partitions",
    "is_ray_available",
]


def is_ray_available() -> bool:
    """Check if Ray is available for distributed search."""
    return _is_ray_available()
