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
Hierarchical Search Module

Three-level search architecture:
- Level 0: Accelerator Design Search (AccelForge co-design)
- Level 1: Hardware Config Search (controls search space)
- Level 2: µGraph Search (constrained by config)

Enhanced features:
- KernelCharacteristics: Bottom-up feedback from Level 2 to Level 0
- BatchSearchAPI: High-throughput batch search
- KernelMigrationEngine: Cross-backend kernel migration via Z3
- PersistentKernelSearchSpace: Persistent kernel RL integration
"""

from .config_space import (
    HardwareConfig,
    SearchSpaceConstraints,
    ConfigActionSpace,
    ConfigObservationSpace,
)

from .graph_space import (
    ConstrainedGraphActionSpace,
    GraphObservationSpace,
    GraphAction,
)

from .hierarchical_env import (
    HierarchicalSearchEnv,
    ConfigEnv,
    ConstrainedGraphEnv,
)

from .hierarchical_trainer import (
    HierarchicalTrainer,
    HierarchicalSearchCoordinator,
)

from .accelerator_space import (
    AcceleratorEnv,
    AcceleratorActionSpace,
    AcceleratorObservationSpace,
    AcceleratorDesignConstraints,
    ParetoFrontTracker,
    ParetoPoint,
    KernelCharacteristics,
)

from .batch_search import (
    BatchSearchAPI,
    BatchSearchConfig,
    KernelSearchResult,
)

from .cross_backend import (
    KernelMigrationEngine,
    MigrationResult,
    BackendCapability,
    BACKEND_CAPABILITIES,
    PersistentKernelConfig,
    PersistentKernelSearchSpace,
)

__all__ = [
    # Config Space (Level 1)
    "HardwareConfig",
    "SearchSpaceConstraints",
    "ConfigActionSpace",
    "ConfigObservationSpace",
    # Graph Space (Level 2)
    "ConstrainedGraphActionSpace",
    "GraphObservationSpace",
    "GraphAction",
    # Environments
    "HierarchicalSearchEnv",
    "ConfigEnv",
    "ConstrainedGraphEnv",
    # Training
    "HierarchicalTrainer",
    "HierarchicalSearchCoordinator",
    # Accelerator Design Space (Level 0)
    "AcceleratorEnv",
    "AcceleratorActionSpace",
    "AcceleratorObservationSpace",
    "AcceleratorDesignConstraints",
    "ParetoFrontTracker",
    "ParetoPoint",
    # Bottom-up Feedback (Problem 1)
    "KernelCharacteristics",
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
]
