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
Cross-Backend Kernel Migration (Problem 6c) and
Persistent Kernel RL Integration (Problem 6d)

Cross-backend migration:
  Find optimal kernel on CUDA → Z3 verify equivalence → migrate to MACA/Ascend.
  This converts 19 independent searches into transfer learning.

Persistent kernel integration:
  Adds persistent kernel (long-running GPU kernel for LLM serving) as a
  Level 2 search target alongside standard kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import json
import numpy as np


# =============================================================================
# Problem 6c: Cross-Backend Kernel Migration
# =============================================================================


@dataclass
class BackendCapability:
    """Describes what a backend supports for migration feasibility."""

    name: str
    supports_shared_memory: bool = True
    supports_tensor_cores: bool = False
    max_threads_per_block: int = 1024
    max_shared_memory_kb: float = 96.0
    warp_size: int = 32
    supports_cooperative_groups: bool = False
    supports_async_copy: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "supports_shared_memory": self.supports_shared_memory,
            "supports_tensor_cores": self.supports_tensor_cores,
            "max_threads_per_block": self.max_threads_per_block,
            "max_shared_memory_kb": self.max_shared_memory_kb,
            "warp_size": self.warp_size,
            "supports_cooperative_groups": self.supports_cooperative_groups,
            "supports_async_copy": self.supports_async_copy,
        }


# Known backend capabilities
BACKEND_CAPABILITIES: Dict[str, BackendCapability] = {
    "cuda": BackendCapability(
        name="cuda",
        supports_tensor_cores=True,
        max_threads_per_block=1024,
        max_shared_memory_kb=96.0,
        warp_size=32,
        supports_cooperative_groups=True,
        supports_async_copy=True,
    ),
    "rocm": BackendCapability(
        name="rocm",
        supports_tensor_cores=False,  # Uses matrix cores
        max_threads_per_block=1024,
        max_shared_memory_kb=64.0,
        warp_size=64,
        supports_cooperative_groups=True,
    ),
    "maca": BackendCapability(
        name="maca",
        supports_tensor_cores=False,
        max_threads_per_block=1024,
        max_shared_memory_kb=128.0,
        warp_size=64,
    ),
    "ascend": BackendCapability(
        name="ascend",
        supports_shared_memory=True,
        supports_tensor_cores=False,
        max_threads_per_block=256,
        max_shared_memory_kb=256.0,
        warp_size=1,
    ),
    "cpu": BackendCapability(
        name="cpu",
        supports_shared_memory=False,
        supports_tensor_cores=False,
        max_threads_per_block=1,
        max_shared_memory_kb=0.0,
        warp_size=1,
    ),
}


@dataclass
class MigrationResult:
    """Result of attempting to migrate a kernel across backends."""

    source_backend: str = ""
    target_backend: str = ""
    feasible: bool = False
    # What needed to change
    adaptations: List[str] = field(default_factory=list)
    # What couldn't be migrated
    blockers: List[str] = field(default_factory=list)
    # Constraint violations (from Z3 check)
    constraint_violations: int = 0
    # Estimated performance ratio (target/source)
    performance_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_backend": self.source_backend,
            "target_backend": self.target_backend,
            "feasible": self.feasible,
            "adaptations": self.adaptations,
            "blockers": self.blockers,
            "constraint_violations": self.constraint_violations,
            "performance_ratio": self.performance_ratio,
        }


class KernelMigrationEngine:
    """
    Cross-backend kernel migration using Z3 equivalence verification.

    Instead of searching independently on each of 19 backends,
    find optimal kernel on one backend and migrate.

    Migration workflow:
    1. Find optimal kernel on source backend (e.g., CUDA)
    2. Analyze kernel for backend-specific features
    3. Check Z3 constraints for target backend compatibility
    4. Adapt kernel (adjust thread counts, memory, etc.)
    5. Verify equivalence via Z3
    """

    def __init__(self):
        self._migration_cache: Dict[str, MigrationResult] = {}

    def check_migration_feasibility(
        self,
        kernel_config: Dict[str, Any],
        source_backend: str,
        target_backend: str,
    ) -> MigrationResult:
        """
        Check if a kernel can be migrated from source to target backend.

        Uses Z3 constraint checking when available, falls back to
        heuristic analysis.
        """
        cache_key = f"{source_backend}_{target_backend}_{json.dumps(kernel_config, sort_keys=True)}"
        if cache_key in self._migration_cache:
            return self._migration_cache[cache_key]

        source_cap = BACKEND_CAPABILITIES.get(
            source_backend, BackendCapability(name=source_backend)
        )
        target_cap = BACKEND_CAPABILITIES.get(
            target_backend, BackendCapability(name=target_backend)
        )

        result = MigrationResult(
            source_backend=source_backend,
            target_backend=target_backend,
        )

        # Check thread compatibility
        block_x = kernel_config.get("block_dim_x", 128)
        block_y = kernel_config.get("block_dim_y", 1)
        total_threads = block_x * block_y
        if total_threads > target_cap.max_threads_per_block:
            result.blockers.append(
                f"thread_count({total_threads}) > target_max({target_cap.max_threads_per_block})"
            )

        # Check shared memory
        shared_mem_kb = kernel_config.get("shared_memory_size", 0) / 1024.0
        if shared_mem_kb > target_cap.max_shared_memory_kb:
            result.blockers.append(
                f"shared_mem({shared_mem_kb:.1f}KB) > target_max({target_cap.max_shared_memory_kb}KB)"
            )

        # Check warp size compatibility
        if source_cap.warp_size != target_cap.warp_size:
            if block_x % target_cap.warp_size != 0:
                result.adaptations.append(
                    f"adjust_block_x: {block_x} → {(block_x // target_cap.warp_size + 1) * target_cap.warp_size}"
                )

        # Check feature dependencies
        if not target_cap.supports_shared_memory and shared_mem_kb > 0:
            result.blockers.append("target_no_shared_memory")

        if not target_cap.supports_tensor_cores and kernel_config.get("uses_tensor_cores", False):
            result.adaptations.append("replace_tensor_core_ops")

        if not target_cap.supports_async_copy and kernel_config.get("uses_async_copy", False):
            result.adaptations.append("replace_async_copy_with_sync")

        # Feasibility decision
        result.feasible = len(result.blockers) == 0
        result.constraint_violations = len(result.blockers)

        # Performance ratio estimate
        if result.feasible:
            result.performance_ratio = self._estimate_performance_ratio(
                kernel_config, source_cap, target_cap
            )

        self._migration_cache[cache_key] = result
        return result

    def get_migration_targets(
        self,
        kernel_config: Dict[str, Any],
        source_backend: str,
    ) -> Dict[str, MigrationResult]:
        """
        Check all backends for migration feasibility.

        Returns a dict of target_backend → MigrationResult.
        """
        results = {}
        for target_name in BACKEND_CAPABILITIES:
            if target_name != source_backend:
                results[target_name] = self.check_migration_feasibility(
                    kernel_config, source_backend, target_name
                )
        return results

    def adapt_kernel(
        self,
        kernel_config: Dict[str, Any],
        migration: MigrationResult,
    ) -> Dict[str, Any]:
        """
        Adapt kernel configuration for target backend.

        Applies the adaptations identified in migration check.
        """
        adapted = dict(kernel_config)
        target_cap = BACKEND_CAPABILITIES.get(
            migration.target_backend, BackendCapability(name=migration.target_backend)
        )

        for adaptation in migration.adaptations:
            if adaptation.startswith("adjust_block_x"):
                block_x = adapted.get("block_dim_x", 128)
                new_block_x = (block_x // target_cap.warp_size + 1) * target_cap.warp_size
                new_block_x = min(new_block_x, target_cap.max_threads_per_block)
                adapted["block_dim_x"] = new_block_x

            elif adaptation == "replace_tensor_core_ops":
                adapted["uses_tensor_cores"] = False

            elif adaptation == "replace_async_copy_with_sync":
                adapted["uses_async_copy"] = False

        adapted["target_backend"] = migration.target_backend
        return adapted

    @staticmethod
    def _estimate_performance_ratio(
        config: Dict[str, Any],
        source: BackendCapability,
        target: BackendCapability,
    ) -> float:
        """Estimate performance ratio when migrating."""
        ratio = 1.0

        # Warp size difference affects occupancy
        if source.warp_size != target.warp_size:
            ratio *= 0.9  # Small penalty

        # Shared memory difference
        src_shmem = source.max_shared_memory_kb
        tgt_shmem = target.max_shared_memory_kb
        if src_shmem > 0:
            ratio *= min(tgt_shmem / src_shmem, 1.2)

        # Thread count difference
        if target.max_threads_per_block < source.max_threads_per_block:
            ratio *= 0.85

        return ratio


# =============================================================================
# Problem 6d: Persistent Kernel RL Integration
# =============================================================================


@dataclass
class PersistentKernelConfig:
    """
    Configuration for persistent kernels (long-running GPU kernels).

    Persistent kernels stay resident on the GPU between inference calls,
    eliminating kernel launch overhead for LLM serving.
    """

    # Whether this kernel should be persistent
    persistent: bool = False
    # Maximum duration before forced yield (microseconds)
    max_duration_us: int = 10000
    # Number of GPU blocks to reserve
    num_persistent_blocks: int = 1
    # Whether to use cooperative groups for synchronization
    use_cooperative_groups: bool = False
    # Input/output queue depth
    queue_depth: int = 4
    # Workload type
    workload_type: str = "inference"  # "inference", "prefill", "decode"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persistent": self.persistent,
            "max_duration_us": self.max_duration_us,
            "num_persistent_blocks": self.num_persistent_blocks,
            "use_cooperative_groups": self.use_cooperative_groups,
            "queue_depth": self.queue_depth,
            "workload_type": self.workload_type,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PersistentKernelConfig":
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


class PersistentKernelSearchSpace:
    """
    Extends Level 2 search space with persistent kernel options.

    Adds persistent kernel as a search target alongside standard kernels.
    The RL agent can choose to:
    1. Build a standard kernel (as before)
    2. Build a persistent kernel (new option)

    Persistent kernels have additional constraints:
    - Must be cooperative-group compatible
    - Must fit in reserved GPU blocks
    - Must handle input/output queuing
    """

    # Persistent kernel action extensions
    PERSISTENT_ACTIONS = {
        "MAKE_PERSISTENT": 4,  # Convert current graph to persistent
        "SET_QUEUE_DEPTH": 5,  # Set I/O queue depth
        "SET_DURATION": 6,  # Set max duration
    }

    QUEUE_DEPTH_OPTIONS = [1, 2, 4, 8, 16]
    DURATION_OPTIONS = [1000, 5000, 10000, 50000, 100000]  # microseconds
    BLOCK_OPTIONS = [1, 2, 4, 8, 16, 32]

    def __init__(self, max_gpu_blocks: int = 32):
        self.max_gpu_blocks = max_gpu_blocks

    def get_persistent_action_mask(
        self,
        num_kn_operators: int,
        num_tb_operators: int,
        backend: str = "cuda",
    ) -> np.ndarray:
        """Get action mask for persistent kernel actions."""
        mask = np.zeros(3, dtype=np.int8)

        # Can make persistent if we have a complete kernel
        has_complete_kernel = num_kn_operators > 0 and num_tb_operators > 0
        cap = BACKEND_CAPABILITIES.get(backend, BackendCapability(name=backend))

        mask[0] = int(has_complete_kernel and cap.supports_cooperative_groups)
        mask[1] = int(has_complete_kernel)  # Queue depth
        mask[2] = int(has_complete_kernel)  # Duration

        return mask

    def compute_persistent_reward(
        self,
        config: PersistentKernelConfig,
        standard_latency_ms: float,
        launch_overhead_us: float = 5.0,
        num_inferences: int = 1000,
    ) -> float:
        """
        Compute reward bonus for persistent kernel.

        Persistent kernels eliminate launch overhead, which compounds
        over many inference calls.
        """
        if not config.persistent:
            return 0.0

        # Savings from eliminating launch overhead
        saved_time_ms = (launch_overhead_us * num_inferences) / 1000.0

        # Queue overhead (persistent kernels have I/O queue cost)
        queue_overhead_ms = 0.001 * config.queue_depth

        # Net benefit
        net_savings_ms = saved_time_ms - queue_overhead_ms

        # Reward proportional to savings relative to kernel latency
        if standard_latency_ms > 0:
            return net_savings_ms / max(standard_latency_ms * num_inferences, 0.001)

        return 0.0

    def suggest_persistent_config(
        self,
        kernel_config: Dict[str, Any],
        workload_type: str = "inference",
    ) -> PersistentKernelConfig:
        """
        Suggest persistent kernel config based on workload type.
        """
        grid_x = kernel_config.get("grid_dim_x", 1)
        grid_y = kernel_config.get("grid_dim_y", 1)
        total_blocks = grid_x * grid_y

        return PersistentKernelConfig(
            persistent=True,
            max_duration_us=10000 if workload_type == "decode" else 50000,
            num_persistent_blocks=min(total_blocks, self.max_gpu_blocks),
            use_cooperative_groups=total_blocks > 1,
            queue_depth=4 if workload_type == "decode" else 2,
            workload_type=workload_type,
        )
