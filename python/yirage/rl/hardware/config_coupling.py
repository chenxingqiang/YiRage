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
Hardware-Config coupling for optimal search configuration.

Automatically generates search configurations based on hardware capabilities.
"""

from typing import List, Tuple, Dict, Any, Optional
import logging
import math
import numpy as np

from .profile import HardwareProfile, WorkloadSpec, PerformanceEstimate
from ..search.config_space import HardwareConfig


logger = logging.getLogger(__name__)
MIN_STANDARD_BLOCK_SIZE = 64
STANDARD_BLOCK_SIZE_CANDIDATES = [MIN_STANDARD_BLOCK_SIZE, 128, 256, 512, 1024]


class ConfigGenerator:
    """
    Generates optimal configurations based on hardware and workload.
    """

    def __init__(self, hardware: HardwareProfile):
        self.hardware = hardware

    def generate(
        self,
        workload: Optional[WorkloadSpec] = None,
        optimization_target: str = "latency",
    ) -> HardwareConfig:
        """
        Generate optimal configuration.

        Args:
            workload: Workload specification
            optimization_target: "latency", "throughput", or "memory"

        Returns:
            Optimal HardwareConfig for this hardware/workload combination
        """
        hw = self.hardware

        # AccelForge backend: derive config from PE array / buffer / dataflow
        if hw.backend == "accelforge":
            return self._generate_for_accelforge(workload, optimization_target)

        # Start with defaults
        block_x = min(hw.max_threads_per_block, 256)
        grid_x = max(1, hw.total_cores // 4)

        if workload:
            # Adjust based on workload
            total_elements = workload.batch_size * workload.sequence_length * workload.hidden_dim

            # Grid size: enough blocks to keep SMs busy
            block_x = self._choose_block_size(workload)
            grid_x = math.ceil(total_elements / block_x)
            grid_x = min(grid_x, 65535)

            # For matmul: 2D grid
            if workload.primary_op == "matmul":
                M = workload.batch_size * workload.sequence_length
                N = workload.hidden_dim

                tile_m = 128  # Tile size
                tile_n = 128

                grid_x = math.ceil(M / tile_m)
                grid_y = math.ceil(N / tile_n)
            else:
                grid_y = 1
        else:
            grid_y = 1

        # Forloop range based on memory
        forloop_range = self._choose_forloop_range(workload)

        # Reduction dimension
        reduction_dimx = hw.warp_size

        # Shared memory
        smem = min(hw.max_shared_memory_per_block, 49152)

        return HardwareConfig(
            grid_dim_x=min(grid_x, hw.max_grid_dim[0]),
            grid_dim_y=min(grid_y if workload else 1, hw.max_grid_dim[1]),
            grid_dim_z=1,
            block_dim_x=block_x,
            block_dim_y=1,
            block_dim_z=1,
            forloop_range=forloop_range,
            reduction_dimx=reduction_dimx,
            shared_memory_size=smem,
            num_registers=min(hw.max_registers_per_thread, 64),
        )

    def _choose_block_size(self, workload: Optional[WorkloadSpec]) -> int:
        """Choose optimal block size."""
        hw = self.hardware

        # Common block sizes
        candidates = STANDARD_BLOCK_SIZE_CANDIDATES
        candidates = [c for c in candidates if c <= hw.max_threads_per_block]

        if not workload:
            # Default to 128 for general workloads
            return min(128, hw.max_threads_per_block)

        # For memory-bound: larger blocks for better memory coalescing
        if workload.is_memory_bound(hw):
            return max(candidates)

        # For compute-bound: balance occupancy
        # Aim for 50% occupancy
        target_threads = hw.max_threads_per_block // 2

        best = candidates[0]
        for c in candidates:
            if abs(c - target_threads) < abs(best - target_threads):
                best = c

        return best

    def _choose_forloop_range(self, workload: Optional[WorkloadSpec]) -> int:
        """Choose forloop range for tiling."""
        if not workload:
            return 1

        # For large workloads, use tiling
        total_elements = workload.batch_size * workload.sequence_length * workload.hidden_dim

        if total_elements > 1e6:
            return 16
        elif total_elements > 1e5:
            return 8
        elif total_elements > 1e4:
            return 4
        else:
            return 1

    def _generate_for_accelforge(
        self,
        workload: Optional[WorkloadSpec],
        optimization_target: str,
    ) -> HardwareConfig:
        """
        Generate config derived from AccelForge hardware design parameters.

        Maps PE array dimensions, buffer sizes, and dataflow to appropriate
        grid/block/forloop parameters, ensuring the config is coupled to the
        actual accelerator design instead of using GPU-style heuristics.
        """
        hw = self.hardware
        af_design = hw.extensions.get("accelforge_design", {})

        pe_rows = af_design.get("pe_array_rows", hw.total_cores)
        pe_cols = af_design.get("pe_array_cols", 1)
        l1_buffer_kb = af_design.get("l1_buffer_kb", hw.shared_memory_kb)
        dataflow = af_design.get("dataflow", "output_stationary")

        # Map PE array columns → block dimension (parallel execution width)
        block_x = min(pe_cols, hw.max_threads_per_block)

        # Map PE array rows → grid dimension (number of parallel groups)
        grid_x = pe_rows

        grid_y = 1
        if workload and workload.primary_op == "matmul":
            M = workload.batch_size * workload.sequence_length
            N = workload.hidden_dim
            tile_m = pe_rows
            tile_n = pe_cols
            grid_x = max(1, math.ceil(M / max(tile_m, 1)))
            grid_y = max(1, math.ceil(N / max(tile_n, 1)))
            grid_x = min(grid_x, 65535)
            grid_y = min(grid_y, 65535)

        # Forloop range: derived from buffer capacity
        smem_bytes = int(l1_buffer_kb * 1024)
        # Bytes per tile iteration for default forloop sizing
        _BYTES_PER_TILE_ITER = 8192
        if workload:
            total_elements = (
                workload.batch_size * workload.sequence_length * workload.hidden_dim
            )
            # Tile elements that fit in L1 buffer (assuming 2 bytes per element)
            tile_elements = smem_bytes // 2
            forloop_range = max(1, math.ceil(total_elements / max(tile_elements, 1)))
            forloop_range = min(forloop_range, 64)
        else:
            forloop_range = max(1, smem_bytes // _BYTES_PER_TILE_ITER)
            forloop_range = min(forloop_range, 16)

        # Reduction dimension from dataflow preference
        if dataflow == "row_stationary":
            reduction_dimx = pe_cols
        elif dataflow == "weight_stationary":
            reduction_dimx = pe_rows
        else:  # output_stationary
            reduction_dimx = min(pe_cols, 32)

        return HardwareConfig(
            grid_dim_x=max(1, grid_x),
            grid_dim_y=max(1, grid_y),
            grid_dim_z=1,
            block_dim_x=max(1, block_x),
            block_dim_y=1,
            block_dim_z=1,
            forloop_range=max(1, forloop_range),
            reduction_dimx=max(1, reduction_dimx),
            shared_memory_size=smem_bytes,
            num_registers=min(hw.max_registers_per_thread, 64),
        )

    def generate_search_space(
        self,
        workload: Optional[WorkloadSpec] = None,
    ) -> List[HardwareConfig]:
        """
        Generate a set of configurations for search.

        Returns a diverse set of valid configurations for this hardware.
        """
        hw = self.hardware
        configs = []

        if hw.backend == "accelforge":
            base = self.generate(workload)
            variants = [base]
            for scale in (0.5, 2.0):
                block_x = max(1, min(hw.max_threads_per_block, int(base.block_dim_x * scale)))
                grid_x = max(1, min(hw.max_grid_dim[0], int(base.grid_dim_x * scale)))
                variants.append(
                    HardwareConfig(
                        grid_dim_x=grid_x,
                        grid_dim_y=base.grid_dim_y,
                        grid_dim_z=base.grid_dim_z,
                        block_dim_x=block_x,
                        block_dim_y=base.block_dim_y,
                        block_dim_z=base.block_dim_z,
                        forloop_range=base.forloop_range,
                        reduction_dimx=base.reduction_dimx,
                        shared_memory_size=base.shared_memory_size,
                        num_registers=base.num_registers,
                    )
                )
            return variants

        # Block sizes to try
        block_sizes = [b for b in STANDARD_BLOCK_SIZE_CANDIDATES if b <= hw.max_threads_per_block]
        if not block_sizes:
            # Accelerators with max_threads_per_block below MIN_STANDARD_BLOCK_SIZE
            # expose less parallelism than the standard block-size candidates.
            logger.warning(
                "Using non-standard block size for hardware with max_threads_per_block=%s",
                hw.max_threads_per_block,
            )
            block_sizes = [max(1, hw.max_threads_per_block)]

        # Grid sizes
        grid_sizes = [1, 2, 4, 8, 16, 32, 64]

        # Forloop ranges
        franges = [1, 2, 4, 8, 16]

        for block_x in block_sizes[:3]:  # Limit to top 3
            for grid_x in grid_sizes[:5]:
                for frange in franges[:3]:
                    config = HardwareConfig(
                        grid_dim_x=grid_x,
                        block_dim_x=block_x,
                        forloop_range=frange,
                        reduction_dimx=hw.warp_size,
                        shared_memory_size=min(hw.max_shared_memory_per_block, 49152),
                        num_registers=min(hw.max_registers_per_thread, 64),
                    )
                    configs.append(config)

        return configs


class HardwareSearchCoupling:
    """
    Couples hardware profiles with search space constraints.
    """

    def __init__(self, hardware: HardwareProfile):
        self.hardware = hardware
        self.config_generator = ConfigGenerator(hardware)

    def get_valid_configs(self) -> List[HardwareConfig]:
        """Get all valid configurations for this hardware."""
        return self.config_generator.generate_search_space()

    def get_constraints(
        self,
        config: HardwareConfig,
        accelerator_constraints: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Get search constraints from hardware + config.

        When accelerator_constraints (from Level 0 AccelForge) are provided,
        they are incorporated to further restrict the search space based on
        the actual accelerator design.

        Returns constraints that limit the µGraph search space.
        """
        hw = self.hardware

        # Valid imaps based on grid dimensions
        valid_imaps = self._compute_valid_imaps(config)

        # Valid forloop ranges
        valid_franges = self._compute_valid_franges(config)

        # Max operators based on resources
        max_ops = self._compute_max_operators(config)

        # Max tensor size based on shared memory
        max_tensor_elements = config.shared_memory_size // 2  # Assuming FP16

        constraints = {
            "valid_imaps": valid_imaps,
            "valid_franges": valid_franges,
            "max_operators": max_ops,
            "max_tensor_elements": max_tensor_elements,
            "warp_size": hw.warp_size,
            "max_shared_memory": config.shared_memory_size,
        }

        # Incorporate AccelForge accelerator design constraints
        if accelerator_constraints is not None:
            constraints = self._apply_accelerator_constraints(
                constraints, accelerator_constraints
            )
        elif hw.backend == "accelforge" and hw.extensions:
            # Auto-derive constraints from AccelForge hardware profile
            constraints = self._apply_accelforge_profile_constraints(constraints)

        return constraints

    def _apply_accelerator_constraints(
        self,
        constraints: Dict[str, Any],
        accel_constraints: Any,
    ) -> Dict[str, Any]:
        """Apply Level 0 AcceleratorDesignConstraints to restrict search space."""
        # Tighten max_operators based on PE parallelism
        if hasattr(accel_constraints, "max_parallelism"):
            pe_limit = accel_constraints.max_parallelism
            constraints["max_operators"] = min(
                constraints["max_operators"], pe_limit
            )

        # Tighten shared memory based on buffer
        if hasattr(accel_constraints, "max_shared_memory_kb"):
            smem_bytes = int(accel_constraints.max_shared_memory_kb * 1024)
            constraints["max_shared_memory"] = min(
                constraints["max_shared_memory"], smem_bytes
            )
            constraints["max_tensor_elements"] = min(
                constraints["max_tensor_elements"],
                smem_bytes // 2,
            )

        # Tighten max_tile_size
        if hasattr(accel_constraints, "max_tile_size"):
            constraints["max_tile_size"] = accel_constraints.max_tile_size

        # Propagate precision info
        if hasattr(accel_constraints, "supported_precisions"):
            constraints["supported_precisions"] = (
                accel_constraints.supported_precisions
            )

        # Propagate dataflow reuse info
        if hasattr(accel_constraints, "supports_weight_reuse"):
            constraints["supports_weight_reuse"] = (
                accel_constraints.supports_weight_reuse
            )
            constraints["supports_output_reuse"] = (
                accel_constraints.supports_output_reuse
            )

        return constraints

    def _apply_accelforge_profile_constraints(
        self, constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Derive constraints from AccelForge hardware profile extensions."""
        hw = self.hardware
        af_design = hw.extensions.get("accelforge_design", {})
        if not af_design:
            return constraints

        pe_rows = af_design.get("pe_array_rows", 0)
        pe_cols = af_design.get("pe_array_cols", 0)
        l1_kb = af_design.get("l1_buffer_kb", 0)

        if pe_rows > 0 and pe_cols > 0:
            total_pes = pe_rows * pe_cols
            constraints["max_operators"] = min(
                constraints["max_operators"], total_pes
            )

        if l1_kb > 0:
            smem_bytes = int(l1_kb * 1024)
            constraints["max_shared_memory"] = min(
                constraints["max_shared_memory"], smem_bytes
            )
            constraints["max_tensor_elements"] = min(
                constraints["max_tensor_elements"],
                smem_bytes // 2,
            )

        dataflow = af_design.get("dataflow", "")
        if dataflow:
            constraints["supports_weight_reuse"] = dataflow in (
                "weight_stationary", "row_stationary"
            )
            constraints["supports_output_reuse"] = dataflow in (
                "output_stationary", "row_stationary"
            )

        precision = af_design.get("data_precision", "")
        if precision:
            constraints["supported_precisions"] = [precision]

        return constraints

    def _compute_valid_imaps(self, config: HardwareConfig) -> List[Tuple[int, int, int]]:
        """Compute valid input mappings based on grid dimensions."""
        valid = []

        for ix in [-1, 0, 1]:
            for iy in [-1, 0, 1]:
                for iz in [-1, 0, 1]:
                    is_valid = True

                    # Check grid dimension constraints
                    if ix == 0 and config.grid_dim_x <= 1:
                        is_valid = False
                    if iy == 1 and config.grid_dim_y <= 1:
                        is_valid = False
                    if iz == 2 and config.grid_dim_z <= 1:
                        is_valid = False

                    if is_valid:
                        valid.append((ix, iy, iz))

        return valid

    def _compute_valid_franges(self, config: HardwareConfig) -> List[int]:
        """Compute valid forloop ranges."""
        fr = config.forloop_range
        valid = []

        for f in [1, 2, 4, 8, 16, 32, 64]:
            if fr >= f and fr % f == 0:
                valid.append(f)

        return valid if valid else [1]

    def _compute_max_operators(self, config: HardwareConfig) -> int:
        """Compute maximum operators based on resources."""
        hw = self.hardware

        # Rough estimate: each operator uses some shared memory and registers
        smem_per_op = 2048  # 2KB per operator
        reg_per_op = 16

        smem_limit = config.shared_memory_size // smem_per_op
        reg_limit = (hw.max_registers_per_thread * config.total_threads) // reg_per_op

        return min(smem_limit, reg_limit, 30)

    def estimate_performance(
        self,
        config: HardwareConfig,
        graph_features: Dict[str, Any],
    ) -> PerformanceEstimate:
        """
        Estimate kernel performance without execution.

        Uses analytical model based on hardware and graph characteristics.
        For AccelForge backends, uses AccelForge's modeling for higher fidelity.
        """
        hw = self.hardware

        # AccelForge backend: use AccelForge bridge for estimation
        if hw.backend == "accelforge":
            return self._estimate_accelforge(config, graph_features)

        # Extract features
        num_ops = graph_features.get("num_operators", 1)
        total_flops = graph_features.get("theoretical_flops", 0)
        total_memory = graph_features.get("memory_bytes", 0)

        # Compute utilization
        total_threads = config.total_blocks * config.total_threads
        theoretical_parallelism = hw.total_cores * hw.max_threads_per_block
        compute_utilization = min(1.0, total_threads / theoretical_parallelism)

        # Estimate occupancy
        threads_per_sm = config.total_threads
        max_threads_per_sm = hw.max_threads_per_block * hw.max_blocks_per_sm
        occupancy = min(1.0, threads_per_sm / max_threads_per_sm)

        # Estimate latency
        if total_flops > 0:
            # Compute time
            peak_flops = hw.peak_tflops_fp16 * 1e12
            compute_time = total_flops / (peak_flops * compute_utilization)

            # Memory time
            bandwidth = hw.memory_bandwidth_gbps * 1e9
            memory_time = total_memory / bandwidth

            # Total latency (max of compute and memory)
            latency_s = max(compute_time, memory_time) / occupancy
            latency_ms = latency_s * 1000
        else:
            latency_ms = 0.1  # Default

        return PerformanceEstimate(
            estimated_latency_ms=latency_ms,
            estimated_tflops=total_flops / (latency_ms * 1e9) if latency_ms > 0 else 0,
            theoretical_peak_tflops=hw.peak_tflops_fp16,
            compute_utilization=compute_utilization,
            theoretical_occupancy=occupancy,
            achieved_occupancy=occupancy * 0.8,  # Conservative
            confidence=0.6,  # Medium confidence for analytical model
        )

    def _estimate_accelforge(
        self,
        config: HardwareConfig,
        graph_features: Dict[str, Any],
    ) -> PerformanceEstimate:
        """
        Estimate performance using AccelForge modeling.

        Provides higher fidelity estimates including energy, area, and power.
        """
        from .accelforge_bridge import AccelForgeBridge, AccelForgeDesignPoint

        hw = self.hardware
        bridge = AccelForgeBridge()

        # Get design point from hardware profile extensions
        af_design_dict = hw.extensions.get("accelforge_design", {})
        design = AccelForgeDesignPoint.from_dict(af_design_dict) if af_design_dict else None

        # Build workload spec for AccelForge
        workload = {
            "estimated_flops": graph_features.get("theoretical_flops", 0),
            "memory_bytes": graph_features.get("memory_bytes", 0),
            "num_operators": graph_features.get("num_operators", 1),
        }

        if design:
            metrics = bridge.evaluate(design, workload)
        else:
            metrics = bridge.evaluate(AccelForgeDesignPoint(), workload)

        # Compute utilization from config
        total_threads = config.total_blocks * config.total_threads
        theoretical_parallelism = hw.total_cores * hw.max_threads_per_block
        compute_utilization = min(1.0, total_threads / max(theoretical_parallelism, 1))

        return PerformanceEstimate(
            estimated_latency_ms=metrics.latency_ms,
            estimated_tflops=metrics.achieved_tops,
            theoretical_peak_tflops=metrics.peak_tops,
            compute_utilization=compute_utilization,
            theoretical_occupancy=metrics.pe_utilization,
            achieved_occupancy=metrics.pe_utilization * 0.9,
            energy_pj=metrics.energy_per_op_pj,
            area_mm2=metrics.area_mm2,
            power_mw=metrics.total_power_mw,
            leak_power_mw=metrics.leak_power_mw,
            confidence=metrics.confidence,
        )

    def to_feature_vector(self, config: HardwareConfig) -> np.ndarray:
        """
        Convert hardware + config to feature vector for RL.

        Returns:
            Combined feature vector of shape (64,)
        """
        # Hardware features (32,)
        hw_features = self.hardware.to_feature_vector()

        # Config features (32,)
        config_features = np.zeros(32, dtype=np.float32)

        config_features[0] = np.log2(config.grid_dim_x + 1) / 10
        config_features[1] = np.log2(config.grid_dim_y + 1) / 10
        config_features[2] = np.log2(config.grid_dim_z + 1) / 10
        config_features[3] = np.log2(config.block_dim_x + 1) / 10
        config_features[4] = np.log2(config.block_dim_y + 1) / 10
        config_features[5] = np.log2(config.forloop_range + 1) / 6
        config_features[6] = np.log2(config.reduction_dimx + 1) / 6
        config_features[7] = config.shared_memory_size / 65536

        # Derived features
        config_features[8] = config.total_threads / self.hardware.max_threads_per_block
        config_features[9] = config.total_blocks / 1024

        # Concatenate
        return np.concatenate([hw_features, config_features])


# Convenience function
def get_optimal_config(
    hardware: Optional[HardwareProfile] = None,
    workload: Optional[WorkloadSpec] = None,
    optimization_target: str = "latency",
) -> HardwareConfig:
    """
    Get optimal configuration for given hardware and workload.

    Args:
        hardware: Hardware profile (auto-detected if None)
        workload: Workload specification
        optimization_target: Optimization goal

    Returns:
        Optimal HardwareConfig
    """
    if hardware is None:
        from .registry import detect_hardware

        hardware = detect_hardware()

    generator = ConfigGenerator(hardware)
    return generator.generate(workload, optimization_target)
