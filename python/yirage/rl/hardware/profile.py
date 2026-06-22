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
Hardware profile definitions.

Provides unified representation of hardware capabilities across backends.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import json
import numpy as np


@dataclass
class HardwareProfile:
    """
    Unified hardware profile for heterogeneous computing.

    Captures capabilities of GPU, NPU, CPU, and other accelerators
    in a normalized format for use in RL training and config generation.
    """

    # Basic identification
    backend: str = "cpu"  # cuda, maca, ascend, cpu, mps
    device_name: str = "Unknown"
    device_id: int = 0
    device_count: int = 1
    driver_version: str = ""

    # Compute capabilities
    compute_capability: Tuple[int, int] = (0, 0)  # (major, minor)
    total_cores: int = 1  # CUDA cores / AI cores / CPU cores
    tensor_core_count: int = 0
    warp_size: int = 32  # 32 for CUDA, 64 for MACA, 1 for CPU

    # Memory specifications
    global_memory_gb: float = 0.0
    shared_memory_kb: float = 48.0
    l1_cache_kb: float = 0.0
    l2_cache_mb: float = 0.0
    memory_bandwidth_gbps: float = 0.0

    # Execution limits
    max_threads_per_block: int = 1024
    max_blocks_per_sm: int = 32
    max_shared_memory_per_block: int = 49152
    max_registers_per_thread: int = 255
    max_grid_dim: Tuple[int, int, int] = (2147483647, 65535, 65535)
    max_block_dim: Tuple[int, int, int] = (1024, 1024, 64)

    # Performance characteristics
    peak_tflops_fp16: float = 0.0
    peak_tflops_fp32: float = 0.0
    peak_tflops_tf32: float = 0.0
    peak_tflops_int8: float = 0.0
    memory_clock_ghz: float = 0.0

    # Special features
    supports_tensor_cores: bool = False
    supports_async_copy: bool = False
    supports_cooperative_groups: bool = False
    supports_unified_memory: bool = False

    # Backend-specific extensions
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "device_id": self.device_id,
            "device_count": self.device_count,
            "compute_capability": list(self.compute_capability),
            "total_cores": self.total_cores,
            "tensor_core_count": self.tensor_core_count,
            "warp_size": self.warp_size,
            "global_memory_gb": self.global_memory_gb,
            "shared_memory_kb": self.shared_memory_kb,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "max_threads_per_block": self.max_threads_per_block,
            "max_shared_memory_per_block": self.max_shared_memory_per_block,
            "peak_tflops_fp16": self.peak_tflops_fp16,
            "peak_tflops_fp32": self.peak_tflops_fp32,
            "supports_tensor_cores": self.supports_tensor_cores,
            "extensions": self.extensions,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HardwareProfile":
        """Create from dictionary."""
        cc = d.get("compute_capability", [0, 0])
        return cls(
            backend=d.get("backend", "cpu"),
            device_name=d.get("device_name", "Unknown"),
            device_id=d.get("device_id", 0),
            device_count=d.get("device_count", 1),
            compute_capability=tuple(cc) if isinstance(cc, list) else cc,
            total_cores=d.get("total_cores", 1),
            tensor_core_count=d.get("tensor_core_count", 0),
            warp_size=d.get("warp_size", 32),
            global_memory_gb=d.get("global_memory_gb", 0.0),
            shared_memory_kb=d.get("shared_memory_kb", 48.0),
            memory_bandwidth_gbps=d.get("memory_bandwidth_gbps", 0.0),
            max_threads_per_block=d.get("max_threads_per_block", 1024),
            max_shared_memory_per_block=d.get("max_shared_memory_per_block", 49152),
            peak_tflops_fp16=d.get("peak_tflops_fp16", 0.0),
            peak_tflops_fp32=d.get("peak_tflops_fp32", 0.0),
            supports_tensor_cores=d.get("supports_tensor_cores", False),
            extensions=d.get("extensions", {}),
        )

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert to normalized feature vector for RL model input.

        Returns:
            numpy array of shape (32,) with normalized features
        """
        features = np.zeros(32, dtype=np.float32)

        # Backend encoding (one-hot, indices 0-5)
        backends = ["cuda", "maca", "ascend", "cpu", "mps", "accelforge"]
        if self.backend in backends:
            features[backends.index(self.backend)] = 1.0

        # Compute capabilities (normalized)
        features[6] = self.compute_capability[0] / 10.0
        features[7] = self.compute_capability[1] / 10.0

        # Core counts (log scale)
        features[8] = np.log10(max(self.total_cores, 1)) / 5.0
        features[9] = np.log10(max(self.tensor_core_count + 1, 1)) / 4.0

        # Warp size
        features[10] = self.warp_size / 64.0

        # Memory (log scale)
        features[11] = np.log10(max(self.global_memory_gb, 0.1)) / 2.0
        features[12] = self.shared_memory_kb / 256.0
        features[13] = np.log10(max(self.memory_bandwidth_gbps, 1)) / 4.0

        # Execution limits (log scale)
        features[14] = np.log10(self.max_threads_per_block) / 4.0
        features[15] = np.log10(max(self.max_shared_memory_per_block, 1)) / 6.0

        # Performance (log scale)
        features[16] = np.log10(max(self.peak_tflops_fp16, 0.001) + 1) / 3.0
        features[17] = np.log10(max(self.peak_tflops_fp32, 0.001) + 1) / 3.0

        # Boolean features
        features[18] = 1.0 if self.supports_tensor_cores else 0.0
        features[19] = 1.0 if self.supports_async_copy else 0.0
        features[20] = 1.0 if self.supports_unified_memory else 0.0

        # AccelForge hardware design features (indices 21-31)
        if self.is_accelforge and self.extensions:
            from .accelforge_bridge import (
                DATAFLOW_ENCODING,
                NOC_TOPOLOGY_ENCODING,
                DATA_PRECISION_ENCODING,
                MAX_PE_ARRAY_LOG2,
            )

            af_design = self.extensions.get("accelforge_design", {})
            af_metrics = self.extensions.get("accelforge_metrics", {})

            # PE array size (normalized)
            pe_rows = af_design.get("pe_array_rows", 0)
            pe_cols = af_design.get("pe_array_cols", 0)
            features[21] = np.log2(max(pe_rows, 1)) / MAX_PE_ARRAY_LOG2
            features[22] = np.log2(max(pe_cols, 1)) / MAX_PE_ARRAY_LOG2

            # Buffer sizes (log scale, normalized)
            features[23] = np.log2(max(af_design.get("l1_buffer_kb", 1), 1)) / 12.0
            features[24] = np.log2(max(af_design.get("l2_buffer_kb", 1), 1)) / 16.0

            # Dataflow encoding
            features[25] = DATAFLOW_ENCODING.get(af_design.get("dataflow", ""), 0.0)

            # Area (normalized)
            features[26] = min(af_metrics.get("area_mm2", 0) / 100.0, 1.0)

            # Energy (normalized)
            features[27] = min(af_metrics.get("energy_per_op_pj", 0) / 10.0, 1.0)

            # Power (normalized)
            features[28] = min(af_metrics.get("total_power_mw", 0) / 10000.0, 1.0)

            # Utilization
            features[29] = af_metrics.get("pe_utilization", 0)

            # NoC topology encoding
            features[30] = NOC_TOPOLOGY_ENCODING.get(
                af_design.get("noc_topology", ""), 0.0
            )

            # Precision encoding
            features[31] = DATA_PRECISION_ENCODING.get(
                af_design.get("data_precision", ""), 0.0
            )

        return features

    @classmethod
    def from_accelforge(cls, af_model: Any) -> "HardwareProfile":
        """
        Create HardwareProfile from an AccelForge accelerator model.

        Bridges AccelForge's hardware design space with YiRage's RL pipeline.

        Args:
            af_model: AccelForge model instance, AccelForgeDesignPoint,
                      or a dict with design parameters.

        Returns:
            HardwareProfile populated with accelerator specs
        """
        from .accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )

        bridge = AccelForgeBridge()

        if isinstance(af_model, AccelForgeDesignPoint):
            design = af_model
        elif isinstance(af_model, dict):
            design = AccelForgeDesignPoint.from_dict(af_model)
        else:
            # Assume it's an AccelForge library model object
            design_dict = {}
            for attr in ["pe_array_rows", "pe_array_cols", "dataflow",
                         "data_precision", "noc_topology"]:
                if hasattr(af_model, attr):
                    design_dict[attr] = getattr(af_model, attr)
            design = AccelForgeDesignPoint.from_dict(design_dict)

        return bridge.to_hardware_profile(design)

    @property
    def is_accelforge(self) -> bool:
        """Check if this profile represents an AccelForge-modeled accelerator."""
        return self.backend == "accelforge"

    @property
    def accelforge_metrics(self) -> Optional[Dict[str, Any]]:
        """Get AccelForge-specific metrics from extensions."""
        if not self.is_accelforge:
            return None
        return self.extensions.get("accelforge_metrics")

    @property
    def is_gpu(self) -> bool:
        """Check if this is a GPU device."""
        return self.backend in ["cuda", "maca", "mps"]

    @property
    def is_npu(self) -> bool:
        """Check if this is an NPU device."""
        return self.backend == "ascend"

    @property
    def effective_parallelism(self) -> int:
        """Estimate effective parallelism for this device."""
        if self.backend == "cpu":
            return self.total_cores
        else:
            return self.total_cores * self.max_threads_per_block


@dataclass
class WorkloadSpec:
    """
    Specification of a compute workload.

    Used to determine optimal configuration for given hardware.
    """

    # Input dimensions
    batch_size: int = 1
    sequence_length: int = 1024
    hidden_dim: int = 4096

    # Operation types
    primary_op: str = "matmul"  # matmul, attention, mlp, etc.
    has_reduction: bool = False
    has_elementwise: bool = False

    # Memory access pattern
    memory_bound: bool = False
    compute_bound: bool = True

    # Precision
    dtype: str = "float16"

    # Additional info
    num_inputs: int = 2
    num_outputs: int = 1
    estimated_flops: float = 0.0
    estimated_memory_bytes: float = 0.0

    def estimate_flops(self) -> float:
        """Estimate FLOPs for this workload."""
        if self.primary_op == "matmul":
            # GEMM: 2 * M * N * K
            M = self.batch_size * self.sequence_length
            K = self.hidden_dim
            N = self.hidden_dim
            return 2.0 * M * N * K
        elif self.primary_op == "attention":
            # Self-attention: ~4 * B * S^2 * D
            B = self.batch_size
            S = self.sequence_length
            D = self.hidden_dim
            return 4.0 * B * S * S * D
        else:
            # Elementwise: B * S * D
            return self.batch_size * self.sequence_length * self.hidden_dim

    def estimate_memory(self) -> float:
        """Estimate memory bytes for this workload."""
        dtype_sizes = {"float16": 2, "float32": 4, "bfloat16": 2, "int8": 1}
        dtype_size = dtype_sizes.get(self.dtype, 2)

        # Input + output tensors
        elements = self.batch_size * self.sequence_length * self.hidden_dim
        return elements * dtype_size * (self.num_inputs + self.num_outputs)

    def is_memory_bound(self, hardware: HardwareProfile) -> bool:
        """Determine if workload is memory-bound on given hardware."""
        flops = self.estimate_flops()
        memory = self.estimate_memory()

        # Arithmetic intensity = FLOPs / Memory
        ai = flops / max(memory, 1)

        # Ridge point = Peak FLOPS / Memory Bandwidth
        peak_flops = hardware.peak_tflops_fp16 * 1e12
        bandwidth = hardware.memory_bandwidth_gbps * 1e9
        ridge_point = peak_flops / max(bandwidth, 1)

        return ai < ridge_point


@dataclass
class PerformanceEstimate:
    """
    Performance estimate for a kernel on specific hardware.
    """

    # Latency
    estimated_latency_ms: float = 0.0
    latency_lower_bound_ms: float = 0.0
    latency_upper_bound_ms: float = 0.0

    # Throughput
    estimated_tflops: float = 0.0
    theoretical_peak_tflops: float = 0.0
    compute_utilization: float = 0.0

    # Memory
    estimated_memory_bandwidth_gbps: float = 0.0
    memory_utilization: float = 0.0

    # Occupancy
    theoretical_occupancy: float = 0.0
    achieved_occupancy: float = 0.0

    # Resource usage
    registers_per_thread: int = 0
    shared_memory_bytes: int = 0

    # AccelForge metrics — energy, area, power
    energy_pj: float = 0.0  # Energy per operation (picojoules)
    area_mm2: float = 0.0  # Chip area (mm²)
    power_mw: float = 0.0  # Total power (milliwatts)
    leak_power_mw: float = 0.0  # Leakage power (milliwatts)

    # Confidence
    confidence: float = 0.0  # 0-1, how confident is this estimate

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimated_latency_ms": self.estimated_latency_ms,
            "estimated_tflops": self.estimated_tflops,
            "compute_utilization": self.compute_utilization,
            "memory_utilization": self.memory_utilization,
            "achieved_occupancy": self.achieved_occupancy,
            "energy_pj": self.energy_pj,
            "area_mm2": self.area_mm2,
            "power_mw": self.power_mw,
            "leak_power_mw": self.leak_power_mw,
            "confidence": self.confidence,
        }
