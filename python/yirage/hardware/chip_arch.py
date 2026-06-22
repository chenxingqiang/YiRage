# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Chip Architecture Specification

Defines the unified `ChipArchitecture` dataclass that captures hardware
characteristics of any accelerator chip.  New chips are registered at
runtime via :pymod:`yirage.hardware.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChipVendor(str, Enum):
    """Known chip vendors."""

    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    HUAWEI = "huawei"
    METAX = "metax"
    APPLE = "apple"
    GOOGLE = "google"
    XILINX = "xilinx"
    AWS = "aws"
    QUALCOMM = "qualcomm"
    SAMSUNG = "samsung"
    OTHER = "other"


class ChipCategory(str, Enum):
    """Broad chip category."""

    GPU = "gpu"
    CPU = "cpu"
    NPU = "npu"
    TPU = "tpu"
    FPGA = "fpga"
    DSA = "dsa"  # Domain-Specific Accelerator
    OTHER = "other"


class MemoryType(str, Enum):
    """Device memory technology."""

    HBM = "hbm"
    HBM2 = "hbm2"
    HBM2E = "hbm2e"
    HBM3 = "hbm3"
    HBM3E = "hbm3e"
    GDDR6 = "gddr6"
    GDDR6X = "gddr6x"
    LPDDR4 = "lpddr4"
    LPDDR5 = "lpddr5"
    DDR4 = "ddr4"
    DDR5 = "ddr5"
    UNIFIED = "unified"
    SRAM = "sram"
    OTHER = "other"


@dataclass
class MemorySpec:
    """Device-level memory specification."""

    capacity_gb: float = 0.0
    bandwidth_gbps: float = 0.0
    memory_type: MemoryType = MemoryType.OTHER
    bus_width_bits: int = 0
    ecc: bool = False


@dataclass
class ComputeSpec:
    """Compute-unit specification."""

    # Thread/warp/wavefront
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_threads_per_cu: int = 2048
    max_blocks_per_cu: int = 32

    # Shared / on-chip memory
    shared_mem_per_block_kb: int = 48
    shared_mem_per_cu_kb: int = 96
    l2_cache_mb: float = 0.0

    # Register file
    registers_per_block: int = 65536

    # Compute units
    num_compute_units: int = 0  # SMs / CUs / AI Cores / etc.

    # TFLOPS (FP32 unless noted)
    peak_tflops_fp32: float = 0.0
    peak_tflops_fp16: float = 0.0
    peak_tflops_bf16: float = 0.0
    peak_tflops_int8: float = 0.0


@dataclass
class FeatureFlags:
    """Hardware feature capability flags."""

    tensor_cores: bool = False
    tensor_core_gen: int = 0
    matrix_units: bool = False
    tma: bool = False  # Tensor Memory Accelerator (Hopper+)
    fp8: bool = False
    fp4: bool = False
    bf16: bool = True
    fp16: bool = True
    int8: bool = True
    int4: bool = False
    sparsity: bool = False
    nvlink: bool = False
    infinity_fabric: bool = False
    rdma: bool = False
    multi_instance: bool = False  # e.g. MIG on NVIDIA
    dynamic_parallelism: bool = False


@dataclass
class ChipArchitecture:
    """
    Unified representation of a chip architecture.

    Every hardware backend — from NVIDIA Blackwell to a future custom ASIC —
    can be described by an instance of this class.  The :pymod:`HardwareRegistry`
    stores and indexes these instances so that the rest of YiRage can query
    chip capabilities at runtime.

    Attributes:
        chip_id:    Unique human-readable identifier, e.g. ``"nvidia_h100"``.
        chip_name:  Marketing name, e.g. ``"NVIDIA H100 SXM5"``.
        vendor:     :class:`ChipVendor` enum.
        category:   :class:`ChipCategory` enum.
        arch_name:  Microarchitecture name, e.g. ``"Hopper"``, ``"CDNA3"``.
        arch_code:  Vendor-specific code, e.g. ``"sm_90"``, ``"gfx942"``.
        backend:    YiRage backend name this maps to (``"cuda"``, ``"maca"``, …).
        memory:     :class:`MemorySpec`.
        compute:    :class:`ComputeSpec`.
        features:   :class:`FeatureFlags`.
        search_config_overrides:
            Extra key/value pairs that override the default search
            configuration for this chip.
        metadata:   Free-form metadata dict for anything not covered above.
    """

    # Identity
    chip_id: str = ""
    chip_name: str = ""
    vendor: ChipVendor = ChipVendor.OTHER
    category: ChipCategory = ChipCategory.OTHER
    arch_name: str = ""
    arch_code: str = ""

    # Backend binding
    backend: str = ""

    # Specs
    memory: MemorySpec = field(default_factory=MemorySpec)
    compute: ComputeSpec = field(default_factory=ComputeSpec)
    features: FeatureFlags = field(default_factory=FeatureFlags)

    # Search tuning
    search_config_overrides: dict[str, Any] = field(default_factory=dict)

    # Free-form metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ utils

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        import dataclasses

        def _convert(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _convert(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, Enum):
                return obj.value
            return obj

        return _convert(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChipArchitecture:
        """Deserialise from a plain dict."""
        memory = MemorySpec(**{
            k: MemoryType(v) if k == "memory_type" else v
            for k, v in data.pop("memory", {}).items()
        })
        compute = ComputeSpec(**data.pop("compute", {}))
        features = FeatureFlags(**data.pop("features", {}))
        vendor = ChipVendor(data.pop("vendor", "other"))
        category = ChipCategory(data.pop("category", "other"))
        return cls(
            memory=memory,
            compute=compute,
            features=features,
            vendor=vendor,
            category=category,
            **data,
        )

    def summary(self) -> str:
        """One-line human-readable summary."""
        parts = [self.chip_name or self.chip_id]
        if self.compute.num_compute_units:
            parts.append(f"{self.compute.num_compute_units} CUs")
        if self.memory.capacity_gb:
            parts.append(f"{self.memory.capacity_gb:.0f}GB {self.memory.memory_type.value.upper()}")
        if self.compute.peak_tflops_fp16:
            parts.append(f"{self.compute.peak_tflops_fp16:.0f} TFLOPS FP16")
        return " | ".join(parts)
