# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Built-in chip architecture definitions.

Call :func:`register_builtin_chips` to populate the
:class:`~yirage.hardware.registry.HardwareRegistry` with all chips that YiRage
ships knowledge about.  This function is invoked automatically when the
``yirage.hardware`` package is first imported.
"""

from __future__ import annotations

from .chip_arch import (
    ChipArchitecture,
    ChipCategory,
    ChipVendor,
    ComputeSpec,
    FeatureFlags,
    MemorySpec,
    MemoryType,
)
from .registry import HardwareRegistry


def register_builtin_chips(registry: HardwareRegistry | None = None) -> int:
    """
    Register all built-in chip definitions.

    Args:
        registry: Target registry.  Defaults to the global singleton.

    Returns:
        Number of chips registered.
    """
    if registry is None:
        registry = HardwareRegistry.instance()

    chips = (
        _nvidia_chips()
        + _amd_chips()
        + _intel_chips()
        + _huawei_chips()
        + _metax_chips()
        + _apple_chips()
        + _google_chips()
        + _fpga_chips()
        + _aws_chips()
    )

    count = 0
    for chip in chips:
        if registry.register(chip, overwrite=False):
            count += 1
    return count


# ============================================================================
# NVIDIA GPUs
# ============================================================================

def _nvidia_chips() -> list[ChipArchitecture]:
    return [
        # Volta
        ChipArchitecture(
            chip_id="nvidia_v100",
            chip_name="NVIDIA V100",
            vendor=ChipVendor.NVIDIA,
            category=ChipCategory.GPU,
            arch_name="Volta",
            arch_code="sm_70",
            backend="cuda",
            memory=MemorySpec(capacity_gb=32, bandwidth_gbps=900, memory_type=MemoryType.HBM2, bus_width_bits=4096, ecc=True),
            compute=ComputeSpec(warp_size=32, max_threads_per_block=1024, shared_mem_per_block_kb=48, shared_mem_per_cu_kb=96, l2_cache_mb=6, num_compute_units=80, peak_tflops_fp32=15.7, peak_tflops_fp16=125),
            features=FeatureFlags(tensor_cores=True, tensor_core_gen=1, nvlink=True),
        ),
        # Turing
        ChipArchitecture(
            chip_id="nvidia_t4",
            chip_name="NVIDIA T4",
            vendor=ChipVendor.NVIDIA,
            category=ChipCategory.GPU,
            arch_name="Turing",
            arch_code="sm_75",
            backend="cuda",
            memory=MemorySpec(capacity_gb=16, bandwidth_gbps=320, memory_type=MemoryType.GDDR6, bus_width_bits=256),
            compute=ComputeSpec(warp_size=32, max_threads_per_block=1024, shared_mem_per_block_kb=64, shared_mem_per_cu_kb=64, l2_cache_mb=6, num_compute_units=40, peak_tflops_fp32=8.1, peak_tflops_fp16=65),
            features=FeatureFlags(tensor_cores=True, tensor_core_gen=2, int8=True),
        ),
        # Ampere (data-centre)
        ChipArchitecture(
            chip_id="nvidia_a100",
            chip_name="NVIDIA A100 80GB",
            vendor=ChipVendor.NVIDIA,
            category=ChipCategory.GPU,
            arch_name="Ampere",
            arch_code="sm_80",
            backend="cuda",
            memory=MemorySpec(capacity_gb=80, bandwidth_gbps=2039, memory_type=MemoryType.HBM2E, bus_width_bits=5120, ecc=True),
            compute=ComputeSpec(warp_size=32, max_threads_per_block=1024, shared_mem_per_block_kb=164, shared_mem_per_cu_kb=164, l2_cache_mb=40, num_compute_units=108, peak_tflops_fp32=19.5, peak_tflops_fp16=312, peak_tflops_bf16=312),
            features=FeatureFlags(tensor_cores=True, tensor_core_gen=3, bf16=True, sparsity=True, nvlink=True, multi_instance=True),
        ),
        # Ampere (consumer)
        ChipArchitecture(
            chip_id="nvidia_rtx3090",
            chip_name="NVIDIA RTX 3090",
            vendor=ChipVendor.NVIDIA,
            category=ChipCategory.GPU,
            arch_name="Ampere",
            arch_code="sm_86",
            backend="cuda",
            memory=MemorySpec(capacity_gb=24, bandwidth_gbps=936, memory_type=MemoryType.GDDR6X, bus_width_bits=384),
            compute=ComputeSpec(warp_size=32, max_threads_per_block=1024, shared_mem_per_block_kb=100, shared_mem_per_cu_kb=100, l2_cache_mb=6, num_compute_units=82, peak_tflops_fp32=35.6, peak_tflops_fp16=71),
            features=FeatureFlags(tensor_cores=True, tensor_core_gen=3),
        ),
        # Ada Lovelace
        ChipArchitecture(
            chip_id="nvidia_rtx4090",
            chip_name="NVIDIA RTX 4090",
            vendor=ChipVendor.NVIDIA,
            category=ChipCategory.GPU,
            arch_name="Ada Lovelace",
            arch_code="sm_89",
            backend="cuda",
            memory=MemorySpec(capacity_gb=24, bandwidth_gbps=1008, memory_type=MemoryType.GDDR6X, bus_width_bits=384),
            compute=ComputeSpec(warp_size=32, max_threads_per_block=1024, shared_mem_per_block_kb=100, shared_mem_per_cu_kb=100, l2_cache_mb=72, num_compute_units=128, peak_tflops_fp32=82.6, peak_tflops_fp16=165),
            features=FeatureFlags(tensor_cores=True, tensor_core_gen=4, fp8=True),
        ),
        # Hopper
        ChipArchitecture(
            chip_id="nvidia_h100",
            chip_name="NVIDIA H100 SXM5",
            vendor=ChipVendor.NVIDIA,
            category=ChipCategory.GPU,
            arch_name="Hopper",
            arch_code="sm_90",
            backend="cuda",
            memory=MemorySpec(capacity_gb=80, bandwidth_gbps=3350, memory_type=MemoryType.HBM3, bus_width_bits=5120, ecc=True),
            compute=ComputeSpec(warp_size=32, max_threads_per_block=1024, shared_mem_per_block_kb=228, shared_mem_per_cu_kb=228, l2_cache_mb=50, num_compute_units=132, peak_tflops_fp32=67, peak_tflops_fp16=989, peak_tflops_bf16=989),
            features=FeatureFlags(tensor_cores=True, tensor_core_gen=4, tma=True, fp8=True, bf16=True, sparsity=True, nvlink=True, multi_instance=True, dynamic_parallelism=True),
        ),
        # Blackwell
        ChipArchitecture(
            chip_id="nvidia_b200",
            chip_name="NVIDIA B200",
            vendor=ChipVendor.NVIDIA,
            category=ChipCategory.GPU,
            arch_name="Blackwell",
            arch_code="sm_100",
            backend="cuda",
            memory=MemorySpec(capacity_gb=192, bandwidth_gbps=8000, memory_type=MemoryType.HBM3E, bus_width_bits=8192, ecc=True),
            compute=ComputeSpec(warp_size=32, max_threads_per_block=1024, shared_mem_per_block_kb=256, shared_mem_per_cu_kb=256, l2_cache_mb=64, num_compute_units=160, peak_tflops_fp32=90, peak_tflops_fp16=1800, peak_tflops_bf16=1800),
            features=FeatureFlags(tensor_cores=True, tensor_core_gen=5, tma=True, fp8=True, fp4=True, bf16=True, sparsity=True, nvlink=True, multi_instance=True, dynamic_parallelism=True),
        ),
    ]


# ============================================================================
# AMD GPUs (ROCm)
# ============================================================================

def _amd_chips() -> list[ChipArchitecture]:
    return [
        ChipArchitecture(
            chip_id="amd_mi250x",
            chip_name="AMD Instinct MI250X",
            vendor=ChipVendor.AMD,
            category=ChipCategory.GPU,
            arch_name="CDNA2",
            arch_code="gfx90a",
            backend="rocm",
            memory=MemorySpec(capacity_gb=128, bandwidth_gbps=3200, memory_type=MemoryType.HBM2E, bus_width_bits=8192, ecc=True),
            compute=ComputeSpec(warp_size=64, max_threads_per_block=1024, shared_mem_per_block_kb=64, num_compute_units=220, peak_tflops_fp32=47.9, peak_tflops_fp16=383),
            features=FeatureFlags(matrix_units=True, infinity_fabric=True, bf16=True),
        ),
        ChipArchitecture(
            chip_id="amd_mi300x",
            chip_name="AMD Instinct MI300X",
            vendor=ChipVendor.AMD,
            category=ChipCategory.GPU,
            arch_name="CDNA3",
            arch_code="gfx942",
            backend="rocm",
            memory=MemorySpec(capacity_gb=192, bandwidth_gbps=5300, memory_type=MemoryType.HBM3, bus_width_bits=8192, ecc=True),
            compute=ComputeSpec(warp_size=64, max_threads_per_block=1024, shared_mem_per_block_kb=64, num_compute_units=304, peak_tflops_fp32=163, peak_tflops_fp16=1307, peak_tflops_bf16=1307),
            features=FeatureFlags(matrix_units=True, infinity_fabric=True, bf16=True, fp8=True, sparsity=True),
        ),
    ]


# ============================================================================
# Intel XPU
# ============================================================================

def _intel_chips() -> list[ChipArchitecture]:
    return [
        ChipArchitecture(
            chip_id="intel_pvc",
            chip_name="Intel Data Center GPU Max 1550 (Ponte Vecchio)",
            vendor=ChipVendor.INTEL,
            category=ChipCategory.GPU,
            arch_name="Xe HPC",
            arch_code="pvc",
            backend="xpu",
            memory=MemorySpec(capacity_gb=128, bandwidth_gbps=3200, memory_type=MemoryType.HBM2E, bus_width_bits=8192, ecc=True),
            compute=ComputeSpec(warp_size=16, max_threads_per_block=1024, num_compute_units=128, peak_tflops_fp32=52, peak_tflops_fp16=420, peak_tflops_bf16=420),
            features=FeatureFlags(matrix_units=True, bf16=True),
        ),
    ]


# ============================================================================
# Huawei Ascend NPU
# ============================================================================

def _huawei_chips() -> list[ChipArchitecture]:
    return [
        ChipArchitecture(
            chip_id="ascend_910",
            chip_name="Huawei Ascend 910",
            vendor=ChipVendor.HUAWEI,
            category=ChipCategory.NPU,
            arch_name="Da Vinci",
            arch_code="ascend910",
            backend="ascend",
            memory=MemorySpec(capacity_gb=32, bandwidth_gbps=1200, memory_type=MemoryType.HBM2, ecc=True),
            compute=ComputeSpec(shared_mem_per_block_kb=256, num_compute_units=32, peak_tflops_fp16=320),
            features=FeatureFlags(matrix_units=True, bf16=True, rdma=True),
        ),
        ChipArchitecture(
            chip_id="ascend_910b",
            chip_name="Huawei Ascend 910B",
            vendor=ChipVendor.HUAWEI,
            category=ChipCategory.NPU,
            arch_name="Da Vinci",
            arch_code="ascend910b",
            backend="ascend",
            memory=MemorySpec(capacity_gb=64, bandwidth_gbps=1600, memory_type=MemoryType.HBM2E, ecc=True),
            compute=ComputeSpec(shared_mem_per_block_kb=512, num_compute_units=32, peak_tflops_fp16=320),
            features=FeatureFlags(matrix_units=True, bf16=True, rdma=True),
        ),
        ChipArchitecture(
            chip_id="ascend_310p",
            chip_name="Huawei Ascend 310P",
            vendor=ChipVendor.HUAWEI,
            category=ChipCategory.NPU,
            arch_name="Da Vinci",
            arch_code="ascend310p",
            backend="ascend",
            memory=MemorySpec(capacity_gb=8, bandwidth_gbps=400, memory_type=MemoryType.HBM2),
            compute=ComputeSpec(shared_mem_per_block_kb=128, num_compute_units=8, peak_tflops_fp16=70),
            features=FeatureFlags(matrix_units=True, bf16=True),
        ),
    ]


# ============================================================================
# MetaX MACA
# ============================================================================

def _metax_chips() -> list[ChipArchitecture]:
    return [
        ChipArchitecture(
            chip_id="metax_c500",
            chip_name="MetaX C500",
            vendor=ChipVendor.METAX,
            category=ChipCategory.GPU,
            arch_name="MACA",
            arch_code="mxc500",
            backend="maca",
            memory=MemorySpec(capacity_gb=64, bandwidth_gbps=2000, memory_type=MemoryType.HBM2E, bus_width_bits=4096),
            compute=ComputeSpec(warp_size=64, max_threads_per_block=1024, shared_mem_per_block_kb=64, registers_per_block=131072, num_compute_units=104, peak_tflops_fp32=50, peak_tflops_fp16=256),
            features=FeatureFlags(tensor_cores=False, bf16=True),
        ),
        ChipArchitecture(
            chip_id="metax_c500_pro",
            chip_name="MetaX C500 Pro",
            vendor=ChipVendor.METAX,
            category=ChipCategory.GPU,
            arch_name="MACA",
            arch_code="mxc500pro",
            backend="maca",
            memory=MemorySpec(capacity_gb=64, bandwidth_gbps=2000, memory_type=MemoryType.HBM2E, bus_width_bits=4096),
            compute=ComputeSpec(warp_size=64, max_threads_per_block=1024, shared_mem_per_block_kb=64, registers_per_block=131072, num_compute_units=104, peak_tflops_fp32=50, peak_tflops_fp16=256),
            features=FeatureFlags(tensor_cores=False, bf16=True),
        ),
    ]


# ============================================================================
# Apple Silicon (MPS)
# ============================================================================

def _apple_chips() -> list[ChipArchitecture]:
    """Return all known Apple Silicon GPU variants (M1–M5).

    Specs represent the maximum-configuration variant of each chip
    (e.g. top GPU core count, max unified memory).
    """

    def _mk(chip_id, chip_name, arch_code, num_cores, fp32_tflops,
            capacity_gb, bw_gbps):
        return ChipArchitecture(
            chip_id=chip_id,
            chip_name=chip_name,
            vendor=ChipVendor.APPLE,
            category=ChipCategory.GPU,
            arch_name="Apple GPU",
            arch_code=arch_code,
            backend="mps",
            memory=MemorySpec(
                capacity_gb=capacity_gb,
                bandwidth_gbps=bw_gbps,
                memory_type=MemoryType.UNIFIED,
            ),
            compute=ComputeSpec(
                warp_size=32,
                max_threads_per_block=1024,
                shared_mem_per_block_kb=32,
                registers_per_block=32768,
                num_compute_units=num_cores,
                peak_tflops_fp32=fp32_tflops,
                peak_tflops_fp16=fp32_tflops,
            ),
            features=FeatureFlags(bf16=True),
        )

    return [
        # M1 family (Apple G13G)
        _mk("apple_m1",       "Apple M1",       "apple_g13g",  8,  2.6,   16,  68.25),
        _mk("apple_m1_pro",   "Apple M1 Pro",   "apple_g13g", 16,  5.2,   32, 200),
        _mk("apple_m1_max",   "Apple M1 Max",   "apple_g13g", 32, 10.4,   64, 400),
        _mk("apple_m1_ultra", "Apple M1 Ultra", "apple_g13g", 64, 21.0,  128, 800),

        # M2 family (Apple G14G)
        _mk("apple_m2",       "Apple M2",       "apple_g14g", 10,  3.6,   24, 100),
        _mk("apple_m2_pro",   "Apple M2 Pro",   "apple_g14g", 19,  6.8,   32, 200),
        _mk("apple_m2_max",   "Apple M2 Max",   "apple_g14g", 38, 13.6,   96, 400),
        _mk("apple_m2_ultra", "Apple M2 Ultra", "apple_g14g", 76, 27.2,  192, 800),

        # M3 family (Apple G15P)
        _mk("apple_m3",       "Apple M3",       "apple_g15p", 10,  4.0,   24, 100),
        _mk("apple_m3_pro",   "Apple M3 Pro",   "apple_g15p", 18,  7.0,   36, 150),
        _mk("apple_m3_max",   "Apple M3 Max",   "apple_g15p", 40, 14.2,  128, 400),
        _mk("apple_m3_ultra", "Apple M3 Ultra", "apple_g15p", 80, 27.2,  192, 800),

        # M4 family (Apple G16P)
        _mk("apple_m4",       "Apple M4",       "apple_g16p", 10,  4.0,   32, 120),
        _mk("apple_m4_pro",   "Apple M4 Pro",   "apple_g16p", 20,  7.2,   48, 273),
        _mk("apple_m4_max",   "Apple M4 Max",   "apple_g16p", 40, 18.0,  128, 546),

        # M5 family (Apple G17P) — projected specs based on generational trends
        _mk("apple_m5",       "Apple M5",       "apple_g17p", 12,  5.0,   32, 150),
        _mk("apple_m5_pro",   "Apple M5 Pro",   "apple_g17p", 22,  9.0,   48, 300),
        _mk("apple_m5_max",   "Apple M5 Max",   "apple_g17p", 44, 22.0,  128, 600),
        _mk("apple_m5_ultra", "Apple M5 Ultra", "apple_g17p", 80, 36.0,  256, 1000),
    ]


# ============================================================================
# Google TPU
# ============================================================================

def _google_chips() -> list[ChipArchitecture]:
    return [
        ChipArchitecture(
            chip_id="google_tpu_v4",
            chip_name="Google TPU v4",
            vendor=ChipVendor.GOOGLE,
            category=ChipCategory.TPU,
            arch_name="TPU",
            arch_code="tpu_v4",
            backend="tpu",
            memory=MemorySpec(capacity_gb=32, bandwidth_gbps=1200, memory_type=MemoryType.HBM2E),
            compute=ComputeSpec(num_compute_units=2, peak_tflops_fp32=275, peak_tflops_bf16=275),
            features=FeatureFlags(matrix_units=True, bf16=True, rdma=True),
        ),
        ChipArchitecture(
            chip_id="google_tpu_v5e",
            chip_name="Google TPU v5e",
            vendor=ChipVendor.GOOGLE,
            category=ChipCategory.TPU,
            arch_name="TPU",
            arch_code="tpu_v5e",
            backend="tpu",
            memory=MemorySpec(capacity_gb=16, bandwidth_gbps=1600, memory_type=MemoryType.HBM2E),
            compute=ComputeSpec(num_compute_units=1, peak_tflops_fp32=197, peak_tflops_bf16=197),
            features=FeatureFlags(matrix_units=True, bf16=True, int8=True, rdma=True),
        ),
    ]


# ============================================================================
# FPGA
# ============================================================================

def _fpga_chips() -> list[ChipArchitecture]:
    return [
        ChipArchitecture(
            chip_id="xilinx_alveo_u250",
            chip_name="Xilinx Alveo U250",
            vendor=ChipVendor.XILINX,
            category=ChipCategory.FPGA,
            arch_name="Alveo",
            arch_code="u250",
            backend="fpga",
            memory=MemorySpec(capacity_gb=64, bandwidth_gbps=77, memory_type=MemoryType.DDR4),
            compute=ComputeSpec(num_compute_units=4, peak_tflops_fp32=4),
            features=FeatureFlags(),
        ),
    ]


# ============================================================================
# AWS Trainium / Inferentia
# ============================================================================

def _aws_chips() -> list[ChipArchitecture]:
    return [
        ChipArchitecture(
            chip_id="aws_trainium2",
            chip_name="AWS Trainium2",
            vendor=ChipVendor.AWS,
            category=ChipCategory.DSA,
            arch_name="Trainium",
            arch_code="trn2",
            backend="nki",
            memory=MemorySpec(capacity_gb=96, bandwidth_gbps=3200, memory_type=MemoryType.HBM2E),
            compute=ComputeSpec(num_compute_units=2, peak_tflops_fp16=380, peak_tflops_bf16=380),
            features=FeatureFlags(matrix_units=True, bf16=True, fp8=True, rdma=True),
        ),
    ]
