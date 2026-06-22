"""
AMD ROCm-specific search configuration
Optimized search space for AMD GPUs (CDNA, RDNA architectures)

Based on AMD ROCm specifications:
- wavefrontSize: 64 (CDNA) or 32 (RDNA in wave32 mode)
- LDS (Local Data Share): 64KB per workgroup
- Matrix Cores (MFMA): Available on CDNA
"""

import os
import subprocess
import multiprocessing
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class ROCmArch(str, Enum):
    """AMD GPU architectures."""

    CDNA1 = "gfx908"  # MI100
    CDNA2 = "gfx90a"  # MI200 series
    CDNA3 = "gfx942"  # MI300 series
    RDNA2 = "gfx1030"  # RX 6000 series
    RDNA3 = "gfx1100"  # RX 7000 series
    UNKNOWN = "unknown"


# Key difference from NVIDIA: wavefrontSize = 64 on CDNA
ROCM_WARP_SIZE = 64


@dataclass
class ROCmArchSpecs:
    """ROCm architecture specifications."""

    arch: ROCmArch
    wavefront_size: int = 64
    max_threads_per_workgroup: int = 1024
    lds_per_workgroup_kb: int = 64
    has_matrix_cores: bool = True
    mfma_m: int = 16
    mfma_n: int = 16
    mfma_k: int = 16


ROCM_ARCH_SPECS: Dict[ROCmArch, ROCmArchSpecs] = {
    ROCmArch.CDNA1: ROCmArchSpecs(arch=ROCmArch.CDNA1, mfma_m=32, mfma_n=32),
    ROCmArch.CDNA2: ROCmArchSpecs(arch=ROCmArch.CDNA2, mfma_m=32, mfma_n=32),
    ROCmArch.CDNA3: ROCmArchSpecs(arch=ROCmArch.CDNA3, mfma_m=64, mfma_n=64),
    ROCmArch.RDNA2: ROCmArchSpecs(arch=ROCmArch.RDNA2, wavefront_size=32, has_matrix_cores=False),
    ROCmArch.RDNA3: ROCmArchSpecs(arch=ROCmArch.RDNA3, wavefront_size=32, has_matrix_cores=True),
}


def get_rocm_search_config(arch: ROCmArch = None) -> Dict[str, Any]:
    """
    Get optimized search configuration for ROCm backend.

    AMD ROCm characteristics:
    - wavefrontSize: 64 (CDNA) - NOT 32 like NVIDIA
    - LDS: 64KB per workgroup
    - MFMA: Matrix Fused Multiply-Add on CDNA

    Returns:
        dict: Search configuration optimized for ROCm
    """
    if arch is None:
        device_info = get_rocm_device_info()
        if device_info and device_info.get("available"):
            arch = device_info.get("arch", ROCmArch.CDNA3)
        else:
            arch = ROCmArch.CDNA3

    specs = ROCM_ARCH_SPECS.get(arch, ROCM_ARCH_SPECS[ROCmArch.CDNA3])
    wavefront_size = specs.wavefront_size

    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))

    # Block sizes must be multiples of wavefrontSize
    if wavefront_size == 64:  # CDNA
        block_dims = [
            (64, 1, 1),
            (128, 1, 1),
            (192, 1, 1),
            (256, 1, 1),
            (320, 1, 1),
            (384, 1, 1),
            (448, 1, 1),
            (512, 1, 1),
            (640, 1, 1),
            (768, 1, 1),
            (1024, 1, 1),
        ]
    else:  # RDNA (wavefront 32)
        block_dims = [
            (32, 1, 1),
            (64, 1, 1),
            (128, 1, 1),
            (256, 1, 1),
            (512, 1, 1),
            (1024, 1, 1),
        ]

    return {
        # Architecture info
        "arch": arch.value,
        "wavefront_size": wavefront_size,
        "has_matrix_cores": specs.has_matrix_cores,
        # Search parameters
        "max_num_threadblock_graph_op": 8,
        "max_num_kernel_graph_op": 5,
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        # Search space
        "grid_dims_to_explore": [
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
            (16, 1, 1),
            (32, 1, 1),
            (64, 1, 1),
            (110, 1, 1),  # MI300X has 110 CUs
            (8, 8, 1),
        ],
        "block_dims_to_explore": block_dims,
        "fmaps_to_explore": [-1, 0, 1, 2],
        "franges_to_explore": [4, 8, 16, 32],
        # Hardware limits
        "max_threads_per_workgroup": specs.max_threads_per_workgroup,
        "lds_per_workgroup_kb": specs.lds_per_workgroup_kb,
    }


def get_rocm_memory_config() -> Dict[str, Any]:
    """Get ROCm memory configuration."""
    device_info = get_rocm_device_info()

    if device_info and device_info.get("available"):
        return {
            "device_name": device_info.get("device_name", "Unknown"),
            "hbm_gb": device_info.get("hbm_gb", 0),
            "wavefront_size": ROCM_WARP_SIZE,
            "note": f"{device_info.get('device_name', 'GPU')} detected",
        }

    return {"device_name": "Unknown", "note": "ROCm device not detected"}


def get_rocm_device_info() -> Optional[Dict[str, Any]]:
    """Detect ROCm device via rocm-smi."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            output = result.stdout
            device_info = {"available": True, "device_name": "AMD GPU"}

            if "MI300" in output:
                device_info.update(
                    {"device_name": "AMD MI300", "arch": ROCmArch.CDNA3, "hbm_gb": 192}
                )
            elif "MI250" in output or "MI210" in output:
                device_info.update(
                    {"device_name": "AMD MI200", "arch": ROCmArch.CDNA2, "hbm_gb": 128}
                )
            elif "MI100" in output:
                device_info.update(
                    {"device_name": "AMD MI100", "arch": ROCmArch.CDNA1, "hbm_gb": 32}
                )

            return device_info
    except:
        pass
    return None


def is_rocm_available() -> bool:
    """Check if ROCm is available."""
    device_info = get_rocm_device_info()
    return device_info is not None and device_info.get("available", False)
