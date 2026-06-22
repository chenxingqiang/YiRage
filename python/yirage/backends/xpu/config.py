"""
Intel XPU-specific search configuration
Optimized for Intel Data Center GPU Max (Ponte Vecchio) and Arc GPUs

Based on Intel oneAPI specifications:
- Sub-group size: 16 or 32
- Shared Local Memory (SLM): 64-128KB per sub-slice
- XMX (Xe Matrix eXtensions): Matrix acceleration
- DPAS (Dot Product Accumulate Systolic): Systolic array
"""

import os
import subprocess
import multiprocessing
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class XPUArch(str, Enum):
    """Intel XPU architectures."""

    PONTE_VECCHIO = "pvc"  # Data Center GPU Max Series
    ARC_A770 = "dg2"  # Arc A-Series (consumer)
    ARC_A380 = "dg2"  # Arc A-Series (entry)
    FLEX = "ats"  # Flex Series (Data Center)
    UNKNOWN = "unknown"


@dataclass
class XPUArchSpecs:
    """XPU architecture specifications."""

    arch: XPUArch
    subgroup_size: int = 16
    max_work_group_size: int = 1024
    slm_per_ss_kb: int = 64
    has_xmx: bool = True
    has_dpas: bool = True
    eu_count: int = 512


XPU_ARCH_SPECS: Dict[XPUArch, XPUArchSpecs] = {
    XPUArch.PONTE_VECCHIO: XPUArchSpecs(
        arch=XPUArch.PONTE_VECCHIO,
        subgroup_size=16,
        slm_per_ss_kb=128,
        has_xmx=True,
        has_dpas=True,
        eu_count=512,
    ),
    XPUArch.ARC_A770: XPUArchSpecs(
        arch=XPUArch.ARC_A770,
        subgroup_size=32,
        slm_per_ss_kb=64,
        has_xmx=True,
        has_dpas=True,
        eu_count=32,
    ),
    XPUArch.FLEX: XPUArchSpecs(
        arch=XPUArch.FLEX,
        subgroup_size=16,
        slm_per_ss_kb=64,
        has_xmx=True,
        has_dpas=True,
        eu_count=128,
    ),
}


def get_xpu_search_config(arch: XPUArch = None) -> Dict[str, Any]:
    """
    Get optimized search configuration for Intel XPU backend.

    Intel XPU characteristics:
    - Sub-group size: 16 (PVC) or 32 (Arc)
    - SLM: 64-128KB Shared Local Memory
    - XMX: Xe Matrix eXtensions for matrix ops
    - DPAS: Systolic array for dot products

    Returns:
        dict: Search configuration optimized for Intel XPU
    """
    if arch is None:
        device_info = get_xpu_info()
        if device_info.get("available"):
            arch = device_info.get("arch", XPUArch.PONTE_VECCHIO)
        else:
            arch = XPUArch.PONTE_VECCHIO

    specs = XPU_ARCH_SPECS.get(arch, XPU_ARCH_SPECS[XPUArch.PONTE_VECCHIO])
    subgroup_size = specs.subgroup_size

    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))

    # Work group sizes should be multiples of subgroup size
    if subgroup_size == 16:
        block_dims = [
            (16, 1, 1),
            (32, 1, 1),
            (64, 1, 1),
            (128, 1, 1),
            (256, 1, 1),
            (512, 1, 1),
            (1024, 1, 1),
        ]
    else:  # 32
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
        "subgroup_size": subgroup_size,
        "has_xmx": specs.has_xmx,
        "has_dpas": specs.has_dpas,
        "eu_count": specs.eu_count,
        # Search parameters
        "max_num_threadblock_graph_op": 6,
        "max_num_kernel_graph_op": 4,
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
        ],
        "block_dims_to_explore": block_dims,
        "fmaps_to_explore": [-1, 0, 1, 2],
        "franges_to_explore": [4, 8, 16, 32],
        # Hardware limits
        "max_work_group_size": specs.max_work_group_size,
        "slm_per_ss_kb": specs.slm_per_ss_kb,
        # SYCL options
        "use_sycl": True,
        "multi_tile": specs.eu_count > 128,
    }


def get_xpu_info() -> Dict[str, Any]:
    """Get Intel XPU device information."""
    try:
        result = subprocess.run(["sycl-ls"], capture_output=True, text=True, timeout=5)

        if result.returncode == 0:
            output = result.stdout
            device_info = {"available": True}

            if "Data Center GPU Max" in output or "PVC" in output:
                device_info.update(
                    {
                        "device_name": "Intel Data Center GPU Max",
                        "arch": XPUArch.PONTE_VECCHIO,
                    }
                )
            elif "Arc" in output:
                device_info.update(
                    {
                        "device_name": "Intel Arc",
                        "arch": XPUArch.ARC_A770,
                    }
                )
            elif "Flex" in output:
                device_info.update(
                    {
                        "device_name": "Intel Flex",
                        "arch": XPUArch.FLEX,
                    }
                )
            else:
                device_info["device_name"] = "Intel XPU"

            return device_info
    except:
        pass

    # Try Intel Extension for PyTorch
    try:
        import intel_extension_for_pytorch as ipex
        import torch

        if torch.xpu.is_available():
            return {
                "available": True,
                "device_name": torch.xpu.get_device_name(0),
                "device_count": torch.xpu.device_count(),
            }
    except:
        pass

    return {"available": False}


def is_xpu_available() -> bool:
    """Check if Intel XPU is available."""
    info = get_xpu_info()
    return info.get("available", False)
