"""
FPGA-specific search configuration
Optimized for Xilinx (AMD) and Intel FPGAs

Based on FPGA characteristics:
- HLS (High-Level Synthesis) for kernel generation
- BRAM/URAM on-chip memory
- DSP slices for computation
- HBM for high-bandwidth models
"""

import os
import subprocess
import multiprocessing
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class FPGAVendor(str, Enum):
    """FPGA vendor types."""

    XILINX = "xilinx"  # AMD Xilinx
    INTEL = "intel"  # Intel (Altera)
    UNKNOWN = "unknown"


class FPGADevice(str, Enum):
    """FPGA device families."""

    # Xilinx/AMD
    ALVEO_U50 = "u50"
    ALVEO_U200 = "u200"
    ALVEO_U250 = "u250"
    ALVEO_U280 = "u280"
    VERSAL_AI = "versal_ai"

    # Intel
    STRATIX_10 = "stratix10"
    AGILEX_7 = "agilex7"

    UNKNOWN = "unknown"


@dataclass
class FPGASpecs:
    """FPGA device specifications."""

    device: FPGADevice
    vendor: FPGAVendor
    lut_count: int = 1000000
    ff_count: int = 2000000
    bram_mb: int = 32
    uram_mb: int = 32
    dsp_count: int = 6000
    hbm_gb: float = 0.0
    has_hbm: bool = False
    ddr_channels: int = 4


FPGA_SPECS: Dict[FPGADevice, FPGASpecs] = {
    FPGADevice.ALVEO_U50: FPGASpecs(
        device=FPGADevice.ALVEO_U50,
        vendor=FPGAVendor.XILINX,
        lut_count=870000,
        ff_count=1743000,
        bram_mb=18,
        uram_mb=27,
        dsp_count=5952,
        hbm_gb=8.0,
        has_hbm=True,
        ddr_channels=0,
    ),
    FPGADevice.ALVEO_U280: FPGASpecs(
        device=FPGADevice.ALVEO_U280,
        vendor=FPGAVendor.XILINX,
        lut_count=1303000,
        ff_count=2607000,
        bram_mb=32,
        uram_mb=45,
        dsp_count=9024,
        hbm_gb=8.0,
        has_hbm=True,
        ddr_channels=2,
    ),
    FPGADevice.VERSAL_AI: FPGASpecs(
        device=FPGADevice.VERSAL_AI,
        vendor=FPGAVendor.XILINX,
        lut_count=900000,
        ff_count=1800000,
        bram_mb=30,
        uram_mb=80,
        dsp_count=1968,
        hbm_gb=32.0,
        has_hbm=True,
        ddr_channels=4,
    ),
    FPGADevice.AGILEX_7: FPGASpecs(
        device=FPGADevice.AGILEX_7,
        vendor=FPGAVendor.INTEL,
        lut_count=2100000,
        ff_count=4200000,
        bram_mb=64,
        uram_mb=0,
        dsp_count=8736,
        hbm_gb=32.0,
        has_hbm=True,
        ddr_channels=4,
    ),
}


def get_fpga_search_config(device: FPGADevice = None) -> Dict[str, Any]:
    """
    Get optimized search configuration for FPGA backend.

    FPGA characteristics:
    - Spatial architecture (vs temporal like GPU)
    - HLS for kernel synthesis
    - Fixed-point optimization available
    - Pipeline parallelism

    Returns:
        dict: Search configuration optimized for FPGA
    """
    if device is None:
        device_info = get_fpga_info()
        if device_info.get("available"):
            device = device_info.get("device", FPGADevice.ALVEO_U280)
        else:
            device = FPGADevice.ALVEO_U280

    specs = FPGA_SPECS.get(device, FPGA_SPECS[FPGADevice.ALVEO_U280])

    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))

    return {
        # Device info
        "device": device.value,
        "vendor": specs.vendor.value,
        "has_hbm": specs.has_hbm,
        "hbm_gb": specs.hbm_gb,
        # Resource limits
        "lut_count": specs.lut_count,
        "ff_count": specs.ff_count,
        "bram_mb": specs.bram_mb,
        "uram_mb": specs.uram_mb,
        "dsp_count": specs.dsp_count,
        # Search parameters
        "max_num_threadblock_graph_op": 4,
        "max_num_kernel_graph_op": 3,
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        # FPGA-specific: parallelism factors
        "grid_dims_to_explore": [
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
        ],
        "block_dims_to_explore": [
            (16, 1, 1),
            (32, 1, 1),
            (64, 1, 1),
            (128, 1, 1),
        ],
        "fmaps_to_explore": [-1, 0, 1],
        "franges_to_explore": [4, 8, 16],
        # HLS directives
        "pipeline_ii": 1,  # Initiation interval
        "unroll_factor": [1, 2, 4, 8],
        "array_partition": True,
        # Data types
        "use_fixed_point": False,
        "fixed_point_bits": 16,
        # Compilation
        "generate_hls": True,
        "target_frequency_mhz": 300,
    }


def get_fpga_info() -> Dict[str, Any]:
    """Get FPGA device information."""
    # Try Xilinx XRT
    try:
        result = subprocess.run(["xbutil", "examine"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            output = result.stdout
            device_info = {"available": True, "vendor": FPGAVendor.XILINX}

            if "U280" in output:
                device_info["device"] = FPGADevice.ALVEO_U280
                device_info["device_name"] = "Xilinx Alveo U280"
            elif "U50" in output:
                device_info["device"] = FPGADevice.ALVEO_U50
                device_info["device_name"] = "Xilinx Alveo U50"
            elif "Versal" in output:
                device_info["device"] = FPGADevice.VERSAL_AI
                device_info["device_name"] = "Xilinx Versal AI"
            else:
                device_info["device_name"] = "Xilinx FPGA"

            return device_info
    except:
        pass

    # Try Intel FPGA
    try:
        result = subprocess.run(["aocl", "diagnose"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return {
                "available": True,
                "vendor": FPGAVendor.INTEL,
                "device_name": "Intel FPGA",
            }
    except:
        pass

    return {"available": False}


def is_fpga_available() -> bool:
    """Check if FPGA is available."""
    info = get_fpga_info()
    return info.get("available", False)
