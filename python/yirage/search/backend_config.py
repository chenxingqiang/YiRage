# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Backend-specific COMET configurations.

Each hardware backend has different characteristics that affect
COMET's cost model and search space.
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional
from .comet_search import COMETSearchConfig, SchedulingStrategy


@dataclass
class BackendHardwareProfile:
    """Hardware profile for a specific backend."""
    
    name: str
    dram_bandwidth_gbps: float
    onchip_bandwidth_gbps: float
    peak_tflops: float
    noc_bandwidth_gbps: float
    tile_sizes: list
    scheduling_strategies: list
    max_fusion_depth: int
    supports_collectives: bool


# =============================================================================
# Backend Hardware Profiles
# =============================================================================

BACKEND_PROFILES: Dict[str, BackendHardwareProfile] = {
    # NVIDIA GPUs
    "cuda_h100": BackendHardwareProfile(
        name="NVIDIA H100",
        dram_bandwidth_gbps=3350.0,
        onchip_bandwidth_gbps=33000.0,
        peak_tflops=989.0,
        noc_bandwidth_gbps=900.0,
        tile_sizes=[64, 128, 256],
        scheduling_strategies=[SchedulingStrategy.PIPELINED, SchedulingStrategy.PARALLEL],
        max_fusion_depth=6,
        supports_collectives=True,
    ),
    "cuda_a100": BackendHardwareProfile(
        name="NVIDIA A100",
        dram_bandwidth_gbps=2039.0,
        onchip_bandwidth_gbps=19000.0,
        peak_tflops=312.0,
        noc_bandwidth_gbps=600.0,
        tile_sizes=[64, 128, 256],
        scheduling_strategies=[SchedulingStrategy.PIPELINED, SchedulingStrategy.PARALLEL],
        max_fusion_depth=6,
        supports_collectives=True,
    ),
    "cuda_v100": BackendHardwareProfile(
        name="NVIDIA V100",
        dram_bandwidth_gbps=900.0,
        onchip_bandwidth_gbps=12000.0,
        peak_tflops=125.0,
        noc_bandwidth_gbps=300.0,
        tile_sizes=[32, 64, 128],
        scheduling_strategies=[SchedulingStrategy.PIPELINED, SchedulingStrategy.PARALLEL],
        max_fusion_depth=5,
        supports_collectives=True,
    ),
    
    # AMD GPUs
    "rocm_mi300x": BackendHardwareProfile(
        name="AMD MI300X",
        dram_bandwidth_gbps=5300.0,
        onchip_bandwidth_gbps=25000.0,
        peak_tflops=1307.0,
        noc_bandwidth_gbps=896.0,
        tile_sizes=[64, 128, 256],
        scheduling_strategies=[SchedulingStrategy.PIPELINED, SchedulingStrategy.PARALLEL],
        max_fusion_depth=6,
        supports_collectives=True,
    ),
    "rocm_mi250x": BackendHardwareProfile(
        name="AMD MI250X",
        dram_bandwidth_gbps=3200.0,
        onchip_bandwidth_gbps=20000.0,
        peak_tflops=383.0,
        noc_bandwidth_gbps=400.0,
        tile_sizes=[64, 128, 256],
        scheduling_strategies=[SchedulingStrategy.PIPELINED, SchedulingStrategy.PARALLEL],
        max_fusion_depth=5,
        supports_collectives=True,
    ),
    
    # Intel GPUs
    "xpu_pvc": BackendHardwareProfile(
        name="Intel Ponte Vecchio",
        dram_bandwidth_gbps=3200.0,
        onchip_bandwidth_gbps=10000.0,
        peak_tflops=420.0,
        noc_bandwidth_gbps=200.0,
        tile_sizes=[32, 64, 128, 256],
        scheduling_strategies=[SchedulingStrategy.PIPELINED, SchedulingStrategy.PARALLEL],
        max_fusion_depth=5,
        supports_collectives=True,
    ),
    
    # Huawei Ascend
    "ascend_910b": BackendHardwareProfile(
        name="Huawei Ascend 910B",
        dram_bandwidth_gbps=1600.0,
        onchip_bandwidth_gbps=12000.0,
        peak_tflops=320.0,
        noc_bandwidth_gbps=392.0,
        tile_sizes=[64, 128, 256],
        scheduling_strategies=[SchedulingStrategy.PIPELINED, SchedulingStrategy.SEQUENTIAL],
        max_fusion_depth=5,
        supports_collectives=True,
    ),
    
    # Google TPU
    "tpu_v5e": BackendHardwareProfile(
        name="Google TPU v5e",
        dram_bandwidth_gbps=1600.0,
        onchip_bandwidth_gbps=20000.0,
        peak_tflops=197.0,
        noc_bandwidth_gbps=1600.0,
        tile_sizes=[128, 256, 512],
        scheduling_strategies=[SchedulingStrategy.PIPELINED],
        max_fusion_depth=8,
        supports_collectives=True,
    ),
    "tpu_v4": BackendHardwareProfile(
        name="Google TPU v4",
        dram_bandwidth_gbps=1200.0,
        onchip_bandwidth_gbps=15000.0,
        peak_tflops=275.0,
        noc_bandwidth_gbps=1200.0,
        tile_sizes=[128, 256, 512],
        scheduling_strategies=[SchedulingStrategy.PIPELINED],
        max_fusion_depth=8,
        supports_collectives=True,
    ),
    
    # MetaX MACA
    "maca_mxc500": BackendHardwareProfile(
        name="MetaX MXC500",
        dram_bandwidth_gbps=2000.0,
        onchip_bandwidth_gbps=15000.0,
        peak_tflops=256.0,
        noc_bandwidth_gbps=400.0,
        tile_sizes=[64, 128, 256],
        scheduling_strategies=[SchedulingStrategy.PIPELINED, SchedulingStrategy.PARALLEL],
        max_fusion_depth=5,
        supports_collectives=True,
    ),
    
    # Apple Silicon
    "mps_m3_max": BackendHardwareProfile(
        name="Apple M3 Max",
        dram_bandwidth_gbps=400.0,
        onchip_bandwidth_gbps=2000.0,
        peak_tflops=14.2,
        noc_bandwidth_gbps=400.0,
        tile_sizes=[32, 64, 128],
        scheduling_strategies=[SchedulingStrategy.SEQUENTIAL, SchedulingStrategy.PIPELINED],
        max_fusion_depth=4,
        supports_collectives=False,
    ),
    "mps_m2_ultra": BackendHardwareProfile(
        name="Apple M2 Ultra",
        dram_bandwidth_gbps=800.0,
        onchip_bandwidth_gbps=3000.0,
        peak_tflops=27.2,
        noc_bandwidth_gbps=800.0,
        tile_sizes=[32, 64, 128],
        scheduling_strategies=[SchedulingStrategy.SEQUENTIAL, SchedulingStrategy.PIPELINED],
        max_fusion_depth=4,
        supports_collectives=False,
    ),
    
    # CPU
    "cpu_xeon": BackendHardwareProfile(
        name="Intel Xeon (Server)",
        dram_bandwidth_gbps=200.0,
        onchip_bandwidth_gbps=1000.0,
        peak_tflops=4.0,
        noc_bandwidth_gbps=50.0,
        tile_sizes=[32, 64, 128, 256],
        scheduling_strategies=[SchedulingStrategy.SEQUENTIAL, SchedulingStrategy.PARALLEL],
        max_fusion_depth=4,
        supports_collectives=False,
    ),
    "cpu_epyc": BackendHardwareProfile(
        name="AMD EPYC (Server)",
        dram_bandwidth_gbps=460.0,
        onchip_bandwidth_gbps=1200.0,
        peak_tflops=5.0,
        noc_bandwidth_gbps=100.0,
        tile_sizes=[32, 64, 128, 256],
        scheduling_strategies=[SchedulingStrategy.SEQUENTIAL, SchedulingStrategy.PARALLEL],
        max_fusion_depth=4,
        supports_collectives=False,
    ),
    
    # FPGA
    "fpga_alveo": BackendHardwareProfile(
        name="Xilinx Alveo U250",
        dram_bandwidth_gbps=77.0,
        onchip_bandwidth_gbps=500.0,
        peak_tflops=4.0,
        noc_bandwidth_gbps=100.0,
        tile_sizes=[16, 32, 64, 128],
        scheduling_strategies=[SchedulingStrategy.PIPELINED],
        max_fusion_depth=10,
        supports_collectives=False,
    ),
}


def get_backend_config(
    backend: str,
    variant: Optional[str] = None
) -> COMETSearchConfig:
    """
    Get COMET configuration for a specific backend.
    
    Args:
        backend: Backend name (cuda, rocm, xpu, ascend, tpu, maca, mps, cpu, fpga)
        variant: Optional variant (e.g., "h100", "a100", "mi300x")
        
    Returns:
        COMETSearchConfig optimized for the backend
    """
    # Build profile key
    if variant:
        key = f"{backend}_{variant}"
    else:
        # Default variants
        defaults = {
            "cuda": "cuda_a100",
            "rocm": "rocm_mi300x",
            "xpu": "xpu_pvc",
            "ascend": "ascend_910b",
            "tpu": "tpu_v5e",
            "maca": "maca_mxc500",
            "mps": "mps_m3_max",
            "cpu": "cpu_xeon",
            "fpga": "fpga_alveo",
        }
        key = defaults.get(backend.lower(), f"{backend}_default")
    
    profile = BACKEND_PROFILES.get(key)
    
    if profile is None:
        # Return default config
        return COMETSearchConfig()
    
    return COMETSearchConfig(
        dram_bandwidth_gbps=profile.dram_bandwidth_gbps,
        onchip_bandwidth_gbps=profile.onchip_bandwidth_gbps,
        peak_tflops=profile.peak_tflops,
        noc_bandwidth_gbps=profile.noc_bandwidth_gbps,
        tile_sizes=profile.tile_sizes,
        scheduling_options=profile.scheduling_strategies,
        max_fusion_depth=profile.max_fusion_depth,
        optimize_collectives=profile.supports_collectives,
    )


def get_auto_detected_config() -> COMETSearchConfig:
    """
    Auto-detect hardware and return appropriate COMET config.
    
    Returns:
        COMETSearchConfig for detected hardware
    """
    # Try to detect CUDA
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cc = props.major * 10 + props.minor
            
            if cc >= 90:
                return get_backend_config("cuda", "h100")
            elif cc >= 80:
                return get_backend_config("cuda", "a100")
            else:
                return get_backend_config("cuda", "v100")
    except ImportError:
        pass
    
    # Try to detect ROCm
    try:
        import torch
        if hasattr(torch, 'hip') and torch.cuda.is_available():
            return get_backend_config("rocm", "mi300x")
    except ImportError:
        pass
    
    # Try to detect MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return get_backend_config("mps", "m3_max")
    except ImportError:
        pass
    
    # Default to CPU
    return get_backend_config("cpu")


def list_supported_backends() -> Dict[str, list]:
    """
    List all supported backends and their variants.
    
    Returns:
        Dictionary mapping backend names to list of variants
    """
    backends = {}
    for key in BACKEND_PROFILES.keys():
        parts = key.split("_", 1)
        backend = parts[0]
        variant = parts[1] if len(parts) > 1 else "default"
        
        if backend not in backends:
            backends[backend] = []
        backends[backend].append(variant)
    
    return backends
