"""
Google TPU-specific search configuration
Optimized for TPU v2/v3/v4/v5e/v5p

Based on TPU specifications:
- MXU (Matrix Multiply Unit): 128x128
- VMEM: 16-32MB per core
- BF16 optimized
- XLA/Pallas kernel generation
"""

import os
import multiprocessing
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum


class TPUVersion(str, Enum):
    """TPU version types."""

    V2 = "v2"
    V3 = "v3"
    V4 = "v4"
    V5E = "v5e"
    V5P = "v5p"
    UNKNOWN = "unknown"


@dataclass
class TPUSpecs:
    """TPU version specifications."""

    version: TPUVersion
    mxu_size: int = 128
    vmem_mb: int = 16
    cmem_mb: int = 4
    bf16_tflops: float = 45.0
    int8_tops: float = 90.0
    hbm_gb: int = 8


TPU_SPECS: Dict[TPUVersion, TPUSpecs] = {
    TPUVersion.V2: TPUSpecs(version=TPUVersion.V2, vmem_mb=8, bf16_tflops=45, hbm_gb=8),
    TPUVersion.V3: TPUSpecs(version=TPUVersion.V3, vmem_mb=16, bf16_tflops=90, hbm_gb=16),
    TPUVersion.V4: TPUSpecs(version=TPUVersion.V4, vmem_mb=32, bf16_tflops=275, hbm_gb=32),
    TPUVersion.V5E: TPUSpecs(version=TPUVersion.V5E, vmem_mb=32, bf16_tflops=197, hbm_gb=16),
    TPUVersion.V5P: TPUSpecs(version=TPUVersion.V5P, vmem_mb=64, bf16_tflops=459, hbm_gb=95),
}


def get_tpu_search_config(version: TPUVersion = None) -> Dict[str, Any]:
    """
    Get optimized search configuration for TPU backend.

    TPU characteristics:
    - MXU: 128x128 systolic array
    - BF16 native support
    - XLA compilation
    - Mesh parallelism via ICI

    Returns:
        dict: Search configuration optimized for TPU
    """
    if version is None:
        version = TPUVersion.V4

    specs = TPU_SPECS.get(version, TPU_SPECS[TPUVersion.V4])

    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))

    return {
        # TPU info
        "version": version.value,
        "mxu_size": specs.mxu_size,
        "vmem_mb": specs.vmem_mb,
        "bf16_tflops": specs.bf16_tflops,
        # Search parameters
        "max_num_threadblock_graph_op": 6,
        "max_num_kernel_graph_op": 4,
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        # Search space (TPU uses MXU-sized tiles)
        "grid_dims_to_explore": [
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
        ],
        "block_dims_to_explore": [
            (128, 1, 1),
            (256, 1, 1),
            (512, 1, 1),
        ],
        "fmaps_to_explore": [-1, 0, 1],
        "franges_to_explore": [4, 8, 16],
        # MXU-aligned tiling
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 128,
        # XLA/Pallas options
        "generate_xla": True,
        "generate_pallas": False,
    }


def get_tpu_info() -> Dict[str, Any]:
    """Get TPU information."""
    try:
        import jax

        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == "tpu"]
        if tpu_devices:
            return {
                "available": True,
                "num_devices": len(tpu_devices),
                "device_kind": str(tpu_devices[0].device_kind),
            }
    except:
        pass

    return {"available": False}


def is_tpu_available() -> bool:
    """Check if TPU is available."""
    info = get_tpu_info()
    return info.get("available", False)
