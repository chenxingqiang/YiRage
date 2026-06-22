"""
NVIDIA CUDA-specific search configuration
Optimized search space for NVIDIA GPUs (Volta, Turing, Ampere, Hopper, Blackwell)

Based on NVIDIA CUDA specifications:
- warpSize: 32
- Shared memory: 48KB-228KB per SM (architecture dependent)
- Tensor Cores: Available on Volta+ (sm70+)
- L2 Cache: Architecture dependent
"""

import os
import subprocess
import multiprocessing
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from enum import Enum


class CUDAArch(str, Enum):
    """NVIDIA GPU architectures."""

    VOLTA = "sm_70"  # V100
    TURING = "sm_75"  # RTX 20xx, T4
    AMPERE = "sm_80"  # A100
    AMPERE_CONSUMER = "sm_86"  # RTX 30xx
    ADA = "sm_89"  # RTX 40xx
    HOPPER = "sm_90"  # H100
    BLACKWELL = "sm_100"  # B100, B200
    UNKNOWN = "unknown"


@dataclass
class CUDAArchSpecs:
    """CUDA architecture specifications."""

    arch: CUDAArch
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_threads_per_sm: int = 2048
    max_blocks_per_sm: int = 32
    shared_mem_per_sm_kb: int = 96
    shared_mem_per_block_kb: int = 48
    l2_cache_mb: float = 6.0
    has_tensor_cores: bool = True
    tensor_core_gen: int = 1  # 1=Volta, 2=Turing, 3=Ampere, 4=Hopper, 5=Blackwell
    has_tma: bool = False  # Tensor Memory Accelerator (Hopper+)
    has_fp8: bool = False  # FP8 support (Hopper+)


CUDA_ARCH_SPECS: Dict[CUDAArch, CUDAArchSpecs] = {
    CUDAArch.VOLTA: CUDAArchSpecs(
        arch=CUDAArch.VOLTA,
        shared_mem_per_sm_kb=96,
        shared_mem_per_block_kb=48,
        l2_cache_mb=6.0,
        tensor_core_gen=1,
    ),
    CUDAArch.TURING: CUDAArchSpecs(
        arch=CUDAArch.TURING,
        shared_mem_per_sm_kb=64,
        shared_mem_per_block_kb=64,
        l2_cache_mb=6.0,
        tensor_core_gen=2,
    ),
    CUDAArch.AMPERE: CUDAArchSpecs(
        arch=CUDAArch.AMPERE,
        shared_mem_per_sm_kb=164,
        shared_mem_per_block_kb=164,
        l2_cache_mb=40.0,
        tensor_core_gen=3,
    ),
    CUDAArch.AMPERE_CONSUMER: CUDAArchSpecs(
        arch=CUDAArch.AMPERE_CONSUMER,
        shared_mem_per_sm_kb=100,
        shared_mem_per_block_kb=100,
        l2_cache_mb=6.0,
        tensor_core_gen=3,
    ),
    CUDAArch.ADA: CUDAArchSpecs(
        arch=CUDAArch.ADA,
        shared_mem_per_sm_kb=100,
        shared_mem_per_block_kb=100,
        l2_cache_mb=72.0,
        tensor_core_gen=4,
        has_fp8=True,
    ),
    CUDAArch.HOPPER: CUDAArchSpecs(
        arch=CUDAArch.HOPPER,
        shared_mem_per_sm_kb=228,
        shared_mem_per_block_kb=228,
        l2_cache_mb=50.0,
        tensor_core_gen=4,
        has_tma=True,
        has_fp8=True,
    ),
    CUDAArch.BLACKWELL: CUDAArchSpecs(
        arch=CUDAArch.BLACKWELL,
        shared_mem_per_sm_kb=256,
        shared_mem_per_block_kb=256,
        l2_cache_mb=64.0,
        tensor_core_gen=5,
        has_tma=True,
        has_fp8=True,
    ),
}


def get_cuda_search_config(arch: CUDAArch = None) -> Dict[str, Any]:
    """
    Get optimized search configuration for CUDA backend.

    Args:
        arch: Optional specific architecture. Auto-detected if None.

    Returns:
        dict: Search configuration optimized for the detected/specified arch
    """
    if arch is None:
        device_info = get_cuda_device_info()
        if device_info and device_info.get("available"):
            arch = device_info.get("arch", CUDAArch.AMPERE)
        else:
            arch = CUDAArch.AMPERE

    specs = CUDA_ARCH_SPECS.get(arch, CUDA_ARCH_SPECS[CUDAArch.AMPERE])

    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))

    # Architecture-dependent configurations
    if specs.has_tma:  # Hopper+
        grid_dims = [
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
            (16, 1, 1),
            (32, 1, 1),
            (64, 1, 1),
            (128, 1, 1),
            (4, 4, 1),
            (8, 8, 1),
            (16, 8, 1),
        ]
        block_dims = [
            (128, 1, 1),
            (256, 1, 1),
            (384, 1, 1),
            (512, 1, 1),
            (768, 1, 1),
            (1024, 1, 1),
        ]
        franges = [4, 8, 16, 32, 64]
    elif specs.tensor_core_gen >= 3:  # Ampere+
        grid_dims = [
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
            (16, 1, 1),
            (32, 1, 1),
            (64, 1, 1),
            (108, 1, 1),
            (4, 4, 1),
            (8, 8, 1),
        ]
        block_dims = [
            (128, 1, 1),
            (256, 1, 1),
            (384, 1, 1),
            (512, 1, 1),
            (768, 1, 1),
            (1024, 1, 1),
        ]
        franges = [4, 8, 16, 32]
    else:  # Volta/Turing
        grid_dims = [
            (1, 1, 1),
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
            (16, 1, 1),
            (32, 1, 1),
            (64, 1, 1),
            (4, 4, 1),
            (8, 8, 1),
        ]
        block_dims = [
            (128, 1, 1),
            (256, 1, 1),
            (512, 1, 1),
        ]
        franges = [4, 8, 16]

    return {
        # Architecture info
        "arch": arch.value,
        "warp_size": specs.warp_size,
        "has_tensor_cores": specs.has_tensor_cores,
        "has_tma": specs.has_tma,
        "has_fp8": specs.has_fp8,
        # Search parameters
        "max_num_threadblock_graph_op": 8,
        "max_num_kernel_graph_op": 5,
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        # Search space
        "grid_dims_to_explore": grid_dims,
        "block_dims_to_explore": block_dims,
        "fmaps_to_explore": [-1, 0, 1, 2],
        "franges_to_explore": franges,
        # Hardware limits
        "max_threads_per_block": specs.max_threads_per_block,
        "shared_mem_per_block_kb": specs.shared_mem_per_block_kb,
    }


def get_cuda_memory_config() -> Dict[str, Any]:
    """Get CUDA memory configuration."""
    device_info = get_cuda_device_info()

    if device_info and device_info.get("available"):
        return {
            "device_name": device_info.get("device_name", "Unknown"),
            "total_memory_gb": device_info.get("total_memory_gb", 0),
            "free_memory_gb": device_info.get("free_memory_gb", 0),
            "sm_count": device_info.get("sm_count", 0),
            "note": f"{device_info.get('device_name', 'GPU')} detected",
        }

    return {"device_name": "Unknown", "total_memory_gb": 0, "note": "CUDA device not detected"}


def get_cuda_device_info() -> Optional[Dict[str, Any]]:
    """Detect CUDA device via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if lines:
                parts = lines[0].split(", ")
                if len(parts) >= 4:
                    device_name = parts[0].strip()
                    total_mem = float(parts[1]) / 1024  # MB to GB
                    free_mem = float(parts[2]) / 1024
                    compute_cap = parts[3].strip()

                    # Determine architecture
                    cc = float(compute_cap)
                    if cc >= 10.0:
                        arch = CUDAArch.BLACKWELL
                    elif cc >= 9.0:
                        arch = CUDAArch.HOPPER
                    elif cc >= 8.9:
                        arch = CUDAArch.ADA
                    elif cc >= 8.6:
                        arch = CUDAArch.AMPERE_CONSUMER
                    elif cc >= 8.0:
                        arch = CUDAArch.AMPERE
                    elif cc >= 7.5:
                        arch = CUDAArch.TURING
                    elif cc >= 7.0:
                        arch = CUDAArch.VOLTA
                    else:
                        arch = CUDAArch.UNKNOWN

                    return {
                        "available": True,
                        "device_name": device_name,
                        "total_memory_gb": total_mem,
                        "free_memory_gb": free_mem,
                        "compute_capability": compute_cap,
                        "arch": arch,
                    }
    except:
        pass

    return None


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    device_info = get_cuda_device_info()
    return device_info is not None and device_info.get("available", False)
