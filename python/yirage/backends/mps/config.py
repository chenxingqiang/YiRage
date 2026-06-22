"""
MPS Configuration for Apple Silicon (M1 - M5 Series)

Comprehensive configuration for all Apple Silicon chips with Metal Performance Shaders.

References:
- Apple Metal Feature Set Tables: https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
- Apple Silicon specifications from official Apple documentation

Metal API Specifications (All M-series):
- SIMD Width: 32 threads per SIMD group
- Max Threads per Threadgroup: 1024
- Max Threadgroup Memory: 32 KB
- Threadgroup Memory Alignment: 16 bytes
- Unified Memory Architecture (UMA)
"""

import os
import subprocess
import multiprocessing
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


# =============================================================================
# Apple Silicon Chip Specifications
# =============================================================================


class AppleChipFamily(str, Enum):
    """Apple Silicon chip families."""

    M1 = "M1"
    M1_PRO = "M1 Pro"
    M1_MAX = "M1 Max"
    M1_ULTRA = "M1 Ultra"
    M2 = "M2"
    M2_PRO = "M2 Pro"
    M2_MAX = "M2 Max"
    M2_ULTRA = "M2 Ultra"
    M3 = "M3"
    M3_PRO = "M3 Pro"
    M3_MAX = "M3 Max"
    M3_ULTRA = "M3 Ultra"
    M4 = "M4"
    M4_PRO = "M4 Pro"
    M4_MAX = "M4 Max"
    M5 = "M5"
    M5_PRO = "M5 Pro"
    M5_MAX = "M5 Max"
    M5_ULTRA = "M5 Ultra"
    UNKNOWN = "Unknown"


@dataclass
class AppleSiliconSpecs:
    """
    Detailed specifications for Apple Silicon chips.

    Data compiled from Apple official specifications and benchmarks.
    """

    # Identification
    family: AppleChipFamily = AppleChipFamily.UNKNOWN
    chip_name: str = ""

    # CPU specifications
    cpu_cores_performance: int = 0
    cpu_cores_efficiency: int = 0
    cpu_cores_total: int = 0

    # GPU specifications
    gpu_cores: int = 0
    gpu_cores_min: int = 0  # For chips with configurable cores
    gpu_cores_max: int = 0

    # Memory specifications
    memory_bandwidth_gbps: float = 0.0
    max_memory_gb: int = 0
    memory_configurations: List[int] = field(default_factory=list)

    # Neural Engine
    neural_engine_cores: int = 0
    neural_engine_tops: float = 0.0  # Trillion operations per second

    # Metal specifications
    simd_width: int = 32
    max_threads_per_threadgroup: int = 1024
    threadgroup_memory_kb: int = 32

    # Process node
    process_nm: int = 0

    # Features
    has_ray_tracing: bool = False
    has_mesh_shading: bool = False
    has_dynamic_caching: bool = False

    # Performance estimates (relative to M1 base = 1.0)
    gpu_performance_factor: float = 1.0

    def get_optimal_threadgroup_size(self) -> int:
        """Get optimal threadgroup size based on GPU cores."""
        # Base: 256 threads, scale with GPU cores
        if self.gpu_cores >= 60:
            return 512  # Ultra chips
        elif self.gpu_cores >= 30:
            return 384  # Max chips
        elif self.gpu_cores >= 14:
            return 256  # Pro chips
        else:
            return 192  # Base chips

    def get_optimal_grid_size(self) -> int:
        """Get optimal grid size for parallelism."""
        # Aim for ~4x occupancy per GPU core
        return min(self.gpu_cores * 4, 256)


# Complete specifications database
APPLE_SILICON_SPECS: Dict[AppleChipFamily, AppleSiliconSpecs] = {
    # ===================
    # M1 Series (2020)
    # ===================
    AppleChipFamily.M1: AppleSiliconSpecs(
        family=AppleChipFamily.M1,
        chip_name="Apple M1",
        cpu_cores_performance=4,
        cpu_cores_efficiency=4,
        cpu_cores_total=8,
        gpu_cores=8,
        gpu_cores_min=7,
        gpu_cores_max=8,
        memory_bandwidth_gbps=68.25,
        max_memory_gb=16,
        memory_configurations=[8, 16],
        neural_engine_cores=16,
        neural_engine_tops=11.0,
        process_nm=5,
        gpu_performance_factor=1.0,
    ),
    AppleChipFamily.M1_PRO: AppleSiliconSpecs(
        family=AppleChipFamily.M1_PRO,
        chip_name="Apple M1 Pro",
        cpu_cores_performance=8,
        cpu_cores_efficiency=2,
        cpu_cores_total=10,
        gpu_cores=16,
        gpu_cores_min=14,
        gpu_cores_max=16,
        memory_bandwidth_gbps=200.0,
        max_memory_gb=32,
        memory_configurations=[16, 32],
        neural_engine_cores=16,
        neural_engine_tops=11.0,
        process_nm=5,
        gpu_performance_factor=2.0,
    ),
    AppleChipFamily.M1_MAX: AppleSiliconSpecs(
        family=AppleChipFamily.M1_MAX,
        chip_name="Apple M1 Max",
        cpu_cores_performance=8,
        cpu_cores_efficiency=2,
        cpu_cores_total=10,
        gpu_cores=32,
        gpu_cores_min=24,
        gpu_cores_max=32,
        memory_bandwidth_gbps=400.0,
        max_memory_gb=64,
        memory_configurations=[32, 64],
        neural_engine_cores=16,
        neural_engine_tops=11.0,
        process_nm=5,
        gpu_performance_factor=4.0,
    ),
    AppleChipFamily.M1_ULTRA: AppleSiliconSpecs(
        family=AppleChipFamily.M1_ULTRA,
        chip_name="Apple M1 Ultra",
        cpu_cores_performance=16,
        cpu_cores_efficiency=4,
        cpu_cores_total=20,
        gpu_cores=64,
        gpu_cores_min=48,
        gpu_cores_max=64,
        memory_bandwidth_gbps=800.0,
        max_memory_gb=128,
        memory_configurations=[64, 128],
        neural_engine_cores=32,
        neural_engine_tops=22.0,
        process_nm=5,
        gpu_performance_factor=8.0,
    ),
    # ===================
    # M2 Series (2022)
    # ===================
    AppleChipFamily.M2: AppleSiliconSpecs(
        family=AppleChipFamily.M2,
        chip_name="Apple M2",
        cpu_cores_performance=4,
        cpu_cores_efficiency=4,
        cpu_cores_total=8,
        gpu_cores=10,
        gpu_cores_min=8,
        gpu_cores_max=10,
        memory_bandwidth_gbps=100.0,
        max_memory_gb=24,
        memory_configurations=[8, 16, 24],
        neural_engine_cores=16,
        neural_engine_tops=15.8,
        process_nm=5,  # Second-gen 5nm
        gpu_performance_factor=1.25,
    ),
    AppleChipFamily.M2_PRO: AppleSiliconSpecs(
        family=AppleChipFamily.M2_PRO,
        chip_name="Apple M2 Pro",
        cpu_cores_performance=8,
        cpu_cores_efficiency=4,
        cpu_cores_total=12,
        gpu_cores=19,
        gpu_cores_min=16,
        gpu_cores_max=19,
        memory_bandwidth_gbps=200.0,
        max_memory_gb=32,
        memory_configurations=[16, 32],
        neural_engine_cores=16,
        neural_engine_tops=15.8,
        process_nm=5,
        gpu_performance_factor=2.4,
    ),
    AppleChipFamily.M2_MAX: AppleSiliconSpecs(
        family=AppleChipFamily.M2_MAX,
        chip_name="Apple M2 Max",
        cpu_cores_performance=8,
        cpu_cores_efficiency=4,
        cpu_cores_total=12,
        gpu_cores=38,
        gpu_cores_min=30,
        gpu_cores_max=38,
        memory_bandwidth_gbps=400.0,
        max_memory_gb=96,
        memory_configurations=[32, 64, 96],
        neural_engine_cores=16,
        neural_engine_tops=15.8,
        process_nm=5,
        gpu_performance_factor=4.75,
    ),
    AppleChipFamily.M2_ULTRA: AppleSiliconSpecs(
        family=AppleChipFamily.M2_ULTRA,
        chip_name="Apple M2 Ultra",
        cpu_cores_performance=16,
        cpu_cores_efficiency=8,
        cpu_cores_total=24,
        gpu_cores=76,
        gpu_cores_min=60,
        gpu_cores_max=76,
        memory_bandwidth_gbps=800.0,
        max_memory_gb=192,
        memory_configurations=[64, 128, 192],
        neural_engine_cores=32,
        neural_engine_tops=31.6,
        process_nm=5,
        gpu_performance_factor=9.5,
    ),
    # ===================
    # M3 Series (2023)
    # ===================
    AppleChipFamily.M3: AppleSiliconSpecs(
        family=AppleChipFamily.M3,
        chip_name="Apple M3",
        cpu_cores_performance=4,
        cpu_cores_efficiency=4,
        cpu_cores_total=8,
        gpu_cores=10,
        gpu_cores_min=8,
        gpu_cores_max=10,
        memory_bandwidth_gbps=100.0,
        max_memory_gb=24,
        memory_configurations=[8, 16, 24],
        neural_engine_cores=16,
        neural_engine_tops=18.0,
        process_nm=3,
        has_ray_tracing=True,
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=1.5,
    ),
    AppleChipFamily.M3_PRO: AppleSiliconSpecs(
        family=AppleChipFamily.M3_PRO,
        chip_name="Apple M3 Pro",
        cpu_cores_performance=6,
        cpu_cores_efficiency=6,
        cpu_cores_total=12,
        gpu_cores=18,
        gpu_cores_min=14,
        gpu_cores_max=18,
        memory_bandwidth_gbps=150.0,
        max_memory_gb=36,
        memory_configurations=[18, 36],
        neural_engine_cores=16,
        neural_engine_tops=18.0,
        process_nm=3,
        has_ray_tracing=True,
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=2.7,
    ),
    AppleChipFamily.M3_MAX: AppleSiliconSpecs(
        family=AppleChipFamily.M3_MAX,
        chip_name="Apple M3 Max",
        cpu_cores_performance=12,
        cpu_cores_efficiency=4,
        cpu_cores_total=16,
        gpu_cores=40,
        gpu_cores_min=30,
        gpu_cores_max=40,
        memory_bandwidth_gbps=400.0,
        max_memory_gb=128,
        memory_configurations=[36, 48, 64, 96, 128],
        neural_engine_cores=16,
        neural_engine_tops=18.0,
        process_nm=3,
        has_ray_tracing=True,
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=6.0,
    ),
    AppleChipFamily.M3_ULTRA: AppleSiliconSpecs(
        family=AppleChipFamily.M3_ULTRA,
        chip_name="Apple M3 Ultra",
        cpu_cores_performance=24,
        cpu_cores_efficiency=8,
        cpu_cores_total=32,
        gpu_cores=80,
        gpu_cores_min=60,
        gpu_cores_max=80,
        memory_bandwidth_gbps=819.0,
        max_memory_gb=512,
        memory_configurations=[128, 192, 256, 384, 512],
        neural_engine_cores=32,
        neural_engine_tops=36.0,
        process_nm=3,
        has_ray_tracing=True,
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=12.0,
    ),
    # ===================
    # M4 Series (2024)
    # ===================
    AppleChipFamily.M4: AppleSiliconSpecs(
        family=AppleChipFamily.M4,
        chip_name="Apple M4",
        cpu_cores_performance=4,
        cpu_cores_efficiency=6,
        cpu_cores_total=10,
        gpu_cores=10,
        gpu_cores_min=10,
        gpu_cores_max=10,
        memory_bandwidth_gbps=120.0,
        max_memory_gb=32,
        memory_configurations=[16, 24, 32],
        neural_engine_cores=16,
        neural_engine_tops=38.0,
        process_nm=3,  # Second-gen 3nm
        has_ray_tracing=True,
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=1.85,
    ),
    AppleChipFamily.M4_PRO: AppleSiliconSpecs(
        family=AppleChipFamily.M4_PRO,
        chip_name="Apple M4 Pro",
        cpu_cores_performance=10,
        cpu_cores_efficiency=4,
        cpu_cores_total=14,
        gpu_cores=20,
        gpu_cores_min=16,
        gpu_cores_max=20,
        memory_bandwidth_gbps=273.0,
        max_memory_gb=64,
        memory_configurations=[24, 48, 64],
        neural_engine_cores=16,
        neural_engine_tops=38.0,
        process_nm=3,
        has_ray_tracing=True,
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=3.7,
    ),
    AppleChipFamily.M4_MAX: AppleSiliconSpecs(
        family=AppleChipFamily.M4_MAX,
        chip_name="Apple M4 Max",
        cpu_cores_performance=12,
        cpu_cores_efficiency=4,
        cpu_cores_total=16,
        gpu_cores=40,
        gpu_cores_min=32,
        gpu_cores_max=40,
        memory_bandwidth_gbps=546.0,
        max_memory_gb=128,
        memory_configurations=[36, 48, 64, 96, 128],
        neural_engine_cores=16,
        neural_engine_tops=38.0,
        process_nm=3,
        has_ray_tracing=True,
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=7.4,
    ),
    # ===================
    # M5 Series (2025)
    # ===================
    AppleChipFamily.M5: AppleSiliconSpecs(
        family=AppleChipFamily.M5,
        chip_name="Apple M5",
        cpu_cores_performance=4,
        cpu_cores_efficiency=6,
        cpu_cores_total=10,
        gpu_cores=10,
        gpu_cores_min=10,
        gpu_cores_max=10,
        memory_bandwidth_gbps=153.0,
        max_memory_gb=32,
        memory_configurations=[16, 24, 32],
        neural_engine_cores=16,
        neural_engine_tops=50.0,  # Estimated
        process_nm=3,  # N3E or N3P
        has_ray_tracing=True,  # 3rd gen
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=2.7,  # ~45% over M4
    ),
    AppleChipFamily.M5_PRO: AppleSiliconSpecs(
        family=AppleChipFamily.M5_PRO,
        chip_name="Apple M5 Pro",
        cpu_cores_performance=10,
        cpu_cores_efficiency=4,
        cpu_cores_total=14,
        gpu_cores=20,
        gpu_cores_min=16,
        gpu_cores_max=20,
        memory_bandwidth_gbps=300.0,  # Estimated
        max_memory_gb=64,
        memory_configurations=[24, 48, 64],
        neural_engine_cores=16,
        neural_engine_tops=50.0,
        process_nm=3,
        has_ray_tracing=True,
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=5.4,
    ),
    AppleChipFamily.M5_MAX: AppleSiliconSpecs(
        family=AppleChipFamily.M5_MAX,
        chip_name="Apple M5 Max",
        cpu_cores_performance=14,
        cpu_cores_efficiency=4,
        cpu_cores_total=18,
        gpu_cores=48,
        gpu_cores_min=36,
        gpu_cores_max=48,
        memory_bandwidth_gbps=700.0,  # Estimated
        max_memory_gb=192,
        memory_configurations=[48, 64, 96, 128, 192],
        neural_engine_cores=16,
        neural_engine_tops=50.0,
        process_nm=3,
        has_ray_tracing=True,
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=10.8,
    ),
    AppleChipFamily.M5_ULTRA: AppleSiliconSpecs(
        family=AppleChipFamily.M5_ULTRA,
        chip_name="Apple M5 Ultra",
        cpu_cores_performance=28,
        cpu_cores_efficiency=8,
        cpu_cores_total=36,
        gpu_cores=96,
        gpu_cores_min=72,
        gpu_cores_max=96,
        memory_bandwidth_gbps=1400.0,  # Estimated
        max_memory_gb=512,
        memory_configurations=[128, 192, 256, 384, 512],
        neural_engine_cores=32,
        neural_engine_tops=100.0,  # Estimated
        process_nm=3,
        has_ray_tracing=True,
        has_mesh_shading=True,
        has_dynamic_caching=True,
        gpu_performance_factor=21.6,
    ),
}

# Default unknown specs
APPLE_SILICON_SPECS[AppleChipFamily.UNKNOWN] = AppleSiliconSpecs(
    family=AppleChipFamily.UNKNOWN,
    chip_name="Unknown Apple Silicon",
    gpu_cores=8,
    memory_bandwidth_gbps=100.0,
    max_memory_gb=16,
    neural_engine_cores=16,
    gpu_performance_factor=1.0,
)


# =============================================================================
# Metal Constants (Common to all M-series)
# =============================================================================

METAL_CONSTANTS = {
    # SIMD configuration
    "simd_width": 32,
    "simd_lanes": 32,
    # Threadgroup limits
    "max_threads_per_threadgroup": 1024,
    "max_threadgroup_memory_bytes": 32 * 1024,  # 32 KB
    "threadgroup_memory_alignment": 16,  # 16 bytes
    # Texture limits
    "max_texture_size_1d": 16384,
    "max_texture_size_2d": 16384,
    "max_texture_size_3d": 2048,
    # Buffer limits
    "max_buffer_length": 256 * 1024 * 1024 * 1024,  # 256 GB (UMA)
    # Argument buffer tier
    "argument_buffer_tier": 2,
    # Read-write texture tier
    "read_write_texture_tier": 2,
}


# =============================================================================
# Chip Detection
# =============================================================================


def detect_apple_silicon() -> Tuple[AppleChipFamily, AppleSiliconSpecs]:
    """
    Detect the current Apple Silicon chip.

    Returns:
        Tuple of (chip_family, specs)
    """
    try:
        # Method 1: Use sysctl
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2
        )
        chip_string = result.stdout.strip()

        # Also get system profiler for more details
        result2 = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, timeout=5
        )
        hw_info = result2.stdout

        # Combine information
        full_info = chip_string + " " + hw_info

        # Detect chip family (check most specific first)
        for family in [
            # M5 series
            AppleChipFamily.M5_ULTRA,
            AppleChipFamily.M5_MAX,
            AppleChipFamily.M5_PRO,
            AppleChipFamily.M5,
            # M4 series
            AppleChipFamily.M4_MAX,
            AppleChipFamily.M4_PRO,
            AppleChipFamily.M4,
            # M3 series
            AppleChipFamily.M3_ULTRA,
            AppleChipFamily.M3_MAX,
            AppleChipFamily.M3_PRO,
            AppleChipFamily.M3,
            # M2 series
            AppleChipFamily.M2_ULTRA,
            AppleChipFamily.M2_MAX,
            AppleChipFamily.M2_PRO,
            AppleChipFamily.M2,
            # M1 series
            AppleChipFamily.M1_ULTRA,
            AppleChipFamily.M1_MAX,
            AppleChipFamily.M1_PRO,
            AppleChipFamily.M1,
        ]:
            # Create search pattern
            family_name = family.value.replace(" ", "")  # e.g., "M1Pro"
            family_name_space = family.value  # e.g., "M1 Pro"

            if family_name in full_info.replace(" ", "") or family_name_space in full_info:
                return family, APPLE_SILICON_SPECS[family]

        # If we detect Apple but can't identify specific chip
        if "Apple" in chip_string:
            return AppleChipFamily.UNKNOWN, APPLE_SILICON_SPECS[AppleChipFamily.UNKNOWN]

    except Exception as e:
        pass

    return AppleChipFamily.UNKNOWN, APPLE_SILICON_SPECS[AppleChipFamily.UNKNOWN]


def get_apple_gpu_info() -> Dict:
    """
    Get comprehensive Apple GPU information.

    Returns:
        dict: GPU information including chip specs and memory
    """
    family, specs = detect_apple_silicon()

    # Get actual system memory
    total_mem_gb = get_system_memory_gb()
    usable_mem_gb = int(total_mem_gb * 0.75) if total_mem_gb else 12

    return {
        "family": family.value,
        "chip_name": specs.chip_name,
        "gpu_cores": specs.gpu_cores,
        "memory_bandwidth_gbps": specs.memory_bandwidth_gbps,
        "total_memory_gb": total_mem_gb or specs.max_memory_gb,
        "usable_memory_gb": usable_mem_gb,
        "neural_engine_tops": specs.neural_engine_tops,
        "process_nm": specs.process_nm,
        "has_ray_tracing": specs.has_ray_tracing,
        "has_mesh_shading": specs.has_mesh_shading,
        "has_dynamic_caching": specs.has_dynamic_caching,
        "gpu_performance_factor": specs.gpu_performance_factor,
        "simd_width": specs.simd_width,
        "max_threads_per_threadgroup": specs.max_threads_per_threadgroup,
        "threadgroup_memory_kb": specs.threadgroup_memory_kb,
    }


# =============================================================================
# Search Configuration
# =============================================================================


def get_mps_search_config(chip_family: AppleChipFamily = None) -> Dict:
    """
    Get optimized search configuration for MPS backend.

    Args:
        chip_family: Optional specific chip family. Auto-detected if None.

    Returns:
        dict: Search configuration optimized for the detected/specified chip
    """
    # Detect chip if not specified
    if chip_family is None:
        chip_family, specs = detect_apple_silicon()
    else:
        specs = APPLE_SILICON_SPECS.get(chip_family, APPLE_SILICON_SPECS[AppleChipFamily.UNKNOWN])

    # CPU cores for search (search is CPU-bound)
    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))

    # Determine optimal configurations based on chip tier
    gpu_cores = specs.gpu_cores

    # Grid dimensions optimized for chip tier
    if gpu_cores >= 60:  # Ultra chips
        grid_dims = [
            (64, 1, 1),
            (128, 1, 1),
            (192, 1, 1),
            (256, 1, 1),
            (384, 1, 1),
            (512, 1, 1),
            (768, 1, 1),
            (1024, 1, 1),
            (64, 2, 1),
            (128, 2, 1),
            (64, 4, 1),
            (128, 4, 1),
        ]
        block_dims = [
            (128, 1, 1),
            (192, 1, 1),
            (256, 1, 1),
            (384, 1, 1),
            (512, 1, 1),
            (768, 1, 1),
            (1024, 1, 1),
        ]
        franges = [4, 8, 16, 32]
    elif gpu_cores >= 30:  # Max chips
        grid_dims = [
            (32, 1, 1),
            (64, 1, 1),
            (128, 1, 1),
            (192, 1, 1),
            (256, 1, 1),
            (384, 1, 1),
            (512, 1, 1),
            (32, 2, 1),
            (64, 2, 1),
            (32, 4, 1),
            (64, 4, 1),
        ]
        block_dims = [
            (64, 1, 1),
            (128, 1, 1),
            (192, 1, 1),
            (256, 1, 1),
            (384, 1, 1),
            (512, 1, 1),
        ]
        franges = [4, 8, 16]
    elif gpu_cores >= 14:  # Pro chips
        grid_dims = [
            (32, 1, 1),
            (64, 1, 1),
            (96, 1, 1),
            (128, 1, 1),
            (192, 1, 1),
            (256, 1, 1),
            (384, 1, 1),
            (32, 2, 1),
            (64, 2, 1),
            (32, 4, 1),
        ]
        block_dims = [
            (64, 1, 1),
            (96, 1, 1),
            (128, 1, 1),
            (192, 1, 1),
            (256, 1, 1),
            (384, 1, 1),
            (512, 1, 1),
        ]
        franges = [4, 8, 16]
    else:  # Base chips (M1, M2, M3, M4, M5)
        grid_dims = [
            (32, 1, 1),
            (64, 1, 1),
            (96, 1, 1),
            (128, 1, 1),
            (160, 1, 1),
            (192, 1, 1),
            (256, 1, 1),
            (32, 2, 1),
            (64, 2, 1),
        ]
        block_dims = [
            (32, 1, 1),
            (64, 1, 1),
            (96, 1, 1),
            (128, 1, 1),
            (160, 1, 1),
            (192, 1, 1),
            (256, 1, 1),
        ]
        franges = [4, 8, 16]

    return {
        # Chip info
        "chip_family": chip_family.value,
        "chip_name": specs.chip_name,
        "gpu_cores": gpu_cores,
        "memory_bandwidth_gbps": specs.memory_bandwidth_gbps,
        # Search parameters
        "max_num_threadblock_graph_op": 6,
        "max_num_kernel_graph_op": 4,
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        # Optimized search space
        "grid_dims_to_explore": grid_dims,
        "block_dims_to_explore": block_dims,
        "fmaps_to_explore": [-1, 0, 1, 2],
        "franges_to_explore": franges,
        # Metal constants
        "simd_width": METAL_CONSTANTS["simd_width"],
        "max_threads_per_threadgroup": METAL_CONSTANTS["max_threads_per_threadgroup"],
        "max_threadgroup_memory_bytes": METAL_CONSTANTS["max_threadgroup_memory_bytes"],
        # Features
        "has_ray_tracing": specs.has_ray_tracing,
        "has_mesh_shading": specs.has_mesh_shading,
        "has_dynamic_caching": specs.has_dynamic_caching,
    }


def get_chip_optimized_config(chip_family: AppleChipFamily) -> Dict:
    """
    Get chip-specific optimized configuration.

    This provides fine-tuned parameters for specific chip generations.

    Args:
        chip_family: The Apple Silicon chip family

    Returns:
        dict: Chip-specific optimizations
    """
    specs = APPLE_SILICON_SPECS.get(chip_family, APPLE_SILICON_SPECS[AppleChipFamily.UNKNOWN])

    config = {
        "chip_family": chip_family.value,
        "chip_name": specs.chip_name,
    }

    # M3+ specific: Dynamic Caching optimizations
    if specs.has_dynamic_caching:
        config.update(
            {
                "use_dynamic_caching": True,
                "register_pressure_threshold": 0.85,  # Higher due to dynamic allocation
                "shared_memory_preference": "adaptive",
            }
        )
    else:
        config.update(
            {
                "use_dynamic_caching": False,
                "register_pressure_threshold": 0.75,
                "shared_memory_preference": "static",
            }
        )

    # M3+ specific: Ray tracing and mesh shading
    if specs.has_ray_tracing:
        config["ray_tracing_available"] = True
    if specs.has_mesh_shading:
        config["mesh_shading_available"] = True

    # Memory bandwidth optimizations
    bw = specs.memory_bandwidth_gbps
    if bw >= 800:  # Ultra chips
        config["prefetch_distance"] = 4
        config["memory_coalescing"] = "aggressive"
    elif bw >= 400:  # Max chips
        config["prefetch_distance"] = 3
        config["memory_coalescing"] = "balanced"
    elif bw >= 200:  # Pro chips
        config["prefetch_distance"] = 2
        config["memory_coalescing"] = "balanced"
    else:  # Base chips
        config["prefetch_distance"] = 1
        config["memory_coalescing"] = "conservative"

    # Neural Engine integration hints
    config["neural_engine_tops"] = specs.neural_engine_tops
    if specs.neural_engine_tops >= 38:  # M4+
        config["offload_to_neural_engine"] = True
    else:
        config["offload_to_neural_engine"] = False

    return config


# =============================================================================
# CPU Search Configuration
# =============================================================================


def get_cpu_search_config() -> Dict:
    """
    Get optimized search configuration for CPU backend.

    Returns:
        dict: Search configuration optimized for CPU execution
    """
    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))

    return {
        "max_num_threadblock_graph_op": 5,
        "max_num_kernel_graph_op": 3,
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        "grid_dims_to_explore": [
            (8, 1, 1),
            (16, 1, 1),
            (32, 1, 1),
            (64, 1, 1),
        ],
        "block_dims_to_explore": [
            (16, 1, 1),
            (32, 1, 1),
            (64, 1, 1),
            (128, 1, 1),
        ],
        "franges_to_explore": [2, 4, 8],
    }


# =============================================================================
# Memory Configuration
# =============================================================================


def get_system_memory_gb() -> Optional[int]:
    """
    Get total system memory in GB.

    Returns:
        int: Total system memory in GB, or None if detection fails
    """
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            mem_bytes = int(result.stdout.strip())
            return mem_bytes // (1024**3)
    except:
        pass

    try:
        import psutil

        return psutil.virtual_memory().total // (1024**3)
    except:
        pass

    return None


def get_mps_memory_config() -> Dict:
    """
    Get MPS memory configuration based on system memory.

    Returns:
        dict: Memory configuration with usable GPU memory
    """
    total_mem_gb = get_system_memory_gb()

    if total_mem_gb is None:
        return {
            "total_gb": 16,
            "usable_gb": 12,
            "note": "Using default estimate (unable to detect system memory)",
        }

    usable_gb = int(total_mem_gb * 0.75)

    return {
        "total_gb": total_mem_gb,
        "usable_gb": usable_gb,
        "note": f"{total_mem_gb}GB unified memory, ~{usable_gb}GB usable for GPU",
    }


# =============================================================================
# Backend Configuration Helper
# =============================================================================


def apply_backend_config(config_dict: Dict, backend: str) -> Dict:
    """
    Apply backend-specific optimizations to search config.

    Args:
        config_dict: Base configuration dictionary
        backend: Backend name ('mps', 'cpu', 'cuda')

    Returns:
        dict: Updated configuration dictionary
    """
    if backend == "mps":
        mps_config = get_mps_search_config()
        config_dict.update(mps_config)

        print(f"  [MPS Config] {mps_config['chip_name']}")
        print(f"    - GPU Cores: {mps_config['gpu_cores']}")
        print(f"    - Memory BW: {mps_config['memory_bandwidth_gbps']} GB/s")
        print(f"    - Search threads: {mps_config['search_thread']}")

        if mps_config.get("has_dynamic_caching"):
            print(f"    - Dynamic Caching: Enabled (M3+)")

        mem_info = get_mps_memory_config()
        print(f"    - Memory: {mem_info['note']}")

    elif backend == "cpu":
        cpu_config = get_cpu_search_config()
        config_dict.update(cpu_config)
        print(f"  [CPU Config] Using {cpu_config['search_thread']} search threads")

    return config_dict


# =============================================================================
# Convenience Functions
# =============================================================================


def print_apple_silicon_info():
    """Print detailed Apple Silicon information."""
    family, specs = detect_apple_silicon()

    print("=" * 60)
    print("  Apple Silicon Information")
    print("=" * 60)
    print(f"  Chip: {specs.chip_name}")
    print(f"  Family: {family.value}")
    print()
    print("  CPU:")
    print(f"    - Performance cores: {specs.cpu_cores_performance}")
    print(f"    - Efficiency cores: {specs.cpu_cores_efficiency}")
    print(f"    - Total cores: {specs.cpu_cores_total}")
    print()
    print("  GPU:")
    print(f"    - GPU cores: {specs.gpu_cores}")
    print(f"    - SIMD width: {specs.simd_width}")
    print(f"    - Max threads/threadgroup: {specs.max_threads_per_threadgroup}")
    print(f"    - Threadgroup memory: {specs.threadgroup_memory_kb} KB")
    print()
    print("  Memory:")
    mem_info = get_mps_memory_config()
    print(f"    - {mem_info['note']}")
    print(f"    - Bandwidth: {specs.memory_bandwidth_gbps} GB/s")
    print()
    print("  Neural Engine:")
    print(f"    - Cores: {specs.neural_engine_cores}")
    print(f"    - Performance: {specs.neural_engine_tops} TOPS")
    print()
    print("  Features:")
    print(f"    - Ray Tracing: {'Yes' if specs.has_ray_tracing else 'No'}")
    print(f"    - Mesh Shading: {'Yes' if specs.has_mesh_shading else 'No'}")
    print(f"    - Dynamic Caching: {'Yes' if specs.has_dynamic_caching else 'No'}")
    print(f"    - Process: {specs.process_nm}nm")
    print("=" * 60)


def get_all_chip_specs() -> Dict[str, AppleSiliconSpecs]:
    """
    Get specifications for all known Apple Silicon chips.

    Returns:
        dict: Mapping from chip name to specs
    """
    return {family.value: specs for family, specs in APPLE_SILICON_SPECS.items()}
