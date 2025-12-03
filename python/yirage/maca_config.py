"""
MetaX MACA-specific search configuration for MetaX GPUs
Optimized search space for MetaX C500/C500 Pro GPUs

Based on MetaX MACA specifications:
- warpSize: 64 (vs NVIDIA's 32)
- Shared memory: 64KB per block
- L2 Cache: 8MB
- Registers: 131072 per block
- SMs: 104 (C500)
- HBM: 64GB

Key differences from CUDA:
- 64-thread warps require different block size configurations
- Block sizes must be multiples of 64
- Warp-level operations use 6 shuffle iterations (log2(64))
"""

import os
import multiprocessing
from typing import Dict, Any, Optional, List


# MACA hardware constants
MACA_WARP_SIZE = 64
MACA_MAX_THREADS_PER_BLOCK = 1024
MACA_MAX_WARPS_PER_SM = 32  # 2048 / 64
MACA_SHARED_MEM_PER_BLOCK = 65536  # 64 KB
MACA_REGISTERS_PER_SM = 131072
MACA_SM_COUNT_C500 = 104


def get_maca_search_config() -> Dict[str, Any]:
    """
    Get optimized search configuration for MACA backend
    
    MetaX MACA GPU characteristics:
    - warpSize: 64 (NOT 32 like NVIDIA!)
    - Shared memory: 64KB per block
    - Registers: 131072 per block (2x NVIDIA)
    - SMs: 104 on C500
    - HBM: 64GB
    - Memory bus: 4096-bit
    
    Returns:
        dict: Search configuration optimized for MACA GPUs
    """
    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))
    
    return {
        # Search space optimized for MACA architecture
        "max_num_threadblock_graph_op": 8,   # Good fusion capability
        "max_num_kernel_graph_op": 5,
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        
        # Grid dimensions
        # MACA has 104 SMs, so explore up to that
        "grid_dims_to_explore": [
            (1, 1, 1),    # Single block
            (2, 1, 1),
            (4, 1, 1),
            (8, 1, 1),
            (16, 1, 1),
            (26, 1, 1),   # 1/4 of C500 SMs
            (52, 1, 1),   # 1/2 of C500 SMs
            (104, 1, 1),  # Full C500 SMs
            # 2D configurations
            (8, 8, 1),    # 64 blocks
            (13, 8, 1),   # 104 blocks
        ],
        
        # Block dimensions - MUST be multiples of 64 (warpSize)
        "block_dims_to_explore": [
            (64, 1, 1),   # 1 warp
            (128, 1, 1),  # 2 warps
            (192, 1, 1),  # 3 warps
            (256, 1, 1),  # 4 warps (commonly optimal)
            (320, 1, 1),  # 5 warps
            (384, 1, 1),  # 6 warps
            (448, 1, 1),  # 7 warps
            (512, 1, 1),  # 8 warps
            (640, 1, 1),  # 10 warps
            (768, 1, 1),  # 12 warps
            (1024, 1, 1), # 16 warps (max)
            # 2D configurations
            (64, 2, 1),   # 2 warps, 2D
            (64, 4, 1),   # 4 warps, 2D
            (128, 2, 1),  # 4 warps, 2D
            (128, 4, 1),  # 8 warps, 2D
            (256, 2, 1),  # 8 warps, 2D
            (256, 4, 1),  # 16 warps, 2D (max)
        ],
        
        # Forloop dimension mappings
        "fmaps_to_explore": [-1, 0, 1, 2],
        
        # Forloop ranges
        "franges_to_explore": [4, 8, 16, 32],
    }


def get_maca_matmul_config() -> Dict[str, Any]:
    """
    Get optimized matrix multiplication configuration for MACA
    
    Returns:
        dict: Matmul tile configuration
    """
    return {
        # Tile sizes optimized for MACA's 64-thread warps
        "tile_sizes": [
            {"tile_m": 64, "tile_n": 64, "tile_k": 32},
            {"tile_m": 128, "tile_n": 64, "tile_k": 32},
            {"tile_m": 64, "tile_n": 128, "tile_k": 32},
            {"tile_m": 128, "tile_n": 128, "tile_k": 32},
            {"tile_m": 256, "tile_n": 128, "tile_k": 32},
            {"tile_m": 128, "tile_n": 256, "tile_k": 32},
        ],
        # Warp tile sizes (adjusted for 64-thread warps)
        "warp_tile_m": 64,
        "warp_tile_n": 64,
        # Pipeline stages
        "num_stages": [2, 3, 4],
    }


def get_maca_memory_config() -> Dict[str, Any]:
    """
    Get MACA memory configuration
    
    Returns:
        dict: Memory configuration
    """
    # Try to detect via mx-smi if available
    device_info = get_maca_device_info()
    
    if device_info and device_info.get('available'):
        return {
            'hbm_gb': device_info.get('hbm_gb', 64),
            'shared_mem_kb': MACA_SHARED_MEM_PER_BLOCK // 1024,
            'l2_cache_mb': 8,
            'registers_per_block': MACA_REGISTERS_PER_SM,
            'warp_size': MACA_WARP_SIZE,
            'note': f"MetaX {device_info.get('device_type', 'C500')} detected"
        }
    
    # Default: MetaX C500 configuration
    return {
        'hbm_gb': 64,
        'shared_mem_kb': 64,
        'l2_cache_mb': 8,
        'registers_per_block': 131072,
        'warp_size': 64,
        'note': 'MetaX C500 default (64GB HBM, 64-thread warps)'
    }


def get_maca_device_info() -> Optional[Dict[str, Any]]:
    """
    Try to detect MetaX MACA device via mx-smi
    
    Returns:
        dict: Device information or None
    """
    try:
        import subprocess
        # Try to detect via mx-smi (MetaX System Management Interface)
        result = subprocess.run(
            ['mx-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            output = result.stdout
            
            device_info = {
                'available': True,
                'device_type': 'Unknown',
                'hbm_gb': 64,
                'sm_count': 104,
                'warp_size': 64,
            }
            
            # Parse mx-smi output
            if 'MetaX C500' in output or 'C500' in output:
                device_info['device_type'] = 'MetaX C500'
                device_info['hbm_gb'] = 64
                device_info['sm_count'] = 104
            elif 'MetaX C500 Pro' in output:
                device_info['device_type'] = 'MetaX C500 Pro'
                device_info['hbm_gb'] = 64
                device_info['sm_count'] = 104
            
            # Try to parse memory usage
            import re
            mem_match = re.search(r'(\d+)/(\d+)\s*MiB', output)
            if mem_match:
                device_info['used_mem_mib'] = int(mem_match.group(1))
                device_info['total_mem_mib'] = int(mem_match.group(2))
                device_info['hbm_gb'] = device_info['total_mem_mib'] // 1024
            
            return device_info
    except FileNotFoundError:
        # mx-smi not found - MACA not installed
        pass
    except Exception as e:
        import sys
        print(f"Warning: Error detecting MACA device: {e}", file=sys.stderr)
    
    return None


def is_maca_available() -> bool:
    """
    Check if MACA backend is available
    
    Returns:
        bool: True if MACA is available
    """
    # Check environment variables
    maca_home = os.environ.get('MACA_HOME') or os.environ.get('MACA_PATH')
    if maca_home and os.path.exists(maca_home):
        return True
    
    # Check common installation paths
    common_paths = ['/opt/maca', '/usr/local/maca']
    for path in common_paths:
        if os.path.exists(path):
            return True
    
    # Try to detect device
    device_info = get_maca_device_info()
    return device_info is not None and device_info.get('available', False)


def get_maca_sdk_path() -> Optional[str]:
    """
    Get MACA SDK installation path
    
    Returns:
        str: Path to MACA SDK or None
    """
    # Check environment variables first
    maca_home = os.environ.get('MACA_HOME')
    if maca_home and os.path.exists(maca_home):
        return maca_home
    
    maca_path = os.environ.get('MACA_PATH')
    if maca_path and os.path.exists(maca_path):
        return maca_path
    
    # Check common installation paths
    common_paths = ['/opt/maca', '/usr/local/maca']
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None


def apply_maca_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply MACA-specific optimizations to search config
    
    Args:
        config_dict: Base configuration dictionary
    
    Returns:
        dict: Updated configuration dictionary optimized for MACA
    """
    maca_config = get_maca_search_config()
    config_dict.update(maca_config)
    
    # Show configuration info
    print(f"  [MACA Config] Using {config_dict['search_thread']} search threads")
    print(f"  [MACA Config] warpSize = 64 (NOT 32 like NVIDIA)")
    
    # Show memory info if available
    mem_info = get_maca_memory_config()
    if mem_info:
        print(f"  [MACA Memory] {mem_info['note']}")
    
    # Show device info
    device_info = get_maca_device_info()
    if device_info and device_info.get('available'):
        print(f"  [MACA Device] {device_info.get('device_type', 'Unknown')}")
        if 'total_mem_mib' in device_info:
            total_gb = device_info['total_mem_mib'] / 1024
            used_gb = device_info.get('used_mem_mib', 0) / 1024
            print(f"  [MACA Memory] {used_gb:.1f}/{total_gb:.1f} GB used")
    
    return config_dict


def validate_block_size(block_size: int) -> bool:
    """
    Validate that block size is appropriate for MACA
    
    Args:
        block_size: Total number of threads in block
    
    Returns:
        bool: True if valid for MACA
    """
    if block_size <= 0 or block_size > MACA_MAX_THREADS_PER_BLOCK:
        return False
    
    # Must be multiple of warpSize (64)
    if block_size % MACA_WARP_SIZE != 0:
        return False
    
    return True


def get_optimal_block_size(problem_size: int) -> int:
    """
    Get optimal block size for given problem size on MACA
    
    Args:
        problem_size: Total number of elements to process
    
    Returns:
        int: Optimal block size (multiple of 64)
    """
    if problem_size < 64:
        return 64  # Minimum 1 warp
    elif problem_size < 256:
        return 128  # 2 warps
    elif problem_size < 1024:
        return 256  # 4 warps
    elif problem_size < 4096:
        return 512  # 8 warps
    else:
        return 1024  # 16 warps (max)


# Export key constants
__all__ = [
    'MACA_WARP_SIZE',
    'MACA_MAX_THREADS_PER_BLOCK',
    'MACA_MAX_WARPS_PER_SM',
    'MACA_SHARED_MEM_PER_BLOCK',
    'MACA_REGISTERS_PER_SM',
    'MACA_SM_COUNT_C500',
    'get_maca_search_config',
    'get_maca_matmul_config',
    'get_maca_memory_config',
    'get_maca_device_info',
    'is_maca_available',
    'get_maca_sdk_path',
    'apply_maca_config',
    'validate_block_size',
    'get_optimal_block_size',
]

