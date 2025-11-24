"""
MPS-specific search configuration for Apple Silicon
Optimized search space for M1/M2/M3 GPUs

Based on actual Apple Silicon Metal specifications:
- Threadgroup memory: 32KB (32768 bytes) for all M-series chips
- SIMD width: 32 threads per SIMD group
- Max threads per threadgroup: 1024
- Unified memory architecture with zero-copy overhead
"""

import os
import multiprocessing

def get_mps_search_config():
    """
    Get optimized search configuration for MPS backend
    
    Apple Silicon GPUs have unique characteristics:
    - Unified memory: No CPU<->GPU data transfer overhead
    - SIMD width: 32 (all operations should be multiples of 32)
    - Threadgroup memory: 32KB per threadgroup (M1/M2/M3)
    - Max threads per threadgroup: 1024
    - Tile-based deferred rendering architecture
    
    Returns:
        dict: Search configuration optimized for MPS backend
    """
    # Determine optimal search threads based on CPU cores (not GPU cores)
    # Search is CPU-bound, so use CPU core count
    cpu_count = multiprocessing.cpu_count()
    # Use 75% of CPU cores for search to leave room for system tasks
    search_threads = max(4, int(cpu_count * 0.75))
    
    return {
        # Search space optimized for Apple Silicon (Phase 1-3 aligned)
        "max_num_threadblock_graph_op": 6,
        "max_num_kernel_graph_op": 4,
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        
        # Grid dimensions (extended to match C++ Phase 1 improvements)
        # Now includes fine-grained options for better optimization
        "grid_dims_to_explore": [
            (32, 1, 1),   # 1 SIMD group
            (64, 1, 1),   # 2 SIMD groups
            (96, 1, 1),   # 3 SIMD groups (new)
            (128, 1, 1),  # 4 SIMD groups
            (160, 1, 1),  # 5 SIMD groups (new)
            (192, 1, 1),  # 6 SIMD groups (new - optimal range)
            (224, 1, 1),  # 7 SIMD groups (new)
            (256, 1, 1),  # 8 SIMD groups
            (320, 1, 1),  # 10 SIMD groups (new)
            (384, 1, 1),  # 12 SIMD groups (new)
            (512, 1, 1),  # 16 SIMD groups
            # 2D configurations for matrix operations
            (32, 2, 1),
            (64, 2, 1),
            (32, 4, 1),
            (64, 4, 1),   # New for better 2D coverage
        ],
        
        # Block dimensions (extended to match threadgroup configs)
        # Aligned with Phase 2 optimal ranges (192-512 sweet spot)
        "block_dims_to_explore": [
            (32, 1, 1),   # 1 SIMD group (minimal)
            (64, 1, 1),   # 2 SIMD groups
            (96, 1, 1),   # 3 SIMD groups (new)
            (128, 1, 1),  # 4 SIMD groups
            (160, 1, 1),  # 5 SIMD groups (new)
            (192, 1, 1),  # 6 SIMD groups (new - optimal start)
            (224, 1, 1),  # 7 SIMD groups (new)
            (256, 1, 1),  # 8 SIMD groups (optimal)
            (320, 1, 1),  # 10 SIMD groups (new)
            (384, 1, 1),  # 12 SIMD groups (new - optimal)
            (448, 1, 1),  # 14 SIMD groups (new)
            (512, 1, 1),  # 16 SIMD groups (optimal end)
        ],
        
        # Forloop dimension mappings
        # -1: no forloop, 0/1/2: forloop on dim 0/1/2
        # Apple Silicon benefits from forloop on outer dimensions
        "fmaps_to_explore": [-1, 0, 1, 2],
        
        # Forloop ranges (matches C++ Phase 1)
        "franges_to_explore": [4, 8, 16],
    }

def get_cpu_search_config():
    """
    Get optimized search configuration for CPU backend
    
    CPU-specific considerations:
    - Cache hierarchy: L1 (32-64KB), L2 (256KB-1MB), L3 (shared)
    - Vector instructions: AVX2 (256-bit) / AVX-512 (512-bit) on x86
    - NEON (128-bit) on ARM
    - Thread-based parallelism instead of SIMD groups
    
    Returns:
        dict: Search configuration optimized for CPU execution
    """
    cpu_count = multiprocessing.cpu_count()
    # For CPU, use more threads since search is CPU-bound
    search_threads = max(4, int(cpu_count * 0.75))
    
    return {
        # CPU has more flexible execution model
        "max_num_threadblock_graph_op": 5,   # Fewer ops for better cache usage
        "max_num_kernel_graph_op": 3,        # Focus on simple fusion
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,     # Based on actual CPU cores
        
        # CPU-optimized dimensions (cache-line friendly)
        # Typical cache line: 64 bytes = 16 float32 or 32 float16
        "grid_dims_to_explore": [
            (8, 1, 1),    # Very small: good for L1 cache
            (16, 1, 1),   # Small: fits in L1 cache
            (32, 1, 1),   # Medium: good cache-line alignment
            (64, 1, 1),   # Large: balance between cache and parallelism
        ],
        
        "block_dims_to_explore": [
            (16, 1, 1),   # Minimal threading
            (32, 1, 1),   # Light threading (cache-friendly)
            (64, 1, 1),   # Moderate threading
            (128, 1, 1),  # Heavy threading (for many-core CPUs)
        ],
        
        # Conservative unrolling for CPU compilers
        "franges_to_explore": [2, 4, 8],  # Smaller for better instruction cache
    }

def apply_backend_config(config_dict, backend):
    """
    Apply backend-specific optimizations to search config
    
    This function adapts the search configuration to match the characteristics
    of different backend hardware architectures.
    
    Args:
        config_dict: Base configuration dictionary
        backend: Backend name ('mps', 'cpu', 'cuda')
    
    Returns:
        dict: Updated configuration dictionary optimized for the target backend
    """
    if backend == "mps":
        mps_config = get_mps_search_config()
        config_dict.update(mps_config)
        
        # Show configuration info
        print(f"  [MPS Config] Using {config_dict['search_thread']} search threads (CPU cores)")
        
        # Show memory info if available
        mem_info = get_mps_memory_config()
        if mem_info:
            print(f"  [MPS Memory] {mem_info['note']}")
            
    elif backend == "cpu":
        cpu_config = get_cpu_search_config()
        config_dict.update(cpu_config)
        print(f"  [CPU Config] Using {config_dict['search_thread']} search threads")
    # CUDA uses default configuration (already optimized for NVIDIA GPUs)
    
    return config_dict


def get_system_memory_size():
    """
    Get total system memory size in bytes (macOS unified memory)
    
    Returns:
        int: Total system memory in bytes, or None if detection fails
    """
    try:
        import subprocess
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except:
        pass
    
    # Fallback: try psutil if available
    try:
        import psutil
        return psutil.virtual_memory().total
    except:
        pass
    
    return None


def get_mps_memory_config():
    """
    Get MPS memory configuration based on system memory
    
    Returns a conservative value (75% of system RAM) to leave room for:
    - macOS system processes
    - Other applications
    - Memory fragmentation
    
    Returns:
        int: Recommended MAX_DMEM_SIZE in bytes
    """
    total_mem = get_system_memory_size()
    
    if total_mem is None:
        # Fallback: use conservative 8GB (lowest common M-series config)
        print("  [MPS Memory] Warning: Cannot detect system memory, using 8GB default")
        return 8 * 1024 * 1024 * 1024
    
    # Use 75% of total memory for safety
    # This leaves room for macOS and other processes
    usable_mem = int(total_mem * 0.75)
    
    # Round down to nearest GB for cleaner values
    usable_gb = usable_mem // (1024 * 1024 * 1024)
    
    print(f"  [MPS Memory] Detected {total_mem / (1024**3):.1f} GB total, "
          f"using {usable_gb} GB ({usable_gb * 1024} MB)")
    
    return usable_gb * 1024 * 1024 * 1024


def get_system_memory_gb():
    """
    Get total system memory in GB (for unified memory architecture)
    
    Returns:
        int: Total system memory in GB, or None if cannot detect
    """
    try:
        import subprocess
        # Use sysctl to get hardware memory size
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'],
            capture_output=True,
            text=True,
            timeout=1
        )
        if result.returncode == 0:
            mem_bytes = int(result.stdout.strip())
            mem_gb = mem_bytes // (1024 ** 3)
            return mem_gb
    except:
        pass
    
    # Fallback: try psutil
    try:
        import psutil
        mem_bytes = psutil.virtual_memory().total
        mem_gb = mem_bytes // (1024 ** 3)
        return mem_gb
    except:
        pass
    
    return None


def get_mps_memory_config():
    """
    Get MPS memory configuration based on system memory
    
    Apple Silicon unified memory configurations:
    - M1/M2/M3: 8GB, 16GB, 24GB
    - M1/M2/M3 Pro: 16GB, 18GB, 32GB, 36GB
    - M1/M2/M3 Max: 32GB, 36GB, 64GB, 96GB, 128GB
    - M1/M2 Ultra: 64GB, 128GB, 192GB
    
    Returns:
        dict: Memory configuration with usable GPU memory
    """
    total_mem_gb = get_system_memory_gb()
    
    if total_mem_gb is None:
        # Conservative default
        return {
            'total_gb': 16,
            'usable_gb': 12,  # Assume 75% usable
            'note': 'Using default estimate (unable to detect system memory)'
        }
    
    # For unified memory, typically 75% can be used for GPU workloads
    # (OS and other processes use the rest)
    usable_gb = int(total_mem_gb * 0.75)
    
    return {
        'total_gb': total_mem_gb,
        'usable_gb': usable_gb,
        'note': f'{total_mem_gb}GB unified memory, ~{usable_gb}GB usable for GPU'
    }


def get_apple_gpu_info():
    """
    Try to detect Apple GPU family and core count (for informational purposes)
    
    Note: This is primarily for logging/debugging. The actual search configuration
    is based on CPU cores since search is CPU-bound.
    
    Returns:
        dict: GPU information if available, None otherwise
    """
    try:
        import subprocess
        result = subprocess.run(
            ['system_profiler', 'SPDisplaysDataType'],
            capture_output=True,
            text=True,
            timeout=2
        )
        output = result.stdout
        
        # Try to detect GPU family
        gpu_info = {
            'family': 'Unknown',
            'chip': 'Unknown'
        }
        
        if 'Apple M1' in output:
            gpu_info['chip'] = 'M1'
            if 'Ultra' in output:
                gpu_info['family'] = 'M1 Ultra'
            elif 'Max' in output:
                gpu_info['family'] = 'M1 Max'
            elif 'Pro' in output:
                gpu_info['family'] = 'M1 Pro'
            else:
                gpu_info['family'] = 'M1'
        elif 'Apple M2' in output:
            gpu_info['chip'] = 'M2'
            if 'Ultra' in output:
                gpu_info['family'] = 'M2 Ultra'
            elif 'Max' in output:
                gpu_info['family'] = 'M2 Max'
            elif 'Pro' in output:
                gpu_info['family'] = 'M2 Pro'
            else:
                gpu_info['family'] = 'M2'
        elif 'Apple M3' in output:
            gpu_info['chip'] = 'M3'
            if 'Ultra' in output:
                gpu_info['family'] = 'M3 Ultra'
            elif 'Max' in output:
                gpu_info['family'] = 'M3 Max'
            elif 'Pro' in output:
                gpu_info['family'] = 'M3 Pro'
            else:
                gpu_info['family'] = 'M3'
        elif 'Apple M4' in output:
            gpu_info['chip'] = 'M4'
            if 'Max' in output:
                gpu_info['family'] = 'M4 Max'
            elif 'Pro' in output:
                gpu_info['family'] = 'M4 Pro'
            else:
                gpu_info['family'] = 'M4'
        
        # Add memory info
        mem_config = get_mps_memory_config()
        gpu_info.update(mem_config)
        
        return gpu_info
    except:
        return None