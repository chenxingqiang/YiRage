"""
MPS-specific search configuration for Apple Silicon
Optimized search space for M1/M2/M3 GPUs
"""

def get_mps_search_config():
    """
    Get optimized search configuration for MPS backend
    
    Apple Silicon GPUs have different characteristics than NVIDIA GPUs:
    - Threadgroup memory: 32KB (M1) or 64KB (M2/M3)
    - SIMD width: 32
    - Unified memory architecture
    - Different optimal tile sizes
    
    Returns:
        dict: Search configuration optimized for MPS
    """
    return {
        # Reduce search space for faster optimization
        "max_num_threadblock_graph_op": 7,  # Reduced from 9
        "max_num_kernel_graph_op": 4,       # Reduced from 5
        "max_num_threadblock_graphs": 1,
        "search_thread": 8,                  # M3 has ~10 GPU cores
        
        # MPS-optimized thread block dimensions
        # Apple GPUs work well with multiples of 32 (SIMD width)
        "grid_dims_to_explore": [
            (32, 1, 1),
            (64, 1, 1),
            (128, 1, 1),
            (32, 2, 1),
            (64, 2, 1),
        ],
        
        "block_dims_to_explore": [
            (32, 1, 1),   # 1 SIMD group
            (64, 1, 1),   # 2 SIMD groups
            (128, 1, 1),  # 4 SIMD groups
            (256, 1, 1),  # 8 SIMD groups
        ],
        
        # Forloop ranges optimized for MPS
        "franges_to_explore": [4, 16, 32],  # Reduced from [4, 16, 64]
    }

def get_cpu_search_config():
    """
    Get optimized search configuration for CPU backend
    
    Returns:
        dict: Search configuration optimized for CPU
    """
    return {
        # CPU has different optimal configurations
        "max_num_threadblock_graph_op": 6,
        "max_num_kernel_graph_op": 4,
        "max_num_threadblock_graphs": 1,
        "search_thread": 8,  # Use fewer threads for search
        
        # CPU-optimized dimensions (cache-friendly)
        "grid_dims_to_explore": [
            (16, 1, 1),
            (32, 1, 1),
            (64, 1, 1),
        ],
        
        "block_dims_to_explore": [
            (32, 1, 1),
            (64, 1, 1),
            (128, 1, 1),
        ],
        
        "franges_to_explore": [4, 8, 16],
    }

def apply_backend_config(config_dict, backend):
    """
    Apply backend-specific optimizations to search config
    
    Args:
        config_dict: Base configuration dictionary
        backend: Backend name ('mps', 'cpu', 'cuda')
    
    Returns:
        Updated configuration dictionary
    """
    if backend == "mps":
        mps_config = get_mps_search_config()
        config_dict.update(mps_config)
    elif backend == "cpu":
        cpu_config = get_cpu_search_config()
        config_dict.update(cpu_config)
    # CUDA uses default configuration
    
    return config_dict

