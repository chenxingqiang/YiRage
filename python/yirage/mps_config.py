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
        # Aggressive reduction for faster search
        "max_num_threadblock_graph_op": 5,  # Reduced from 9 -> 7 -> 5
        "max_num_kernel_graph_op": 3,       # Reduced from 5 -> 4 -> 3
        "max_num_threadblock_graphs": 1,
        "search_thread": 8,                  # M3 has ~10 GPU cores
        
        # Early stopping when good solutions found
        "early_stop_threshold": 10,          # Stop after finding 10 valid graphs
        "search_timeout_seconds": 300,       # 5 minute timeout
        
        # MPS-optimized thread block dimensions (minimal set)
        # Apple GPUs work well with multiples of 32 (SIMD width)
        "grid_dims_to_explore": [
            (32, 1, 1),   # Small
            (64, 1, 1),   # Medium
            (128, 1, 1),  # Large
        ],
        
        "block_dims_to_explore": [
            (64, 1, 1),   # 2 SIMD groups (optimal for most cases)
            (128, 1, 1),  # 4 SIMD groups
        ],
        
        # Minimal forloop ranges for faster search
        "franges_to_explore": [4, 16],  # Reduced from [4, 16, 32]
    }

def get_cpu_search_config():
    """
    Get optimized search configuration for CPU backend
    
    Returns:
        dict: Search configuration optimized for CPU
    """
    return {
        # Aggressive reduction for CPU
        "max_num_threadblock_graph_op": 5,
        "max_num_kernel_graph_op": 3,
        "max_num_threadblock_graphs": 1,
        "search_thread": 8,
        
        # Early stopping
        "early_stop_threshold": 10,
        "search_timeout_seconds": 300,
        
        # Minimal CPU-optimized dimensions
        "grid_dims_to_explore": [
            (32, 1, 1),
            (64, 1, 1),
        ],
        
        "block_dims_to_explore": [
            (64, 1, 1),
            (128, 1, 1),
        ],
        
        # Reduced forloop ranges
        "franges_to_explore": [4, 16],
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

