"""
Ascend-specific search configuration for Huawei NPUs
Optimized search space for Ascend 910/910B/310P

Based on Huawei Ascend CANN specifications:
- AI Core local memory (L1): 256KB (910), 512KB (910B)
- AI Core count: 32 (910/910B), 8 (310P)
- Block size: Configurable AI Cores per block
- Cube unit: Matrix multiplication acceleration (16x16 tiles)
- Vector unit: Element-wise operations
"""

import multiprocessing

def get_ascend_search_config():
    """
    Get optimized search configuration for Ascend backend
    
    Huawei Ascend NPU characteristics:
    - AI Cores: Specialized tensor processing units
    - Cube operations: Accelerated matrix multiplication (16x16 native)
    - Vector operations: Element-wise operations
    - L1 Buffer: 256KB-512KB per AI Core
    - HBM: 32GB (910), 64GB (910B)
    
    Returns:
        dict: Search configuration optimized for Ascend NPU
    """
    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))
    
    return {
        # Search space optimized for Ascend architecture
        "max_num_threadblock_graph_op": 8,   # Ascend supports complex fusion
        "max_num_kernel_graph_op": 5,        # Good fusion capability
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        
        # Grid dimensions (AI Core blocks)
        # Ascend has 32 AI Cores (910/910B) or 8 (310P)
        "grid_dims_to_explore": [
            (1, 1, 1),    # Single block
            (2, 1, 1),    # 2 blocks
            (4, 1, 1),    # 4 blocks
            (8, 1, 1),    # 8 blocks
            (16, 1, 1),   # 16 blocks
            (32, 1, 1),   # 32 blocks (full 910/910B)
            (2, 2, 1),    # 2D: 4 blocks
            (4, 4, 1),    # 2D: 16 blocks
            (8, 4, 1),    # 2D: 32 blocks
        ],
        
        # Block dimensions (AI Cores per block)
        "block_dims_to_explore": [
            (1, 1, 1),    # 1 AI Core
            (2, 1, 1),    # 2 AI Cores
            (4, 1, 1),    # 4 AI Cores
            (8, 1, 1),    # 8 AI Cores (good balance)
            (16, 1, 1),   # 16 AI Cores
            (32, 1, 1),   # 32 AI Cores (all cores in one block)
        ],
        
        # Forloop dimension mappings
        "fmaps_to_explore": [-1, 0, 1, 2],
        
        # Forloop ranges - Cube operations work well with 16x multiples
        "franges_to_explore": [4, 8, 16],
    }


def get_ascend_memory_config():
    """
    Get Ascend memory configuration
    
    Returns:
        dict: Memory configuration
    """
    # Try to detect via ACL if available
    try:
        # TODO: Add actual ACL detection when CANN is available
        pass
    except:
        pass
    
    # Default: Ascend 910B configuration
    return {
        'hbm_gb': 64,      # HBM2e memory
        'l1_kb': 512,      # L1 buffer per AI Core
        'ai_cores': 32,    # Total AI Cores
        'note': 'Ascend 910B default (64GB HBM, 32 AI Cores)'
    }


def get_ascend_device_info():
    """
    Try to detect Ascend device type
    
    Returns:
        dict: Device information or None
    """
    try:
        import subprocess
        # Try to detect via npu-smi
        result = subprocess.run(
            ['npu-smi', 'info'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode == 0:
            output = result.stdout
            
            device_info = {
                'available': True,
                'device_type': 'Unknown'
            }
            
            if 'Ascend 910B' in output or '910B' in output:
                device_info['device_type'] = 'Ascend 910B'
                device_info['ai_cores'] = 32
                device_info['hbm_gb'] = 64
                device_info['l1_kb'] = 512
            elif 'Ascend 910' in output:
                device_info['device_type'] = 'Ascend 910'
                device_info['ai_cores'] = 32
                device_info['hbm_gb'] = 32
                device_info['l1_kb'] = 256
            elif 'Ascend 310P' in output or '310P' in output:
                device_info['device_type'] = 'Ascend 310P'
                device_info['ai_cores'] = 8
                device_info['hbm_gb'] = 8
                device_info['l1_kb'] = 128
            
            return device_info
    except:
        pass
    
    return None

