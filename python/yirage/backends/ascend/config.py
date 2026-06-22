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
        "max_num_threadblock_graph_op": 8,  # Ascend supports complex fusion
        "max_num_kernel_graph_op": 5,  # Good fusion capability
        "max_num_threadblock_graphs": 1,
        "search_thread": search_threads,
        # =====================================================================
        # IMPORTANT: imap/omap values are DIMENSION INDICES, not sizes!
        # - (-1, -1, -1) = no dimension mapping (required when grid_dim=1)
        # - (0, -1, -1) = map tensor dim[0] to grid.x
        # - (1, -1, -1) = map tensor dim[1] to grid.x
        # For 2D tensors like (batch, hidden), valid indices are 0 and 1
        # =====================================================================
        # Input dimension mappings
        "imaps_to_explore": [
            (-1, -1, -1),  # No mapping - required for grid=(1,1,1)
            (0, -1, -1),  # Map dim 0 to grid.x (batch dimension)
            (1, -1, -1),  # Map dim 1 to grid.x (hidden dimension)
        ],
        # Output dimension mappings
        "omaps_to_explore": [
            (-1, -1, -1),  # No mapping
            (0, -1, -1),  # Map dim 0 to grid.x
        ],
        # Grid dimensions (AI Core blocks)
        # Ascend has 32 AI Cores (910/910B) or 8 (310P)
        "grid_dims_to_explore": [
            (1, 1, 1),  # Single block - works with all imaps
            (2, 1, 1),  # 2 blocks
            (4, 1, 1),  # 4 blocks
            (8, 1, 1),  # 8 blocks
            (16, 1, 1),  # 16 blocks
        ],
        # Block dimensions (AI Cores per block)
        "block_dims_to_explore": [
            (1, 1, 1),  # 1 AI Core
            (2, 1, 1),  # 2 AI Cores
            (4, 1, 1),  # 4 AI Cores
        ],
        # Forloop dimension mappings
        # -1 = don't split this dimension
        # 0, 1, 2 = dimension indices to split
        "fmaps_to_explore": [-1, 0, 1],
        # Forloop ranges - Cube operations work well with 16x multiples
        "franges_to_explore": [4, 16, 64],
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
        "hbm_gb": 64,  # HBM2e memory
        "l1_kb": 512,  # L1 buffer per AI Core
        "ai_cores": 32,  # Total AI Cores
        "note": "Ascend 910B default (64GB HBM, 32 AI Cores)",
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
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=2)

        if result.returncode == 0:
            output = result.stdout

            device_info = {"available": True, "device_type": "Unknown"}

            if "Ascend 910B" in output or "910B" in output:
                device_info["device_type"] = "Ascend 910B"
                device_info["ai_cores"] = 32
                device_info["hbm_gb"] = 64
                device_info["l1_kb"] = 512
            elif "Ascend 910" in output:
                device_info["device_type"] = "Ascend 910"
                device_info["ai_cores"] = 32
                device_info["hbm_gb"] = 32
                device_info["l1_kb"] = 256
            elif "Ascend 310P" in output or "310P" in output:
                device_info["device_type"] = "Ascend 310P"
                device_info["ai_cores"] = 8
                device_info["hbm_gb"] = 8
                device_info["l1_kb"] = 128

            return device_info
    except:
        pass

    return None


# =============================================================================
# torch_npu Dependency Check
# =============================================================================

_TORCH_NPU_CHECK_DONE = False
_TORCH_NPU_AVAILABLE = False


def check_torch_npu(warn: bool = True) -> bool:
    """
    Check if torch_npu is installed and working.
    
    Args:
        warn: If True, print warning with installation guide when not available.
        
    Returns:
        True if torch_npu is available, False otherwise.
    """
    global _TORCH_NPU_CHECK_DONE, _TORCH_NPU_AVAILABLE
    
    if _TORCH_NPU_CHECK_DONE:
        return _TORCH_NPU_AVAILABLE
    
    _TORCH_NPU_CHECK_DONE = True
    
    try:
        import torch
        import torch_npu
        
        if hasattr(torch, 'npu') and torch.npu.is_available():
            _TORCH_NPU_AVAILABLE = True
            return True
        else:
            if warn:
                print("Warning: torch_npu installed but NPU not accessible")
                print("  Check: npu-smi info")
            return False
            
    except ImportError:
        if warn:
            _print_torch_npu_install_guide()
        return False
    except Exception as e:
        if warn:
            print(f"Warning: torch_npu error: {e}")
        return False


def _print_torch_npu_install_guide():
    """Print installation guide for torch_npu."""
    import sys
    print("\n" + "=" * 60, file=sys.stderr)
    print("torch_npu is required for Ascend NPU support", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("""
Install torch_npu using one of these methods:

1. Huawei Ascend Repository (Recommended):
   pip install torch-npu -i https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/ascend-repo/simple/

2. Helper script:
   ./scripts/install_torch_npu.sh

3. From source (Gitee):
   git clone https://gitee.com/ascend/pytorch.git
   cd pytorch && pip install -e .

Prerequisites:
- CANN toolkit installed
- PyTorch installed (matching version)
""", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)


def require_torch_npu():
    """
    Ensure torch_npu is available. Raises ImportError if not.
    
    Use this decorator or call directly before NPU operations:
    
        from yirage.backends.ascend.config import require_torch_npu
        require_torch_npu()  # Raises with install guide if missing
    """
    if not check_torch_npu(warn=False):
        _print_torch_npu_install_guide()
        raise ImportError(
            "torch_npu is required for Ascend NPU operations. "
            "Install with: pip install torch-npu -i https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/ascend-repo/simple/"
        )


def is_ascend_available() -> bool:
    """
    Check if Ascend NPU is available (both hardware and torch_npu).
    
    Returns:
        True if Ascend NPU can be used, False otherwise.
    """
    # First check hardware
    device_info = get_ascend_device_info()
    if device_info is None:
        return False
    
    # Then check torch_npu (without warning)
    return check_torch_npu(warn=False)
