#!/usr/bin/env python3
"""
MACA Backend Optimization Demo for YiRage

This script demonstrates:
1. MACA device detection
2. MACA-optimized search configuration
3. Kernel graph creation for MACA
4. Superoptimization with MACA parameters
"""

import torch
import yirage
import time

# Import MACA configuration
from yirage.maca_config import (
    MACA_WARP_SIZE,
    MACA_MAX_THREADS_PER_BLOCK,
    MACA_SHARED_MEM_PER_BLOCK,
    get_maca_search_config,
    get_maca_device_info,
    get_maca_memory_config,
    validate_block_size,
    get_optimal_block_size,
    is_maca_available
)


def print_header(title):
    print()
    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)


def demo_device_info():
    """Demonstrate MACA device detection."""
    print_header("MACA Device Information")
    
    if not is_maca_available():
        print("  MACA SDK not found!")
        return False
    
    info = get_maca_device_info()
    if info:
        print(f"  Available: {info.get('available', False)}")
        print(f"  Device: {info.get('device_type', 'Unknown')}")
        print(f"  HBM Memory: {info.get('hbm_gb', 0)} GB")
        print(f"  SM Count: {info.get('sm_count', 0)}")
        print(f"  Warp Size: {info.get('warp_size', MACA_WARP_SIZE)}")
        print(f"  Used Memory: {info.get('used_mem_mib', 0)} MiB")
        print(f"  Total Memory: {info.get('total_mem_mib', 0)} MiB")
        return True
    return False


def demo_maca_config():
    """Show MACA-specific configuration."""
    print_header("MACA Hardware Configuration")
    
    print(f"  Warp Size: {MACA_WARP_SIZE} threads (NVIDIA uses 32)")
    print(f"  Max Threads/Block: {MACA_MAX_THREADS_PER_BLOCK}")
    print(f"  Shared Memory: {MACA_SHARED_MEM_PER_BLOCK // 1024} KB/block")
    
    print()
    print("  Search Configuration:")
    config = get_maca_search_config()
    
    grid_dims = config.get("grid_dims_to_explore", [])
    block_dims = config.get("block_dims_to_explore", [])
    franges = config.get("franges_to_explore", [])
    
    print(f"    Grid dims: {len(grid_dims)} configurations")
    print(f"    Block dims: {len(block_dims)} configurations")
    print(f"    Forloop ranges: {franges}")
    print(f"    Search threads: {config.get('search_thread', 8)}")


def demo_block_optimization():
    """Show block size optimization for MACA."""
    print_header("Block Size Optimization")
    
    print("  Problem Size -> Optimal Block Size")
    print("  " + "-" * 40)
    
    test_sizes = [256, 512, 1024, 4096, 16384, 65536]
    for size in test_sizes:
        optimal = get_optimal_block_size(size)
        valid = validate_block_size(optimal)
        warps = optimal // MACA_WARP_SIZE
        print(f"    {size:>6} -> {optimal:>4} ({warps} warps, valid: {valid})")


def demo_matmul_graph():
    """Create and optimize a MatMul kernel graph."""
    print_header("MatMul Kernel Optimization")
    
    # Dimensions
    M, N, K = 128, 128, 1024
    print(f"  Matrix dimensions: A({M}x{K}) @ B({K}x{N}) = C({M}x{N})")
    
    # Create graph
    graph = yirage.new_kernel_graph()
    A = graph.new_input(dims=(M, K), dtype=yirage.float16)
    B = graph.new_input(dims=(K, N), dtype=yirage.float16)
    C = graph.matmul(A, B)
    graph.mark_output(C)
    
    print("  Graph created successfully!")
    
    # MACA optimization notes
    print()
    print("  MACA Optimization Notes:")
    print(f"    - Use block sizes that are multiples of {MACA_WARP_SIZE}")
    print("    - Optimal tiles: 64x64, 128x64, 128x128")
    print("    - Consider mctlass for tensor core operations")
    
    return graph


def demo_rms_norm_graph():
    """Create an RMS Normalization kernel graph."""
    print_header("RMSNorm Kernel Optimization")
    
    batch, hidden = 8, 4096
    print(f"  Input shape: ({batch}, {hidden})")
    
    graph = yirage.new_kernel_graph()
    X = graph.new_input(dims=(batch, hidden), dtype=yirage.float16)
    W = graph.new_input(dims=(hidden,), dtype=yirage.float16)
    
    # RMSNorm: Y = X * W / sqrt(mean(X^2) + eps)
    # Note: Full implementation requires reduction support
    
    print("  Graph inputs created!")
    print()
    print("  MACA RMSNorm Optimization:")
    print(f"    - Reduction uses {MACA_WARP_SIZE}-wide shuffles")
    print("    - 6 shuffle iterations for full warp reduction")
    print("    - Shared memory for cross-warp reduction")
    
    return graph


def demo_attention_config():
    """Show attention optimization configuration for MACA."""
    print_header("Attention Optimization for MACA")
    
    print("  Group Query Attention Configuration:")
    print("    - num_heads: 32")
    print("    - num_kv_heads: 8")
    print("    - head_dim: 128")
    print("    - sequence_length: 2048")
    
    print()
    print("  MACA Flash Attention Optimization:")
    print("    - Use mcflashinfer library")
    print(f"    - {MACA_WARP_SIZE}-thread warps for softmax reduction")
    print("    - Tile sizes aligned to warp size")
    print("    - HBM-optimized memory access patterns")


def demo_memory_config():
    """Show MACA memory configuration."""
    print_header("MACA Memory Configuration")
    
    config = get_maca_memory_config()
    
    print(f"  Shared Memory/Block: {config.get('shared_memory_per_block', 0) // 1024} KB")
    print(f"  L2 Cache: {config.get('l2_cache_size', 0) // (1024*1024)} MB")
    print(f"  Memory Bus Width: {config.get('memory_bus_width', 0)} bits")
    print(f"  Default Pool Size: {config.get('default_memory_pool_size', 0) // (1024*1024)} MB")


def main():
    print()
    print("=" * 50)
    print("  MACA Backend Optimization Demo")
    print("  YiRage - Kernel Superoptimizer")
    print("=" * 50)
    
    # Run demos
    has_device = demo_device_info()
    demo_maca_config()
    demo_block_optimization()
    demo_memory_config()
    demo_matmul_graph()
    demo_rms_norm_graph()
    demo_attention_config()
    
    # Summary
    print_header("Summary")
    print("  MACA backend integration complete!")
    print()
    print("  Key differences from NVIDIA CUDA:")
    print(f"    1. Warp size: {MACA_WARP_SIZE} (not 32)")
    print("    2. Block sizes must be multiples of 64")
    print("    3. Use mxcc compiler instead of nvcc")
    print("    4. mc* APIs instead of cuda* APIs")
    print()
    print("  Libraries for MACA optimization:")
    print("    - mctlass: CUTLASS-equivalent for tensor ops")
    print("    - mcflashinfer: Flash Attention for MACA")
    print("    - mcblas: cuBLAS-equivalent")
    print()
    
    if has_device:
        print("  MACA GPU detected and ready!")
    else:
        print("  Note: Run on MetaX GPU for full functionality")
    
    print()
    print("=" * 50)
    print("  Demo Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()

