#!/usr/bin/env python3
"""
MACA Superoptimization Test
"""
import yirage
import time
from yirage.maca_config import get_maca_search_config, MACA_WARP_SIZE

print("=" * 50)
print("  MACA Kernel Superoptimization Test")
print("=" * 50)

# Create MatMul Graph
print()
print("1. Creating MatMul Graph...")
graph = yirage.new_kernel_graph()

M, N, K = 64, 64, 256
A = graph.new_input(dims=(M, K), dtype=yirage.float16)
B = graph.new_input(dims=(K, N), dtype=yirage.float16)
C = graph.matmul(A, B)
graph.mark_output(C)

print(f"   Matmul: ({M}x{K}) @ ({K}x{N}) = ({M}x{N})")

# Get MACA configuration
print()
print("2. MACA Search Configuration:")
config = get_maca_search_config()
block_dims = config.get("block_dims_to_explore", [])
grid_dims = config.get("grid_dims_to_explore", [])
franges = config.get("franges_to_explore", [])
imaps = config.get("imap_to_explore", [])
omaps = config.get("omap_to_explore", [])
fmaps = config.get("fmap_to_explore", [])

print(f"   Block dims: {len(block_dims)} configs")
print(f"   Grid dims: {len(grid_dims)} configs")
print(f"   Forloop ranges: {franges}")

# Run superoptimization with MACA parameters
print()
print("3. Running Superoptimization with MACA config...")
start = time.time()

try:
    result = graph.superoptimize(
        imaps=imaps if imaps else None,
        omaps=omaps if omaps else None,
        griddims=grid_dims if grid_dims else None,
        blockdims=block_dims if block_dims else None,
        fmaps=fmaps if fmaps else None,
        franges=franges if franges else None,
        backend="cpu",  # Use CPU for search, MACA for execution
        verbose=True
    )
    elapsed = time.time() - start
    print(f"   Completed in {elapsed:.2f}s")
    
    if result:
        print("   Found optimized kernel!")
        
except Exception as e:
    elapsed = time.time() - start
    print(f"   After {elapsed:.2f}s: {e}")

# Summary
print()
print("4. MACA Optimization Summary:")
print(f"   Warp Size: {MACA_WARP_SIZE}")
print("   Compiler: mxcc -x maca")
print("   Target: MetaX C500 GPU")

print()
print("=" * 50)
print("  Test Complete!")
print("=" * 50)

