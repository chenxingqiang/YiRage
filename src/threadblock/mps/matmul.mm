/* Copyright 2025 YiRage Team */

#include "threadblock/graph.h"
#include "threadblock/matmul.h"
#include "threadblock/mps/matmul.h"

#ifdef YIRAGE_USE_MPS

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace yirage {
namespace threadblock {
namespace mps {

// Metal compute pipeline for matmul fingerprint
void launch_matmul_fingerprint(
    id<MTLBuffer> A_buffer,
    id<MTLBuffer> B_buffer,
    id<MTLBuffer> C_buffer,
    int m, int n, int k,
    id<MTLCommandQueue> queue) {
    
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        
        // Set buffers and dispatch
        // 32-thread SIMD groups
        MTLSize threadgroupSize = MTLSizeMake(32, 1, 1);
        MTLSize gridSize = MTLSizeMake((m * n + 31) / 32, 1, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmdBuffer commit];
    }
}

}  // namespace mps
}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_USE_MPS
