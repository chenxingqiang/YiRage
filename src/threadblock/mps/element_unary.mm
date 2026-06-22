/* Copyright 2025 YiRage Team */

#include "threadblock/element_unary.h"
#include "threadblock/mps/element_unary.h"

#ifdef YIRAGE_USE_MPS

#import <Metal/Metal.h>

namespace yirage {
namespace threadblock {
namespace mps {

void launch_element_unary_fingerprint(
    id<MTLBuffer> input_buffer,
    id<MTLBuffer> output_buffer,
    int num_elements, int op_type,
    id<MTLCommandQueue> queue) {
    
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuffer computeCommandEncoder];
        
        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
        MTLSize gridSize = MTLSizeMake((num_elements + 255) / 256, 1, 1);
        
        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmdBuffer commit];
    }
}

}  // namespace mps
}  // namespace threadblock
}  // namespace yirage

#endif
