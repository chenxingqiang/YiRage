/* Copyright 2025 YiRage Team */

#include "threadblock/mps/input_loader.h"

#ifdef YIRAGE_USE_MPS

#import <Metal/Metal.h>

namespace yirage {
namespace threadblock {
namespace mps {

void launch_input_loader(
    id<MTLBuffer> src_buffer,
    id<MTLBuffer> dst_buffer,
    int num_elements,
    id<MTLCommandQueue> queue) {
    
    @autoreleasepool {
        id<MTLCommandBuffer> cmdBuffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [cmdBuffer blitCommandEncoder];
        
        [blitEncoder copyFromBuffer:src_buffer
                       sourceOffset:0
                           toBuffer:dst_buffer
                  destinationOffset:0
                               size:num_elements * sizeof(float)];
        
        [blitEncoder endEncoding];
        [cmdBuffer commit];
    }
}

}  // namespace mps
}  // namespace threadblock
}  // namespace yirage

#endif
