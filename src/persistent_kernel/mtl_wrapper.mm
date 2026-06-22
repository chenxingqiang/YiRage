/* Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Metal (MPS) wrapper functions for YiRage persistent kernel.
 * 
 * This file provides the Objective-C++ bridge to Apple's Metal framework.
 */

#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

extern "C" {

void* mtl_create_default_device() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device) {
        return (__bridge_retained void*)device;
    }
    return nullptr;
}

void mtl_release_device(void* device) {
    if (device) {
        id<MTLDevice> mtlDevice = (__bridge_transfer id<MTLDevice>)device;
        (void)mtlDevice; // Release handled by transfer
    }
}

void* mtl_create_command_queue(void* device) {
    if (!device) return nullptr;
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLCommandQueue> queue = [mtlDevice newCommandQueue];
    if (queue) {
        return (__bridge_retained void*)queue;
    }
    return nullptr;
}

void mtl_release_command_queue(void* queue) {
    if (queue) {
        id<MTLCommandQueue> mtlQueue = (__bridge_transfer id<MTLCommandQueue>)queue;
        (void)mtlQueue; // Release handled by transfer
    }
}

void* mtl_allocate_buffer(void* device, size_t size) {
    if (!device || size == 0) return nullptr;
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLBuffer> buffer = [mtlDevice newBufferWithLength:size 
                                    options:MTLResourceStorageModeShared];
    if (buffer) {
        return (__bridge_retained void*)buffer;
    }
    return nullptr;
}

void mtl_release_buffer(void* buffer) {
    if (buffer) {
        id<MTLBuffer> mtlBuffer = (__bridge_transfer id<MTLBuffer>)buffer;
        (void)mtlBuffer; // Release handled by transfer
    }
}

void mtl_synchronize_device(void* device) {
    // Metal doesn't have a direct device synchronization.
    // Synchronization is done through command buffers.
    // This is a no-op for basic compatibility.
    (void)device;
}

size_t mtl_get_max_threadgroup_memory(void* device) {
    if (!device) return 0;
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    return [mtlDevice maxThreadgroupMemoryLength];
}

const char* mtl_get_device_name(void* device) {
    if (!device) return "Unknown";
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    static thread_local char nameBuffer[256];
    NSString* name = [mtlDevice name];
    [name getCString:nameBuffer maxLength:sizeof(nameBuffer) encoding:NSUTF8StringEncoding];
    return nameBuffer;
}

bool mtl_is_device_available() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    bool available = (device != nil);
    device = nil; // Release
    return available;
}

// Additional Metal helper functions

void* mtl_create_buffer_from_data(void* device, const void* data, size_t size) {
    if (!device || !data || size == 0) return nullptr;
    id<MTLDevice> mtlDevice = (__bridge id<MTLDevice>)device;
    id<MTLBuffer> buffer = [mtlDevice newBufferWithBytes:data 
                                    length:size 
                                    options:MTLResourceStorageModeShared];
    if (buffer) {
        return (__bridge_retained void*)buffer;
    }
    return nullptr;
}

void* mtl_buffer_contents(void* buffer) {
    if (!buffer) return nullptr;
    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
    return [mtlBuffer contents];
}

size_t mtl_buffer_length(void* buffer) {
    if (!buffer) return 0;
    id<MTLBuffer> mtlBuffer = (__bridge id<MTLBuffer>)buffer;
    return [mtlBuffer length];
}

void* mtl_create_command_buffer(void* queue) {
    if (!queue) return nullptr;
    id<MTLCommandQueue> mtlQueue = (__bridge id<MTLCommandQueue>)queue;
    id<MTLCommandBuffer> cmdBuffer = [mtlQueue commandBuffer];
    if (cmdBuffer) {
        return (__bridge_retained void*)cmdBuffer;
    }
    return nullptr;
}

void mtl_commit_command_buffer(void* cmdBuffer) {
    if (!cmdBuffer) return;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    [mtlCmdBuffer commit];
}

void mtl_wait_until_completed(void* cmdBuffer) {
    if (!cmdBuffer) return;
    id<MTLCommandBuffer> mtlCmdBuffer = (__bridge id<MTLCommandBuffer>)cmdBuffer;
    [mtlCmdBuffer waitUntilCompleted];
}

void mtl_release_command_buffer(void* cmdBuffer) {
    if (cmdBuffer) {
        id<MTLCommandBuffer> mtlCmdBuffer = (__bridge_transfer id<MTLCommandBuffer>)cmdBuffer;
        (void)mtlCmdBuffer; // Release handled by transfer
    }
}

} // extern "C"

#else // !__APPLE__

// Stub implementations for non-Apple platforms
extern "C" {

void* mtl_create_default_device() { return nullptr; }
void mtl_release_device(void* device) { (void)device; }
void* mtl_create_command_queue(void* device) { (void)device; return nullptr; }
void mtl_release_command_queue(void* queue) { (void)queue; }
void* mtl_allocate_buffer(void* device, size_t size) { (void)device; (void)size; return nullptr; }
void mtl_release_buffer(void* buffer) { (void)buffer; }
void mtl_synchronize_device(void* device) { (void)device; }
size_t mtl_get_max_threadgroup_memory(void* device) { (void)device; return 0; }
const char* mtl_get_device_name(void* device) { (void)device; return "Unavailable"; }
bool mtl_is_device_available() { return false; }
void* mtl_create_buffer_from_data(void* device, const void* data, size_t size) { 
    (void)device; (void)data; (void)size; return nullptr; 
}
void* mtl_buffer_contents(void* buffer) { (void)buffer; return nullptr; }
size_t mtl_buffer_length(void* buffer) { (void)buffer; return 0; }
void* mtl_create_command_buffer(void* queue) { (void)queue; return nullptr; }
void mtl_commit_command_buffer(void* cmdBuffer) { (void)cmdBuffer; }
void mtl_wait_until_completed(void* cmdBuffer) { (void)cmdBuffer; }
void mtl_release_command_buffer(void* cmdBuffer) { (void)cmdBuffer; }

} // extern "C"

#endif // __APPLE__
