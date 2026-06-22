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
 *
 * Persistent Kernel Backend - Simplified interface for new backends
 */

#pragma once

#include "persistent_kernel/pk_backend_interface.h"

namespace yirage {
namespace pk {

// Re-export types from persistent_kernel namespace for convenience
using persistent_kernel::PKBackendType;
using persistent_kernel::PKMode;
using persistent_kernel::PKDataType;
using persistent_kernel::PKCapabilities;
using persistent_kernel::PKMemoryAllocator;
using persistent_kernel::PKAtomicOps;
using persistent_kernel::PKTaskExecutor;
using persistent_kernel::PKTaskDesc;
using persistent_kernel::PKRuntimeConfig;
using persistent_kernel::PKBackendInterface;

/**
 * @brief Kernel launch configuration
 * 
 * This structure holds all parameters needed to launch a kernel.
 */
struct PKKernelConfig {
    const char* kernel_name;
    void* kernel_func;
    
    // Grid/block dimensions
    int grid_x, grid_y, grid_z;
    int block_x, block_y, block_z;
    
    // Memory
    size_t shared_memory_bytes;
    void* stream;
    
    // Arguments
    void** args;
    size_t num_args;
    
    PKKernelConfig() : 
        kernel_name(nullptr), kernel_func(nullptr),
        grid_x(1), grid_y(1), grid_z(1),
        block_x(1), block_y(1), block_z(1),
        shared_memory_bytes(0), stream(nullptr),
        args(nullptr), num_args(0) {}
};

/**
 * @brief Simplified backend interface for new backend implementations
 * 
 * This abstract class provides a simpler interface than PKBackendInterface
 * for implementing new backends that don't need all the advanced features.
 */
class PKBackend {
public:
    virtual ~PKBackend() = default;
    
    // ========== Initialization ==========
    virtual bool initialize(int device_id = 0) = 0;
    virtual void shutdown() = 0;
    virtual bool is_initialized() const = 0;
    
    // ========== Memory Management ==========
    virtual void* allocate_memory(size_t size) { return nullptr; }
    virtual void free_memory(void* ptr) = 0;
    virtual bool copy_to_device(void* dst, const void* src, size_t size) = 0;
    virtual bool copy_to_host(void* dst, const void* src, size_t size) = 0;
    
    // ========== Kernel Execution ==========
    virtual bool launch_kernel(const PKKernelConfig& config) = 0;
    virtual void synchronize() = 0;
    
    // ========== Info ==========
    virtual const char* get_name() const { return "Unknown"; }
    virtual PKBackendType get_type() const { return PKBackendType::CPU; }
};

} // namespace pk
} // namespace yirage
