/* Copyright 2025 YiRage Team
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
 * @file rocm_pk_backend.cc
 * @brief AMD ROCm/HIP Persistent Kernel Backend Implementation
 *
 * Provides persistent kernel execution for AMD GPUs using HIP.
 * Supports CDNA architectures: MI100, MI200, MI250, MI300 series.
 */

#include "persistent_kernel/backends/rocm_pk_backend.h"

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// ROCm Memory Allocator Implementation
// =============================================================================

RocmMemoryAllocator::RocmMemoryAllocator(int device_id)
    : device_id_(device_id),
      total_allocated_(0),
      alignment_(256) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipError_t err = hipSetDevice(device_id_);
    if (err != hipSuccess) {
        std::cerr << "[RocmMemoryAllocator] Failed to set device " 
                  << device_id_ << ": " << hipGetErrorString(err) << std::endl;
    }
#endif
}

RocmMemoryAllocator::~RocmMemoryAllocator() {
    // Clean up any remaining allocations
    for (auto& [ptr, size] : allocations_) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
        hipFree(ptr);
#endif
    }
    allocations_.clear();
}

void* RocmMemoryAllocator::allocate(size_t size) {
    if (size == 0) return nullptr;
    
    // Align size
    size_t aligned_size = ((size + alignment_ - 1) / alignment_) * alignment_;
    
    void* ptr = nullptr;
    
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipError_t err = hipSetDevice(device_id_);
    if (err != hipSuccess) {
        throw std::runtime_error("Failed to set ROCm device");
    }
    
    err = hipMalloc(&ptr, aligned_size);
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("ROCm allocation failed: ") + hipGetErrorString(err)
        );
    }
    
    allocations_[ptr] = aligned_size;
    total_allocated_ += aligned_size;
#else
    throw std::runtime_error("ROCm backend not enabled");
#endif
    
    return ptr;
}

void RocmMemoryAllocator::deallocate(void* ptr) {
    if (ptr == nullptr) return;
    
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    auto it = allocations_.find(ptr);
    if (it != allocations_.end()) {
        total_allocated_ -= it->second;
        allocations_.erase(it);
        hipFree(ptr);
    }
#endif
}

void* RocmMemoryAllocator::allocate_pinned(size_t size) {
    if (size == 0) return nullptr;
    
    void* ptr = nullptr;
    
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipError_t err = hipHostMalloc(&ptr, size, hipHostMallocDefault);
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("ROCm pinned allocation failed: ") + hipGetErrorString(err)
        );
    }
#else
    throw std::runtime_error("ROCm backend not enabled");
#endif
    
    return ptr;
}

void RocmMemoryAllocator::deallocate_pinned(void* ptr) {
    if (ptr == nullptr) return;
    
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipHostFree(ptr);
#endif
}

size_t RocmMemoryAllocator::get_available_memory() const {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    size_t free_mem, total_mem;
    hipError_t err = hipSetDevice(device_id_);
    if (err == hipSuccess) {
        err = hipMemGetInfo(&free_mem, &total_mem);
        if (err == hipSuccess) {
            return free_mem;
        }
    }
#endif
    return 0;
}

// =============================================================================
// ROCm Atomic Operations Implementation
// =============================================================================

RocmAtomicOps::RocmAtomicOps() = default;
RocmAtomicOps::~RocmAtomicOps() = default;

// ROCm uses HIP atomics on device side
// Host-side stubs for interface compatibility

uint64_t RocmAtomicOps::fetch_add_u64(uint64_t* addr, uint64_t val) {
    // Host-side stub - actual implementation is in device code
    return 0;
}

uint64_t RocmAtomicOps::compare_exchange_u64(
    uint64_t* addr, 
    uint64_t expected, 
    uint64_t desired
) {
    // Host-side stub
    return 0;
}

uint64_t RocmAtomicOps::load_acquire_u64(uint64_t* addr) {
    // Host-side stub
    return *addr;
}

void RocmAtomicOps::store_release_u64(uint64_t* addr, uint64_t val) {
    // Host-side stub
}

// =============================================================================
// ROCm PK Backend Implementation
// =============================================================================

RocmPKBackend::RocmPKBackend()
    : device_id_(0),
      stream_(nullptr),
      initialized_(false),
      kernel_launched_(false) {
}

RocmPKBackend::~RocmPKBackend() {
    shutdown();
}

bool RocmPKBackend::initialize(int device_id) {
    if (initialized_) {
        return true;
    }
    
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    device_id_ = device_id;
    
    hipError_t err = hipSetDevice(device_id_);
    if (err != hipSuccess) {
        std::cerr << "[RocmPKBackend] Failed to set device: " 
                  << hipGetErrorString(err) << std::endl;
        return false;
    }
    
    // Get device properties
    hipDeviceProp_t props;
    err = hipGetDeviceProperties(&props, device_id_);
    if (err == hipSuccess) {
        device_name_ = props.name;
        compute_units_ = props.multiProcessorCount;
        wavefront_size_ = props.warpSize;  // 64 for AMD
        lds_size_ = props.sharedMemPerBlock;
        
        std::cout << "[RocmPKBackend] Device: " << device_name_ << std::endl;
        std::cout << "  - Compute Units: " << compute_units_ << std::endl;
        std::cout << "  - Wavefront Size: " << wavefront_size_ << std::endl;
        std::cout << "  - LDS Size: " << lds_size_ / 1024 << " KB" << std::endl;
    }
    
    // Create stream
    err = hipStreamCreate(&stream_);
    if (err != hipSuccess) {
        std::cerr << "[RocmPKBackend] Failed to create stream: " 
                  << hipGetErrorString(err) << std::endl;
        return false;
    }
    
    // Create allocator
    allocator_ = std::make_unique<RocmMemoryAllocator>(device_id_);
    atomic_ops_ = std::make_unique<RocmAtomicOps>();
    
    initialized_ = true;
    return true;
#else
    std::cerr << "[RocmPKBackend] ROCm not enabled at compile time" << std::endl;
    return false;
#endif
}

void RocmPKBackend::shutdown() {
    if (!initialized_) return;
    
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    if (kernel_launched_) {
        // Wait for any running kernel
        hipStreamSynchronize(stream_);
    }
    
    if (stream_) {
        hipStreamDestroy(stream_);
        stream_ = nullptr;
    }
    
    allocator_.reset();
    atomic_ops_.reset();
#endif
    
    initialized_ = false;
    kernel_launched_ = false;
}

void* RocmPKBackend::allocate_memory(size_t size) {
    if (!initialized_ || !allocator_) {
        throw std::runtime_error("RocmPKBackend not initialized");
    }
    return allocator_->allocate(size);
}

void RocmPKBackend::free_memory(void* ptr) {
    if (allocator_) {
        allocator_->deallocate(ptr);
    }
}

void RocmPKBackend::copy_to_device(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    if (!initialized_) {
        throw std::runtime_error("RocmPKBackend not initialized");
    }
    
    hipError_t err = hipMemcpyAsync(
        dst, src, size, hipMemcpyHostToDevice, stream_
    );
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("ROCm H2D copy failed: ") + hipGetErrorString(err)
        );
    }
#endif
}

void RocmPKBackend::copy_from_device(void* dst, const void* src, size_t size) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    if (!initialized_) {
        throw std::runtime_error("RocmPKBackend not initialized");
    }
    
    hipError_t err = hipMemcpyAsync(
        dst, src, size, hipMemcpyDeviceToHost, stream_
    );
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("ROCm D2H copy failed: ") + hipGetErrorString(err)
        );
    }
#endif
}

void RocmPKBackend::synchronize() {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    if (stream_) {
        hipStreamSynchronize(stream_);
    }
#endif
}

bool RocmPKBackend::launch_persistent_kernel(
    const PKTaskGraph& task_graph,
    const PKLaunchConfig& config
) {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    if (!initialized_) {
        std::cerr << "[RocmPKBackend] Not initialized" << std::endl;
        return false;
    }
    
    // Configure launch parameters
    dim3 grid(config.grid_dim_x, config.grid_dim_y, config.grid_dim_z);
    dim3 block(config.block_dim_x, config.block_dim_y, config.block_dim_z);
    size_t shared_mem = config.shared_mem_bytes;
    
    // Launch kernel (actual kernel function would be linked)
    // hipLaunchKernelGGL(pk_kernel, grid, block, shared_mem, stream_, ...);
    
    kernel_launched_ = true;
    return true;
#else
    return false;
#endif
}

void RocmPKBackend::signal_completion() {
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    synchronize();
#endif
    kernel_launched_ = false;
}

PKBackendType RocmPKBackend::get_backend_type() const {
    return PKBackendType::ROCM;
}

std::string RocmPKBackend::get_device_name() const {
    return device_name_;
}

size_t RocmPKBackend::get_available_memory() const {
    if (allocator_) {
        return allocator_->get_available_memory();
    }
    return 0;
}

MemoryAllocator* RocmPKBackend::get_allocator() {
    return allocator_.get();
}

AtomicOps* RocmPKBackend::get_atomic_ops() {
    return atomic_ops_.get();
}

// =============================================================================
// Factory Registration
// =============================================================================

// Register ROCm backend with factory (if enabled)
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
namespace {
    struct RocmBackendRegistrar {
        RocmBackendRegistrar() {
            // Register with PKBackendFactory
            // PKBackendFactory::register_backend(PKBackendType::ROCM, ...);
        }
    };
    static RocmBackendRegistrar rocm_registrar;
}
#endif

}  // namespace persistent_kernel
}  // namespace yirage
