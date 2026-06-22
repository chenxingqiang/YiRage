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
 * ROCm/HIP Device Memory Manager
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace yirage {
namespace kernel {
namespace rocm {

#define HIP_CHECK(call)                                                       \
    do {                                                                       \
        hipError_t err = call;                                                \
        if (err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, hipGetErrorString(err));              \
            return nullptr;                                                    \
        }                                                                      \
    } while (0)

#define HIP_CHECK_VOID(call)                                                  \
    do {                                                                       \
        hipError_t err = call;                                                \
        if (err != hipSuccess) {                                              \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, hipGetErrorString(err));              \
            return;                                                            \
        }                                                                      \
    } while (0)

// =============================================================================
// Memory Pool for efficient allocation
// =============================================================================

class ROCmMemoryPool {
public:
    ROCmMemoryPool(size_t initial_size = 256 * 1024 * 1024)  // 256 MB default
        : pool_size_(initial_size), used_size_(0), pool_ptr_(nullptr) {
        hipError_t err = hipMalloc(&pool_ptr_, pool_size_);
        if (err != hipSuccess) {
            fprintf(stderr, "Failed to allocate memory pool: %s\n",
                    hipGetErrorString(err));
            pool_ptr_ = nullptr;
            pool_size_ = 0;
        }
    }
    
    ~ROCmMemoryPool() {
        if (pool_ptr_) {
            hipFree(pool_ptr_);
        }
    }
    
    void* allocate(size_t size) {
        // Align to 256 bytes
        size = (size + 255) & ~255;
        
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if we have a free block
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->second >= size) {
                void* ptr = it->first;
                size_t block_size = it->second;
                free_blocks_.erase(it);
                
                // Split block if much larger
                if (block_size > size + 256) {
                    void* remainder = static_cast<char*>(ptr) + size;
                    free_blocks_[remainder] = block_size - size;
                }
                
                allocated_blocks_[ptr] = size;
                return ptr;
            }
        }
        
        // Allocate from pool tail
        if (used_size_ + size <= pool_size_) {
            void* ptr = static_cast<char*>(pool_ptr_) + used_size_;
            used_size_ += size;
            allocated_blocks_[ptr] = size;
            return ptr;
        }
        
        // Pool exhausted, fall back to direct allocation
        void* ptr;
        if (hipMalloc(&ptr, size) != hipSuccess) {
            return nullptr;
        }
        external_allocations_.push_back(ptr);
        allocated_blocks_[ptr] = size;
        return ptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        auto it = allocated_blocks_.find(ptr);
        if (it == allocated_blocks_.end()) {
            return;  // Not from this pool
        }
        
        size_t size = it->second;
        allocated_blocks_.erase(it);
        
        // Check if this is from the pool or external
        if (ptr >= pool_ptr_ && 
            ptr < static_cast<char*>(pool_ptr_) + pool_size_) {
            // Add to free list
            free_blocks_[ptr] = size;
            // TODO: Coalesce adjacent free blocks
        } else {
            // External allocation
            hipFree(ptr);
        }
    }
    
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        used_size_ = 0;
        free_blocks_.clear();
        allocated_blocks_.clear();
        
        for (void* ptr : external_allocations_) {
            hipFree(ptr);
        }
        external_allocations_.clear();
    }
    
    size_t get_used_size() const { return used_size_; }
    size_t get_pool_size() const { return pool_size_; }

private:
    void* pool_ptr_;
    size_t pool_size_;
    size_t used_size_;
    std::mutex mutex_;
    std::unordered_map<void*, size_t> free_blocks_;
    std::unordered_map<void*, size_t> allocated_blocks_;
    std::vector<void*> external_allocations_;
};

// =============================================================================
// Global Memory Manager
// =============================================================================

class ROCmDeviceMemoryManager {
public:
    static ROCmDeviceMemoryManager& instance() {
        static ROCmDeviceMemoryManager mgr;
        return mgr;
    }
    
    // Device memory allocation
    void* allocate(size_t size) {
        void* ptr;
        if (hipMalloc(&ptr, size) != hipSuccess) {
            return nullptr;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        allocations_[ptr] = size;
        total_allocated_ += size;
        return ptr;
    }
    
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            total_allocated_ -= it->second;
            allocations_.erase(it);
        }
        hipFree(ptr);
    }
    
    // Host-pinned memory
    void* allocate_pinned(size_t size) {
        void* ptr;
        if (hipHostMalloc(&ptr, size, hipHostMallocDefault) != hipSuccess) {
            return nullptr;
        }
        return ptr;
    }
    
    void deallocate_pinned(void* ptr) {
        hipHostFree(ptr);
    }
    
    // Managed memory (unified)
    void* allocate_managed(size_t size) {
        void* ptr;
        if (hipMallocManaged(&ptr, size, hipMemAttachGlobal) != hipSuccess) {
            return nullptr;
        }
        return ptr;
    }
    
    // Memory copy
    void copy_h2d(void* dst, const void* src, size_t size, hipStream_t stream = nullptr) {
        if (stream) {
            hipMemcpyAsync(dst, src, size, hipMemcpyHostToDevice, stream);
        } else {
            hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
        }
    }
    
    void copy_d2h(void* dst, const void* src, size_t size, hipStream_t stream = nullptr) {
        if (stream) {
            hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToHost, stream);
        } else {
            hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
        }
    }
    
    void copy_d2d(void* dst, const void* src, size_t size, hipStream_t stream = nullptr) {
        if (stream) {
            hipMemcpyAsync(dst, src, size, hipMemcpyDeviceToDevice, stream);
        } else {
            hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
        }
    }
    
    // Memory set
    void memset(void* ptr, int value, size_t size, hipStream_t stream = nullptr) {
        if (stream) {
            hipMemsetAsync(ptr, value, size, stream);
        } else {
            hipMemset(ptr, value, size);
        }
    }
    
    // Prefetch for managed memory
    void prefetch_to_device(void* ptr, size_t size, int device_id, hipStream_t stream = nullptr) {
        hipMemPrefetchAsync(ptr, size, device_id, stream);
    }
    
    void prefetch_to_host(void* ptr, size_t size, hipStream_t stream = nullptr) {
        hipMemPrefetchAsync(ptr, size, hipCpuDeviceId, stream);
    }
    
    // Query
    size_t get_total_allocated() const { return total_allocated_; }
    
    size_t get_free_memory() const {
        size_t free, total;
        hipMemGetInfo(&free, &total);
        return free;
    }
    
    size_t get_total_memory() const {
        size_t free, total;
        hipMemGetInfo(&free, &total);
        return total;
    }

private:
    ROCmDeviceMemoryManager() : total_allocated_(0) {}
    ~ROCmDeviceMemoryManager() {
        // Note: Don't free here - let HIP runtime clean up
    }
    
    std::mutex mutex_;
    std::unordered_map<void*, size_t> allocations_;
    size_t total_allocated_;
};

// =============================================================================
// C API Wrappers
// =============================================================================

extern "C" {

void* rocm_malloc(size_t size) {
    return ROCmDeviceMemoryManager::instance().allocate(size);
}

void rocm_free(void* ptr) {
    ROCmDeviceMemoryManager::instance().deallocate(ptr);
}

void* rocm_malloc_pinned(size_t size) {
    return ROCmDeviceMemoryManager::instance().allocate_pinned(size);
}

void rocm_free_pinned(void* ptr) {
    ROCmDeviceMemoryManager::instance().deallocate_pinned(ptr);
}

void* rocm_malloc_managed(size_t size) {
    return ROCmDeviceMemoryManager::instance().allocate_managed(size);
}

void rocm_memcpy_h2d(void* dst, const void* src, size_t size, hipStream_t stream) {
    ROCmDeviceMemoryManager::instance().copy_h2d(dst, src, size, stream);
}

void rocm_memcpy_d2h(void* dst, const void* src, size_t size, hipStream_t stream) {
    ROCmDeviceMemoryManager::instance().copy_d2h(dst, src, size, stream);
}

void rocm_memcpy_d2d(void* dst, const void* src, size_t size, hipStream_t stream) {
    ROCmDeviceMemoryManager::instance().copy_d2d(dst, src, size, stream);
}

void rocm_memset(void* ptr, int value, size_t size, hipStream_t stream) {
    ROCmDeviceMemoryManager::instance().memset(ptr, value, size, stream);
}

size_t rocm_get_free_memory() {
    return ROCmDeviceMemoryManager::instance().get_free_memory();
}

size_t rocm_get_total_memory() {
    return ROCmDeviceMemoryManager::instance().get_total_memory();
}

}  // extern "C"

}  // namespace rocm
}  // namespace kernel
}  // namespace yirage
