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

#pragma once

#include "persistent_kernel/pk_backend_interface.h"
#include <atomic>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// MPS Memory Allocator
// =============================================================================

/**
 * @brief Apple Metal Performance Shaders memory allocator
 * 
 * Uses Metal API for GPU memory management on Apple Silicon.
 */
class MpsMemoryAllocator : public PKMemoryAllocator {
public:
    MpsMemoryAllocator();
    ~MpsMemoryAllocator() override;
    
    void* allocate(size_t size) override;
    void free(void* ptr) override;
    void copy_h2d(void* dst, const void* src, size_t size) override;
    void copy_d2h(void* dst, const void* src, size_t size) override;
    void copy_d2d(void* dst, const void* src, size_t size) override;
    void copy_h2d_async(void* dst, const void* src, size_t size, 
                        void* stream) override;
    void memset(void* ptr, int value, size_t size) override;
    size_t get_total_memory() const override;
    size_t get_free_memory() const override;
    
private:
    void* mtl_device_;     // MTLDevice*
    void* default_queue_;  // MTLCommandQueue*
};

// =============================================================================
// MPS Atomic Operations
// =============================================================================

/**
 * @brief MPS-specific atomic operations
 * 
 * Metal uses different atomic semantics; these provide compatibility.
 */
class MpsAtomicOps : public PKAtomicOps {
public:
    MpsAtomicOps();
    ~MpsAtomicOps() override;
    
    uint64_t fetch_add_u64(uint64_t* addr, uint64_t val) override;
    uint64_t fetch_sub_u64(uint64_t* addr, uint64_t val) override;
    uint64_t compare_exchange_u64(uint64_t* addr, uint64_t expected, 
                                   uint64_t desired) override;
    void store_release_u64(uint64_t* addr, uint64_t val) override;
    uint64_t load_acquire_u64(uint64_t* addr) override;
    
    uint32_t fetch_add_u32(uint32_t* addr, uint32_t val) override;
    uint32_t fetch_sub_u32(uint32_t* addr, uint32_t val) override;
    uint32_t compare_exchange_u32(uint32_t* addr, uint32_t expected,
                                   uint32_t desired) override;
    
    void memory_fence() override;
    void thread_fence() override;
};

// =============================================================================
// MPS Task Executor
// =============================================================================

/**
 * @brief MPS task executor using Metal compute shaders
 */
class MpsTaskExecutor : public PKTaskExecutor {
public:
    MpsTaskExecutor();
    ~MpsTaskExecutor() override;
    
    bool supports_task(PKTaskType type) const override;
    void execute(const PKTaskDesc& desc,
                 const PKRuntimeConfig& config,
                 void* shared_memory,
                 size_t shared_memory_size) override;
    size_t get_shared_memory_size(PKTaskType type) const override;
    const char* get_task_name(PKTaskType type) const override;
    
private:
    void* compute_pipeline_cache_;  // Dictionary of MTLComputePipelineState
    void* mtl_device_;
};

// =============================================================================
// MPS Persistent Kernel Backend
// =============================================================================

/**
 * @brief Apple Metal Performance Shaders backend
 * 
 * This backend uses Metal for GPU computation on Apple Silicon (M1/M2/M3).
 * 
 * Supported modes:
 * - EAGER: Immediate execution using Metal command buffers
 * - GRAPH: Metal Performance Graph for optimized execution
 * 
 * Hardware features:
 * - Unified memory architecture (shared CPU/GPU memory)
 * - Metal shader language (MSL)
 * - No tensor cores (uses AMX for matrix ops on CPU)
 */
class MpsPKBackend : public PKBackendInterface {
public:
    explicit MpsPKBackend(int device_id = 0);
    ~MpsPKBackend() override;
    
    // Backend Information
    PKBackendType get_type() const override;
    std::string get_name() const override;
    std::string get_display_name() const override;
    bool is_available() const override;
    PKCapabilities get_capabilities() const override;
    
    // Mode Support
    bool supports_mode(PKMode mode) const override;
    PKMode get_default_mode() const override;
    std::vector<PKMode> get_supported_modes() const override;
    
    // Component Access
    PKMemoryAllocator* get_allocator() override;
    PKAtomicOps* get_atomic_ops() override;
    PKTaskExecutor* get_executor() override;
    
    // Initialization
    bool initialize(const PKRuntimeConfig& config) override;
    void finalize() override;
    void reset() override;
    
    // Stream Management (Command Queue in Metal)
    void* create_stream() override;
    void destroy_stream(void* stream) override;
    void synchronize_stream(void* stream) override;
    void synchronize() override;
    
    // Kernel Launch
    void launch_worker_kernel(const PKRuntimeConfig& config,
                              int num_workers,
                              int threads_per_worker) override;
    void launch_scheduler_kernel(const PKRuntimeConfig& config) override;
    
    // Mode-Specific Operations
    bool prepare_next_batch(PKRuntimeConfig& config) override;
    void process_batch_results(PKRuntimeConfig& config) override;
    
    // Device Management
    bool set_device(int device_id) override;
    int get_device() const override;
    int get_device_count() const override;
    
    // Compilation
    std::vector<std::string> get_compile_flags(PKMode mode) const override;
    std::vector<std::string> get_include_dirs() const override;
    
    // MPS-specific methods
    std::string get_gpu_family() const;
    bool supports_apple_silicon() const;
    size_t get_max_threadgroup_memory() const;
    size_t get_max_threads_per_threadgroup() const;
    
private:
    int device_id_;
    bool initialized_;
    std::string gpu_family_;
    
    std::unique_ptr<MpsMemoryAllocator> allocator_;
    std::unique_ptr<MpsAtomicOps> atomic_ops_;
    std::unique_ptr<MpsTaskExecutor> executor_;
    
    void* mtl_device_;        // MTLDevice*
    void* command_queue_;     // MTLCommandQueue*
    
    void detect_capabilities();
};

} // namespace persistent_kernel
} // namespace yirage
