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

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// MACA Memory Allocator
// =============================================================================

/**
 * @brief MetaX MACA GPU memory allocator
 * 
 * MACA uses a CUDA-compatible API for memory management.
 */
class MacaMemoryAllocator : public PKMemoryAllocator {
public:
    MacaMemoryAllocator();
    ~MacaMemoryAllocator() override;
    
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
    int device_id_;
};

// =============================================================================
// MACA Atomic Operations
// =============================================================================

/**
 * @brief MACA-specific atomic operations
 * 
 * MACA uses a similar atomic API to CUDA.
 */
class MacaAtomicOps : public PKAtomicOps {
public:
    MacaAtomicOps();
    ~MacaAtomicOps() override;
    
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
// MACA Task Executor
// =============================================================================

class MacaTaskExecutor : public PKTaskExecutor {
public:
    MacaTaskExecutor();
    ~MacaTaskExecutor() override;
    
    bool supports_task(PKTaskType type) const override;
    void execute(const PKTaskDesc& desc,
                 const PKRuntimeConfig& config,
                 void* shared_memory,
                 size_t shared_memory_size) override;
    size_t get_shared_memory_size(PKTaskType type) const override;
    const char* get_task_name(PKTaskType type) const override;
    
private:
    int compute_capability_;
};

// =============================================================================
// MACA Persistent Kernel Backend
// =============================================================================

/**
 * @brief MetaX MACA GPU backend for persistent kernel execution
 * 
 * This backend executes kernels on MetaX MACA GPUs.
 * MACA provides a CUDA-compatible programming model.
 */
class MacaPKBackend : public PKBackendInterface {
public:
    explicit MacaPKBackend(int device_id = 0);
    ~MacaPKBackend() override;
    
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
    
    // Stream Management
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
    
    // MACA-specific methods
    int get_compute_capability() const;
    size_t get_max_shared_memory() const;
    
private:
    int device_id_;
    bool initialized_;
    int compute_capability_;
    
    std::unique_ptr<MacaMemoryAllocator> allocator_;
    std::unique_ptr<MacaAtomicOps> atomic_ops_;
    std::unique_ptr<MacaTaskExecutor> executor_;
    
    void* worker_stream_;
    void* scheduler_stream_;
    
    void detect_capabilities();
};

} // namespace persistent_kernel
} // namespace yirage
