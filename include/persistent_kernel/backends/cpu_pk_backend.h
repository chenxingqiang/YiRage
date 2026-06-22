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
#include <thread>
#include <vector>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// CPU Memory Allocator
// =============================================================================

class CpuMemoryAllocator : public PKMemoryAllocator {
public:
    CpuMemoryAllocator();
    ~CpuMemoryAllocator() override;
    
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
    size_t total_allocated_;
    bool use_aligned_alloc_;
    size_t alignment_;
};

// =============================================================================
// CPU Atomic Operations
// =============================================================================

/**
 * @brief CPU-specific atomic operations using C++ std::atomic
 */
class CpuAtomicOps : public PKAtomicOps {
public:
    CpuAtomicOps();
    ~CpuAtomicOps() override;
    
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
// CPU Task Executor
// =============================================================================

class CpuTaskExecutor : public PKTaskExecutor {
public:
    CpuTaskExecutor();
    ~CpuTaskExecutor() override;
    
    bool supports_task(PKTaskType type) const override;
    void execute(const PKTaskDesc& desc,
                 const PKRuntimeConfig& config,
                 void* shared_memory,
                 size_t shared_memory_size) override;
    size_t get_shared_memory_size(PKTaskType type) const override;
    const char* get_task_name(PKTaskType type) const override;
    
private:
    int num_threads_;
    bool use_avx_;
    bool use_avx512_;
};

// =============================================================================
// CPU Persistent Kernel Backend
// =============================================================================

/**
 * @brief CPU backend for persistent kernel execution
 * 
 * This backend executes kernels on CPU using thread pools and SIMD intrinsics.
 * It supports OFFLINE and ONEPASS modes only.
 */
class CpuPKBackend : public PKBackendInterface {
public:
    explicit CpuPKBackend(int num_threads = -1);  // -1 = auto-detect
    ~CpuPKBackend() override;
    
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
    
    // Stream Management (CPU uses thread pools instead)
    void* create_stream() override;
    void destroy_stream(void* stream) override;
    void synchronize_stream(void* stream) override;
    void synchronize() override;
    
    // Kernel Launch (CPU uses thread pool execution)
    void launch_worker_kernel(const PKRuntimeConfig& config,
                              int num_workers,
                              int threads_per_worker) override;
    void launch_scheduler_kernel(const PKRuntimeConfig& config) override;
    
    // Mode-Specific Operations
    bool prepare_next_batch(PKRuntimeConfig& config) override;
    void process_batch_results(PKRuntimeConfig& config) override;
    
    // Device Management (CPU always has device 0)
    bool set_device(int device_id) override;
    int get_device() const override;
    int get_device_count() const override;
    
    // Compilation
    std::vector<std::string> get_compile_flags(PKMode mode) const override;
    std::vector<std::string> get_include_dirs() const override;
    
    // CPU-specific methods
    int get_num_cores() const;
    int get_num_threads() const;
    bool supports_avx() const;
    bool supports_avx512() const;
    std::string get_cpu_name() const;
    
private:
    int num_threads_;
    bool initialized_;
    bool supports_avx_;
    bool supports_avx512_;
    std::string cpu_name_;
    
    std::unique_ptr<CpuMemoryAllocator> allocator_;
    std::unique_ptr<CpuAtomicOps> atomic_ops_;
    std::unique_ptr<CpuTaskExecutor> executor_;
    
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_;
    
    void detect_capabilities();
    void start_worker_threads(int num_workers);
    void stop_worker_threads();
};

} // namespace persistent_kernel
} // namespace yirage
