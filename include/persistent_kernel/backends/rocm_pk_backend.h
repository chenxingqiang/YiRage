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

#pragma once

/**
 * @file rocm_pk_backend.h
 * @brief AMD ROCm/HIP Persistent Kernel Backend
 *
 * This backend provides persistent kernel execution for AMD GPUs using HIP.
 * It supports CDNA architectures (MI100, MI200, MI250, MI300 series).
 *
 * Key features:
 * - HIP-based GPU execution
 * - MFMA (Matrix Fused Multiply-Add) acceleration
 * - 64-thread wavefront operations
 * - LDS (Local Data Share) optimization
 */

#include "persistent_kernel/pk_backend_interface.h"
#include "persistent_kernel/tasks/rocm/task_header.h"

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#include <hip/hip_runtime.h>
#endif

#include <memory>
#include <string>
#include <vector>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// ROCm Memory Allocator
// =============================================================================

/**
 * @brief HIP memory allocator for AMD GPUs
 */
class ROCmMemoryAllocator : public PKMemoryAllocator {
public:
    ROCmMemoryAllocator();
    ~ROCmMemoryAllocator() override;

    void* allocate(size_t size, PKMemoryType type) override;
    void deallocate(void* ptr) override;
    void copy(void* dst, const void* src, size_t size, 
              PKMemoryType dst_type, PKMemoryType src_type) override;
    void memset(void* ptr, int value, size_t size) override;
    void synchronize() override;

    // ROCm-specific
    void* allocate_managed(size_t size);
    void prefetch_to_device(void* ptr, size_t size, int device_id);
    void prefetch_to_host(void* ptr, size_t size);

private:
    int device_id_;
};

// =============================================================================
// ROCm Atomic Operations
// =============================================================================

/**
 * @brief HIP atomic operations for AMD GPUs
 */
class ROCmAtomicOps : public PKAtomicOps {
public:
    ROCmAtomicOps();
    ~ROCmAtomicOps() override;

    int atomic_add(int* addr, int val) override;
    int atomic_cas(int* addr, int compare, int val) override;
    int atomic_max(int* addr, int val) override;
    int atomic_min(int* addr, int val) override;
    void memory_fence() override;

    // ROCm-specific atomics
    int64_t atomic_add_64(int64_t* addr, int64_t val);
    uint32_t atomic_or(uint32_t* addr, uint32_t val);
};

// =============================================================================
// ROCm Task Executor
// =============================================================================

/**
 * @brief HIP task executor for AMD GPUs
 */
class ROCmTaskExecutor : public PKTaskExecutor {
public:
    ROCmTaskExecutor();
    ~ROCmTaskExecutor() override;

    void execute_task(PKTaskType type, const void* params, 
                     const PKRuntimeConfig& config) override;
    bool is_task_supported(PKTaskType type) const override;
    std::vector<PKTaskType> get_supported_tasks() const override;

    // ROCm-specific
    void execute_mfma_gemm(const void* params, const PKRuntimeConfig& config);
    void execute_sparse_gemm(const void* params, const PKRuntimeConfig& config);
    void execute_fp8_gemm(const void* params, const PKRuntimeConfig& config);

private:
    rocm::AMDArch detected_arch_;
};

// =============================================================================
// ROCm Persistent Kernel Backend
// =============================================================================

/**
 * @brief AMD ROCm/HIP Persistent Kernel Backend
 *
 * Features:
 * - CDNA architecture support (MI100-MI300)
 * - MFMA acceleration
 * - Wavefront-level optimization
 * - LDS management
 */
class ROCmPKBackend : public PKBackendInterface {
public:
    explicit ROCmPKBackend(int device_id = 0);
    ~ROCmPKBackend() override;

    // Core interface
    PKBackendType get_type() const override;
    std::string get_name() const override;
    std::string get_display_name() const override;
    PKCapabilities get_capabilities() const override;

    // Initialization
    bool initialize(const PKRuntimeConfig& config) override;
    void finalize() override;
    bool is_initialized() const override;

    // Kernel execution
    void launch_worker_kernel(const PKRuntimeConfig& config,
                             int num_blocks, int block_size) override;
    void launch_scheduler_kernel(const PKRuntimeConfig& config) override;
    void synchronize() override;

    // Memory management
    PKMemoryAllocator* get_allocator() override;
    PKAtomicOps* get_atomic_ops() override;
    PKTaskExecutor* get_executor() override;

    // Stream management
    void create_streams(int num_streams) override;
    void destroy_streams() override;
    void* get_stream(int index) override;

    // Event management
    void create_event(void** event) override;
    void destroy_event(void* event) override;
    void record_event(void* event, void* stream) override;
    void wait_event(void* event, void* stream) override;
    float elapsed_time(void* start, void* end) override;

    // Query methods
    int get_device_count() const override;
    int get_current_device() const override;
    void set_device(int device_id) override;
    size_t get_free_memory() const override;
    size_t get_total_memory() const override;

    // ROCm-specific methods
    rocm::AMDArch get_architecture() const;
    std::string get_gfx_target() const;
    int get_compute_unit_count() const;
    int get_wavefront_size() const;
    size_t get_lds_size() const;
    bool supports_fp8() const;
    bool supports_sparsity() const;

    // Kernel compilation
    bool compile_kernel(const std::string& source, const std::string& kernel_name);
    void* get_compiled_kernel(const std::string& kernel_name);

private:
    int device_id_;
    bool initialized_;
    rocm::AMDArch architecture_;
    
    std::unique_ptr<ROCmMemoryAllocator> allocator_;
    std::unique_ptr<ROCmAtomicOps> atomic_ops_;
    std::unique_ptr<ROCmTaskExecutor> executor_;

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipStream_t worker_stream_;
    hipStream_t scheduler_stream_;
    std::vector<hipStream_t> streams_;
#else
    void* worker_stream_;
    void* scheduler_stream_;
    std::vector<void*> streams_;
#endif

    // Compiled kernels cache
    std::unordered_map<std::string, void*> kernel_cache_;
};

// =============================================================================
// ROCm Backend Factory
// =============================================================================

/**
 * @brief Check if ROCm is available
 */
bool is_rocm_available();

/**
 * @brief Get available ROCm devices
 */
std::vector<int> get_rocm_devices();

/**
 * @brief Create ROCm backend
 */
std::unique_ptr<PKBackendInterface> create_rocm_backend(int device_id = 0);

/**
 * @brief Get ROCm device properties
 */
struct ROCmDeviceProperties {
    std::string name;
    rocm::AMDArch architecture;
    int compute_units;
    size_t total_memory;
    size_t lds_per_cu;
    int wavefront_size;
    int max_threads_per_block;
    bool supports_mfma;
    bool supports_fp8;
    bool supports_sparsity;
};

ROCmDeviceProperties get_rocm_device_properties(int device_id = 0);

}  // namespace persistent_kernel
}  // namespace yirage
