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
 * ROCm Backend for AMD GPUs (MI100, MI200, MI300 series)
 */

#pragma once

#include "backend/backend_interface.h"
#include "type.h"

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

namespace yirage {
namespace backend {

/**
 * @brief ROCm backend for AMD GPUs
 * 
 * Supports:
 * - AMD Instinct MI100/MI200/MI300 series
 * - AMD Radeon Pro series
 * - HIP runtime API
 * - hipBLAS, rocBLAS for optimized kernels
 */
class ROCmBackend : public BackendInterface {
public:
    ROCmBackend();
    ~ROCmBackend() override;

    // ========== Backend Information ==========
    type::BackendType get_type() const override;
    std::string get_name() const override;
    std::string get_display_name() const override;
    bool is_available() const override;
    type::BackendInfo get_info() const override;

    // ========== Compilation ==========
    bool compile(CompileContext const& ctx) override;
    std::string get_compile_flags() const override;
    std::vector<std::string> get_include_dirs() const override;
    std::vector<std::string> get_library_dirs() const override;
    std::vector<std::string> get_link_libraries() const override;

    // ========== Memory Management ==========
    void* allocate_memory(size_t size) override;
    void free_memory(void* ptr) override;
    bool copy_to_device(void* dst, void const* src, size_t size) override;
    bool copy_to_host(void* dst, void const* src, size_t size) override;
    bool copy_device_to_device(void* dst, void const* src, size_t size) override;

    // ========== Synchronization ==========
    void synchronize() override;

    // ========== Capability Query ==========
    size_t get_max_memory() const override;
    size_t get_max_shared_memory() const override;
    bool supports_data_type(type::DataType dt) const override;
    int get_compute_capability() const override;
    int get_num_compute_units() const override;

    // ========== Device Management ==========
    bool set_device(int device_id) override;
    int get_device() const override;
    int get_device_count() const override;

    // ========== ROCm-specific ==========
    
    /**
     * @brief Get GCN architecture name (e.g., "gfx90a" for MI200)
     */
    std::string get_arch_name() const;
    
    /**
     * @brief Get wavefront size (64 for AMD GPUs)
     */
    int get_wavefront_size() const { return 64; }  // AMD uses 64-thread wavefronts
    
    /**
     * @brief Check if Matrix Core (CDNA) is available
     */
    bool has_matrix_cores() const;
    
    /**
     * @brief Get LDS (Local Data Share) size per workgroup
     */
    size_t get_lds_size() const;

private:
    bool check_rocm_availability();
    void query_device_properties();

    bool is_available_;
    int current_device_;
    int device_count_;
    
#ifdef YIRAGE_BACKEND_ROCM_ENABLED
    hipDeviceProp_t device_prop_;
    rocblas_handle rocblas_handle_;
#endif
};

// =============================================================================
// AMD GPU Architecture Info
// =============================================================================

struct AMDGPUInfo {
    std::string arch_name;      // e.g., "gfx90a", "gfx942"
    int compute_units;          // Number of CUs
    int wavefront_size;         // Always 64 for AMD
    size_t global_memory;       // Global memory in bytes
    size_t lds_size;            // LDS per workgroup
    bool has_matrix_cores;      // CDNA matrix cores
    int matrix_core_gen;        // 1=MI100, 2=MI200, 3=MI300
};

AMDGPUInfo get_amd_gpu_info(int device_id = 0);

}  // namespace backend
}  // namespace yirage
