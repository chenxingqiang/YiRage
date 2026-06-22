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
 * Intel XPU Backend (Arc, Data Center GPU Max, Gaudi)
 */

#pragma once

#include "backend/backend_interface.h"
#include "type.h"

#ifdef YIRAGE_BACKEND_XPU_ENABLED
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#endif

namespace yirage {
namespace backend {

/**
 * @brief Intel XPU backend using SYCL/oneAPI
 * 
 * Supports:
 * - Intel Arc GPUs (Alchemist, Battlemage)
 * - Intel Data Center GPU Max (Ponte Vecchio)
 * - Intel Gaudi (HPU)
 * - SYCL/DPC++ programming model
 * - oneMKL, oneDNN optimizations
 */
class XPUBackend : public BackendInterface {
public:
    XPUBackend();
    ~XPUBackend() override;

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

    // ========== XPU-specific ==========
    
    /**
     * @brief Get device type (Arc, Max, Gaudi)
     */
    enum XPUDeviceType {
        XPU_ARC,        // Intel Arc (consumer)
        XPU_MAX,        // Data Center GPU Max
        XPU_GAUDI,      // Habana Gaudi
        XPU_UNKNOWN
    };
    
    XPUDeviceType get_device_type() const;
    
    /**
     * @brief Get number of Xe cores
     */
    int get_xe_cores() const;
    
    /**
     * @brief Check if XMX (Xe Matrix Extensions) available
     */
    bool has_xmx() const;
    
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    /**
     * @brief Get SYCL queue for this device
     */
    sycl::queue& get_queue();
#endif

private:
    bool check_xpu_availability();
    void query_device_properties();

    bool is_available_;
    int current_device_;
    int device_count_;
    XPUDeviceType device_type_;
    
#ifdef YIRAGE_BACKEND_XPU_ENABLED
    std::unique_ptr<sycl::queue> queue_;
    sycl::device device_;
#endif
};

// =============================================================================
// Intel XPU Architecture Info
// =============================================================================

struct IntelXPUInfo {
    std::string name;
    XPUBackend::XPUDeviceType type;
    int xe_cores;           // Number of Xe cores
    int xe_slices;          // Number of Xe slices
    size_t global_memory;   // Global memory in bytes
    size_t local_memory;    // SLM (Shared Local Memory) per workgroup
    bool has_xmx;           // XMX matrix extension support
    int simd_width;         // Preferred SIMD width (16 or 32)
};

IntelXPUInfo get_intel_xpu_info(int device_id = 0);

// Intel XPU architecture constants
namespace xpu {
    constexpr int ARC_SIMD_WIDTH = 16;
    constexpr int MAX_SIMD_WIDTH = 32;
    constexpr size_t SLM_SIZE_KB = 128;  // 128KB SLM per subslice
    constexpr int XMX_SIZE = 8;          // 8x8 XMX operations
}

}  // namespace backend
}  // namespace yirage
