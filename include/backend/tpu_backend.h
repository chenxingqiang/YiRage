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
 * TPU Backend for Google Cloud TPU (v2, v3, v4, v5)
 */

#pragma once

#include "backend/backend_interface.h"
#include "type.h"

namespace yirage {
namespace backend {

/**
 * @brief TPU backend for Google Cloud TPU
 * 
 * Supports:
 * - TPU v2, v3, v4, v5 via libtpu
 * - XLA compilation
 * - PJRT runtime
 */
class TPUBackend : public BackendInterface {
public:
    TPUBackend();
    ~TPUBackend() override;

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

    // ========== TPU-specific ==========
    
    /**
     * @brief Get TPU version (2, 3, 4, 5)
     */
    int get_tpu_version() const;
    
    /**
     * @brief Get number of TPU cores per chip
     */
    int get_cores_per_chip() const;
    
    /**
     * @brief Get HBM memory per chip in bytes
     */
    size_t get_hbm_per_chip() const;
    
    /**
     * @brief Compile XLA HLO to TPU executable
     */
    bool compile_xla(const std::string& hlo_text, std::string& executable);

private:
    bool check_tpu_availability();
    void query_device_properties();

    bool is_available_;
    int current_device_;
    int device_count_;
    int tpu_version_;
    
    // PJRT client handle (opaque)
    void* pjrt_client_;
};

// =============================================================================
// TPU Architecture Info
// =============================================================================

struct TPUInfo {
    int version;            // TPU v2, v3, v4, v5
    int cores_per_chip;     // 2 for v2/v3, 4 for v4, 8 for v5
    size_t hbm_per_chip;    // HBM memory
    int mxu_size;           // Matrix multiply unit size (128x128)
    float peak_tflops_bf16; // BF16 peak performance
};

TPUInfo get_tpu_info();

// TPU architecture constants
namespace tpu {
    constexpr int MXU_SIZE = 128;          // 128x128 systolic array
    constexpr int V2_CORES_PER_CHIP = 2;
    constexpr int V3_CORES_PER_CHIP = 2;
    constexpr int V4_CORES_PER_CHIP = 4;
    constexpr int V5_CORES_PER_CHIP = 8;
    
    constexpr size_t V2_HBM_GB = 8;
    constexpr size_t V3_HBM_GB = 16;
    constexpr size_t V4_HBM_GB = 32;
    constexpr size_t V5_HBM_GB = 96;
}

}  // namespace backend
}  // namespace yirage
