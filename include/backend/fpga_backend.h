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
 * FPGA Backend (Intel/Xilinx/Lattice)
 */

#pragma once

#include "backend/backend_interface.h"
#include "type.h"

namespace yirage {
namespace backend {

/**
 * @brief FPGA backend for various FPGA platforms
 * 
 * Supports:
 * - Intel Stratix/Arria (via OpenCL or oneAPI)
 * - Xilinx Alveo (via Vitis/XRT)
 * - Lattice (future)
 * - OpenCL for FPGA programming model
 */
class FPGABackend : public BackendInterface {
public:
    enum FPGAVendor {
        FPGA_INTEL,
        FPGA_XILINX,
        FPGA_LATTICE,
        FPGA_UNKNOWN
    };
    
    FPGABackend();
    ~FPGABackend() override;

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

    // ========== FPGA-specific ==========
    
    /**
     * @brief Get FPGA vendor
     */
    FPGAVendor get_vendor() const;
    
    /**
     * @brief Get FPGA device name (e.g., "Stratix 10")
     */
    std::string get_fpga_name() const;
    
    /**
     * @brief Load bitstream/xclbin
     */
    bool load_bitstream(const std::string& path);
    
    /**
     * @brief Get number of DSP blocks
     */
    int get_dsp_count() const;
    
    /**
     * @brief Get on-chip memory (BRAM/URAM) size
     */
    size_t get_onchip_memory() const;

private:
    bool check_fpga_availability();
    void query_device_properties();

    bool is_available_;
    int current_device_;
    int device_count_;
    FPGAVendor vendor_;
    std::string fpga_name_;
    
    // OpenCL handles (opaque)
    void* cl_platform_;
    void* cl_device_;
    void* cl_context_;
    void* cl_queue_;
};

// =============================================================================
// FPGA Architecture Info
// =============================================================================

struct FPGAInfo {
    std::string name;
    FPGABackend::FPGAVendor vendor;
    int logic_elements;      // LEs or LUTs
    int dsp_blocks;          // DSP count
    size_t bram_bytes;       // Block RAM
    size_t uram_bytes;       // Ultra RAM (Xilinx)
    size_t ddr_bytes;        // DDR memory
    size_t hbm_bytes;        // HBM (if available)
    int clock_mhz;           // Kernel clock
};

FPGAInfo get_fpga_info(int device_id = 0);

}  // namespace backend
}  // namespace yirage
