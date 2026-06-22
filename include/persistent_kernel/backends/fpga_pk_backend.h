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
 * FPGA Persistent Kernel Backend
 */

#pragma once

#include "persistent_kernel/pk_backend.h"

namespace yirage {
namespace pk {

/**
 * @brief FPGA Persistent Kernel Backend
 * 
 * Architecture features:
 * - Pipeline-based execution
 * - DSP blocks for multiply-accumulate
 * - On-chip BRAM/URAM for local storage
 * - DDR/HBM for global memory
 */
class FPGAPKBackend : public PKBackend {
public:
    FPGAPKBackend();
    ~FPGAPKBackend() override;

    // ========== Initialization ==========
    bool initialize(int device_id = 0) override;
    void shutdown() override;
    bool is_initialized() const override;

    // ========== Memory Management ==========
    void* allocate_ddr(size_t size);
    void* allocate_bram(size_t size);  // On-chip memory
    void free_memory(void* ptr) override;
    
    bool copy_to_device(void* dst, const void* src, size_t size) override;
    bool copy_to_host(void* dst, const void* src, size_t size) override;

    // ========== Kernel Execution ==========
    bool launch_kernel(const PKKernelConfig& config) override;
    void synchronize() override;

    // ========== FPGA-specific ==========
    
    /**
     * @brief Load bitstream (.aocx for Intel, .xclbin for Xilinx)
     */
    bool load_bitstream(const std::string& path);
    
    /**
     * @brief Get pipeline depth
     */
    int get_pipeline_depth() const { return pipeline_depth_; }
    
    /**
     * @brief Get initiation interval (II)
     */
    int get_initiation_interval() const { return ii_; }
    
    /**
     * @brief Get kernel clock frequency
     */
    int get_clock_mhz() const { return clock_mhz_; }

private:
    bool is_initialized_;
    int device_id_;
    int pipeline_depth_;
    int ii_;  // Initiation interval
    int clock_mhz_;
    
    void* cl_context_;
    void* cl_queue_;
    void* cl_kernel_;
};

// =============================================================================
// FPGA-specific PK Configuration
// =============================================================================

struct FPGAPKConfig {
    std::string kernel_name;       // Kernel function name
    int num_compute_units = 1;     // Number of kernel instances
    int pipeline_depth = 16;       // Loop pipeline depth
    bool use_hbm = false;          // Use HBM if available
    size_t bram_usage = 0;         // BRAM to allocate
    size_t uram_usage = 0;         // URAM to allocate (Xilinx)
};

}  // namespace pk
}  // namespace yirage
