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
 * TPU Persistent Kernel Backend
 */

#pragma once

#include "persistent_kernel/pk_backend.h"

namespace yirage {
namespace pk {

/**
 * @brief TPU Persistent Kernel Backend
 * 
 * TPU architecture differences:
 * - Uses XLA/PJRT for execution
 * - 128x128 Matrix Multiply Unit (MXU)
 * - HBM memory with VMEM scratchpad
 * - BF16 native support
 */
class TPUPKBackend : public PKBackend {
public:
    TPUPKBackend();
    ~TPUPKBackend() override;

    // ========== Initialization ==========
    bool initialize(int device_id = 0) override;
    void shutdown() override;
    bool is_initialized() const override;

    // ========== Memory Management ==========
    void* allocate_hbm(size_t size);
    void* allocate_vmem(size_t size);
    void free_memory(void* ptr) override;
    
    bool copy_to_device(void* dst, const void* src, size_t size) override;
    bool copy_to_host(void* dst, const void* src, size_t size) override;

    // ========== Kernel Execution ==========
    bool launch_kernel(const PKKernelConfig& config) override;
    void synchronize() override;

    // ========== TPU-specific ==========
    
    /**
     * @brief Execute XLA computation on TPU
     */
    bool execute_xla(const std::string& hlo_module);
    
    /**
     * @brief Get TPU version
     */
    int get_tpu_version() const { return tpu_version_; }
    
    /**
     * @brief Get MXU configuration
     */
    void get_mxu_config(int& rows, int& cols) const {
        rows = 128; cols = 128;  // Standard MXU size
    }

private:
    bool is_initialized_;
    int device_id_;
    int tpu_version_;
    void* pjrt_client_;
};

// =============================================================================
// TPU-specific PK Configuration
// =============================================================================

struct TPUPKConfig {
    int mxu_tiles_m = 1;      // Number of MXU tiles in M dimension
    int mxu_tiles_n = 1;      // Number of MXU tiles in N dimension
    bool use_vmem = true;     // Use VMEM scratchpad
    bool use_bf16 = true;     // Use BF16 computation
    int pipeline_depth = 2;   // Memory pipeline depth
};

}  // namespace pk
}  // namespace yirage
