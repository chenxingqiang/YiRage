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
 * AWS NKI (Neuron Kernel Interface) Persistent Kernel Backend
 */

#pragma once

#include "persistent_kernel/pk_backend.h"
#include "threadblock/nki/nki_ops.h"

#include <string>

namespace yirage {
namespace pk {

/**
 * @brief NKI Persistent Kernel Backend for AWS Trainium/Inferentia
 * 
 * Architecture features:
 * - NeuronCore with Tensor Engine (128x128)
 * - Vector Engine for element-wise ops
 * - SBUF (24MB on-chip memory per core)
 * - HBM for global memory
 */
class NKIPKBackend : public PKBackend {
public:
    NKIPKBackend();
    ~NKIPKBackend() override;

    // ========== Initialization ==========
    bool initialize(int device_id = 0) override;
    void shutdown() override;
    bool is_initialized() const override;

    // ========== Memory Management ==========
    void* allocate_hbm(size_t size);
    void* allocate_sbuf(size_t size);  // On-chip SBUF
    void free_memory(void* ptr) override;
    
    bool copy_to_device(void* dst, const void* src, size_t size) override;
    bool copy_to_host(void* dst, const void* src, size_t size) override;

    // ========== Kernel Execution ==========
    bool launch_kernel(const PKKernelConfig& config) override;
    void synchronize() override;

    // ========== NKI-specific ==========
    
    /**
     * @brief Compile NKI kernel
     */
    bool compile_nki_kernel(const std::string& kernel_code,
                           const std::string& kernel_name);
    
    /**
     * @brief Launch NKI kernel with configuration
     */
    bool launch_nki_kernel(const std::string& kernel_name,
                          void** args,
                          int num_args,
                          const threadblock::nki::NKITileConfig& config);
    
    /**
     * @brief Get number of NeuronCores
     */
    int get_neuron_core_count() const;
    
    /**
     * @brief Get chip type (Trainium v1, v2, Inferentia)
     */
    enum NeuronChipType {
        TRAINIUM_V1,
        TRAINIUM_V2,
        INFERENTIA_V1,
        INFERENTIA_V2,
        UNKNOWN_CHIP
    };
    
    NeuronChipType get_chip_type() const;

private:
    bool is_initialized_;
    int device_id_;
    int neuron_core_count_;
    NeuronChipType chip_type_;
    
    // Neuron Runtime handle
    void* nrt_model_;
};

// =============================================================================
// NKI-specific PK Configuration
// =============================================================================

struct NKIPKConfig {
    std::string kernel_code;
    std::string kernel_name;
    threadblock::nki::NKITileConfig tile_config;
    int num_cores = 1;              // NeuronCores to use
    bool use_tensor_engine = true;  // Use Tensor Engine
    bool use_vector_engine = true;  // Use Vector Engine
};

}  // namespace pk
}  // namespace yirage
