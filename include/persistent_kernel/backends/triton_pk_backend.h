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
 * Triton Persistent Kernel Backend
 */

#pragma once

#include "persistent_kernel/pk_backend.h"
#include "threadblock/triton/triton_ops.h"

#include <string>

namespace yirage {
namespace pk {

/**
 * @brief Triton Persistent Kernel Backend
 * 
 * Enables using Triton kernels in persistent kernel context
 * by managing kernel compilation and execution.
 */
class TritonPKBackend : public PKBackend {
public:
    TritonPKBackend();
    ~TritonPKBackend() override;

    // ========== Initialization ==========
    bool initialize(int device_id = 0) override;
    void shutdown() override;
    bool is_initialized() const override;

    // ========== Memory Management ==========
    void* allocate_memory(size_t size);
    void free_memory(void* ptr) override;
    
    bool copy_to_device(void* dst, const void* src, size_t size) override;
    bool copy_to_host(void* dst, const void* src, size_t size) override;

    // ========== Kernel Execution ==========
    bool launch_kernel(const PKKernelConfig& config) override;
    void synchronize() override;

    // ========== Triton-specific ==========
    
    /**
     * @brief Compile Triton kernel from Python source
     */
    bool compile_kernel(const std::string& kernel_code,
                       const std::string& kernel_name);
    
    /**
     * @brief Launch a compiled Triton kernel
     */
    bool launch_triton_kernel(const std::string& kernel_name,
                             void** args,
                             int num_args,
                             const threadblock::triton::TritonTileConfig& config);
    
    /**
     * @brief Get target architecture string
     */
    std::string get_target_arch() const;

private:
    bool is_initialized_;
    int device_id_;
    std::string target_arch_;
    
    // Handle to underlying GPU backend (CUDA or HIP)
    void* gpu_context_;
};

// =============================================================================
// Triton-specific PK Configuration
// =============================================================================

struct TritonPKConfig {
    std::string kernel_code;                    // Triton kernel source
    std::string kernel_name;                    // Entry point name
    threadblock::triton::TritonTileConfig tile_config;
    bool enable_autotune = true;
    int max_autotune_iterations = 100;
};

}  // namespace pk
}  // namespace yirage
