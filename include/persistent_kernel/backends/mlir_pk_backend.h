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
 * MLIR Persistent Kernel Backend
 * JIT compilation and execution of MLIR kernels
 */

#pragma once

#include "persistent_kernel/pk_backend.h"
#include "threadblock/mlir/mlir_ops.h"

#include <string>
#include <functional>
#include <memory>

namespace yirage {
namespace pk {

/**
 * @brief MLIR Persistent Kernel Backend
 * 
 * Provides JIT compilation and execution of MLIR kernels.
 * Supports multiple target backends through MLIR's compilation
 * infrastructure.
 */
class MLIRPKBackend : public PKBackend {
public:
    /**
     * @brief Target for JIT compilation
     */
    enum JITTarget {
        JIT_TARGET_CPU,         // LLVM for CPU
        JIT_TARGET_CUDA,        // NVVM for CUDA
        JIT_TARGET_ROCM,        // ROCDL for ROCm
        JIT_TARGET_VULKAN,      // SPIR-V for Vulkan
    };
    
    MLIRPKBackend(JITTarget target = JIT_TARGET_CPU);
    ~MLIRPKBackend() override;

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

    // ========== MLIR-specific ==========
    
    /**
     * @brief JIT compile MLIR module
     */
    bool jit_compile(const std::string& mlir_module);
    
    /**
     * @brief Get function pointer for compiled kernel
     */
    using KernelFunc = std::function<void(void**)>;
    bool get_kernel_func(const std::string& name, KernelFunc& func);
    
    /**
     * @brief Execute JIT-compiled kernel
     */
    bool execute(const std::string& kernel_name, void** args, int num_args);
    
    /**
     * @brief Run optimization passes before compilation
     */
    bool run_optimization_passes(
        const std::string& input_mlir,
        std::string& optimized_mlir,
        const threadblock::mlir_ops::MLIRThreadblockPassConfig& config
    );
    
    /**
     * @brief Set JIT target
     */
    void set_target(JITTarget target);
    JITTarget get_target() const;
    
    /**
     * @brief Enable/disable caching of compiled kernels
     */
    void set_cache_enabled(bool enabled);
    bool is_cache_enabled() const;
    
    /**
     * @brief Get compilation statistics
     */
    struct CompileStats {
        int kernels_compiled;
        int cache_hits;
        double total_compile_time_ms;
        double average_compile_time_ms;
    };
    CompileStats get_compile_stats() const;

private:
    bool compile_to_target(const std::string& mlir_module);
    std::string get_target_triple() const;
    
    bool is_initialized_;
    int device_id_;
    JITTarget target_;
    bool cache_enabled_;
    
    // MLIR execution engine (opaque)
    void* execution_engine_;
    void* mlir_context_;
    
    // Compilation stats
    CompileStats stats_;
    
    // Kernel cache
    std::map<std::string, void*> kernel_cache_;
};

// =============================================================================
// MLIR-specific PK Configuration
// =============================================================================

struct MLIRPKConfig {
    std::string mlir_module;                // MLIR source
    std::string entry_point;                // Kernel entry function
    MLIRPKBackend::JITTarget target = MLIRPKBackend::JIT_TARGET_CPU;
    threadblock::mlir_ops::MLIRTileConfig tile_config;
    threadblock::mlir_ops::MLIRThreadblockPassConfig pass_config;
    bool enable_profiling = false;
    bool enable_caching = true;
};

// =============================================================================
// MLIR Execution Utilities
// =============================================================================

/**
 * @brief Execute MLIR matmul kernel
 */
bool mlir_execute_matmul(
    MLIRPKBackend& backend,
    const void* A, const void* B, void* C,
    int M, int N, int K,
    type::DataType dtype
);

/**
 * @brief Execute MLIR attention kernel
 */
bool mlir_execute_attention(
    MLIRPKBackend& backend,
    const void* Q, const void* K, const void* V, void* Out,
    int batch, int heads, int seq_len, int head_dim,
    bool causal
);

/**
 * @brief Execute MLIR RMS norm kernel
 */
bool mlir_execute_rms_norm(
    MLIRPKBackend& backend,
    const void* input, const void* gamma, void* output,
    int batch, int hidden_dim,
    float epsilon
);

}  // namespace pk
}  // namespace yirage
