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
 * MLIR Backend - Compiler Infrastructure Integration
 */

#pragma once

#include "backend/backend_interface.h"
#include "type.h"

#include <memory>
#include <functional>

namespace yirage {
namespace backend {

/**
 * @brief MLIR Backend for compiler-based code generation
 * 
 * This backend uses MLIR as an intermediate representation
 * to generate optimized code for various target platforms.
 * 
 * Supports lowering to:
 * - LLVM IR (for CPU targets)
 * - NVVM/PTX (for CUDA)
 * - ROCDL (for ROCm)
 * - SPIR-V (for OpenCL/Vulkan)
 */
class MLIRBackend : public BackendInterface {
public:
    /**
     * @brief Target backend for MLIR lowering
     */
    enum MLIRTarget {
        MLIR_TARGET_LLVM,       // LLVM IR for CPU
        MLIR_TARGET_NVVM,       // NVIDIA PTX
        MLIR_TARGET_ROCDL,      // AMD ROCm
        MLIR_TARGET_SPIRV,      // SPIR-V
        MLIR_TARGET_VULKAN,     // Vulkan compute
        MLIR_TARGET_WASM,       // WebAssembly
    };
    
    MLIRBackend(MLIRTarget target = MLIR_TARGET_LLVM);
    ~MLIRBackend() override;

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

    // ========== Memory Management (delegated to target) ==========
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

    // ========== MLIR-specific ==========
    
    /**
     * @brief Set target for code generation
     */
    void set_target(MLIRTarget target);
    MLIRTarget get_target() const;
    
    /**
     * @brief Compile MLIR module to executable
     */
    bool compile_mlir(const std::string& mlir_module,
                     std::string& output);
    
    /**
     * @brief Run MLIR optimization passes
     */
    bool run_passes(const std::string& input_mlir,
                   const std::vector<std::string>& passes,
                   std::string& output_mlir);
    
    /**
     * @brief Lower YiRage dialect to target
     */
    bool lower_yirage_dialect(const std::string& yirage_mlir,
                             std::string& lowered_mlir);
    
    /**
     * @brief JIT compile and get function pointer
     */
    using JITFunction = std::function<void(void**)>;
    bool jit_compile(const std::string& mlir_module,
                    const std::string& entry_point,
                    JITFunction& func);

private:
    bool check_mlir_availability();
    
    MLIRTarget target_;
    bool is_available_;
    int device_id_;
    
    // MLIR context (opaque)
    void* mlir_context_;
    void* execution_engine_;
};

// =============================================================================
// MLIR Pass Pipeline Configuration
// =============================================================================

struct MLIRPassConfig {
    bool canonicalize = true;
    bool cse = true;                    // Common subexpression elimination
    bool loop_fusion = true;
    bool loop_unroll = false;
    int unroll_factor = 4;
    bool vectorize = true;
    int vector_width = 4;
    bool tile_loops = true;
    std::vector<int> tile_sizes = {32, 32, 32};
    bool bufferize = true;
    bool convert_to_llvm = true;
};

/**
 * @brief Build MLIR pass pipeline from config
 */
std::vector<std::string> build_pass_pipeline(const MLIRPassConfig& config);

// =============================================================================
// MLIR Dialect Registration
// =============================================================================

/**
 * @brief Register YiRage dialect with MLIR context
 */
bool register_yirage_dialect(void* mlir_context);

/**
 * @brief Get list of available dialects
 */
std::vector<std::string> get_available_dialects();

}  // namespace backend
}  // namespace yirage
