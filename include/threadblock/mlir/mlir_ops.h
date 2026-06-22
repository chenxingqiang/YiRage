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
 * MLIR Threadblock Operations
 * Generate MLIR IR for threadblock-level operations
 */

#pragma once

#include "type.h"
#include <string>
#include <vector>
#include <map>

namespace yirage {
namespace threadblock {
namespace mlir_ops {

// =============================================================================
// MLIR Operation Types
// =============================================================================

enum MLIROpType {
    MLIR_MATMUL,
    MLIR_ATTENTION,
    MLIR_SOFTMAX,
    MLIR_RMS_NORM,
    MLIR_LAYER_NORM,
    MLIR_ELEMENTWISE,
    MLIR_REDUCE,
    MLIR_BROADCAST,
    MLIR_TRANSPOSE,
    MLIR_GATHER,
    MLIR_SCATTER,
};

// =============================================================================
// MLIR Tile Configuration
// =============================================================================

struct MLIRTileConfig {
    std::vector<int> tile_sizes = {32, 32, 32};
    int vector_width = 4;
    bool enable_vectorization = true;
    bool enable_loop_unroll = false;
    int unroll_factor = 4;
    bool enable_tiling = true;
    bool enable_fusion = true;
};

// =============================================================================
// MLIR Code Generator for Threadblock Operations
// =============================================================================

class MLIRCodeGenerator {
public:
    /**
     * @brief Generate MLIR for MatMul using Linalg dialect
     */
    static std::string generate_matmul(
        int M, int N, int K,
        type::DataType dtype,
        const MLIRTileConfig& config
    );
    
    /**
     * @brief Generate MLIR for Flash Attention
     */
    static std::string generate_flash_attention(
        int batch, int heads, int seq_len, int head_dim,
        bool causal,
        const MLIRTileConfig& config
    );
    
    /**
     * @brief Generate MLIR for RMS Normalization
     */
    static std::string generate_rms_norm(
        int hidden_dim,
        float epsilon,
        type::DataType dtype
    );
    
    /**
     * @brief Generate MLIR for Layer Normalization
     */
    static std::string generate_layer_norm(
        int hidden_dim,
        float epsilon,
        type::DataType dtype
    );
    
    /**
     * @brief Generate MLIR for element-wise operations
     */
    static std::string generate_elementwise(
        MLIROpType op,
        const std::vector<int>& shape,
        type::DataType dtype
    );
    
    /**
     * @brief Generate MLIR for reduction operations
     */
    static std::string generate_reduce(
        const std::string& reduce_op,  // "add", "max", "min"
        const std::vector<int>& shape,
        const std::vector<int>& reduce_dims,
        type::DataType dtype
    );
    
    /**
     * @brief Generate MLIR for SwiGLU activation
     */
    static std::string generate_swiglu(
        int hidden_dim,
        type::DataType dtype
    );
    
    /**
     * @brief Generate MLIR for fused operations
     */
    static std::string generate_fused_ops(
        const std::vector<MLIROpType>& ops,
        const std::vector<std::vector<int>>& shapes,
        type::DataType dtype
    );
};

// =============================================================================
// MLIR Pass Pipeline for Threadblock Optimization
// =============================================================================

struct MLIRThreadblockPassConfig {
    bool tile_and_fuse = true;
    bool vectorize = true;
    bool parallelize = true;
    bool bufferize = true;
    std::vector<int> tile_sizes = {32, 32, 32};
    int vector_width = 4;
    int num_threads = 1;
};

class MLIRPassPipeline {
public:
    /**
     * @brief Build optimization pass pipeline for threadblock ops
     */
    static std::vector<std::string> build_threadblock_pipeline(
        const MLIRThreadblockPassConfig& config
    );
    
    /**
     * @brief Build pass pipeline for specific target
     */
    static std::vector<std::string> build_target_pipeline(
        const std::string& target,  // "llvm", "nvvm", "rocm", "spirv"
        const MLIRThreadblockPassConfig& config
    );
    
    /**
     * @brief Run passes on MLIR module
     */
    static bool run_passes(
        void* mlir_context,
        void* mlir_module,
        const std::vector<std::string>& passes
    );
};

// =============================================================================
// MLIR Kernel Registry
// =============================================================================

class MLIRKernelRegistry {
public:
    static MLIRKernelRegistry& instance();
    
    void register_kernel(const std::string& name, const std::string& mlir_code);
    std::string get_kernel(const std::string& name) const;
    bool has_kernel(const std::string& name) const;
    
    /**
     * @brief Compile MLIR to target (LLVM IR, PTX, etc.)
     */
    bool compile_kernel(const std::string& name,
                       const std::string& target);
    
    /**
     * @brief Get compiled binary for kernel
     */
    std::vector<uint8_t> get_compiled_binary(const std::string& name) const;

private:
    MLIRKernelRegistry() = default;
    std::map<std::string, std::string> kernels_;
    std::map<std::string, std::vector<uint8_t>> compiled_binaries_;
};

// =============================================================================
// MLIR Dialect Utilities
// =============================================================================

namespace dialect {

/**
 * @brief Get type string for MLIR
 */
std::string get_mlir_type(type::DataType dtype);

/**
 * @brief Get tensor type string
 */
std::string get_tensor_type(const std::vector<int>& shape, type::DataType dtype);

/**
 * @brief Get memref type string
 */
std::string get_memref_type(const std::vector<int>& shape, type::DataType dtype);

/**
 * @brief Generate affine map for tiling
 */
std::string generate_affine_map(int num_dims);

}  // namespace dialect

}  // namespace mlir_ops
}  // namespace threadblock
}  // namespace yirage
