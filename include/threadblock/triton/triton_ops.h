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
 * Triton Threadblock Operations
 * Bridge between YiRage and Triton's tile-based programming model
 */

#pragma once

#include "type.h"
#include <string>
#include <vector>
#include <map>

namespace yirage {
namespace threadblock {
namespace triton {

// =============================================================================
// Triton Operation Types
// =============================================================================

enum TritonOpType {
    TRITON_MATMUL,
    TRITON_ATTENTION,
    TRITON_SOFTMAX,
    TRITON_REDUCE,
    TRITON_ELEMENTWISE,
    TRITON_SCAN,
};

// =============================================================================
// Triton Tile Configuration
// =============================================================================

struct TritonTileConfig {
    int block_m = 64;       // Tile size in M dimension
    int block_n = 64;       // Tile size in N dimension
    int block_k = 32;       // Tile size in K dimension
    int num_warps = 4;      // Number of warps
    int num_stages = 2;     // Pipeline stages
    bool use_mma = true;    // Use tensor core MMA
};

// =============================================================================
// Triton Code Generator
// =============================================================================

class TritonCodeGenerator {
public:
    /**
     * @brief Generate Triton kernel code for MatMul
     */
    static std::string generate_matmul(
        int M, int N, int K,
        type::DataType dtype,
        const TritonTileConfig& config
    );
    
    /**
     * @brief Generate Triton kernel code for Flash Attention
     */
    static std::string generate_flash_attention(
        int batch, int heads, int seq_len, int head_dim,
        bool causal,
        const TritonTileConfig& config
    );
    
    /**
     * @brief Generate Triton kernel code for RMS Norm
     */
    static std::string generate_rms_norm(
        int hidden_dim,
        float epsilon,
        type::DataType dtype
    );
    
    /**
     * @brief Generate Triton kernel code for SwiGLU
     */
    static std::string generate_swiglu(
        int hidden_dim,
        type::DataType dtype
    );
    
    /**
     * @brief Generate Triton kernel code for element-wise op
     */
    static std::string generate_elementwise(
        TritonOpType op,
        int size,
        type::DataType dtype
    );
};

// =============================================================================
// Triton Kernel Registry
// =============================================================================

class TritonKernelRegistry {
public:
    static TritonKernelRegistry& instance();
    
    /**
     * @brief Register a Triton kernel
     */
    void register_kernel(const std::string& name, const std::string& code);
    
    /**
     * @brief Get kernel code by name
     */
    std::string get_kernel(const std::string& name) const;
    
    /**
     * @brief Check if kernel exists
     */
    bool has_kernel(const std::string& name) const;
    
    /**
     * @brief Compile kernel to PTX/HSACO
     */
    bool compile_kernel(const std::string& name, 
                       const std::string& target_arch);

private:
    TritonKernelRegistry() = default;
    std::map<std::string, std::string> kernels_;
    std::map<std::string, std::string> compiled_; // name -> binary path
};

// =============================================================================
// Triton Autotuner
// =============================================================================

struct TritonAutotuneConfig {
    std::vector<int> block_m_options = {32, 64, 128};
    std::vector<int> block_n_options = {32, 64, 128};
    std::vector<int> block_k_options = {16, 32, 64};
    std::vector<int> num_warps_options = {2, 4, 8};
    std::vector<int> num_stages_options = {2, 3, 4};
};

class TritonAutotuner {
public:
    /**
     * @brief Find best configuration for MatMul
     */
    static TritonTileConfig autotune_matmul(
        int M, int N, int K,
        type::DataType dtype,
        const std::string& target_arch,
        const TritonAutotuneConfig& search_space = TritonAutotuneConfig()
    );
    
    /**
     * @brief Find best configuration for Attention
     */
    static TritonTileConfig autotune_attention(
        int batch, int heads, int seq_len, int head_dim,
        const std::string& target_arch
    );
};

}  // namespace triton
}  // namespace threadblock
}  // namespace yirage
