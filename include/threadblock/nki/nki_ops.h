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
 * AWS Neuron Kernel Interface (NKI) Operations
 * For AWS Trainium and Inferentia chips
 */

#pragma once

#include "type.h"
#include <string>
#include <vector>

namespace yirage {
namespace threadblock {
namespace nki {

// =============================================================================
// NKI Architecture Constants
// =============================================================================

// AWS Trainium/Inferentia use NeuronCore architecture
// Key features:
// - Tensor Engine for matrix operations (128x128 systolic array)
// - Vector Engine for element-wise operations
// - Scalar Engine for control flow
// - SBUF (State Buffer) for on-chip memory

namespace constants {
    constexpr int TENSOR_ENGINE_SIZE = 128;      // 128x128 systolic array
    constexpr int VECTOR_ENGINE_WIDTH = 128;     // 128 elements per cycle
    constexpr size_t SBUF_SIZE = 24 * 1024 * 1024;  // 24MB SBUF per core
    constexpr int PSUM_BUFFER_SIZE = 32;         // Partial sum buffers
}

// =============================================================================
// NKI Tile Configuration
// =============================================================================

struct NKITileConfig {
    int partition_dim = 128;    // Partition dimension (power of 2, max 128)
    int free_dim = 512;         // Free dimension
    int num_partitions = 1;     // Number of partitions to use
    bool use_psum = true;       // Use partial sum accumulation
};

// =============================================================================
// NKI Code Generator
// =============================================================================

class NKICodeGenerator {
public:
    /**
     * @brief Generate NKI kernel for MatMul
     */
    static std::string generate_matmul(
        int M, int N, int K,
        type::DataType dtype,
        const NKITileConfig& config
    );
    
    /**
     * @brief Generate NKI kernel for Attention
     */
    static std::string generate_attention(
        int batch, int heads, int seq_len, int head_dim,
        bool causal,
        const NKITileConfig& config
    );
    
    /**
     * @brief Generate NKI kernel for RMS Norm
     */
    static std::string generate_rms_norm(
        int hidden_dim,
        float epsilon
    );
    
    /**
     * @brief Generate NKI kernel for element-wise operations
     */
    static std::string generate_elementwise(
        const std::string& op_name,
        int size,
        type::DataType dtype
    );
};

// =============================================================================
// NKI Kernel Registry
// =============================================================================

class NKIKernelRegistry {
public:
    static NKIKernelRegistry& instance();
    
    void register_kernel(const std::string& name, const std::string& code);
    std::string get_kernel(const std::string& name) const;
    bool has_kernel(const std::string& name) const;
    bool compile_kernel(const std::string& name);

private:
    NKIKernelRegistry() = default;
    std::map<std::string, std::string> kernels_;
};

// =============================================================================
// NKI Performance Estimator
// =============================================================================

struct NKIPerformanceEstimate {
    double estimated_tflops;
    double sbuf_utilization;
    double tensor_engine_utilization;
    int estimated_latency_us;
};

NKIPerformanceEstimate estimate_nki_performance(
    const std::string& op_type,
    int M, int N, int K,
    type::DataType dtype
);

}  // namespace nki
}  // namespace threadblock
}  // namespace yirage
