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
 */

/**
 * @file pk_task_kernels.h
 * @brief Task kernel specifications for persistent kernel backends
 * 
 * This header defines kernel parameters, launch configurations, and
 * shared memory requirements for each task type across different backends.
 */

#pragma once

#include "persistent_kernel/pk_backend_interface.h"
#include <cstdint>
#include <cstring>

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// Task Kernel Parameters
// =============================================================================

/**
 * @brief Embedding layer parameters
 */
struct EmbeddingParams {
    const float* embedding_table;  // [vocab_size, hidden_dim]
    const int32_t* input_ids;      // [batch_size, seq_len]
    float* output;                 // [batch_size, seq_len, hidden_dim]
    int vocab_size;
    int hidden_dim;
    int batch_size;
    int seq_len;
};

/**
 * @brief RMS Normalization parameters
 */
struct RMSNormParams {
    const float* input;    // [batch_size, seq_len, hidden_dim]
    const float* weight;   // [hidden_dim]
    float* output;         // [batch_size, seq_len, hidden_dim]
    float eps;
    int batch_size;
    int seq_len;
    int hidden_dim;
};

/**
 * @brief RMS Norm + Linear fused parameters
 */
struct RMSNormLinearParams {
    const float* input;    // [batch_size, seq_len, hidden_dim]
    const float* rms_weight;   // [hidden_dim]
    const float* linear_weight; // [out_dim, hidden_dim]
    const float* linear_bias;   // [out_dim] or nullptr
    float* output;         // [batch_size, seq_len, out_dim]
    float eps;
    int batch_size;
    int seq_len;
    int hidden_dim;
    int out_dim;
};

/**
 * @brief Linear layer parameters
 */
struct LinearParams {
    const float* input;   // [batch_size, seq_len, in_dim]
    const float* weight;  // [out_dim, in_dim]
    const float* bias;    // [out_dim] or nullptr
    float* output;        // [batch_size, seq_len, out_dim]
    int batch_size;
    int seq_len;
    int in_dim;
    int out_dim;
    bool use_bias;
};

/**
 * @brief Linear with residual add parameters
 */
struct LinearResidualParams {
    const float* input;    // [batch_size, seq_len, in_dim]
    const float* residual; // [batch_size, seq_len, out_dim]
    const float* weight;   // [out_dim, in_dim]
    const float* bias;     // [out_dim] or nullptr
    float* output;         // [batch_size, seq_len, out_dim]
    int batch_size;
    int seq_len;
    int in_dim;
    int out_dim;
    float residual_scale;
};

/**
 * @brief Attention parameters
 */
struct AttentionParams {
    const float* q;        // [batch, heads, seq, head_dim]
    const float* k;        // [batch, kv_heads, seq, head_dim]
    const float* v;        // [batch, kv_heads, seq, head_dim]
    const float* mask;     // [batch, 1, seq, seq] or nullptr
    float* output;         // [batch, heads, seq, head_dim]
    int batch_size;
    int num_heads;
    int num_kv_heads;
    int seq_len;
    int head_dim;
    float scale;
    bool is_causal;
};

/**
 * @brief Rotary embedding parameters
 */
struct RotaryEmbeddingParams {
    float* q;              // [batch, heads, seq, head_dim] in-place
    float* k;              // [batch, kv_heads, seq, head_dim] in-place
    const float* cos;      // [seq, head_dim/2]
    const float* sin;      // [seq, head_dim/2]
    int batch_size;
    int num_heads;
    int num_kv_heads;
    int seq_len;
    int head_dim;
    int position_offset;
};

/**
 * @brief Paged attention parameters
 */
struct PagedAttentionParams {
    const float* q;              // [batch, heads, 1, head_dim]
    const float* k_cache;        // [num_blocks, block_size, kv_heads, head_dim]
    const float* v_cache;        // [num_blocks, block_size, kv_heads, head_dim]
    const int32_t* block_table;  // [batch, max_blocks]
    const int32_t* seq_lens;     // [batch]
    float* output;               // [batch, heads, 1, head_dim]
    int batch_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int block_size;
    int max_seq_len;
    float scale;
};

/**
 * @brief SiLU * gate fused parameters
 */
struct SiLUMulParams {
    const float* gate;     // [batch, seq, hidden_dim]
    const float* up;       // [batch, seq, hidden_dim]
    float* output;         // [batch, seq, hidden_dim]
    int batch_size;
    int seq_len;
    int hidden_dim;
};

/**
 * @brief SiLU * gate + Linear fused parameters
 */
struct SiLUMulLinearParams {
    const float* gate;     // [batch, seq, hidden_dim]
    const float* up;       // [batch, seq, hidden_dim]
    const float* weight;   // [out_dim, hidden_dim]
    float* output;         // [batch, seq, out_dim]
    int batch_size;
    int seq_len;
    int hidden_dim;
    int out_dim;
};

/**
 * @brief MoE gating parameters
 */
struct MoEGateParams {
    const float* hidden;       // [batch, seq, hidden_dim]
    const float* gate_weight;  // [num_experts, hidden_dim]
    int32_t* expert_ids;       // [batch, seq, top_k]
    float* expert_weights;     // [batch, seq, top_k]
    int batch_size;
    int seq_len;
    int hidden_dim;
    int num_experts;
    int top_k;
};

/**
 * @brief MoE linear parameters
 *
 * Implements the expert down-projection with weighted accumulation:
 *   output[t] += routing_weight[k] * (expert_weights[expert_id] @ input[t])
 * for each token t and each of its top_k selected experts.
 */
struct MoELinearParams {
    const float* input;           // [batch, seq, hidden_dim]
    const float* expert_weights;  // [num_experts, out_dim, hidden_dim]
    const int32_t* expert_ids;    // [batch, seq, top_k]
    const float* gate_weights;    // [batch, seq, top_k]
    float* output;                // [batch, seq, out_dim]
    int batch_size;
    int seq_len;
    int hidden_dim;
    int out_dim;
    int num_experts;
    int top_k;
};

/**
 * @brief Fused MoE SwiGLU + down-projection parameters
 *
 * Implements the complete expert MLP in a single fused kernel:
 *   gate_act = W_gate[expert_id] @ input[t]           (intermediate_size)
 *   up_act   = W_up[expert_id]   @ input[t]           (intermediate_size)
 *   hidden   = SiLU(gate_act) * up_act                (intermediate_size)
 *   out[t]  += routing_weight[k] * (W_down[expert_id] @ hidden)
 *
 * By fusing gate/up GEMM + SiLU-mul + down GEMM we avoid materialising the
 * intermediate activation buffers across kernel launches and improve cache
 * utilisation on CPU.
 */
struct MoESiLULinearParams {
    const float* input;           // [num_tokens, hidden_dim]  (batch*seq flattened)
    const float* w_gate;          // [num_experts, intermediate_size, hidden_dim]
    const float* w_up;            // [num_experts, intermediate_size, hidden_dim]
    const float* w_down;          // [num_experts, hidden_dim, intermediate_size]
    const int32_t* expert_ids;    // [num_tokens, top_k]
    const float* routing_weights; // [num_tokens, top_k]
    float* output;                // [num_tokens, hidden_dim]
    int num_tokens;               // batch_size * seq_len
    int hidden_dim;
    int intermediate_size;
    int num_experts;
    int top_k;
};

/**
 * @brief Argmax parameters
 */
struct ArgmaxParams {
    const float* input;    // [batch, seq, vocab_size]
    int32_t* output;       // [batch, seq]
    int batch_size;
    int seq_len;
    int vocab_size;
};

/**
 * @brief AllReduce parameters
 */
struct AllReduceParams {
    float* data;           // [size]
    size_t count;          // Number of elements
    int num_ranks;
    int rank;
    void* comm;            // NCCL/HCCL communicator handle
};

// =============================================================================
// Kernel Launch Configuration
// =============================================================================

/**
 * @brief Launch configuration for GPU kernels
 */
struct KernelLaunchConfig {
    int grid_x;
    int grid_y;
    int grid_z;
    int block_x;
    int block_y;
    int block_z;
    size_t shared_memory_bytes;
    void* stream;
    
    KernelLaunchConfig()
        : grid_x(1), grid_y(1), grid_z(1),
          block_x(256), block_y(1), block_z(1),
          shared_memory_bytes(0), stream(nullptr) {}
};

/**
 * @brief Calculate launch config for elementwise operations
 */
inline KernelLaunchConfig get_elementwise_config(int num_elements, 
                                                  int block_size = 256) {
    KernelLaunchConfig config;
    config.block_x = block_size;
    config.grid_x = (num_elements + block_size - 1) / block_size;
    return config;
}

/**
 * @brief Calculate launch config for matrix operations
 */
inline KernelLaunchConfig get_matmul_config(int M, int N, int K,
                                            int tile_m = 64,
                                            int tile_n = 64) {
    KernelLaunchConfig config;
    config.grid_x = (N + tile_n - 1) / tile_n;
    config.grid_y = (M + tile_m - 1) / tile_m;
    config.block_x = 128;  // Typically 128 threads per tile
    config.shared_memory_bytes = (tile_m * K + K * tile_n) * sizeof(float);
    return config;
}

/**
 * @brief Calculate launch config for attention
 */
inline KernelLaunchConfig get_attention_config(int batch_size,
                                                int num_heads,
                                                int seq_len,
                                                int head_dim) {
    KernelLaunchConfig config;
    // One block per (batch, head) pair
    config.grid_x = batch_size;
    config.grid_y = num_heads;
    // Block handles seq_len tokens
    config.block_x = std::min(256, seq_len);
    // Shared memory for Q, K, V tiles
    config.shared_memory_bytes = 3 * seq_len * head_dim * sizeof(float);
    return config;
}

// =============================================================================
// Task Execution Helpers
// =============================================================================

/**
 * @brief Unpack task descriptor parameters
 */
inline void unpack_task_params(const PKTaskDesc& desc, void* params, 
                                PKTaskType type) {
    // Copy parameters based on task type
    switch (type) {
        case PKTaskType::EMBEDDING:
            std::memcpy(params, desc.params, sizeof(EmbeddingParams));
            break;
        case PKTaskType::RMS_NORM:
            std::memcpy(params, desc.params, sizeof(RMSNormParams));
            break;
        case PKTaskType::RMS_NORM_LINEAR:
            std::memcpy(params, desc.params, sizeof(RMSNormLinearParams));
            break;
        case PKTaskType::LINEAR:
            std::memcpy(params, desc.params, sizeof(LinearParams));
            break;
        case PKTaskType::LINEAR_RESIDUAL:
            std::memcpy(params, desc.params, sizeof(LinearResidualParams));
            break;
        case PKTaskType::ATTENTION:
            std::memcpy(params, desc.params, sizeof(AttentionParams));
            break;
        case PKTaskType::PAGED_ATTENTION:
            std::memcpy(params, desc.params, sizeof(PagedAttentionParams));
            break;
        case PKTaskType::ROTARY_EMBEDDING:
            std::memcpy(params, desc.params, sizeof(RotaryEmbeddingParams));
            break;
        case PKTaskType::SILU_MUL:
            std::memcpy(params, desc.params, sizeof(SiLUMulParams));
            break;
        case PKTaskType::SILU_MUL_LINEAR:
            std::memcpy(params, desc.params, sizeof(SiLUMulLinearParams));
            break;
        case PKTaskType::MOE_GATE:
            std::memcpy(params, desc.params, sizeof(MoEGateParams));
            break;
        case PKTaskType::MOE_LINEAR:
            std::memcpy(params, desc.params, sizeof(MoELinearParams));
            break;
        case PKTaskType::MOE_SILU_LINEAR:
            std::memcpy(params, desc.params, sizeof(MoESiLULinearParams));
            break;
        case PKTaskType::ARGMAX:
            std::memcpy(params, desc.params, sizeof(ArgmaxParams));
            break;
        case PKTaskType::ALLREDUCE:
            std::memcpy(params, desc.params, sizeof(AllReduceParams));
            break;
        default:
            break;
    }
}

/**
 * @brief Get shared memory requirements for a task
 */
inline size_t get_task_shared_memory(PKTaskType type, const void* params) {
    switch (type) {
        case PKTaskType::RMS_NORM: {
            auto* p = static_cast<const RMSNormParams*>(params);
            // Need to store partial sums for reduction
            return p->hidden_dim * sizeof(float);
        }
        case PKTaskType::ATTENTION: {
            auto* p = static_cast<const AttentionParams*>(params);
            // QK^T and softmax for one head
            return p->seq_len * p->seq_len * sizeof(float);
        }
        case PKTaskType::LINEAR: {
            // Tile storage: two 64×64 FP32 tiles
            return 2 * 64 * 64 * sizeof(float);
        }
        case PKTaskType::MOE_LINEAR: {
            auto* p = static_cast<const MoELinearParams*>(params);
            // Scratch for one token row + one output row per thread
            return static_cast<size_t>(p->out_dim) * sizeof(float);
        }
        case PKTaskType::MOE_SILU_LINEAR: {
            auto* p = static_cast<const MoESiLULinearParams*>(params);
            // Scratch for gate_act + up_act buffers per token
            return static_cast<size_t>(p->intermediate_size) * 2 * sizeof(float);
        }
        default:
            return 0;
    }
}

// =============================================================================
// Backend-Specific Kernel Dispatchers
// =============================================================================

/**
 * @brief CUDA kernel dispatcher declaration
 */
#ifdef YIRAGE_BACKEND_CUDA
void dispatch_cuda_task(PKTaskType type, const void* params,
                        const KernelLaunchConfig& config);
#endif

/**
 * @brief CPU kernel dispatcher declaration
 */
void dispatch_cpu_task(PKTaskType type, const void* params);

/**
 * @brief Ascend kernel dispatcher declaration
 */
#ifdef YIRAGE_BACKEND_ASCEND
void dispatch_ascend_task(PKTaskType type, const void* params, void* stream);
#endif

/**
 * @brief MACA kernel dispatcher declaration
 */
#ifdef YIRAGE_BACKEND_MACA
void dispatch_maca_task(PKTaskType type, const void* params,
                        const KernelLaunchConfig& config);
#endif

/**
 * @brief MPS kernel dispatcher declaration
 */
#ifdef YIRAGE_BACKEND_MPS
void dispatch_mps_task(PKTaskType type, const void* params, void* encoder);
#endif

} // namespace persistent_kernel
} // namespace yirage
