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
 * @file pk_cpu_kernels.cc
 * @brief CPU reference implementations of persistent kernel tasks
 * 
 * These implementations serve as:
 * 1. Reference for correctness verification
 * 2. Fallback when no GPU is available
 * 3. Template for other backend implementations
 */

#include "persistent_kernel/pk_task_kernels.h"
#include <cmath>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace yirage {
namespace persistent_kernel {

// =============================================================================
// Embedding Kernel
// =============================================================================

static void cpu_embedding(const EmbeddingParams& params) {
    const int batch_size = params.batch_size;
    const int seq_len = params.seq_len;
    const int hidden_dim = params.hidden_dim;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int token_id = params.input_ids[b * seq_len + s];
            const float* emb_row = params.embedding_table + 
                                   token_id * hidden_dim;
            float* out_row = params.output + 
                             (b * seq_len + s) * hidden_dim;
            
            // Copy embedding vector
            std::memcpy(out_row, emb_row, hidden_dim * sizeof(float));
        }
    }
}

// =============================================================================
// RMS Normalization Kernel
// =============================================================================

static void cpu_rms_norm(const RMSNormParams& params) {
    const int batch_size = params.batch_size;
    const int seq_len = params.seq_len;
    const int hidden_dim = params.hidden_dim;
    const float eps = params.eps;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const float* in_row = params.input + 
                                  (b * seq_len + s) * hidden_dim;
            float* out_row = params.output + 
                             (b * seq_len + s) * hidden_dim;
            
            // Compute RMS
            float sum_sq = 0.0f;
            for (int h = 0; h < hidden_dim; ++h) {
                sum_sq += in_row[h] * in_row[h];
            }
            float rms = std::sqrt(sum_sq / hidden_dim + eps);
            float inv_rms = 1.0f / rms;
            
            // Apply normalization with weight
            for (int h = 0; h < hidden_dim; ++h) {
                out_row[h] = in_row[h] * inv_rms * params.weight[h];
            }
        }
    }
}

// =============================================================================
// Linear Kernel
// =============================================================================

static void cpu_linear(const LinearParams& params) {
    const int batch_size = params.batch_size;
    const int seq_len = params.seq_len;
    const int in_dim = params.in_dim;
    const int out_dim = params.out_dim;
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int o = 0; o < out_dim; ++o) {
                const float* in_row = params.input + 
                                      (b * seq_len + s) * in_dim;
                const float* weight_row = params.weight + o * in_dim;
                
                float sum = 0.0f;
                for (int i = 0; i < in_dim; ++i) {
                    sum += in_row[i] * weight_row[i];
                }
                
                if (params.use_bias && params.bias) {
                    sum += params.bias[o];
                }
                
                params.output[(b * seq_len + s) * out_dim + o] = sum;
            }
        }
    }
}

// =============================================================================
// Linear with Residual Kernel
// =============================================================================

static void cpu_linear_residual(const LinearResidualParams& params) {
    const int batch_size = params.batch_size;
    const int seq_len = params.seq_len;
    const int in_dim = params.in_dim;
    const int out_dim = params.out_dim;
    const float residual_scale = params.residual_scale;
    
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int o = 0; o < out_dim; ++o) {
                const float* in_row = params.input + 
                                      (b * seq_len + s) * in_dim;
                const float* weight_row = params.weight + o * in_dim;
                
                float sum = 0.0f;
                for (int i = 0; i < in_dim; ++i) {
                    sum += in_row[i] * weight_row[i];
                }
                
                if (params.bias) {
                    sum += params.bias[o];
                }
                
                // Add scaled residual
                float residual_val = params.residual[
                    (b * seq_len + s) * out_dim + o];
                params.output[(b * seq_len + s) * out_dim + o] = 
                    sum + residual_scale * residual_val;
            }
        }
    }
}

// =============================================================================
// Attention Kernel
// =============================================================================

static void cpu_attention(const AttentionParams& params) {
    const int batch_size = params.batch_size;
    const int num_heads = params.num_heads;
    const int num_kv_heads = params.num_kv_heads;
    const int seq_len = params.seq_len;
    const int head_dim = params.head_dim;
    const float scale = params.scale;
    const bool is_causal = params.is_causal;
    
    const int kv_head_ratio = num_heads / num_kv_heads;  // GQA ratio
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            int kv_h = h / kv_head_ratio;
            
            // Attention scores buffer
            std::vector<float> scores(seq_len * seq_len);
            
            // Compute Q @ K^T
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
                    // Causal mask
                    if (is_causal && k_pos > q_pos) {
                        scores[q_pos * seq_len + k_pos] = -1e9f;
                        continue;
                    }
                    
                    float dot = 0.0f;
                    const float* q = params.q + 
                        ((b * num_heads + h) * seq_len + q_pos) * head_dim;
                    const float* k = params.k + 
                        ((b * num_kv_heads + kv_h) * seq_len + k_pos) * head_dim;
                    
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q[d] * k[d];
                    }
                    scores[q_pos * seq_len + k_pos] = dot * scale;
                }
            }
            
            // Apply softmax per row
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                float max_val = -1e9f;
                for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
                    max_val = std::max(max_val, 
                                       scores[q_pos * seq_len + k_pos]);
                }
                
                float sum_exp = 0.0f;
                for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
                    float val = std::exp(scores[q_pos * seq_len + k_pos] 
                                         - max_val);
                    scores[q_pos * seq_len + k_pos] = val;
                    sum_exp += val;
                }
                
                for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
                    scores[q_pos * seq_len + k_pos] /= sum_exp;
                }
            }
            
            // Compute output = scores @ V
            for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
                float* out = params.output + 
                    ((b * num_heads + h) * seq_len + q_pos) * head_dim;
                
                for (int d = 0; d < head_dim; ++d) {
                    out[d] = 0.0f;
                }
                
                for (int k_pos = 0; k_pos < seq_len; ++k_pos) {
                    float attn_weight = scores[q_pos * seq_len + k_pos];
                    const float* v = params.v + 
                        ((b * num_kv_heads + kv_h) * seq_len + k_pos) * head_dim;
                    
                    for (int d = 0; d < head_dim; ++d) {
                        out[d] += attn_weight * v[d];
                    }
                }
            }
        }
    }
}

// =============================================================================
// Rotary Embedding Kernel
// =============================================================================

static void cpu_rotary_embedding(const RotaryEmbeddingParams& params) {
    const int batch_size = params.batch_size;
    const int num_heads = params.num_heads;
    const int num_kv_heads = params.num_kv_heads;
    const int seq_len = params.seq_len;
    const int head_dim = params.head_dim;
    const int half_dim = head_dim / 2;
    
    // Apply to Q
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                float* q = params.q + 
                    ((b * num_heads + h) * seq_len + s) * head_dim;
                
                int pos = s + params.position_offset;
                const float* cos_val = params.cos + pos * half_dim;
                const float* sin_val = params.sin + pos * half_dim;
                
                for (int d = 0; d < half_dim; ++d) {
                    float x0 = q[d];
                    float x1 = q[d + half_dim];
                    q[d] = x0 * cos_val[d] - x1 * sin_val[d];
                    q[d + half_dim] = x1 * cos_val[d] + x0 * sin_val[d];
                }
            }
        }
    }
    
    // Apply to K
    #pragma omp parallel for collapse(3)
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int s = 0; s < seq_len; ++s) {
                float* k = params.k + 
                    ((b * num_kv_heads + h) * seq_len + s) * head_dim;
                
                int pos = s + params.position_offset;
                const float* cos_val = params.cos + pos * half_dim;
                const float* sin_val = params.sin + pos * half_dim;
                
                for (int d = 0; d < half_dim; ++d) {
                    float x0 = k[d];
                    float x1 = k[d + half_dim];
                    k[d] = x0 * cos_val[d] - x1 * sin_val[d];
                    k[d + half_dim] = x1 * cos_val[d] + x0 * sin_val[d];
                }
            }
        }
    }
}

// =============================================================================
// SiLU * gate Kernel
// =============================================================================

static void cpu_silu_mul(const SiLUMulParams& params) {
    const int total = params.batch_size * params.seq_len * params.hidden_dim;
    
    #pragma omp parallel for
    for (int i = 0; i < total; ++i) {
        float gate_val = params.gate[i];
        float up_val = params.up[i];
        
        // SiLU(gate) = gate * sigmoid(gate)
        float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_val));
        float silu_gate = gate_val * sigmoid_gate;
        
        params.output[i] = silu_gate * up_val;
    }
}

// =============================================================================
// Argmax Kernel
// =============================================================================

static void cpu_argmax(const ArgmaxParams& params) {
    const int batch_size = params.batch_size;
    const int seq_len = params.seq_len;
    const int vocab_size = params.vocab_size;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const float* logits = params.input + 
                (b * seq_len + s) * vocab_size;
            
            int max_idx = 0;
            float max_val = logits[0];
            
            for (int v = 1; v < vocab_size; ++v) {
                if (logits[v] > max_val) {
                    max_val = logits[v];
                    max_idx = v;
                }
            }
            
            params.output[b * seq_len + s] = max_idx;
        }
    }
}

// =============================================================================
// MoE Gate Kernel
// =============================================================================

/**
 * @brief Reorder tokens by assigned expert for cache-friendly batched GEMM.
 *
 * Populates the `expert_tokens` output parameter — a vector indexed by expert
 * ID, each element being a list of (token_index, routing_weight) pairs for
 * the tokens assigned to that expert.  Grouping tokens this way allows the
 * downstream GEMM to process a contiguous block of tokens per expert, which
 * maximises B-tile reuse and reduces DRAM bandwidth.
 */
static void cpu_moe_reorder_tokens(
        int num_tokens,
        int top_k,
        int num_experts,
        const int32_t* expert_ids,
        const float*   routing_weights,
        std::vector<std::vector<std::pair<int,float>>>& expert_tokens) {
    expert_tokens.assign(num_experts, {});
    for (int t = 0; t < num_tokens; ++t) {
        for (int k = 0; k < top_k; ++k) {
            int   eid = expert_ids[t * top_k + k];
            float w   = routing_weights[t * top_k + k];
            if (eid >= 0 && eid < num_experts) {
                expert_tokens[eid].emplace_back(t, w);
            }
        }
    }
}

static void cpu_moe_gate(const MoEGateParams& params) {
    const int batch_size = params.batch_size;
    const int seq_len = params.seq_len;
    const int hidden_dim = params.hidden_dim;
    const int num_experts = params.num_experts;
    const int top_k = params.top_k;
    
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            const float* hidden = params.hidden + 
                (b * seq_len + s) * hidden_dim;
            
            // Compute gate scores for all experts
            std::vector<std::pair<float, int>> scores(num_experts);
            for (int e = 0; e < num_experts; ++e) {
                float score = 0.0f;
                const float* gate_row = params.gate_weight + e * hidden_dim;
                
                for (int h = 0; h < hidden_dim; ++h) {
                    score += hidden[h] * gate_row[h];
                }
                scores[e] = {score, e};
            }
            
            // Sort to find top-k
            std::partial_sort(scores.begin(), 
                              scores.begin() + top_k,
                              scores.end(),
                              [](const auto& a, const auto& b) {
                                  return a.first > b.first;
                              });
            
            // Compute softmax over top-k
            float max_score = scores[0].first;
            float sum_exp = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                sum_exp += std::exp(scores[k].first - max_score);
            }
            
            // Store results
            int32_t* expert_ids = params.expert_ids + 
                (b * seq_len + s) * top_k;
            float* expert_weights = params.expert_weights + 
                (b * seq_len + s) * top_k;
            
            for (int k = 0; k < top_k; ++k) {
                expert_ids[k] = scores[k].second;
                expert_weights[k] = std::exp(scores[k].first - max_score) 
                                    / sum_exp;
            }
        }
    }
}

// =============================================================================
// MoE Linear Kernel (expert-parallel batched GEMM with weighted accumulation)
// =============================================================================

static void cpu_moe_linear(const MoELinearParams& params) {
    const int num_tokens  = params.batch_size * params.seq_len;
    const int hidden_dim  = params.hidden_dim;
    const int out_dim     = params.out_dim;
    const int num_experts = params.num_experts;
    const int top_k       = params.top_k;

    // Zero output buffer
    std::fill(params.output, params.output + num_tokens * out_dim, 0.0f);

    // Group tokens by their selected experts for cache-friendly batched GEMM
    std::vector<std::vector<std::pair<int,float>>> expert_tokens;
    cpu_moe_reorder_tokens(num_tokens, top_k, num_experts,
                           params.expert_ids, params.gate_weights,
                           expert_tokens);

    // Process each expert's assigned tokens in parallel.
    // Each (token, expert) pair is unique so the per-token output accumulation
    // across experts is race-free when done with separate per-expert passes
    // and a final reduction.  We use a simple two-phase approach:
    //  1) accumulate per-expert contributions into a local buffer per thread,
    //  2) a serial scatter-add back to the shared output array.
    #pragma omp parallel for schedule(dynamic)
    for (int e = 0; e < num_experts; ++e) {
        const auto& tokens = expert_tokens[e];
        if (tokens.empty()) continue;

        // expert_weights layout: [num_experts, out_dim, hidden_dim]
        const float* W = params.expert_weights +
                         static_cast<ptrdiff_t>(e) * out_dim * hidden_dim;

        for (const auto& [t, routing_w] : tokens) {
            const float* in_row  = params.input  + t * hidden_dim;
            float*       out_row = params.output + t * out_dim;

            for (int o = 0; o < out_dim; ++o) {
                const float* weight_row = W + o * hidden_dim;
                float sum = 0.0f;
                for (int h = 0; h < hidden_dim; ++h) {
                    sum += in_row[h] * weight_row[h];
                }
                // Weighted accumulation.  Different experts for the same token
                // may run on different threads; protect with an atomic update.
                #pragma omp atomic
                out_row[o] += routing_w * sum;
            }
        }
    }
}

// =============================================================================
// Fused MoE SwiGLU + Down-Projection Kernel
// =============================================================================

/**
 * Fused expert MLP: SiLU(W_gate @ x) * (W_up @ x) then W_down @ result.
 *
 * Avoids materializing intermediate activations across separate kernel calls
 * and keeps the hot input token row in L1/L2 cache across all three GEMMs.
 */
static void cpu_moe_silu_linear(const MoESiLULinearParams& params) {
    const int num_tokens  = params.num_tokens;
    const int hidden_dim  = params.hidden_dim;
    const int inter_size  = params.intermediate_size;
    const int num_experts = params.num_experts;
    const int top_k       = params.top_k;

    // Zero output buffer
    std::fill(params.output, params.output + num_tokens * hidden_dim, 0.0f);

    #pragma omp parallel
    {
        // Per-thread scratch buffers (avoids heap allocation per token)
        std::vector<float> gate_buf(inter_size);
        std::vector<float> up_buf(inter_size);
        std::vector<float> act_buf(inter_size);  // SiLU(gate) * up

        #pragma omp for schedule(static)
        for (int t = 0; t < num_tokens; ++t) {
            const float* in_row  = params.input  + t * hidden_dim;
            float*       out_row = params.output + t * hidden_dim;

            for (int k = 0; k < top_k; ++k) {
                int   eid     = params.expert_ids[t * top_k + k];
                float route_w = params.routing_weights[t * top_k + k];

                if (eid < 0 || eid >= num_experts) continue;

                // Weight layout: [num_experts, rows, hidden_dim]
                const float* Wg = params.w_gate +
                    static_cast<ptrdiff_t>(eid) * inter_size * hidden_dim;
                const float* Wu = params.w_up +
                    static_cast<ptrdiff_t>(eid) * inter_size * hidden_dim;
                // W_down layout: [num_experts, hidden_dim, inter_size]
                const float* Wd = params.w_down +
                    static_cast<ptrdiff_t>(eid) * hidden_dim * inter_size;

                // Gate projection: gate_buf = Wg @ in_row
                for (int i = 0; i < inter_size; ++i) {
                    float val = 0.0f;
                    const float* row = Wg + i * hidden_dim;
                    for (int h = 0; h < hidden_dim; ++h) val += row[h] * in_row[h];
                    gate_buf[i] = val;
                }

                // Up projection: up_buf = Wu @ in_row
                for (int i = 0; i < inter_size; ++i) {
                    float val = 0.0f;
                    const float* row = Wu + i * hidden_dim;
                    for (int h = 0; h < hidden_dim; ++h) val += row[h] * in_row[h];
                    up_buf[i] = val;
                }

                // SiLU(gate) * up
                for (int i = 0; i < inter_size; ++i) {
                    float g   = gate_buf[i];
                    float sig = 1.0f / (1.0f + std::exp(-g));
                    act_buf[i] = g * sig * up_buf[i];
                }

                // Down projection: out_row += route_w * (Wd @ act_buf)
                // Token-level parallelism means different threads never
                // share the same out_row, so no atomic needed here.
                for (int h = 0; h < hidden_dim; ++h) {
                    float val = 0.0f;
                    const float* row = Wd + h * inter_size;
                    for (int i = 0; i < inter_size; ++i) val += row[i] * act_buf[i];
                    out_row[h] += route_w * val;
                }
            }
        }
    }
}

// =============================================================================
// CPU Task Dispatcher
// =============================================================================

void dispatch_cpu_task(PKTaskType type, const void* params) {
    switch (type) {
        case PKTaskType::EMBEDDING:
            cpu_embedding(*static_cast<const EmbeddingParams*>(params));
            break;
        case PKTaskType::RMS_NORM:
            cpu_rms_norm(*static_cast<const RMSNormParams*>(params));
            break;
        case PKTaskType::LINEAR:
            cpu_linear(*static_cast<const LinearParams*>(params));
            break;
        case PKTaskType::LINEAR_RESIDUAL:
            cpu_linear_residual(
                *static_cast<const LinearResidualParams*>(params));
            break;
        case PKTaskType::ATTENTION:
            cpu_attention(*static_cast<const AttentionParams*>(params));
            break;
        case PKTaskType::ROTARY_EMBEDDING:
            cpu_rotary_embedding(
                *static_cast<const RotaryEmbeddingParams*>(params));
            break;
        case PKTaskType::SILU_MUL:
            cpu_silu_mul(*static_cast<const SiLUMulParams*>(params));
            break;
        case PKTaskType::ARGMAX:
            cpu_argmax(*static_cast<const ArgmaxParams*>(params));
            break;
        case PKTaskType::MOE_GATE:
            cpu_moe_gate(*static_cast<const MoEGateParams*>(params));
            break;
        case PKTaskType::MOE_LINEAR:
            cpu_moe_linear(*static_cast<const MoELinearParams*>(params));
            break;
        case PKTaskType::MOE_SILU_LINEAR:
            cpu_moe_silu_linear(*static_cast<const MoESiLULinearParams*>(params));
            break;
        case PKTaskType::TERMINATE:
        case PKTaskType::BEGIN_TASK_GRAPH:
            // No-op for control tasks
            break;
        default:
            // Unsupported task
            break;
    }
}

} // namespace persistent_kernel
} // namespace yirage
