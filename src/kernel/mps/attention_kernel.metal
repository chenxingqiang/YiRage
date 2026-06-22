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
 * Metal Attention Kernels for Apple Silicon
 * 
 * Flash Attention style implementation for memory efficiency.
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Parameters
// =============================================================================

struct AttentionParams {
    uint batch_size;
    uint num_heads;
    uint seq_len;
    uint head_dim;
    float scale;
    bool causal;
};

// =============================================================================
// SIMD Helpers
// =============================================================================

inline float simd_max(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

inline float simd_sum(float val) {
    val += simd_shuffle_xor(val, 16);
    val += simd_shuffle_xor(val, 8);
    val += simd_shuffle_xor(val, 4);
    val += simd_shuffle_xor(val, 2);
    val += simd_shuffle_xor(val, 1);
    return val;
}

// =============================================================================
// Dot Product Attention (single head)
// =============================================================================

kernel void dot_product_attention_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_max[8];
    threadgroup float shared_sum[8];
    threadgroup float scores[256];  // Max seq_len per tile
    
    uint batch = gid.z / params.num_heads;
    uint head = gid.z % params.num_heads;
    uint q_pos = gid.y;
    
    if (batch >= params.batch_size || q_pos >= params.seq_len) return;
    
    uint head_offset = ((batch * params.num_heads) + head) * params.seq_len * params.head_dim;
    
    device const half* q_vec = Q + head_offset + q_pos * params.head_dim;
    device const half* k_base = K + head_offset;
    device const half* v_base = V + head_offset;
    device half* out_vec = output + head_offset + q_pos * params.head_dim;
    
    // Phase 1: Compute attention scores and find max
    float local_max = -INFINITY;
    
    for (uint k_pos = tid; k_pos < params.seq_len; k_pos += 256) {
        // Causal masking
        if (params.causal && k_pos > q_pos) {
            scores[k_pos] = -INFINITY;
            continue;
        }
        
        // Dot product Q·K
        float score = 0.0f;
        device const half* k_vec = k_base + k_pos * params.head_dim;
        for (uint d = 0; d < params.head_dim; d++) {
            score += float(q_vec[d]) * float(k_vec[d]);
        }
        score *= params.scale;
        
        scores[k_pos] = score;
        local_max = max(local_max, score);
    }
    
    // Reduce max
    local_max = simd_max(local_max);
    if (simd_lane == 0) shared_max[simd_id] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_max = shared_max[simd_lane];
        local_max = simd_max(local_max);
        if (simd_lane == 0) shared_max[0] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float row_max = shared_max[0];
    
    // Phase 2: Compute exp and sum
    float local_sum = 0.0f;
    for (uint k_pos = tid; k_pos < params.seq_len; k_pos += 256) {
        float exp_score = exp(scores[k_pos] - row_max);
        scores[k_pos] = exp_score;
        local_sum += exp_score;
    }
    
    local_sum = simd_sum(local_sum);
    if (simd_lane == 0) shared_sum[simd_id] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_id == 0 && simd_lane < 8) {
        local_sum = shared_sum[simd_lane];
        local_sum = simd_sum(local_sum);
        if (simd_lane == 0) shared_sum[0] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    float inv_sum = 1.0f / shared_sum[0];
    
    // Phase 3: Compute weighted sum of V
    for (uint d = tid; d < params.head_dim; d += 256) {
        float out_val = 0.0f;
        for (uint k_pos = 0; k_pos < params.seq_len; k_pos++) {
            float weight = scores[k_pos] * inv_sum;
            out_val += weight * float(v_base[k_pos * params.head_dim + d]);
        }
        out_vec[d] = half(out_val);
    }
}

// =============================================================================
// Flash Attention Style (tiled for memory efficiency)
// =============================================================================

kernel void flash_attention_f16(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* output [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    constexpr uint Q_TILE = 32;
    constexpr uint KV_TILE = 64;
    
    threadgroup half Q_shared[Q_TILE][128 + 1];
    threadgroup half K_shared[KV_TILE][128 + 1];
    threadgroup half V_shared[KV_TILE][128 + 1];
    threadgroup float row_max[Q_TILE];
    threadgroup float row_sum[Q_TILE];
    threadgroup float acc[Q_TILE][128];
    
    uint batch_head = tg_id.y;
    uint batch = batch_head / params.num_heads;
    uint head = batch_head % params.num_heads;
    uint q_tile_start = tg_id.x * Q_TILE;
    
    if (batch >= params.batch_size) return;
    
    uint head_offset = ((batch * params.num_heads) + head) * params.seq_len * params.head_dim;
    
    // Initialize
    for (uint i = tid; i < Q_TILE; i += 256) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
        for (uint d = 0; d < params.head_dim; d++) {
            acc[i][d] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Load Q tile
    for (uint i = tid; i < Q_TILE * params.head_dim; i += 256) {
        uint q_row = i / params.head_dim;
        uint d = i % params.head_dim;
        uint q_pos = q_tile_start + q_row;
        if (q_pos < params.seq_len && d < params.head_dim) {
            Q_shared[q_row][d] = Q[head_offset + q_pos * params.head_dim + d];
        } else {
            Q_shared[q_row][d] = half(0.0f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Iterate over KV tiles
    uint num_kv_tiles = (params.seq_len + KV_TILE - 1) / KV_TILE;
    
    for (uint kv_t = 0; kv_t < num_kv_tiles; kv_t++) {
        uint kv_start = kv_t * KV_TILE;
        
        // Load K, V tiles
        for (uint i = tid; i < KV_TILE * params.head_dim; i += 256) {
            uint kv_row = i / params.head_dim;
            uint d = i % params.head_dim;
            uint kv_pos = kv_start + kv_row;
            if (kv_pos < params.seq_len && d < params.head_dim) {
                K_shared[kv_row][d] = K[head_offset + kv_pos * params.head_dim + d];
                V_shared[kv_row][d] = V[head_offset + kv_pos * params.head_dim + d];
            } else {
                K_shared[kv_row][d] = half(0.0f);
                V_shared[kv_row][d] = half(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute scores and update accumulator
        for (uint q_row = simd_id; q_row < Q_TILE; q_row += 8) {
            uint q_pos = q_tile_start + q_row;
            if (q_pos >= params.seq_len) continue;
            
            float old_max = row_max[q_row];
            float new_max = old_max;
            float block_sum = 0.0f;
            
            // Compute scores for this Q row
            for (uint kv_row = simd_lane; kv_row < KV_TILE; kv_row += 32) {
                uint k_pos = kv_start + kv_row;
                
                // Causal mask
                if (params.causal && k_pos > q_pos) continue;
                if (k_pos >= params.seq_len) continue;
                
                // Dot product
                float score = 0.0f;
                for (uint d = 0; d < params.head_dim; d++) {
                    score += float(Q_shared[q_row][d]) * float(K_shared[kv_row][d]);
                }
                score *= params.scale;
                
                new_max = max(new_max, score);
            }
            
            // Reduce max within SIMD
            new_max = simd_max(new_max);
            
            // Rescale old accumulator
            float scale_old = exp(old_max - new_max);
            row_sum[q_row] *= scale_old;
            for (uint d = 0; d < params.head_dim; d++) {
                acc[q_row][d] *= scale_old;
            }
            
            // Add new contributions
            for (uint kv_row = simd_lane; kv_row < KV_TILE; kv_row += 32) {
                uint k_pos = kv_start + kv_row;
                if (params.causal && k_pos > q_pos) continue;
                if (k_pos >= params.seq_len) continue;
                
                float score = 0.0f;
                for (uint d = 0; d < params.head_dim; d++) {
                    score += float(Q_shared[q_row][d]) * float(K_shared[kv_row][d]);
                }
                score *= params.scale;
                
                float weight = exp(score - new_max);
                block_sum += weight;
                
                for (uint d = 0; d < params.head_dim; d++) {
                    acc[q_row][d] += weight * float(V_shared[kv_row][d]);
                }
            }
            
            // Reduce sum
            block_sum = simd_sum(block_sum);
            row_sum[q_row] += block_sum;
            row_max[q_row] = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write output
    for (uint i = tid; i < Q_TILE * params.head_dim; i += 256) {
        uint q_row = i / params.head_dim;
        uint d = i % params.head_dim;
        uint q_pos = q_tile_start + q_row;
        
        if (q_pos < params.seq_len && d < params.head_dim) {
            float out_val = acc[q_row][d] / row_sum[q_row];
            output[head_offset + q_pos * params.head_dim + d] = half(out_val);
        }
    }
}

// =============================================================================
// RoPE (Rotary Position Embedding)
// =============================================================================

kernel void apply_rope_f16(
    device half* query [[buffer(0)]],
    device half* key [[buffer(1)]],
    device const float* cos_cache [[buffer(2)]],
    device const float* sin_cache [[buffer(3)]],
    constant AttentionParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint batch = gid.z / params.num_heads;
    uint head = gid.z % params.num_heads;
    uint pos = gid.y;
    uint pair_idx = gid.x;  // Which dimension pair
    
    if (batch >= params.batch_size || pos >= params.seq_len) return;
    if (pair_idx * 2 + 1 >= params.head_dim) return;
    
    uint head_offset = ((batch * params.num_heads) + head) * params.seq_len * params.head_dim;
    uint base_idx = head_offset + pos * params.head_dim + pair_idx * 2;
    
    float cos_val = cos_cache[pos * (params.head_dim / 2) + pair_idx];
    float sin_val = sin_cache[pos * (params.head_dim / 2) + pair_idx];
    
    // Apply RoPE to query
    float q0 = float(query[base_idx]);
    float q1 = float(query[base_idx + 1]);
    query[base_idx] = half(q0 * cos_val - q1 * sin_val);
    query[base_idx + 1] = half(q0 * sin_val + q1 * cos_val);
    
    // Apply RoPE to key
    float k0 = float(key[base_idx]);
    float k1 = float(key[base_idx + 1]);
    key[base_idx] = half(k0 * cos_val - k1 * sin_val);
    key[base_idx + 1] = half(k0 * sin_val + k1 * cos_val);
}
