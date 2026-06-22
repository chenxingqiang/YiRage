/* Copyright 2025 YiRage Team
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

#pragma once

/**
 * @file attention.metal.h
 * @brief MPS Attention kernels
 *
 * Multi-head attention kernels for transformer models.
 * Includes both standard attention and grouped-query attention (GQA).
 */

namespace yirage {
namespace persistent_kernel {
namespace mps {

constexpr const char* ATTENTION_KERNEL_SOURCE = R"(
// Attention score kernel: scores = Q @ K^T * scale
// Q, K: [batch, num_heads, seq_len, head_dim]
kernel void attention_score_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& num_heads [[buffer(4)]],
    constant int& seq_len [[buffer(5)]],
    constant int& head_dim [[buffer(6)]],
    constant float& scale [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.z / num_heads;
    int h = tid.z % num_heads;
    int q_pos = tid.y;
    int k_pos = tid.x;
    
    if (b >= batch_size || q_pos >= seq_len || k_pos >= seq_len) return;
    
    device const float* q = Q + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
    device const float* k = K + ((b * num_heads + h) * seq_len + k_pos) * head_dim;
    
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += q[d] * k[d];
    }
    
    scores[((b * num_heads + h) * seq_len + q_pos) * seq_len + k_pos] = dot * scale;
}

// Causal mask application
// Sets attention scores to -inf where k_pos > q_pos
kernel void apply_causal_mask_kernel(
    device float* scores [[buffer(0)]],
    constant int& batch_size [[buffer(1)]],
    constant int& num_heads [[buffer(2)]],
    constant int& seq_len [[buffer(3)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.z / num_heads;
    int h = tid.z % num_heads;
    int q_pos = tid.y;
    int k_pos = tid.x;
    
    if (b >= batch_size || q_pos >= seq_len || k_pos >= seq_len) return;
    
    if (k_pos > q_pos) {
        scores[((b * num_heads + h) * seq_len + q_pos) * seq_len + k_pos] = -INFINITY;
    }
}

// Attention output: output = softmax(scores) @ V
kernel void attention_output_kernel(
    device const float* scores [[buffer(0)]],
    device const float* V [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& num_heads [[buffer(4)]],
    constant int& seq_len [[buffer(5)]],
    constant int& head_dim [[buffer(6)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.z / num_heads;
    int h = tid.z % num_heads;
    int pos = tid.y;
    int d = tid.x;
    
    if (b >= batch_size || pos >= seq_len || d >= head_dim) return;
    
    device const float* score_row = scores + ((b * num_heads + h) * seq_len + pos) * seq_len;
    device const float* v_head = V + (b * num_heads + h) * seq_len * head_dim;
    
    float sum = 0.0f;
    for (int k = 0; k < seq_len; k++) {
        sum += score_row[k] * v_head[k * head_dim + d];
    }
    
    output[((b * num_heads + h) * seq_len + pos) * head_dim + d] = sum;
}

// Grouped Query Attention (GQA) score kernel
// num_kv_heads < num_heads, multiple Q heads share same K/V
kernel void gqa_score_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& num_heads [[buffer(4)]],
    constant int& num_kv_heads [[buffer(5)]],
    constant int& seq_len [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.z / num_heads;
    int h = tid.z % num_heads;
    int q_pos = tid.y;
    int k_pos = tid.x;
    
    if (b >= batch_size || q_pos >= seq_len || k_pos >= seq_len) return;
    
    // Map query head to KV head
    int kv_head = h / (num_heads / num_kv_heads);
    
    device const float* q = Q + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
    device const float* k = K + ((b * num_kv_heads + kv_head) * seq_len + k_pos) * head_dim;
    
    float dot = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        dot += q[d] * k[d];
    }
    
    scores[((b * num_heads + h) * seq_len + q_pos) * seq_len + k_pos] = dot * scale;
}

// Flash Attention style kernel (simplified)
// Computes attention in tiles to reduce memory
kernel void flash_attention_kernel(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& num_heads [[buffer(5)]],
    constant int& seq_len [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant float& scale [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_position_in_threadgroup]]
) {
    constexpr int TILE_SIZE = 64;
    
    threadgroup float tile_k[TILE_SIZE * 64];  // [TILE_SIZE, head_dim]
    threadgroup float tile_v[TILE_SIZE * 64];
    threadgroup float tile_scores[TILE_SIZE];
    
    int b = tgid.y / num_heads;
    int h = tgid.y % num_heads;
    int q_tile = tgid.x;
    
    if (b >= batch_size) return;
    
    int q_start = q_tile * TILE_SIZE;
    
    // Process each query position in this tile
    for (int q_off = 0; q_off < TILE_SIZE && q_start + q_off < seq_len; q_off++) {
        int q_pos = q_start + q_off;
        
        device const float* q = Q + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
        device float* out = output + ((b * num_heads + h) * seq_len + q_pos) * head_dim;
        
        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        
        // Initialize output accumulator
        float acc[64];
        for (int d = 0; d < head_dim; d++) {
            acc[d] = 0.0f;
        }
        
        // Process K/V in tiles
        int num_k_tiles = (seq_len + TILE_SIZE - 1) / TILE_SIZE;
        for (int kt = 0; kt < num_k_tiles; kt++) {
            int k_start = kt * TILE_SIZE;
            
            // Load K tile (cooperative load)
            for (int i = tid_in_tg; i < TILE_SIZE * head_dim; i += 256) {
                int ki = i / head_dim;
                int d = i % head_dim;
                int k_pos = k_start + ki;
                if (k_pos < seq_len) {
                    tile_k[i] = K[((b * num_heads + h) * seq_len + k_pos) * head_dim + d];
                    tile_v[i] = V[((b * num_heads + h) * seq_len + k_pos) * head_dim + d];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Compute attention for valid K positions
            for (int ki = 0; ki < TILE_SIZE && k_start + ki < seq_len; ki++) {
                int k_pos = k_start + ki;
                if (k_pos > q_pos) continue;  // Causal mask
                
                // Compute Q @ K score
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += q[d] * tile_k[ki * head_dim + d];
                }
                score *= scale;
                
                // Online softmax update
                float old_max = max_score;
                max_score = max(max_score, score);
                float exp_diff = exp(old_max - max_score);
                sum_exp = sum_exp * exp_diff + exp(score - max_score);
                
                // Update accumulator
                float weight = exp(score - max_score);
                for (int d = 0; d < head_dim; d++) {
                    acc[d] = acc[d] * exp_diff + weight * tile_v[ki * head_dim + d];
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Normalize and write output
        for (int d = 0; d < head_dim; d++) {
            out[d] = acc[d] / sum_exp;
        }
    }
}
)";

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
