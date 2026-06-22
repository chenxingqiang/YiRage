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
 * @file rotary_embedding.metal.h
 * @brief MPS Rotary Position Embedding (RoPE) kernel
 *
 * Applies rotary position embeddings to Q and K tensors.
 * Used in LLaMA, Qwen, and other modern transformers.
 */

namespace yirage {
namespace persistent_kernel {
namespace mps {

constexpr const char* ROTARY_EMBEDDING_KERNEL_SOURCE = R"(
// Rotary Position Embedding (RoPE)
// Applies rotation to each pair of dimensions based on position
kernel void rotary_embedding_kernel(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* cos_cache [[buffer(2)]],
    device const float* sin_cache [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& num_heads [[buffer(5)]],
    constant int& seq_len [[buffer(6)]],
    constant int& head_dim [[buffer(7)]],
    constant int& position_offset [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.z;
    int h = tid.y;
    int s = tid.x;
    
    if (b >= batch_size || h >= num_heads || s >= seq_len) return;
    
    int half_dim = head_dim / 2;
    int pos = s + position_offset;
    
    device float* q_ptr = q + ((b * num_heads + h) * seq_len + s) * head_dim;
    device const float* cos_ptr = cos_cache + pos * half_dim;
    device const float* sin_ptr = sin_cache + pos * half_dim;
    
    // Apply rotation to Q
    for (int d = 0; d < half_dim; d++) {
        float x0 = q_ptr[d];
        float x1 = q_ptr[d + half_dim];
        float cos_val = cos_ptr[d];
        float sin_val = sin_ptr[d];
        
        q_ptr[d] = x0 * cos_val - x1 * sin_val;
        q_ptr[d + half_dim] = x1 * cos_val + x0 * sin_val;
    }
}

// RoPE for K tensor (separate kernel for flexibility)
kernel void rotary_embedding_k_kernel(
    device float* k [[buffer(0)]],
    device const float* cos_cache [[buffer(1)]],
    device const float* sin_cache [[buffer(2)]],
    constant int& batch_size [[buffer(3)]],
    constant int& num_kv_heads [[buffer(4)]],
    constant int& seq_len [[buffer(5)]],
    constant int& head_dim [[buffer(6)]],
    constant int& position_offset [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.z;
    int h = tid.y;
    int s = tid.x;
    
    if (b >= batch_size || h >= num_kv_heads || s >= seq_len) return;
    
    int half_dim = head_dim / 2;
    int pos = s + position_offset;
    
    device float* k_ptr = k + ((b * num_kv_heads + h) * seq_len + s) * head_dim;
    device const float* cos_ptr = cos_cache + pos * half_dim;
    device const float* sin_ptr = sin_cache + pos * half_dim;
    
    for (int d = 0; d < half_dim; d++) {
        float x0 = k_ptr[d];
        float x1 = k_ptr[d + half_dim];
        float cos_val = cos_ptr[d];
        float sin_val = sin_ptr[d];
        
        k_ptr[d] = x0 * cos_val - x1 * sin_val;
        k_ptr[d + half_dim] = x1 * cos_val + x0 * sin_val;
    }
}

// Fused Q and K RoPE (more efficient)
kernel void rotary_embedding_fused_kernel(
    device float* q [[buffer(0)]],
    device float* k [[buffer(1)]],
    device const float* cos_cache [[buffer(2)]],
    device const float* sin_cache [[buffer(3)]],
    constant int& batch_size [[buffer(4)]],
    constant int& num_q_heads [[buffer(5)]],
    constant int& num_kv_heads [[buffer(6)]],
    constant int& seq_len [[buffer(7)]],
    constant int& head_dim [[buffer(8)]],
    constant int& position_offset [[buffer(9)]],
    uint3 tid [[thread_position_in_grid]]
) {
    int b = tid.z;
    int s = tid.x;
    
    if (b >= batch_size || s >= seq_len) return;
    
    int half_dim = head_dim / 2;
    int pos = s + position_offset;
    
    device const float* cos_ptr = cos_cache + pos * half_dim;
    device const float* sin_ptr = sin_cache + pos * half_dim;
    
    // Apply to all Q heads
    for (int h = 0; h < num_q_heads; h++) {
        device float* q_ptr = q + ((b * num_q_heads + h) * seq_len + s) * head_dim;
        
        for (int d = 0; d < half_dim; d++) {
            float x0 = q_ptr[d];
            float x1 = q_ptr[d + half_dim];
            q_ptr[d] = x0 * cos_ptr[d] - x1 * sin_ptr[d];
            q_ptr[d + half_dim] = x1 * cos_ptr[d] + x0 * sin_ptr[d];
        }
    }
    
    // Apply to all KV heads
    for (int h = 0; h < num_kv_heads; h++) {
        device float* k_ptr = k + ((b * num_kv_heads + h) * seq_len + s) * head_dim;
        
        for (int d = 0; d < half_dim; d++) {
            float x0 = k_ptr[d];
            float x1 = k_ptr[d + half_dim];
            k_ptr[d] = x0 * cos_ptr[d] - x1 * sin_ptr[d];
            k_ptr[d + half_dim] = x1 * cos_ptr[d] + x0 * sin_ptr[d];
        }
    }
}

// Precompute RoPE frequencies
kernel void compute_rope_freqs_kernel(
    device float* cos_cache [[buffer(0)]],
    device float* sin_cache [[buffer(1)]],
    constant int& max_seq_len [[buffer(2)]],
    constant int& head_dim [[buffer(3)]],
    constant float& base [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int pos = tid.y;
    int d = tid.x;
    
    if (pos >= max_seq_len || d >= head_dim / 2) return;
    
    float freq = 1.0f / pow(base, float(2 * d) / float(head_dim));
    float angle = float(pos) * freq;
    
    int idx = pos * (head_dim / 2) + d;
    cos_cache[idx] = cos(angle);
    sin_cache[idx] = sin(angle);
}
)";

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
