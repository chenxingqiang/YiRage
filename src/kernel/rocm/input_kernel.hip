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
 * ROCm/HIP Input Kernels
 * 
 * Kernels for loading and preprocessing input data.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// =============================================================================
// Token Embedding Lookup
// =============================================================================

/**
 * @brief Embedding lookup with FP16 table
 */
__global__ void embedding_lookup_f16_hip(
    const int* __restrict__ token_ids,
    const half* __restrict__ embedding_table,
    half* __restrict__ output,
    int num_tokens,
    int embedding_dim,
    int vocab_size
) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (token_idx >= num_tokens) return;
    
    int token_id = token_ids[token_idx];
    if (token_id < 0 || token_id >= vocab_size) return;
    
    // Copy embedding vector (coalesced access within warp)
    for (int d = dim_idx; d < embedding_dim; d += blockDim.x) {
        output[token_idx * embedding_dim + d] = 
            embedding_table[token_id * embedding_dim + d];
    }
}

/**
 * @brief Embedding lookup with FP32 table
 */
__global__ void embedding_lookup_f32_hip(
    const int* __restrict__ token_ids,
    const float* __restrict__ embedding_table,
    float* __restrict__ output,
    int num_tokens,
    int embedding_dim,
    int vocab_size
) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (token_idx >= num_tokens) return;
    
    int token_id = token_ids[token_idx];
    if (token_id < 0 || token_id >= vocab_size) return;
    
    for (int d = dim_idx; d < embedding_dim; d += blockDim.x) {
        output[token_idx * embedding_dim + d] = 
            embedding_table[token_id * embedding_dim + d];
    }
}

// =============================================================================
// Positional Embedding
// =============================================================================

/**
 * @brief Add positional encoding (learned)
 */
__global__ void add_position_embedding_f16_hip(
    half* __restrict__ embeddings,
    const half* __restrict__ position_table,
    const int* __restrict__ positions,
    int num_tokens,
    int embedding_dim,
    int max_positions
) {
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (token_idx >= num_tokens) return;
    
    int pos = positions[token_idx];
    if (pos < 0 || pos >= max_positions) return;
    
    for (int d = dim_idx; d < embedding_dim; d += blockDim.x) {
        float emb = __half2float(embeddings[token_idx * embedding_dim + d]);
        float pos_emb = __half2float(position_table[pos * embedding_dim + d]);
        embeddings[token_idx * embedding_dim + d] = __float2half(emb + pos_emb);
    }
}

/**
 * @brief RoPE (Rotary Position Embedding)
 */
__global__ void apply_rope_f16_hip(
    half* __restrict__ query,
    half* __restrict__ key,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int rotary_dim
) {
    int batch = blockIdx.z;
    int head = blockIdx.y;
    int pos = blockIdx.x;
    int tid = threadIdx.x;
    
    if (pos >= seq_len) return;
    
    int half_rotary = rotary_dim / 2;
    
    // Process pairs of dimensions
    for (int i = tid; i < half_rotary; i += blockDim.x) {
        int base = ((batch * num_heads + head) * seq_len + pos) * head_dim;
        
        float cos_val = cos_cache[pos * half_rotary + i];
        float sin_val = sin_cache[pos * half_rotary + i];
        
        // Query
        float q0 = __half2float(query[base + i * 2]);
        float q1 = __half2float(query[base + i * 2 + 1]);
        query[base + i * 2] = __float2half(q0 * cos_val - q1 * sin_val);
        query[base + i * 2 + 1] = __float2half(q0 * sin_val + q1 * cos_val);
        
        // Key
        float k0 = __half2float(key[base + i * 2]);
        float k1 = __half2float(key[base + i * 2 + 1]);
        key[base + i * 2] = __float2half(k0 * cos_val - k1 * sin_val);
        key[base + i * 2 + 1] = __float2half(k0 * sin_val + k1 * cos_val);
    }
}

// =============================================================================
// Input Preprocessing
// =============================================================================

/**
 * @brief Normalize input (mean=0, std=1)
 */
__global__ void normalize_input_f32_hip(
    float* __restrict__ input,
    int num_samples,
    int feature_dim,
    float mean,
    float std
) {
    int sample = blockIdx.x;
    int tid = threadIdx.x;
    
    if (sample >= num_samples) return;
    
    float inv_std = 1.0f / std;
    
    for (int d = tid; d < feature_dim; d += blockDim.x) {
        int idx = sample * feature_dim + d;
        input[idx] = (input[idx] - mean) * inv_std;
    }
}

/**
 * @brief Pad input sequence
 */
__global__ void pad_sequence_f16_hip(
    const half* __restrict__ input,
    half* __restrict__ output,
    const int* __restrict__ lengths,
    int batch_size,
    int max_len,
    int feature_dim,
    half pad_value
) {
    int batch = blockIdx.y;
    int pos = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch >= batch_size || pos >= max_len) return;
    
    int seq_len = lengths[batch];
    
    for (int d = tid; d < feature_dim; d += blockDim.x) {
        int out_idx = (batch * max_len + pos) * feature_dim + d;
        
        if (pos < seq_len) {
            // Copy actual data (assume input is packed)
            // Need cumulative sum of lengths for proper indexing
            output[out_idx] = input[out_idx];  // Simplified
        } else {
            output[out_idx] = pad_value;
        }
    }
}

// =============================================================================
// Kernel Launch Wrappers
// =============================================================================

extern "C" {

void launch_embedding_lookup_f16_hip(
    const int* token_ids, const half* embedding_table, half* output,
    int num_tokens, int embedding_dim, int vocab_size,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_tokens);
    
    hipLaunchKernelGGL(embedding_lookup_f16_hip, grid, block, 0, stream,
                       token_ids, embedding_table, output,
                       num_tokens, embedding_dim, vocab_size);
}

void launch_embedding_lookup_f32_hip(
    const int* token_ids, const float* embedding_table, float* output,
    int num_tokens, int embedding_dim, int vocab_size,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_tokens);
    
    hipLaunchKernelGGL(embedding_lookup_f32_hip, grid, block, 0, stream,
                       token_ids, embedding_table, output,
                       num_tokens, embedding_dim, vocab_size);
}

void launch_add_position_embedding_f16_hip(
    half* embeddings, const half* position_table, const int* positions,
    int num_tokens, int embedding_dim, int max_positions,
    hipStream_t stream
) {
    dim3 block(256);
    dim3 grid(num_tokens);
    
    hipLaunchKernelGGL(add_position_embedding_f16_hip, grid, block, 0, stream,
                       embeddings, position_table, positions,
                       num_tokens, embedding_dim, max_positions);
}

void launch_apply_rope_f16_hip(
    half* query, half* key,
    const float* cos_cache, const float* sin_cache,
    int batch_size, int num_heads, int seq_len, int head_dim, int rotary_dim,
    hipStream_t stream
) {
    dim3 block(64);  // 1 wavefront
    dim3 grid(seq_len, num_heads, batch_size);
    
    hipLaunchKernelGGL(apply_rope_f16_hip, grid, block, 0, stream,
                       query, key, cos_cache, sin_cache,
                       batch_size, num_heads, seq_len, head_dim, rotary_dim);
}

}  // extern "C"
