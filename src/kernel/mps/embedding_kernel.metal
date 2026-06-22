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
 * Metal Embedding Kernels for Apple Silicon
 */

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Parameters
// =============================================================================

struct EmbeddingParams {
    uint num_tokens;
    uint embedding_dim;
    uint vocab_size;
};

// =============================================================================
// Token Embedding Lookup
// =============================================================================

kernel void embedding_lookup_f16(
    device const int* token_ids [[buffer(0)]],
    device const half* embedding_table [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint dim_idx = gid.x;
    
    if (token_idx >= params.num_tokens || dim_idx >= params.embedding_dim) return;
    
    int token_id = token_ids[token_idx];
    if (token_id < 0 || uint(token_id) >= params.vocab_size) return;
    
    output[token_idx * params.embedding_dim + dim_idx] = 
        embedding_table[token_id * params.embedding_dim + dim_idx];
}

kernel void embedding_lookup_f32(
    device const int* token_ids [[buffer(0)]],
    device const float* embedding_table [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint dim_idx = gid.x;
    
    if (token_idx >= params.num_tokens || dim_idx >= params.embedding_dim) return;
    
    int token_id = token_ids[token_idx];
    if (token_id < 0 || uint(token_id) >= params.vocab_size) return;
    
    output[token_idx * params.embedding_dim + dim_idx] = 
        embedding_table[token_id * params.embedding_dim + dim_idx];
}

// =============================================================================
// Vectorized Embedding Lookup
// =============================================================================

kernel void embedding_lookup_f16_vec4(
    device const int* token_ids [[buffer(0)]],
    device const half4* embedding_table [[buffer(1)]],
    device half4* output [[buffer(2)]],
    constant EmbeddingParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint vec_idx = gid.x;
    uint embedding_dim_vec4 = params.embedding_dim / 4;
    
    if (token_idx >= params.num_tokens || vec_idx >= embedding_dim_vec4) return;
    
    int token_id = token_ids[token_idx];
    if (token_id < 0 || uint(token_id) >= params.vocab_size) return;
    
    output[token_idx * embedding_dim_vec4 + vec_idx] = 
        embedding_table[token_id * embedding_dim_vec4 + vec_idx];
}

// =============================================================================
// Position Embedding
// =============================================================================

struct PositionParams {
    uint num_tokens;
    uint embedding_dim;
    uint max_positions;
};

kernel void add_position_embedding_f16(
    device half* embeddings [[buffer(0)]],
    device const half* position_table [[buffer(1)]],
    device const int* positions [[buffer(2)]],
    constant PositionParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint dim_idx = gid.x;
    
    if (token_idx >= params.num_tokens || dim_idx >= params.embedding_dim) return;
    
    int pos = positions[token_idx];
    if (pos < 0 || uint(pos) >= params.max_positions) return;
    
    uint idx = token_idx * params.embedding_dim + dim_idx;
    float emb = float(embeddings[idx]);
    float pos_emb = float(position_table[pos * params.embedding_dim + dim_idx]);
    embeddings[idx] = half(emb + pos_emb);
}

// =============================================================================
// Sinusoidal Position Embedding (computed on-the-fly)
// =============================================================================

kernel void sinusoidal_position_embedding_f16(
    device half* output [[buffer(0)]],
    constant PositionParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint pos = gid.y;
    uint dim_idx = gid.x;
    
    if (pos >= params.max_positions || dim_idx >= params.embedding_dim) return;
    
    uint half_dim = params.embedding_dim / 2;
    uint pair_idx = dim_idx / 2;
    
    // Frequency: 1 / (10000^(2i/d))
    float freq = 1.0f / pow(10000.0f, float(pair_idx) / float(half_dim));
    float angle = float(pos) * freq;
    
    float value;
    if (dim_idx % 2 == 0) {
        value = sin(angle);
    } else {
        value = cos(angle);
    }
    
    output[pos * params.embedding_dim + dim_idx] = half(value);
}

// =============================================================================
// LM Head (Embedding transpose for output projection)
// =============================================================================

struct LMHeadParams {
    uint batch_size;
    uint hidden_dim;
    uint vocab_size;
};

kernel void lm_head_f16(
    device const half* hidden [[buffer(0)]],
    device const half* weight [[buffer(1)]],
    device float* logits [[buffer(2)]],
    constant LMHeadParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    uint batch = gid.y;
    uint vocab_idx = gid.x;
    
    if (batch >= params.batch_size || vocab_idx >= params.vocab_size) return;
    
    device const half* h = hidden + batch * params.hidden_dim;
    device const half* w = weight + vocab_idx * params.hidden_dim;
    
    // Dot product
    float sum = 0.0f;
    for (uint d = 0; d < params.hidden_dim; d++) {
        sum += float(h[d]) * float(w[d]);
    }
    
    logits[batch * params.vocab_size + vocab_idx] = sum;
}

// =============================================================================
// Argmax (Greedy Decoding)
// =============================================================================

inline float simd_max_val(float val) {
    val = max(val, simd_shuffle_xor(val, 16));
    val = max(val, simd_shuffle_xor(val, 8));
    val = max(val, simd_shuffle_xor(val, 4));
    val = max(val, simd_shuffle_xor(val, 2));
    val = max(val, simd_shuffle_xor(val, 1));
    return val;
}

kernel void argmax_f32(
    device const float* logits [[buffer(0)]],
    device int* output_tokens [[buffer(1)]],
    constant uint& batch_size [[buffer(2)]],
    constant uint& vocab_size [[buffer(3)]],
    uint batch [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_max[8];
    threadgroup int shared_idx[8];
    
    if (batch >= batch_size) return;
    
    device const float* row = logits + batch * vocab_size;
    
    float local_max = -INFINITY;
    int local_idx = 0;
    
    for (uint i = tid; i < vocab_size; i += 256) {
        float val = row[i];
        if (val > local_max) {
            local_max = val;
            local_idx = int(i);
        }
    }
    
    // SIMD reduction
    float simd_max = simd_max_val(local_max);
    
    // Find which lane has the max
    bool has_max = (local_max == simd_max);
    int winning_lane = ctz(simd_ballot(has_max));
    
    if (simd_lane == uint(winning_lane)) {
        shared_max[simd_id] = local_max;
        shared_idx[simd_id] = local_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Final reduction
    if (simd_id == 0 && simd_lane < 8) {
        local_max = shared_max[simd_lane];
        local_idx = shared_idx[simd_lane];
        
        float final_max = simd_max_val(local_max);
        bool is_winner = (local_max == final_max);
        int winner = ctz(simd_ballot(is_winner));
        
        if (simd_lane == 0) {
            output_tokens[batch] = shared_idx[winner];
        }
    }
}
