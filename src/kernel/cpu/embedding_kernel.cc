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
 * CPU Embedding Kernels
 */

#include "kernel/cpu/cpu_kernel_config.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <thread>
#include <vector>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#endif

namespace yirage {
namespace kernel {
namespace cpu {

// =============================================================================
// Embedding Lookup
// =============================================================================

void embedding_lookup_f32(
    const int* token_ids,
    const float* embedding_table,
    float* output,
    int num_tokens,
    int embedding_dim,
    int vocab_size
) {
    for (int t = 0; t < num_tokens; t++) {
        int token_id = token_ids[t];
        if (token_id < 0 || token_id >= vocab_size) continue;
        
        const float* src = embedding_table + token_id * embedding_dim;
        float* dst = output + t * embedding_dim;
        
        std::memcpy(dst, src, embedding_dim * sizeof(float));
    }
}

void embedding_lookup_parallel_f32(
    const int* token_ids,
    const float* embedding_table,
    float* output,
    int num_tokens,
    int embedding_dim,
    int vocab_size,
    int num_threads
) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    std::vector<std::thread> threads;
    int tokens_per_thread = (num_tokens + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        int start = t * tokens_per_thread;
        int end = std::min(start + tokens_per_thread, num_tokens);
        
        if (start >= num_tokens) break;
        
        threads.emplace_back([=]() {
            for (int i = start; i < end; i++) {
                int token_id = token_ids[i];
                if (token_id < 0 || token_id >= vocab_size) continue;
                
                const float* src = embedding_table + token_id * embedding_dim;
                float* dst = output + i * embedding_dim;
                std::memcpy(dst, src, embedding_dim * sizeof(float));
            }
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
}

// =============================================================================
// Position Embedding (Learned)
// =============================================================================

void add_position_embedding_f32(
    float* embeddings,
    const float* position_table,
    const int* positions,
    int num_tokens,
    int embedding_dim,
    int max_positions,
    SIMDType simd_type
) {
    for (int t = 0; t < num_tokens; t++) {
        int pos = positions[t];
        if (pos < 0 || pos >= max_positions) continue;
        
        float* emb = embeddings + t * embedding_dim;
        const float* pos_emb = position_table + pos * embedding_dim;
        
        int d = 0;
        
#ifdef __AVX2__
        if (simd_type == SIMDType::AVX2 || simd_type == SIMDType::AVX512) {
            for (; d <= embedding_dim - 8; d += 8) {
                __m256 e = _mm256_loadu_ps(&emb[d]);
                __m256 p = _mm256_loadu_ps(&pos_emb[d]);
                _mm256_storeu_ps(&emb[d], _mm256_add_ps(e, p));
            }
        }
#endif
#ifdef __aarch64__
        if (simd_type == SIMDType::NEON) {
            for (; d <= embedding_dim - 4; d += 4) {
                float32x4_t e = vld1q_f32(&emb[d]);
                float32x4_t p = vld1q_f32(&pos_emb[d]);
                vst1q_f32(&emb[d], vaddq_f32(e, p));
            }
        }
#endif
        
        for (; d < embedding_dim; d++) {
            emb[d] += pos_emb[d];
        }
    }
}

// =============================================================================
// Sinusoidal Position Embedding
// =============================================================================

void compute_sinusoidal_embedding_f32(
    float* output,
    int max_positions,
    int embedding_dim
) {
    int half_dim = embedding_dim / 2;
    
    for (int pos = 0; pos < max_positions; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / std::pow(10000.0f, static_cast<float>(i) / half_dim);
            float angle = pos * freq;
            
            output[pos * embedding_dim + i * 2] = std::sin(angle);
            output[pos * embedding_dim + i * 2 + 1] = std::cos(angle);
        }
    }
}

// =============================================================================
// RoPE (Rotary Position Embedding)
// =============================================================================

void apply_rope_f32(
    float* query,
    float* key,
    const float* cos_cache,
    const float* sin_cache,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    int rotary_dim
) {
    int half_rotary = rotary_dim / 2;
    
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int pos = 0; pos < seq_len; pos++) {
                int base = ((b * num_heads + h) * seq_len + pos) * head_dim;
                
                for (int i = 0; i < half_rotary; i++) {
                    float cos_val = cos_cache[pos * half_rotary + i];
                    float sin_val = sin_cache[pos * half_rotary + i];
                    
                    // Query
                    float q0 = query[base + i * 2];
                    float q1 = query[base + i * 2 + 1];
                    query[base + i * 2] = q0 * cos_val - q1 * sin_val;
                    query[base + i * 2 + 1] = q0 * sin_val + q1 * cos_val;
                    
                    // Key
                    float k0 = key[base + i * 2];
                    float k1 = key[base + i * 2 + 1];
                    key[base + i * 2] = k0 * cos_val - k1 * sin_val;
                    key[base + i * 2 + 1] = k0 * sin_val + k1 * cos_val;
                }
            }
        }
    }
}

void precompute_rope_cache_f32(
    float* cos_cache,
    float* sin_cache,
    int max_seq_len,
    int head_dim,
    float base
) {
    int half_dim = head_dim / 2;
    
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float freq = 1.0f / std::pow(base, static_cast<float>(i * 2) / head_dim);
            float angle = pos * freq;
            
            cos_cache[pos * half_dim + i] = std::cos(angle);
            sin_cache[pos * half_dim + i] = std::sin(angle);
        }
    }
}

// =============================================================================
// LM Head (Output projection)
// =============================================================================

void lm_head_f32(
    const float* hidden,
    const float* weight,
    float* logits,
    int batch_size,
    int hidden_dim,
    int vocab_size,
    SIMDType simd_type
) {
    // This is essentially hidden @ weight.T
    for (int b = 0; b < batch_size; b++) {
        const float* h = hidden + b * hidden_dim;
        float* out = logits + b * vocab_size;
        
        for (int v = 0; v < vocab_size; v++) {
            const float* w = weight + v * hidden_dim;
            float sum = 0.0f;
            int d = 0;
            
#ifdef __AVX2__
            if (simd_type == SIMDType::AVX2 || simd_type == SIMDType::AVX512) {
                __m256 sum_vec = _mm256_setzero_ps();
                for (; d <= hidden_dim - 8; d += 8) {
                    __m256 hv = _mm256_loadu_ps(&h[d]);
                    __m256 wv = _mm256_loadu_ps(&w[d]);
                    sum_vec = _mm256_fmadd_ps(hv, wv, sum_vec);
                }
                
                __m128 lo = _mm256_castps256_ps128(sum_vec);
                __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
                lo = _mm_add_ps(lo, hi);
                lo = _mm_hadd_ps(lo, lo);
                lo = _mm_hadd_ps(lo, lo);
                sum = _mm_cvtss_f32(lo);
            }
#endif
            
            for (; d < hidden_dim; d++) {
                sum += h[d] * w[d];
            }
            
            out[v] = sum;
        }
    }
}

}  // namespace cpu
}  // namespace kernel
}  // namespace yirage
