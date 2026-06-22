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
 * CPU RMSNorm Kernels
 */

#include "kernel/cpu/cpu_kernel_config.h"

#include <cmath>
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
// Naive RMSNorm
// =============================================================================

void rms_norm_naive_f32(
    const float* input,
    const float* weight,
    float* output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    for (int t = 0; t < num_tokens; t++) {
        const float* in = input + t * hidden_dim;
        float* out = output + t * hidden_dim;
        
        // Compute sum of squares
        float sum_sq = 0.0f;
        for (int d = 0; d < hidden_dim; d++) {
            sum_sq += in[d] * in[d];
        }
        
        // Compute inverse RMS
        float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_dim + eps);
        
        // Normalize and scale
        for (int d = 0; d < hidden_dim; d++) {
            out[d] = in[d] * inv_rms * weight[d];
        }
    }
}

// =============================================================================
// AVX2 RMSNorm
// =============================================================================

#ifdef __AVX2__

void rms_norm_avx2_f32(
    const float* input,
    const float* weight,
    float* output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    for (int t = 0; t < num_tokens; t++) {
        const float* in = input + t * hidden_dim;
        float* out = output + t * hidden_dim;
        
        // Compute sum of squares with AVX2
        __m256 sum_sq_vec = _mm256_setzero_ps();
        int d = 0;
        
        for (; d <= hidden_dim - 8; d += 8) {
            __m256 x = _mm256_loadu_ps(&in[d]);
            sum_sq_vec = _mm256_fmadd_ps(x, x, sum_sq_vec);
        }
        
        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(sum_sq_vec);
        __m128 hi = _mm256_extractf128_ps(sum_sq_vec, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float sum_sq = _mm_cvtss_f32(lo);
        
        // Handle remainder
        for (; d < hidden_dim; d++) {
            sum_sq += in[d] * in[d];
        }
        
        float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_dim + eps);
        __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);
        
        // Normalize and scale
        d = 0;
        for (; d <= hidden_dim - 8; d += 8) {
            __m256 x = _mm256_loadu_ps(&in[d]);
            __m256 w = _mm256_loadu_ps(&weight[d]);
            __m256 y = _mm256_mul_ps(_mm256_mul_ps(x, inv_rms_vec), w);
            _mm256_storeu_ps(&out[d], y);
        }
        
        // Handle remainder
        for (; d < hidden_dim; d++) {
            out[d] = in[d] * inv_rms * weight[d];
        }
    }
}

#endif  // __AVX2__

// =============================================================================
// AVX-512 RMSNorm
// =============================================================================

#ifdef __AVX512F__

void rms_norm_avx512_f32(
    const float* input,
    const float* weight,
    float* output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    for (int t = 0; t < num_tokens; t++) {
        const float* in = input + t * hidden_dim;
        float* out = output + t * hidden_dim;
        
        __m512 sum_sq_vec = _mm512_setzero_ps();
        int d = 0;
        
        for (; d <= hidden_dim - 16; d += 16) {
            __m512 x = _mm512_loadu_ps(&in[d]);
            sum_sq_vec = _mm512_fmadd_ps(x, x, sum_sq_vec);
        }
        
        float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);
        
        // Handle remainder
        for (; d < hidden_dim; d++) {
            sum_sq += in[d] * in[d];
        }
        
        float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_dim + eps);
        __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);
        
        d = 0;
        for (; d <= hidden_dim - 16; d += 16) {
            __m512 x = _mm512_loadu_ps(&in[d]);
            __m512 w = _mm512_loadu_ps(&weight[d]);
            __m512 y = _mm512_mul_ps(_mm512_mul_ps(x, inv_rms_vec), w);
            _mm512_storeu_ps(&out[d], y);
        }
        
        for (; d < hidden_dim; d++) {
            out[d] = in[d] * inv_rms * weight[d];
        }
    }
}

#endif  // __AVX512F__

// =============================================================================
// ARM NEON RMSNorm
// =============================================================================

#ifdef __aarch64__

void rms_norm_neon_f32(
    const float* input,
    const float* weight,
    float* output,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    for (int t = 0; t < num_tokens; t++) {
        const float* in = input + t * hidden_dim;
        float* out = output + t * hidden_dim;
        
        float32x4_t sum_sq_vec = vdupq_n_f32(0.0f);
        int d = 0;
        
        for (; d <= hidden_dim - 4; d += 4) {
            float32x4_t x = vld1q_f32(&in[d]);
            sum_sq_vec = vfmaq_f32(sum_sq_vec, x, x);
        }
        
        // Horizontal sum
        float32x2_t sum_lo = vget_low_f32(sum_sq_vec);
        float32x2_t sum_hi = vget_high_f32(sum_sq_vec);
        sum_lo = vadd_f32(sum_lo, sum_hi);
        float sum_sq = vget_lane_f32(vpadd_f32(sum_lo, sum_lo), 0);
        
        for (; d < hidden_dim; d++) {
            sum_sq += in[d] * in[d];
        }
        
        float inv_rms = 1.0f / std::sqrt(sum_sq / hidden_dim + eps);
        float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);
        
        d = 0;
        for (; d <= hidden_dim - 4; d += 4) {
            float32x4_t x = vld1q_f32(&in[d]);
            float32x4_t w = vld1q_f32(&weight[d]);
            float32x4_t y = vmulq_f32(vmulq_f32(x, inv_rms_vec), w);
            vst1q_f32(&out[d], y);
        }
        
        for (; d < hidden_dim; d++) {
            out[d] = in[d] * inv_rms * weight[d];
        }
    }
}

#endif  // __aarch64__

// =============================================================================
// Multi-threaded RMSNorm
// =============================================================================

void rms_norm_parallel_f32(
    const float* input,
    const float* weight,
    float* output,
    int num_tokens,
    int hidden_dim,
    float eps,
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
            rms_norm_naive_f32(input + start * hidden_dim, weight,
                              output + start * hidden_dim,
                              end - start, hidden_dim, eps);
        });
    }
    
    for (auto& th : threads) {
        th.join();
    }
}

// =============================================================================
// Dispatcher
// =============================================================================

void rms_norm_f32(
    const float* input,
    const float* weight,
    float* output,
    int num_tokens,
    int hidden_dim,
    float eps,
    SIMDType simd_type
) {
    switch (simd_type) {
#ifdef __AVX512F__
        case SIMDType::AVX512:
            rms_norm_avx512_f32(input, weight, output, num_tokens, hidden_dim, eps);
            break;
#endif
#ifdef __AVX2__
        case SIMDType::AVX2:
        case SIMDType::AVX:
            rms_norm_avx2_f32(input, weight, output, num_tokens, hidden_dim, eps);
            break;
#endif
#ifdef __aarch64__
        case SIMDType::NEON:
            rms_norm_neon_f32(input, weight, output, num_tokens, hidden_dim, eps);
            break;
#endif
        default:
            rms_norm_naive_f32(input, weight, output, num_tokens, hidden_dim, eps);
            break;
    }
}

}  // namespace cpu
}  // namespace kernel
}  // namespace yirage
