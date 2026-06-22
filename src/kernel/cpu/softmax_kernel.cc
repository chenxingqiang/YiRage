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
 * CPU Softmax Kernels
 */

#include "kernel/cpu/cpu_kernel_config.h"

#include <cmath>
#include <algorithm>
#include <limits>
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
// Naive Softmax
// =============================================================================

void softmax_naive_f32(
    const float* input,
    float* output,
    int num_rows,
    int row_size
) {
    for (int r = 0; r < num_rows; r++) {
        const float* in_row = input + r * row_size;
        float* out_row = output + r * row_size;
        
        // Find max
        float max_val = in_row[0];
        for (int i = 1; i < row_size; i++) {
            max_val = std::max(max_val, in_row[i]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int i = 0; i < row_size; i++) {
            out_row[i] = std::exp(in_row[i] - max_val);
            sum += out_row[i];
        }
        
        // Normalize
        float inv_sum = 1.0f / sum;
        for (int i = 0; i < row_size; i++) {
            out_row[i] *= inv_sum;
        }
    }
}

// =============================================================================
// AVX2 Softmax
// =============================================================================

#ifdef __AVX2__

// Fast exp approximation for AVX2
inline __m256 exp_avx2(__m256 x) {
    // Clamping for numerical stability
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    
    // exp(x) = 2^(x * log2(e))
    __m256 log2e = _mm256_set1_ps(1.44269504088896341f);
    __m256 t = _mm256_mul_ps(x, log2e);
    
    // Split into integer and fractional parts
    __m256 ti = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT);
    __m256 tf = _mm256_sub_ps(t, ti);
    
    // Polynomial approximation for 2^tf
    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(0.693147180559945f);
    __m256 c2 = _mm256_set1_ps(0.240226506959101f);
    __m256 c3 = _mm256_set1_ps(0.055504108664822f);
    
    __m256 p = _mm256_fmadd_ps(c3, tf, c2);
    p = _mm256_fmadd_ps(p, tf, c1);
    p = _mm256_fmadd_ps(p, tf, c0);
    
    // Scale by 2^ti
    __m256i ti_int = _mm256_cvtps_epi32(ti);
    ti_int = _mm256_add_epi32(ti_int, _mm256_set1_epi32(127));
    ti_int = _mm256_slli_epi32(ti_int, 23);
    __m256 scale = _mm256_castsi256_ps(ti_int);
    
    return _mm256_mul_ps(p, scale);
}

void softmax_avx2_f32(
    const float* input,
    float* output,
    int num_rows,
    int row_size
) {
    for (int r = 0; r < num_rows; r++) {
        const float* in_row = input + r * row_size;
        float* out_row = output + r * row_size;
        
        // Find max with AVX2
        __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
        int i = 0;
        for (; i <= row_size - 8; i += 8) {
            __m256 x = _mm256_loadu_ps(&in_row[i]);
            max_vec = _mm256_max_ps(max_vec, x);
        }
        
        // Horizontal max
        __m128 lo = _mm256_castps256_ps128(max_vec);
        __m128 hi = _mm256_extractf128_ps(max_vec, 1);
        lo = _mm_max_ps(lo, hi);
        lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0x0E));
        lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0x01));
        float max_val = _mm_cvtss_f32(lo);
        
        for (; i < row_size; i++) {
            max_val = std::max(max_val, in_row[i]);
        }
        
        __m256 max_vec_bcast = _mm256_set1_ps(max_val);
        
        // Compute exp and sum
        __m256 sum_vec = _mm256_setzero_ps();
        i = 0;
        for (; i <= row_size - 8; i += 8) {
            __m256 x = _mm256_loadu_ps(&in_row[i]);
            __m256 e = exp_avx2(_mm256_sub_ps(x, max_vec_bcast));
            _mm256_storeu_ps(&out_row[i], e);
            sum_vec = _mm256_add_ps(sum_vec, e);
        }
        
        // Horizontal sum
        lo = _mm256_castps256_ps128(sum_vec);
        hi = _mm256_extractf128_ps(sum_vec, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float sum = _mm_cvtss_f32(lo);
        
        for (; i < row_size; i++) {
            out_row[i] = std::exp(in_row[i] - max_val);
            sum += out_row[i];
        }
        
        // Normalize
        __m256 inv_sum_vec = _mm256_set1_ps(1.0f / sum);
        i = 0;
        for (; i <= row_size - 8; i += 8) {
            __m256 e = _mm256_loadu_ps(&out_row[i]);
            _mm256_storeu_ps(&out_row[i], _mm256_mul_ps(e, inv_sum_vec));
        }
        
        float inv_sum = 1.0f / sum;
        for (; i < row_size; i++) {
            out_row[i] *= inv_sum;
        }
    }
}

#endif  // __AVX2__

// =============================================================================
// ARM NEON Softmax
// =============================================================================

#ifdef __aarch64__

void softmax_neon_f32(
    const float* input,
    float* output,
    int num_rows,
    int row_size
) {
    for (int r = 0; r < num_rows; r++) {
        const float* in_row = input + r * row_size;
        float* out_row = output + r * row_size;
        
        // Find max
        float32x4_t max_vec = vdupq_n_f32(-std::numeric_limits<float>::infinity());
        int i = 0;
        for (; i <= row_size - 4; i += 4) {
            float32x4_t x = vld1q_f32(&in_row[i]);
            max_vec = vmaxq_f32(max_vec, x);
        }
        
        float32x2_t max_lo = vget_low_f32(max_vec);
        float32x2_t max_hi = vget_high_f32(max_vec);
        max_lo = vpmax_f32(max_lo, max_hi);
        max_lo = vpmax_f32(max_lo, max_lo);
        float max_val = vget_lane_f32(max_lo, 0);
        
        for (; i < row_size; i++) {
            max_val = std::max(max_val, in_row[i]);
        }
        
        float32x4_t max_bcast = vdupq_n_f32(max_val);
        
        // Compute exp and sum
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        i = 0;
        for (; i <= row_size - 4; i += 4) {
            float32x4_t x = vld1q_f32(&in_row[i]);
            float32x4_t diff = vsubq_f32(x, max_bcast);
            
            // exp approximation using NEON
            float vals[4];
            vst1q_f32(vals, diff);
            for (int j = 0; j < 4; j++) {
                vals[j] = std::exp(vals[j]);
            }
            float32x4_t e = vld1q_f32(vals);
            
            vst1q_f32(&out_row[i], e);
            sum_vec = vaddq_f32(sum_vec, e);
        }
        
        float32x2_t sum_lo = vget_low_f32(sum_vec);
        float32x2_t sum_hi = vget_high_f32(sum_vec);
        sum_lo = vadd_f32(sum_lo, sum_hi);
        float sum = vget_lane_f32(vpadd_f32(sum_lo, sum_lo), 0);
        
        for (; i < row_size; i++) {
            out_row[i] = std::exp(in_row[i] - max_val);
            sum += out_row[i];
        }
        
        // Normalize
        float32x4_t inv_sum_vec = vdupq_n_f32(1.0f / sum);
        i = 0;
        for (; i <= row_size - 4; i += 4) {
            float32x4_t e = vld1q_f32(&out_row[i]);
            vst1q_f32(&out_row[i], vmulq_f32(e, inv_sum_vec));
        }
        
        float inv_sum = 1.0f / sum;
        for (; i < row_size; i++) {
            out_row[i] *= inv_sum;
        }
    }
}

#endif  // __aarch64__

// =============================================================================
// Dispatcher
// =============================================================================

void softmax_f32(
    const float* input,
    float* output,
    int num_rows,
    int row_size,
    SIMDType simd_type
) {
    switch (simd_type) {
#ifdef __AVX2__
        case SIMDType::AVX512:
        case SIMDType::AVX2:
        case SIMDType::AVX:
            softmax_avx2_f32(input, output, num_rows, row_size);
            break;
#endif
#ifdef __aarch64__
        case SIMDType::NEON:
            softmax_neon_f32(input, output, num_rows, row_size);
            break;
#endif
        default:
            softmax_naive_f32(input, output, num_rows, row_size);
            break;
    }
}

}  // namespace cpu
}  // namespace kernel
}  // namespace yirage
