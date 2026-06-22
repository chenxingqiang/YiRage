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
 * CPU Reduction Kernels
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
// Sum Reduction
// =============================================================================

float reduce_sum_naive_f32(const float* input, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    return sum;
}

#ifdef __AVX2__

float reduce_sum_avx2_f32(const float* input, int size) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    
    for (; i <= size - 8; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        sum_vec = _mm256_add_ps(sum_vec, x);
    }
    
    // Horizontal sum
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float sum = _mm_cvtss_f32(lo);
    
    for (; i < size; i++) {
        sum += input[i];
    }
    
    return sum;
}

#endif

#ifdef __aarch64__

float reduce_sum_neon_f32(const float* input, int size) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    int i = 0;
    
    for (; i <= size - 4; i += 4) {
        float32x4_t x = vld1q_f32(&input[i]);
        sum_vec = vaddq_f32(sum_vec, x);
    }
    
    float32x2_t sum_lo = vget_low_f32(sum_vec);
    float32x2_t sum_hi = vget_high_f32(sum_vec);
    sum_lo = vadd_f32(sum_lo, sum_hi);
    float sum = vget_lane_f32(vpadd_f32(sum_lo, sum_lo), 0);
    
    for (; i < size; i++) {
        sum += input[i];
    }
    
    return sum;
}

#endif

// =============================================================================
// Max Reduction
// =============================================================================

float reduce_max_naive_f32(const float* input, int size) {
    float max_val = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < size; i++) {
        max_val = std::max(max_val, input[i]);
    }
    return max_val;
}

#ifdef __AVX2__

float reduce_max_avx2_f32(const float* input, int size) {
    __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    int i = 0;
    
    for (; i <= size - 8; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        max_vec = _mm256_max_ps(max_vec, x);
    }
    
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0x0E));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, 0x01));
    float max_val = _mm_cvtss_f32(lo);
    
    for (; i < size; i++) {
        max_val = std::max(max_val, input[i]);
    }
    
    return max_val;
}

#endif

// =============================================================================
// Row-wise Reductions
// =============================================================================

void reduce_sum_row_f32(
    const float* input,
    float* output,
    int num_rows,
    int row_size,
    SIMDType simd_type
) {
    for (int r = 0; r < num_rows; r++) {
        const float* row = input + r * row_size;
#ifdef __AVX2__
        if (simd_type == SIMDType::AVX2 || simd_type == SIMDType::AVX512) {
            output[r] = reduce_sum_avx2_f32(row, row_size);
            continue;
        }
#endif
#ifdef __aarch64__
        if (simd_type == SIMDType::NEON) {
            output[r] = reduce_sum_neon_f32(row, row_size);
            continue;
        }
#endif
        output[r] = reduce_sum_naive_f32(row, row_size);
    }
}

void reduce_max_row_f32(
    const float* input,
    float* output,
    int num_rows,
    int row_size,
    SIMDType simd_type
) {
    for (int r = 0; r < num_rows; r++) {
        const float* row = input + r * row_size;
#ifdef __AVX2__
        if (simd_type == SIMDType::AVX2 || simd_type == SIMDType::AVX512) {
            output[r] = reduce_max_avx2_f32(row, row_size);
            continue;
        }
#endif
        output[r] = reduce_max_naive_f32(row, row_size);
    }
}

void reduce_mean_row_f32(
    const float* input,
    float* output,
    int num_rows,
    int row_size,
    SIMDType simd_type
) {
    reduce_sum_row_f32(input, output, num_rows, row_size, simd_type);
    float inv_size = 1.0f / row_size;
    for (int r = 0; r < num_rows; r++) {
        output[r] *= inv_size;
    }
}

// =============================================================================
// Argmax
// =============================================================================

int argmax_naive_f32(const float* input, int size) {
    int max_idx = 0;
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
    }
    return max_idx;
}

void argmax_row_f32(
    const float* input,
    int* output,
    int num_rows,
    int row_size
) {
    for (int r = 0; r < num_rows; r++) {
        output[r] = argmax_naive_f32(input + r * row_size, row_size);
    }
}

// =============================================================================
// Variance (for LayerNorm)
// =============================================================================

void reduce_variance_row_f32(
    const float* input,
    const float* mean,
    float* variance,
    int num_rows,
    int row_size,
    SIMDType simd_type
) {
    for (int r = 0; r < num_rows; r++) {
        const float* row = input + r * row_size;
        float row_mean = mean[r];
        float sum_sq = 0.0f;
        
#ifdef __AVX2__
        if (simd_type == SIMDType::AVX2 || simd_type == SIMDType::AVX512) {
            __m256 mean_vec = _mm256_set1_ps(row_mean);
            __m256 sum_sq_vec = _mm256_setzero_ps();
            int i = 0;
            
            for (; i <= row_size - 8; i += 8) {
                __m256 x = _mm256_loadu_ps(&row[i]);
                __m256 diff = _mm256_sub_ps(x, mean_vec);
                sum_sq_vec = _mm256_fmadd_ps(diff, diff, sum_sq_vec);
            }
            
            __m128 lo = _mm256_castps256_ps128(sum_sq_vec);
            __m128 hi = _mm256_extractf128_ps(sum_sq_vec, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            sum_sq = _mm_cvtss_f32(lo);
            
            for (; i < row_size; i++) {
                float diff = row[i] - row_mean;
                sum_sq += diff * diff;
            }
            
            variance[r] = sum_sq / row_size;
            continue;
        }
#endif
        
        for (int i = 0; i < row_size; i++) {
            float diff = row[i] - row_mean;
            sum_sq += diff * diff;
        }
        variance[r] = sum_sq / row_size;
    }
}

// =============================================================================
// L2 Norm
// =============================================================================

void reduce_l2_norm_row_f32(
    const float* input,
    float* output,
    int num_rows,
    int row_size,
    SIMDType simd_type
) {
    for (int r = 0; r < num_rows; r++) {
        const float* row = input + r * row_size;
        float sum_sq = 0.0f;
        
#ifdef __AVX2__
        if (simd_type == SIMDType::AVX2 || simd_type == SIMDType::AVX512) {
            __m256 sum_sq_vec = _mm256_setzero_ps();
            int i = 0;
            
            for (; i <= row_size - 8; i += 8) {
                __m256 x = _mm256_loadu_ps(&row[i]);
                sum_sq_vec = _mm256_fmadd_ps(x, x, sum_sq_vec);
            }
            
            __m128 lo = _mm256_castps256_ps128(sum_sq_vec);
            __m128 hi = _mm256_extractf128_ps(sum_sq_vec, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            sum_sq = _mm_cvtss_f32(lo);
            
            for (; i < row_size; i++) {
                sum_sq += row[i] * row[i];
            }
            
            output[r] = std::sqrt(sum_sq);
            continue;
        }
#endif
        
        for (int i = 0; i < row_size; i++) {
            sum_sq += row[i] * row[i];
        }
        output[r] = std::sqrt(sum_sq);
    }
}

// =============================================================================
// Dispatchers
// =============================================================================

float reduce_sum_f32(const float* input, int size, SIMDType simd_type) {
#ifdef __AVX2__
    if (simd_type == SIMDType::AVX2 || simd_type == SIMDType::AVX512) {
        return reduce_sum_avx2_f32(input, size);
    }
#endif
#ifdef __aarch64__
    if (simd_type == SIMDType::NEON) {
        return reduce_sum_neon_f32(input, size);
    }
#endif
    return reduce_sum_naive_f32(input, size);
}

float reduce_max_f32(const float* input, int size, SIMDType simd_type) {
#ifdef __AVX2__
    if (simd_type == SIMDType::AVX2 || simd_type == SIMDType::AVX512) {
        return reduce_max_avx2_f32(input, size);
    }
#endif
    return reduce_max_naive_f32(input, size);
}

}  // namespace cpu
}  // namespace kernel
}  // namespace yirage
