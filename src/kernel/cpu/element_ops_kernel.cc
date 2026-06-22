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
 * CPU Element-wise Operations
 */

#include "kernel/cpu/cpu_kernel_config.h"

#include <cmath>
#include <algorithm>

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
// Binary Operations - Naive
// =============================================================================

void add_naive_f32(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

void mul_naive_f32(const float* a, const float* b, float* c, int size) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

void silu_mul_naive_f32(const float* gate, const float* up, float* out, int size) {
    for (int i = 0; i < size; i++) {
        float g = gate[i];
        float silu = g / (1.0f + std::exp(-g));
        out[i] = silu * up[i];
    }
}

// =============================================================================
// Activation Functions - Naive
// =============================================================================

void relu_naive_f32(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

void gelu_naive_f32(const float* input, float* output, int size) {
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;
    
    for (int i = 0; i < size; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x3);
        output[i] = 0.5f * x * (1.0f + std::tanh(inner));
    }
}

void silu_naive_f32(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

void sigmoid_naive_f32(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = 1.0f / (1.0f + std::exp(-input[i]));
    }
}

void tanh_naive_f32(const float* input, float* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = std::tanh(input[i]);
    }
}

// =============================================================================
// AVX2 Binary Operations
// =============================================================================

#ifdef __AVX2__

void add_avx2_f32(const float* a, const float* b, float* c, int size) {
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_add_ps(va, vb));
    }
    for (; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

void mul_avx2_f32(const float* a, const float* b, float* c, int size) {
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_mul_ps(va, vb));
    }
    for (; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

void relu_avx2_f32(const float* input, float* output, int size) {
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        _mm256_storeu_ps(&output[i], _mm256_max_ps(x, zero));
    }
    for (; i < size; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

// Fast sigmoid approximation for AVX2
inline __m256 sigmoid_avx2(__m256 x) {
    // Clamp to avoid overflow
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    
    // Polynomial approximation
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    
    // exp(-x) approximation
    __m256 log2e = _mm256_set1_ps(1.44269504f);
    __m256 t = _mm256_mul_ps(neg_x, log2e);
    __m256 ti = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT);
    __m256 tf = _mm256_sub_ps(t, ti);
    
    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(0.693147f);
    __m256 c2 = _mm256_set1_ps(0.240227f);
    
    __m256 p = _mm256_fmadd_ps(c2, tf, c1);
    p = _mm256_fmadd_ps(p, tf, c0);
    
    __m256i ti_int = _mm256_cvtps_epi32(ti);
    ti_int = _mm256_add_epi32(ti_int, _mm256_set1_epi32(127));
    ti_int = _mm256_slli_epi32(ti_int, 23);
    __m256 exp_neg_x = _mm256_mul_ps(p, _mm256_castsi256_ps(ti_int));
    
    return _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_x));
}

void silu_avx2_f32(const float* input, float* output, int size) {
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 sig = sigmoid_avx2(x);
        _mm256_storeu_ps(&output[i], _mm256_mul_ps(x, sig));
    }
    for (; i < size; i++) {
        float x = input[i];
        output[i] = x / (1.0f + std::exp(-x));
    }
}

void silu_mul_avx2_f32(const float* gate, const float* up, float* out, int size) {
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 g = _mm256_loadu_ps(&gate[i]);
        __m256 u = _mm256_loadu_ps(&up[i]);
        __m256 sig = sigmoid_avx2(g);
        __m256 silu = _mm256_mul_ps(g, sig);
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(silu, u));
    }
    for (; i < size; i++) {
        float g = gate[i];
        float silu = g / (1.0f + std::exp(-g));
        out[i] = silu * up[i];
    }
}

#endif  // __AVX2__

// =============================================================================
// AVX-512 Operations
// =============================================================================

#ifdef __AVX512F__

void add_avx512_f32(const float* a, const float* b, float* c, int size) {
    int i = 0;
    for (; i <= size - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&c[i], _mm512_add_ps(va, vb));
    }
    for (; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

void mul_avx512_f32(const float* a, const float* b, float* c, int size) {
    int i = 0;
    for (; i <= size - 16; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        _mm512_storeu_ps(&c[i], _mm512_mul_ps(va, vb));
    }
    for (; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

void relu_avx512_f32(const float* input, float* output, int size) {
    __m512 zero = _mm512_setzero_ps();
    int i = 0;
    for (; i <= size - 16; i += 16) {
        __m512 x = _mm512_loadu_ps(&input[i]);
        _mm512_storeu_ps(&output[i], _mm512_max_ps(x, zero));
    }
    for (; i < size; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

#endif  // __AVX512F__

// =============================================================================
// ARM NEON Operations
// =============================================================================

#ifdef __aarch64__

void add_neon_f32(const float* a, const float* b, float* c, int size) {
    int i = 0;
    for (; i <= size - 4; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&c[i], vaddq_f32(va, vb));
    }
    for (; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

void mul_neon_f32(const float* a, const float* b, float* c, int size) {
    int i = 0;
    for (; i <= size - 4; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        vst1q_f32(&c[i], vmulq_f32(va, vb));
    }
    for (; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

void relu_neon_f32(const float* input, float* output, int size) {
    float32x4_t zero = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= size - 4; i += 4) {
        float32x4_t x = vld1q_f32(&input[i]);
        vst1q_f32(&output[i], vmaxq_f32(x, zero));
    }
    for (; i < size; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

#endif  // __aarch64__

// =============================================================================
// Dispatchers
// =============================================================================

void add_f32(const float* a, const float* b, float* c, int size, SIMDType simd_type) {
    switch (simd_type) {
#ifdef __AVX512F__
        case SIMDType::AVX512:
            add_avx512_f32(a, b, c, size);
            break;
#endif
#ifdef __AVX2__
        case SIMDType::AVX2:
        case SIMDType::AVX:
            add_avx2_f32(a, b, c, size);
            break;
#endif
#ifdef __aarch64__
        case SIMDType::NEON:
            add_neon_f32(a, b, c, size);
            break;
#endif
        default:
            add_naive_f32(a, b, c, size);
            break;
    }
}

void mul_f32(const float* a, const float* b, float* c, int size, SIMDType simd_type) {
    switch (simd_type) {
#ifdef __AVX512F__
        case SIMDType::AVX512:
            mul_avx512_f32(a, b, c, size);
            break;
#endif
#ifdef __AVX2__
        case SIMDType::AVX2:
        case SIMDType::AVX:
            mul_avx2_f32(a, b, c, size);
            break;
#endif
#ifdef __aarch64__
        case SIMDType::NEON:
            mul_neon_f32(a, b, c, size);
            break;
#endif
        default:
            mul_naive_f32(a, b, c, size);
            break;
    }
}

void relu_f32(const float* input, float* output, int size, SIMDType simd_type) {
    switch (simd_type) {
#ifdef __AVX512F__
        case SIMDType::AVX512:
            relu_avx512_f32(input, output, size);
            break;
#endif
#ifdef __AVX2__
        case SIMDType::AVX2:
        case SIMDType::AVX:
            relu_avx2_f32(input, output, size);
            break;
#endif
#ifdef __aarch64__
        case SIMDType::NEON:
            relu_neon_f32(input, output, size);
            break;
#endif
        default:
            relu_naive_f32(input, output, size);
            break;
    }
}

void gelu_f32(const float* input, float* output, int size, SIMDType simd_type) {
    // GELU uses tanh which is complex to vectorize efficiently
    // Fall back to naive for now
    gelu_naive_f32(input, output, size);
}

void silu_f32(const float* input, float* output, int size, SIMDType simd_type) {
#ifdef __AVX2__
    if (simd_type == SIMDType::AVX2 || simd_type == SIMDType::AVX512) {
        silu_avx2_f32(input, output, size);
        return;
    }
#endif
    silu_naive_f32(input, output, size);
}

void silu_mul_f32(const float* gate, const float* up, float* out, int size, SIMDType simd_type) {
#ifdef __AVX2__
    if (simd_type == SIMDType::AVX2 || simd_type == SIMDType::AVX512) {
        silu_mul_avx2_f32(gate, up, out, size);
        return;
    }
#endif
    silu_mul_naive_f32(gate, up, out, size);
}

}  // namespace cpu
}  // namespace kernel
}  // namespace yirage
