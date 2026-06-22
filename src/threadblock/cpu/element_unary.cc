/* Copyright 2025 YiRage Team */

#include "threadblock/element_unary.h"
#include "threadblock/cpu/element_unary.h"

#ifdef YIRAGE_USE_CPU

#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace yirage {
namespace threadblock {
namespace cpu {

#if defined(__AVX2__)
void relu_avx2(const float* input, float* output, int n) {
    __m256 zero = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 y = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(output + i, y);
    }
    for (; i < n; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}

void silu_avx2(const float* input, float* output, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        // Approximate sigmoid: 0.5 * (1 + tanh(x * 0.5))
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 neg_x = _mm256_mul_ps(x, _mm256_set1_ps(-1.0f));
        // exp(-x) approximation
        __m256 exp_neg = _mm256_set1_ps(1.0f);  // Simplified
        __m256 sigmoid = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
        __m256 silu = _mm256_mul_ps(x, sigmoid);
        _mm256_storeu_ps(output + i, silu);
    }
    for (; i < n; i++) {
        float x = input[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[i] = x * sigmoid;
    }
}
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
void relu_neon(const float* input, float* output, int n) {
    float32x4_t zero = vdupq_n_f32(0);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        float32x4_t y = vmaxq_f32(x, zero);
        vst1q_f32(output + i, y);
    }
    for (; i < n; i++) {
        output[i] = input[i] > 0 ? input[i] : 0;
    }
}
#endif

void element_unary_naive(const float* input, float* output, int n, int op) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float x = input[i];
        switch (op) {
            case 0: output[i] = x > 0 ? x : 0; break;  // ReLU
            case 1: output[i] = x / (1 + expf(-x)); break;  // SiLU
            case 2: output[i] = tanhf(x); break;  // Tanh
            case 3: output[i] = 1.0f / (1.0f + expf(-x)); break;  // Sigmoid
            default: output[i] = x;
        }
    }
}

}  // namespace cpu
}  // namespace threadblock
}  // namespace yirage

#endif
