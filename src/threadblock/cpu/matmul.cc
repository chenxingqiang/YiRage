/* Copyright 2025 YiRage Team */

#include "threadblock/graph.h"
#include "threadblock/matmul.h"
#include "threadblock/cpu/matmul.h"

#ifdef YIRAGE_USE_CPU

#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

namespace yirage {
namespace threadblock {
namespace cpu {

// AVX2 optimized GEMM micro-kernel
#if defined(__AVX2__)
void gemm_avx2_6x16(const float* A, const float* B, float* C,
                    int K, int lda, int ldb, int ldc) {
    __m256 c00 = _mm256_setzero_ps();
    __m256 c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps();
    __m256 c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps();
    __m256 c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps();
    __m256 c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps();
    __m256 c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps();
    __m256 c51 = _mm256_setzero_ps();

    for (int k = 0; k < K; k++) {
        __m256 b0 = _mm256_loadu_ps(B + k * ldb);
        __m256 b1 = _mm256_loadu_ps(B + k * ldb + 8);

        __m256 a0 = _mm256_broadcast_ss(A + 0 * lda + k);
        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);

        __m256 a1 = _mm256_broadcast_ss(A + 1 * lda + k);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);

        __m256 a2 = _mm256_broadcast_ss(A + 2 * lda + k);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);

        __m256 a3 = _mm256_broadcast_ss(A + 3 * lda + k);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);

        __m256 a4 = _mm256_broadcast_ss(A + 4 * lda + k);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);

        __m256 a5 = _mm256_broadcast_ss(A + 5 * lda + k);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    _mm256_storeu_ps(C + 0 * ldc, c00);
    _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc, c10);
    _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc, c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc, c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc, c40);
    _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc, c50);
    _mm256_storeu_ps(C + 5 * ldc + 8, c51);
}
#endif

// NEON optimized GEMM micro-kernel
#if defined(__aarch64__) || defined(_M_ARM64)
void gemm_neon_8x8(const float* A, const float* B, float* C,
                   int K, int lda, int ldb, int ldc) {
    float32x4_t c00 = vdupq_n_f32(0);
    float32x4_t c01 = vdupq_n_f32(0);
    float32x4_t c10 = vdupq_n_f32(0);
    float32x4_t c11 = vdupq_n_f32(0);
    // ... more accumulators ...

    for (int k = 0; k < K; k++) {
        float32x4_t b0 = vld1q_f32(B + k * ldb);
        float32x4_t b1 = vld1q_f32(B + k * ldb + 4);

        float32x4_t a0 = vdupq_n_f32(A[0 * lda + k]);
        c00 = vfmaq_f32(c00, a0, b0);
        c01 = vfmaq_f32(c01, a0, b1);

        float32x4_t a1 = vdupq_n_f32(A[1 * lda + k]);
        c10 = vfmaq_f32(c10, a1, b0);
        c11 = vfmaq_f32(c11, a1, b1);
    }

    vst1q_f32(C + 0 * ldc, c00);
    vst1q_f32(C + 0 * ldc + 4, c01);
    vst1q_f32(C + 1 * ldc, c10);
    vst1q_f32(C + 1 * ldc + 4, c11);
}
#endif

// Naive fallback
void gemm_naive(const float* A, const float* B, float* C,
                int M, int N, int K) {
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

}  // namespace cpu
}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_USE_CPU
