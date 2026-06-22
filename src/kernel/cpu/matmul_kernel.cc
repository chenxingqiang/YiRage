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
 * CPU Matrix Multiplication Kernels
 * 
 * Optimized for x86 (AVX/AVX2/AVX-512) and ARM (NEON)
 */

#include "kernel/cpu/cpu_kernel_config.h"

#include <cstring>
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
// Constants
// =============================================================================

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;

// =============================================================================
// Naive GEMM (reference implementation)
// =============================================================================

void gemm_naive_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// =============================================================================
// AVX2 GEMM (8-wide SIMD)
// =============================================================================

#ifdef __AVX2__

void gemm_avx2_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    __m256 alpha_vec = _mm256_set1_ps(alpha);
    __m256 beta_vec = _mm256_set1_ps(beta);
    
    // Tiled GEMM with AVX2
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                int i_end = std::min(i0 + TILE_M, M);
                int j_end = std::min(j0 + TILE_N, N);
                int k_end = std::min(k0 + TILE_K, K);
                
                for (int i = i0; i < i_end; i++) {
                    for (int j = j0; j < j_end; j += 8) {
                        __m256 c_vec;
                        if (k0 == 0) {
                            c_vec = _mm256_mul_ps(beta_vec, 
                                _mm256_loadu_ps(&C[i * N + j]));
                        } else {
                            c_vec = _mm256_loadu_ps(&C[i * N + j]);
                        }
                        
                        for (int k = k0; k < k_end; k++) {
                            __m256 a_vec = _mm256_set1_ps(A[i * K + k]);
                            __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                            c_vec = _mm256_fmadd_ps(
                                _mm256_mul_ps(alpha_vec, a_vec), b_vec, c_vec);
                        }
                        
                        _mm256_storeu_ps(&C[i * N + j], c_vec);
                    }
                    
                    // Handle remainder
                    for (int j = (j_end / 8) * 8; j < j_end; j++) {
                        float sum = (k0 == 0) ? beta * C[i * N + j] : C[i * N + j];
                        for (int k = k0; k < k_end; k++) {
                            sum += alpha * A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

#endif  // __AVX2__

// =============================================================================
// AVX-512 GEMM (16-wide SIMD)
// =============================================================================

#ifdef __AVX512F__

void gemm_avx512_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    __m512 alpha_vec = _mm512_set1_ps(alpha);
    __m512 beta_vec = _mm512_set1_ps(beta);
    
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                int i_end = std::min(i0 + TILE_M, M);
                int j_end = std::min(j0 + TILE_N, N);
                int k_end = std::min(k0 + TILE_K, K);
                
                for (int i = i0; i < i_end; i++) {
                    for (int j = j0; j < j_end; j += 16) {
                        __m512 c_vec;
                        if (k0 == 0) {
                            c_vec = _mm512_mul_ps(beta_vec,
                                _mm512_loadu_ps(&C[i * N + j]));
                        } else {
                            c_vec = _mm512_loadu_ps(&C[i * N + j]);
                        }
                        
                        for (int k = k0; k < k_end; k++) {
                            __m512 a_vec = _mm512_set1_ps(A[i * K + k]);
                            __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                            c_vec = _mm512_fmadd_ps(
                                _mm512_mul_ps(alpha_vec, a_vec), b_vec, c_vec);
                        }
                        
                        _mm512_storeu_ps(&C[i * N + j], c_vec);
                    }
                }
            }
        }
    }
}

#endif  // __AVX512F__

// =============================================================================
// ARM NEON GEMM (4-wide SIMD)
// =============================================================================

#ifdef __aarch64__

void gemm_neon_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta
) {
    float32x4_t alpha_vec = vdupq_n_f32(alpha);
    float32x4_t beta_vec = vdupq_n_f32(beta);
    
    for (int i0 = 0; i0 < M; i0 += TILE_M) {
        for (int j0 = 0; j0 < N; j0 += TILE_N) {
            for (int k0 = 0; k0 < K; k0 += TILE_K) {
                int i_end = std::min(i0 + TILE_M, M);
                int j_end = std::min(j0 + TILE_N, N);
                int k_end = std::min(k0 + TILE_K, K);
                
                for (int i = i0; i < i_end; i++) {
                    for (int j = j0; j < j_end; j += 4) {
                        float32x4_t c_vec;
                        if (k0 == 0) {
                            c_vec = vmulq_f32(beta_vec, vld1q_f32(&C[i * N + j]));
                        } else {
                            c_vec = vld1q_f32(&C[i * N + j]);
                        }
                        
                        for (int k = k0; k < k_end; k++) {
                            float32x4_t a_vec = vdupq_n_f32(A[i * K + k]);
                            float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                            c_vec = vfmaq_f32(c_vec, 
                                vmulq_f32(alpha_vec, a_vec), b_vec);
                        }
                        
                        vst1q_f32(&C[i * N + j], c_vec);
                    }
                    
                    // Handle remainder
                    for (int j = (j_end / 4) * 4; j < j_end; j++) {
                        float sum = (k0 == 0) ? beta * C[i * N + j] : C[i * N + j];
                        for (int k = k0; k < k_end; k++) {
                            sum += alpha * A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

#endif  // __aarch64__

// =============================================================================
// Multi-threaded GEMM
// =============================================================================

void gemm_parallel_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    int num_threads
) {
    if (num_threads <= 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    std::vector<std::thread> threads;
    int rows_per_thread = (M + num_threads - 1) / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        int start_row = t * rows_per_thread;
        int end_row = std::min(start_row + rows_per_thread, M);
        
        if (start_row >= M) break;
        
        threads.emplace_back([=]() {
            for (int i = start_row; i < end_row; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; k++) {
                        sum += A[i * K + k] * B[k * N + j];
                    }
                    C[i * N + j] = alpha * sum + beta * C[i * N + j];
                }
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

// =============================================================================
// Dispatcher
// =============================================================================

void gemm_f32(
    const float* A, const float* B, float* C,
    int M, int N, int K,
    float alpha, float beta,
    SIMDType simd_type
) {
    if (simd_type == SIMDType::AUTO) {
        gemm_f32(A, B, C, M, N, K, alpha, beta,
                 CPUOptimizer::detect_simd_support());
        return;
    }
    switch (simd_type) {
#ifdef __AVX512F__
        case SIMDType::AVX512:
            gemm_avx512_f32(A, B, C, M, N, K, alpha, beta);
            break;
#endif
#ifdef __AVX2__
        case SIMDType::AVX2:
        case SIMDType::AVX:
            gemm_avx2_f32(A, B, C, M, N, K, alpha, beta);
            break;
#endif
#ifdef __aarch64__
        case SIMDType::NEON:
            gemm_neon_f32(A, B, C, M, N, K, alpha, beta);
            break;
#endif
        default:
            gemm_naive_f32(A, B, C, M, N, K, alpha, beta);
            break;
    }
}

}  // namespace cpu
}  // namespace kernel
}  // namespace yirage
