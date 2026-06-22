/* Copyright 2025 Chen Xingqiang (YiRage Project)
 * SPDX-License-Identifier: Apache-2.0
 *
 * C ABI for Python / Cython CPU GEMM dispatch.
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * GEMM C = A @ B with architecture-detected SIMD (AVX2/AVX-512/NEON).
 *
 * @param A  Row-major [M x K]
 * @param B  Row-major [K x N]
 * @param C  Row-major [M x N] output (beta=0)
 * @param num_threads  <=1: single-thread SIMD; >1: parallel row blocks
 * @return 0 on success, -1 if CPU backend disabled at build time
 */
int yirage_cpu_gemm_f32(const float *A, const float *B, float *C, int M, int N,
                        int K, int num_threads);

/**
 * Fused RMS norm + GEMM: Y = rms_norm(X) @ W (fp32 compute).
 *
 * @param X  Row-major [M x K]
 * @param W  Row-major [K x N]
 * @param Y  Row-major [M x N] output
 * @param epsilon  RMS stabilizer (typically 1e-6)
 * @param num_threads  OpenMP thread count; <=0 uses runtime default
 * @return 0 on success, -1 on invalid args or CPU backend disabled
 */
int yirage_cpu_rms_matmul_f32(const float *X, const float *W, float *Y, int M,
                              int N, int K, float epsilon, int num_threads);

#ifdef __cplusplus
}
#endif
