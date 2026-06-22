/* Copyright 2025 Chen Xingqiang (YiRage Project)
 * SPDX-License-Identifier: Apache-2.0
 *
 * Fused RMS norm + GEMM for CPU: OpenMP row RMS + cblas SGEMM + row scale.
 */

#include "kernel/cpu/cpu_kernel.h"

#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef YIRAGE_CPU_BLAS_ENABLED
#include <cblas.h>
#if defined(YIRAGE_OPENBLAS_BLAS) || defined(OPENBLAS_CONFIG_H)
extern "C" void openblas_set_num_threads(int num_threads);
#define YIRAGE_HAS_OPENBLAS_SET_THREADS 1
#endif
#endif

namespace yirage {
namespace kernel {
namespace cpu {
namespace {

int resolve_num_threads(int num_threads) {
  if (num_threads > 0) {
    return num_threads;
  }
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

void compute_inv_rms_rows(const float *X, float *inv_rms, int M, int K,
                          float epsilon, int num_threads) {
  const int nt = resolve_num_threads(num_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
  for (int m = 0; m < M; ++m) {
    const float *row = X + static_cast<size_t>(m) * static_cast<size_t>(K);
    double ss = 0.0;
    for (int k = 0; k < K; ++k) {
      const double v = static_cast<double>(row[k]);
      ss += v * v;
    }
    const float mean_sq =
        static_cast<float>(ss / static_cast<double>(K));
    inv_rms[m] = 1.0f / std::sqrt(mean_sq + epsilon);
  }
}

void scale_rows_f32(float *Y, const float *inv_rms, int M, int N,
                    int num_threads) {
  const int nt = resolve_num_threads(num_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt)
#endif
  for (int m = 0; m < M; ++m) {
    const float s = inv_rms[m];
    float *row = Y + static_cast<size_t>(m) * static_cast<size_t>(N);
    for (int n = 0; n < N; ++n) {
      row[n] *= s;
    }
  }
}

void gemm_f32_output(const float *X, const float *W, float *Y, int M, int N,
                     int K, int num_threads) {
  const int nt = resolve_num_threads(num_threads);
#ifdef YIRAGE_CPU_BLAS_ENABLED
#ifdef _OPENMP
  const int prev_omp = omp_get_max_threads();
  omp_set_num_threads(1);
#endif
#ifdef YIRAGE_HAS_OPENBLAS_SET_THREADS
  openblas_set_num_threads(nt);
#endif
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, X, K,
              W, N, 0.0f, Y, N);
#ifdef YIRAGE_HAS_OPENBLAS_SET_THREADS
  openblas_set_num_threads(1);
#endif
#ifdef _OPENMP
  omp_set_num_threads(prev_omp);
#endif
#else
  gemm_parallel_f32(X, W, Y, M, N, K, 1.0f, 0.0f, nt);
#endif
}

} // namespace

void rms_matmul_f32(const float *X, const float *W, float *Y, int M, int N,
                    int K, float epsilon, int num_threads) {
  std::vector<float> inv_rms(static_cast<size_t>(M));
  compute_inv_rms_rows(X, inv_rms.data(), M, K, epsilon, num_threads);
  gemm_f32_output(X, W, Y, M, N, K, num_threads);
  scale_rows_f32(Y, inv_rms.data(), M, N, num_threads);
}

} // namespace cpu
} // namespace kernel
} // namespace yirage
