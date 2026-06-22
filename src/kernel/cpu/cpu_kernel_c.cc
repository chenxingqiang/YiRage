/* Copyright 2025 Chen Xingqiang (YiRage Project)
 * SPDX-License-Identifier: Apache-2.0 */

#include "kernel/cpu/cpu_kernel_c.h"

#ifdef YIRAGE_BACKEND_CPU_ENABLED

#include "kernel/cpu/cpu_kernel.h"
#include "kernel/cpu/cpu_kernel_config.h"

#include <algorithm>
#include <thread>
#include <vector>

namespace {

using yirage::kernel::cpu::CPUOptimizer;
using yirage::kernel::cpu::SIMDType;
using yirage::kernel::cpu::gemm_f32;
using yirage::kernel::cpu::rms_matmul_f32;

void gemm_parallel_simd_f32(const float *A, const float *B, float *C, int M,
                            int N, int K, float alpha, float beta,
                            int num_threads, SIMDType simd) {
  if (num_threads <= 0) {
    num_threads = static_cast<int>(std::thread::hardware_concurrency());
  }
  if (num_threads <= 1) {
    gemm_f32(A, B, C, M, N, K, alpha, beta, simd);
    return;
  }

  std::vector<std::thread> threads;
  int rows_per_thread = (M + num_threads - 1) / num_threads;

  for (int t = 0; t < num_threads; t++) {
    int start_row = t * rows_per_thread;
    int end_row = std::min(start_row + rows_per_thread, M);
    if (start_row >= M) {
      break;
    }
    int sub_m = end_row - start_row;
    const float *a_sub = A + start_row * K;
    float *c_sub = C + start_row * N;

    threads.emplace_back([=]() {
      gemm_f32(a_sub, B, c_sub, sub_m, N, K, alpha, beta, simd);
    });
  }

  for (auto &th : threads) {
    th.join();
  }
}

} // namespace

extern "C" int yirage_cpu_gemm_f32(const float *A, const float *B, float *C,
                                   int M, int N, int K, int num_threads) {
  if (!A || !B || !C || M <= 0 || N <= 0 || K <= 0) {
    return -1;
  }
  SIMDType simd = CPUOptimizer::detect_simd_support();
  gemm_parallel_simd_f32(A, B, C, M, N, K, 1.0f, 0.0f, num_threads, simd);
  return 0;
}

extern "C" int yirage_cpu_rms_matmul_f32(const float *X, const float *W,
                                         float *Y, int M, int N, int K,
                                         float epsilon, int num_threads) {
  if (!X || !W || !Y || M <= 0 || N <= 0 || K <= 0) {
    return -1;
  }
  rms_matmul_f32(X, W, Y, M, N, K, epsilon, num_threads);
  return 0;
}

#else

extern "C" int yirage_cpu_gemm_f32(const float *A, const float *B, float *C,
                                   int M, int N, int K, int num_threads) {
  (void)A;
  (void)B;
  (void)C;
  (void)M;
  (void)N;
  (void)K;
  (void)num_threads;
  return -1;
}

extern "C" int yirage_cpu_rms_matmul_f32(const float *X, const float *W,
                                         float *Y, int M, int N, int K,
                                         float epsilon, int num_threads) {
  (void)X;
  (void)W;
  (void)Y;
  (void)M;
  (void)N;
  (void)K;
  (void)epsilon;
  (void)num_threads;
  return -1;
}

#endif
