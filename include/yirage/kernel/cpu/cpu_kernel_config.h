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
 * This file is part of YiRage (Yi Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */


#pragma once

#include "yirage/kernel/common/kernel_interface.h"

#ifdef YIRAGE_BACKEND_CPU_ENABLED

namespace yirage {
namespace kernel {
namespace cpu {

/**
 * @brief SIMD instruction set types
 */
enum class SIMDType {
  NONE,
  SSE,
  SSE2,
  SSE3,
  SSE4_1,
  SSE4_2,
  AVX,
  AVX2,
  AVX512
};

/**
 * @brief CPU-specific kernel configuration
 */
struct CPUKernelConfig : public KernelConfig {
  // OpenMP configuration
  int num_threads = 0; // 0 = auto-detect
  bool use_openmp = true;
  bool dynamic_scheduling = false;

  // SIMD configuration
  SIMDType simd_type = SIMDType::AVX2;
  int vector_width = 8; // Number of elements per SIMD register

  // Cache configuration
  size_t l1_cache_size = 32 * 1024;        // 32 KB
  size_t l2_cache_size = 256 * 1024;       // 256 KB
  size_t l3_cache_size = 8 * 1024 * 1024;  // 8 MB
  int cache_line_size = 64;                // bytes

  // Blocking/tiling configuration
  int tile_m = 64;
  int tile_n = 64;
  int tile_k = 64;
  int micro_tile_m = 8;
  int micro_tile_n = 8;

  // Memory alignment
  int alignment = 64; // bytes (for AVX-512)

  // Prefetching
  bool use_prefetch = true;
  int prefetch_distance = 16; // cache lines ahead

  // Loop unrolling
  int unroll_factor = 4;

  CPUKernelConfig() { backend_type = type::BT_CPU; }

  // Get vector width in bytes
  int get_vector_bytes() const {
    switch (simd_type) {
    case SIMDType::SSE:
    case SIMDType::SSE2:
    case SIMDType::SSE3:
    case SIMDType::SSE4_1:
    case SIMDType::SSE4_2:
      return 16; // 128-bit
    case SIMDType::AVX:
    case SIMDType::AVX2:
      return 32; // 256-bit
    case SIMDType::AVX512:
      return 64; // 512-bit
    default:
      return 4; // scalar
    }
  }
};

/**
 * @brief CPU kernel optimizer
 */
class CPUOptimizer {
public:
  /**
   * @brief Detect available SIMD instruction set
   * @return Highest supported SIMD type
   */
  static SIMDType detect_simd_support();

  /**
   * @brief Get CPU feature string
   * @return String describing CPU features
   */
  static std::string get_cpu_features();

  /**
   * @brief Compute optimal tile sizes based on cache
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param element_size Size of each element in bytes
   * @param config Output configuration
   */
  static void compute_optimal_tiles(int m, int n, int k, size_t element_size,
                                   CPUKernelConfig &config);

  /**
   * @brief Compute optimal number of threads
   * @param problem_size Total problem size
   * @param num_cores Number of CPU cores
   * @param memory_bound Whether the kernel is memory-bound
   * @return Optimal number of threads
   */
  static int compute_optimal_threads(size_t problem_size, int num_cores,
                                     bool memory_bound = false);

  /**
   * @brief Estimate cache efficiency
   * @param config Kernel configuration
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param element_size Size of each element
   * @return Estimated cache hit rate (0.0 - 1.0)
   */
  static float estimate_cache_efficiency(CPUKernelConfig const &config, int m,
                                        int n, int k, size_t element_size);

  /**
   * @brief Estimate vectorization efficiency
   * @param config Kernel configuration
   * @param data_size Size of data to process
   * @return Estimated vectorization efficiency (0.0 - 1.0)
   */
  static float estimate_vectorization_efficiency(
      CPUKernelConfig const &config, size_t data_size);

  /**
   * @brief Optimize configuration for specific CPU
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param config Configuration to optimize
   */
  static void optimize_for_cpu(int m, int n, int k, CPUKernelConfig &config);

  /**
   * @brief Calculate optimal unroll factor
   * @param loop_size Loop iteration count
   * @param simd_type SIMD instruction set
   * @return Optimal unroll factor
   */
  static int compute_unroll_factor(int loop_size, SIMDType simd_type);
};

/**
 * @brief Convert SIMD type to string
 */
inline std::string simd_type_to_string(SIMDType type) {
  switch (type) {
  case SIMDType::NONE:
    return "none";
  case SIMDType::SSE:
    return "sse";
  case SIMDType::SSE2:
    return "sse2";
  case SIMDType::SSE3:
    return "sse3";
  case SIMDType::SSE4_1:
    return "sse4.1";
  case SIMDType::SSE4_2:
    return "sse4.2";
  case SIMDType::AVX:
    return "avx";
  case SIMDType::AVX2:
    return "avx2";
  case SIMDType::AVX512:
    return "avx512";
  default:
    return "unknown";
  }
}

} // namespace cpu
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_CPU_ENABLED
