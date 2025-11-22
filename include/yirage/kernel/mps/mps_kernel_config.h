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

#ifdef YIRAGE_BACKEND_MPS_ENABLED

namespace yirage {
namespace kernel {
namespace mps {

/**
 * @brief Memory access pattern for MPS kernels
 */
enum class MemoryPattern {
  COALESCED,  // Coalesced access
  STRIDED,    // Strided access
  TILED       // Tiled access pattern
};

/**
 * @brief MPS-specific kernel configuration
 */
struct MPSKernelConfig : public KernelConfig {
  // Threadgroup configuration (similar to CUDA blocks)
  int threads_per_threadgroup = 256;
  int simd_width = 32; // Apple GPU SIMD width

  // Threadgroup memory (similar to CUDA shared memory)
  size_t threadgroup_memory_size = 32 * 1024; // 32 KB typical
  
  // Memory access pattern
  MemoryPattern access_pattern = MemoryPattern::COALESCED;

  // Apple GPU family (7 for M1, 8 for M2, 9 for M3)
  int gpu_family = 7;

  // Tile configuration for matrix operations
  int tile_m = 32;
  int tile_n = 32;
  int tile_k = 32;

  // Metal shader configuration
  bool use_fast_math = true;
  int thread_execution_width = 32; // Threads executed together

  MPSKernelConfig() { backend_type = type::BT_MPS; }

  // Get number of SIMD groups per threadgroup
  int get_num_simd_groups() const {
    return (threads_per_threadgroup + simd_width - 1) / simd_width;
  }
};

/**
 * @brief MPS kernel optimizer
 */
class MPSOptimizer {
public:
  /**
   * @brief Detect Apple GPU family
   * @return GPU family number (7, 8, 9...)
   */
  static int detect_gpu_family();

  /**
   * @brief Get GPU core count
   * @return Number of GPU cores
   */
  static int get_gpu_core_count();

  /**
   * @brief Compute optimal threadgroup size
   * @param problem_size Total problem size
   * @param gpu_family Apple GPU family
   * @return Optimal threadgroup size
   */
  static int compute_optimal_threadgroup_size(size_t problem_size,
                                              int gpu_family);

  /**
   * @brief Compute optimal tile sizes
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param gpu_family Apple GPU family
   * @param config Output configuration
   */
  static void compute_optimal_tiles(int m, int n, int k, int gpu_family,
                                    MPSKernelConfig &config);

  /**
   * @brief Select optimal memory access pattern
   * @param data_size Size of data
   * @param stride Access stride
   * @return Optimal memory pattern
   */
  static MemoryPattern select_memory_pattern(size_t data_size, int stride);

  /**
   * @brief Estimate memory bandwidth utilization
   * @param config Kernel configuration
   * @param bytes_accessed Total bytes accessed
   * @param execution_time_ms Execution time
   * @return Memory bandwidth in GB/s
   */
  static float estimate_memory_bandwidth(MPSKernelConfig const &config,
                                        size_t bytes_accessed,
                                        float execution_time_ms);

  /**
   * @brief Optimize configuration for Apple Silicon
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param config Configuration to optimize
   */
  static void optimize_for_apple_silicon(int m, int n, int k,
                                         MPSKernelConfig &config);
};

/**
 * @brief Convert memory pattern to string
 */
inline std::string memory_pattern_to_string(MemoryPattern pattern) {
  switch (pattern) {
  case MemoryPattern::COALESCED:
    return "coalesced";
  case MemoryPattern::STRIDED:
    return "strided";
  case MemoryPattern::TILED:
    return "tiled";
  default:
    return "unknown";
  }
}

} // namespace mps
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_MPS_ENABLED





