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

#ifdef YIRAGE_BACKEND_CUDA_ENABLED

namespace yirage {
namespace kernel {
namespace cuda {

/**
 * @brief Shared memory layout strategies
 */
enum class SmemLayout {
  ROW_MAJOR,    // Row-major layout
  COLUMN_MAJOR, // Column-major layout
  SWIZZLED      // Swizzled to avoid bank conflicts
};

/**
 * @brief Cache preference settings
 */
enum class CachePreference {
  PREFER_NONE,   // No preference
  PREFER_SHARED, // Prefer shared memory
  PREFER_L1,     // Prefer L1 cache
  PREFER_EQUAL   // Equal preference
};

/**
 * @brief CUDA-specific kernel configuration
 */
struct CUDAKernelConfig : public KernelConfig {
  // Warp configuration
  int warp_size = 32;
  int num_warps = 4;

  // Shared memory configuration
  SmemLayout smem_layout = SmemLayout::SWIZZLED;
  int smem_bank_size = 4; // bytes per bank
  int smem_padding = 8;   // padding to avoid bank conflicts

  // Tensor Core configuration
  bool use_tensor_core = false;
  int mma_m = 16; // m-dimension for mma instruction
  int mma_n = 8;  // n-dimension for mma instruction
  int mma_k = 16; // k-dimension for mma instruction

  // Register usage
  int max_registers_per_thread = 255;
  int min_blocks_per_sm = 1;

  // Cache configuration
  CachePreference cache_preference = CachePreference::PREFER_NONE;
  bool prefer_l2_cache = false;

  // Compute capability
  int compute_capability = 80; // Default to Ampere (8.0)

  // Memory coalescing
  int coalesce_width = 128; // bytes

  // Kernel fusion settings
  bool enable_kernel_fusion = true;
  int max_fusion_depth = 3;

  CUDAKernelConfig() { backend_type = type::BT_CUDA; }

  // Get number of warps
  int get_num_warps() const {
    return (get_total_threads() + warp_size - 1) / warp_size;
  }

  // Get shared memory size per warp
  size_t get_smem_per_warp() const {
    int num_warps = get_num_warps();
    return num_warps > 0 ? shared_memory_size / num_warps : 0;
  }
};

/**
 * @brief CUDA kernel optimizer
 */
class CUDAOptimizer {
public:
  /**
   * @brief Compute optimal warp configuration
   * @param problem_size Total problem size
   * @param compute_capability CUDA compute capability
   * @return Optimal number of warps
   */
  static int compute_optimal_warps(size_t problem_size,
                                  int compute_capability);

  /**
   * @brief Compute optimal shared memory configuration
   * @param data_size Size of data to store in shared memory
   * @param layout Memory layout
   * @param padding Additional padding to add
   * @return Optimal shared memory size in bytes
   */
  static size_t compute_optimal_smem(size_t data_size, SmemLayout layout,
                                    int padding = 8);

  /**
   * @brief Check if memory access pattern has bank conflicts
   * @param layout Memory layout
   * @param stride Access stride
   * @param bank_size Size of each bank
   * @return true if bank conflicts exist
   */
  static bool has_bank_conflict(SmemLayout layout, int stride, int bank_size);

  /**
   * @brief Estimate occupancy for given configuration
   * @param config Kernel configuration
   * @param registers_per_thread Number of registers per thread
   * @return Estimated occupancy (0.0 - 1.0)
   */
  static float estimate_occupancy(CUDAKernelConfig const &config,
                                 int registers_per_thread);

  /**
   * @brief Select optimal Tensor Core configuration
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param compute_capability CUDA compute capability
   * @param config Output configuration
   * @return true if Tensor Cores can be used
   */
  static bool select_tensor_core_config(int m, int n, int k,
                                       int compute_capability,
                                       CUDAKernelConfig &config);

  /**
   * @brief Optimize grid and block dimensions
   * @param problem_m M dimension of problem
   * @param problem_n N dimension of problem
   * @param problem_k K dimension of problem (for matmul)
   * @param compute_capability CUDA compute capability
   * @param config Output configuration
   */
  static void optimize_grid_block_dims(int problem_m, int problem_n,
                                      int problem_k, int compute_capability,
                                      CUDAKernelConfig &config);

  /**
   * @brief Estimate memory bandwidth utilization
   * @param config Kernel configuration
   * @param bytes_accessed Total bytes accessed
   * @param execution_time_ms Execution time in milliseconds
   * @return Memory bandwidth in GB/s
   */
  static float estimate_memory_bandwidth(CUDAKernelConfig const &config,
                                        size_t bytes_accessed,
                                        float execution_time_ms);

  /**
   * @brief Estimate compute throughput
   * @param config Kernel configuration
   * @param num_operations Number of floating point operations
   * @param execution_time_ms Execution time in milliseconds
   * @return Compute throughput in TFLOPS
   */
  static float estimate_compute_throughput(CUDAKernelConfig const &config,
                                          size_t num_operations,
                                          float execution_time_ms);
};

} // namespace cuda
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_CUDA_ENABLED





