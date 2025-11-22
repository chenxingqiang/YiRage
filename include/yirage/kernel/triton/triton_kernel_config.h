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

#ifdef YIRAGE_BACKEND_TRITON_ENABLED

namespace yirage {
namespace kernel {
namespace triton {

/**
 * @brief Triton-specific kernel configuration
 * 
 * Triton is a compiler-based approach, so configuration focuses on
 * compiler hints and block-level parameters.
 */
struct TritonKernelConfig : public KernelConfig {
  // Block size configuration (Triton's BLOCK_SIZE)
  int block_size_m = 128;
  int block_size_n = 128;
  int block_size_k = 32;

  // Number of warps per block
  int num_warps = 4;

  // Number of stages for software pipelining
  int num_stages = 3;

  // Split-K configuration for large reductions
  bool use_split_k = false;
  int split_k_factor = 1;

  // Auto-tuning configuration
  bool enable_auto_tune = true;
  int num_auto_tune_trials = 100;

  // Optimization flags
  bool enable_fp32_accumulation = true;
  bool enable_tma = false; // Tensor Memory Accelerator (Hopper+)

  TritonKernelConfig() { backend_type = type::BT_TRITON; }
};

/**
 * @brief Triton kernel optimizer
 * 
 * Provides utilities for configuring Triton kernels
 */
class TritonOptimizer {
public:
  /**
   * @brief Compute optimal block sizes
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param compute_capability CUDA compute capability
   * @param config Output configuration
   */
  static void compute_optimal_blocks(int m, int n, int k,
                                    int compute_capability,
                                    TritonKernelConfig &config);

  /**
   * @brief Select optimal number of warps
   * @param block_size_m M block size
   * @param block_size_n N block size
   * @param compute_capability CUDA compute capability
   * @return Optimal number of warps
   */
  static int select_num_warps(int block_size_m, int block_size_n,
                             int compute_capability);

  /**
   * @brief Select optimal number of stages
   * @param compute_capability CUDA compute capability
   * @return Optimal number of stages
   */
  static int select_num_stages(int compute_capability);

  /**
   * @brief Determine if split-K should be used
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @return true if split-K should be used
   */
  static bool should_use_split_k(int m, int n, int k);
};

} // namespace triton
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_TRITON_ENABLED

