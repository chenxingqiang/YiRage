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


#include "yirage/kernel/triton/triton_kernel_config.h"

#ifdef YIRAGE_BACKEND_TRITON_ENABLED

#include <algorithm>

namespace yirage {
namespace kernel {
namespace triton {

void TritonOptimizer::compute_optimal_blocks(int m, int n, int k,
                                            int compute_capability,
                                            TritonKernelConfig &config) {
  // Triton block sizes typically range from 16 to 256
  // Larger blocks for larger problems

  if (m >= 2048 && n >= 2048) {
    // Large problem
    config.block_size_m = 128;
    config.block_size_n = 256;
    config.block_size_k = 64;
  } else if (m >= 512 && n >= 512) {
    // Medium problem
    config.block_size_m = 64;
    config.block_size_n = 128;
    config.block_size_k = 32;
  } else {
    // Small problem
    config.block_size_m = 32;
    config.block_size_n = 64;
    config.block_size_k = 16;
  }

  // Select number of warps
  config.num_warps = select_num_warps(config.block_size_m,
                                     config.block_size_n,
                                     compute_capability);

  // Select number of stages
  config.num_stages = select_num_stages(compute_capability);

  // Determine if split-K should be used
  config.use_split_k = should_use_split_k(m, n, k);
  if (config.use_split_k) {
    // Determine split factor
    if (k >= 1024) {
      config.split_k_factor = 4;
    } else if (k >= 512) {
      config.split_k_factor = 2;
    }
  }
}

int TritonOptimizer::select_num_warps(int block_size_m, int block_size_n,
                                     int compute_capability) {
  // Triton typically uses 2, 4, or 8 warps
  int block_elements = block_size_m * block_size_n;

  if (block_elements >= 16384) {
    return 8;
  } else if (block_elements >= 8192) {
    return 4;
  } else {
    return 2;
  }
}

int TritonOptimizer::select_num_stages(int compute_capability) {
  // Software pipelining stages
  if (compute_capability >= 80) {
    // Ampere and later benefit from more stages
    return 3;
  } else {
    return 2;
  }
}

bool TritonOptimizer::should_use_split_k(int m, int n, int k) {
  // Split-K is beneficial when:
  // 1. K dimension is large relative to M and N
  // 2. Problem is reduction-heavy
  
  if (k >= 1024 && (k > m || k > n)) {
    return true;
  }

  return false;
}

} // namespace triton
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_TRITON_ENABLED

