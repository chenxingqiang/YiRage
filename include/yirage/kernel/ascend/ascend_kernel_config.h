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
 */

#pragma once

#include "yirage/kernel/common/kernel_interface.h"

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED

namespace yirage {
namespace kernel {
namespace ascend {

/**
 * @brief Ascend-specific kernel configuration
 * 
 * Huawei Ascend NPU architecture:
 * - AI Cores: Tensor processing units (similar to CUDA cores)
 * - Block size: Configurable number of AI Cores per block
 * - L1 Buffer: Local memory per AI Core (256KB-512KB)
 * - Cube operations: Matrix multiplication acceleration
 * - Vector operations: Element-wise operations
 */
struct AscendKernelConfig : public KernelConfig {
  // AI Core block configuration
  int ai_cores_per_block = 8;     // Number of AI Cores per block
  int blocks_per_grid_x = 1;
  int blocks_per_grid_y = 1;
  
  // Memory configuration
  size_t l1_buffer_size = 256 * 1024;  // 256 KB (910), 512 KB (910B)
  
  // Tile configuration for Cube operations
  int tile_m = 16;
  int tile_n = 16;
  int tile_k = 16;
  
  // Device type: 0=910, 1=910B, 2=310P
  int device_type = 0;
  
  // Optimization flags
  bool use_cube_ops = true;      // Use Cube for matmul
  bool use_vector_ops = true;    // Use Vector for element-wise
  bool enable_fusion = true;     // Enable operator fusion
  
  AscendKernelConfig() { backend_type = type::BT_ASCEND; }
  
  // Get total number of AI Cores
  int get_total_ai_cores() const {
    return ai_cores_per_block * blocks_per_grid_x * blocks_per_grid_y;
  }
};

/**
 * @brief Ascend kernel optimizer
 */
class AscendOptimizer {
public:
  /**
   * @brief Detect Ascend device type
   * @return Device type (0=910, 1=910B, 2=310P)
   */
  static int detect_device_type();
  
  /**
   * @brief Get AI Core count
   * @return Number of AI Cores
   */
  static int get_ai_core_count();
  
  /**
   * @brief Compute optimal block size
   * @param problem_size Problem size
   * @param device_type Ascend device type
   * @return Optimal AI cores per block
   */
  static int compute_optimal_block_size(size_t problem_size, int device_type);
  
  /**
   * @brief Compute optimal tile sizes for Cube operations
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param device_type Ascend device type
   * @param config Output configuration
   */
  static void compute_optimal_tiles(int m, int n, int k, int device_type,
                                    AscendKernelConfig &config);
};

} // namespace ascend
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED

