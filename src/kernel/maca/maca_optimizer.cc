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
 * This file is part of YiRage (Yi Revolutionary AGile Engine)
 * 
 * MACA Kernel Optimizer
 * 
 * Optimization utilities for MetaX MACA kernels.
 * Provides heuristics for launch configuration and tuning.
 */

#include "yirage/kernel/maca/maca_kernel_config.h"
#include "yirage/vector_types.h"  // For dim3

#ifdef YIRAGE_BACKEND_MACA_ENABLED

#include <algorithm>
#include <cmath>

namespace yirage {
namespace kernel {
namespace maca {

/**
 * @brief MACA-specific optimization parameters
 */
struct MACAOptParams {
  // MetaX C500 specific values
  static constexpr int WARP_SIZE = 64;
  static constexpr int MAX_THREADS_PER_BLOCK = 1024;
  static constexpr int MAX_WARPS_PER_SM = 32;  // 2048 / 64
  static constexpr int MAX_BLOCKS_PER_SM = 16;
  static constexpr int REGISTERS_PER_SM = 131072;
  static constexpr int SHARED_MEM_PER_SM = 65536;  // 64 KB
  static constexpr int SM_COUNT = 104;  // MetaX C500
};

/**
 * @brief Compute optimal block size for MACA
 * 
 * @param problem_size Total problem size
 * @param compute_intensity Compute to memory ratio
 * @return Recommended block size (multiple of 64)
 */
int compute_optimal_block_size(size_t problem_size, float compute_intensity) {
  // Start with default block size
  int block_size = 256;  // 4 warps of 64 threads
  
  // For small problems, use smaller blocks for better occupancy
  if (problem_size < 1024) {
    block_size = 64;  // 1 warp
  } else if (problem_size < 8192) {
    block_size = 128;  // 2 warps
  } else if (problem_size < 65536) {
    block_size = 256;  // 4 warps
  } else if (problem_size < 262144) {
    block_size = 512;  // 8 warps
  } else {
    block_size = 1024;  // 16 warps
  }
  
  // Adjust for compute intensity
  if (compute_intensity > 10.0f) {
    // Compute-bound: larger blocks for better register utilization
    block_size = std::min(block_size * 2, MACAOptParams::MAX_THREADS_PER_BLOCK);
  } else if (compute_intensity < 1.0f) {
    // Memory-bound: smaller blocks for better occupancy
    block_size = std::max(block_size / 2, MACAOptParams::WARP_SIZE);
  }
  
  // Ensure block size is multiple of warp size
  block_size = (block_size / MACAOptParams::WARP_SIZE) * MACAOptParams::WARP_SIZE;
  
  return block_size;
}

/**
 * @brief Compute optimal grid size for MACA
 * 
 * @param total_elements Total number of elements to process
 * @param block_size Threads per block
 * @return Grid dimensions
 */
dim3 compute_optimal_grid_size(size_t total_elements, int block_size) {
  int num_blocks = (total_elements + block_size - 1) / block_size;
  
  // Limit to reasonable number of blocks
  int max_blocks = MACAOptParams::SM_COUNT * MACAOptParams::MAX_BLOCKS_PER_SM;
  num_blocks = std::min(num_blocks, max_blocks);
  
  // For 2D/3D grids, try to balance dimensions
  if (num_blocks > 65535) {
    int grid_x = 65535;
    int grid_y = (num_blocks + grid_x - 1) / grid_x;
    return dim3(grid_x, grid_y, 1);
  }
  
  return dim3(num_blocks, 1, 1);
}

/**
 * @brief Compute optimal shared memory configuration
 * 
 * @param smem_per_thread Shared memory needed per thread
 * @param block_size Threads per block
 * @return Recommended shared memory size in bytes
 */
size_t compute_optimal_smem_size(size_t smem_per_thread, int block_size) {
  size_t total_smem = smem_per_thread * block_size;
  
  // Round up to 256-byte alignment
  total_smem = ((total_smem + 255) / 256) * 256;
  
  // Cap at available shared memory
  return std::min(total_smem, (size_t)MACAOptParams::SHARED_MEM_PER_SM);
}

/**
 * @brief Compute optimal tile sizes for matrix operations
 * 
 * @param M Matrix M dimension
 * @param N Matrix N dimension
 * @param K Matrix K dimension
 * @param config Output configuration
 */
void compute_optimal_tile_config(int M, int N, int K, MACAMatmulConfig& config) {
  // Default tile sizes for MACA (adjusted for 64-thread warps)
  config.tile_m = 128;
  config.tile_n = 128;
  config.tile_k = 32;
  
  // Adjust based on problem size
  if (M < 128 || N < 128) {
    // Small matrices: use smaller tiles
    config.tile_m = std::min(64, M);
    config.tile_n = std::min(64, N);
    config.num_stages = 2;
  } else if (M >= 4096 && N >= 4096) {
    // Large matrices: use larger tiles
    config.tile_m = 256;
    config.tile_n = 128;
    config.num_stages = 4;
  }
  
  // Warp tile sizes for 64-thread warps
  config.warp_m = std::min(64, config.tile_m);
  config.warp_n = std::min(64, config.tile_n);
  
  // Check shared memory constraints
  size_t smem_needed = 
      (config.tile_m * config.tile_k + config.tile_k * config.tile_n) * 
      sizeof(float) * config.num_stages;
  
  while (smem_needed > MACAOptParams::SHARED_MEM_PER_SM && config.num_stages > 1) {
    config.num_stages--;
    smem_needed = 
        (config.tile_m * config.tile_k + config.tile_k * config.tile_n) * 
        sizeof(float) * config.num_stages;
  }
}

/**
 * @brief Compute occupancy for a given kernel configuration
 * 
 * @param threads_per_block Threads per block
 * @param registers_per_thread Registers used per thread
 * @param smem_per_block Shared memory per block
 * @return Estimated occupancy (0.0 - 1.0)
 */
float compute_occupancy(int threads_per_block, 
                        int registers_per_thread,
                        size_t smem_per_block) {
  // Calculate blocks limited by threads
  int warps_per_block = threads_per_block / MACAOptParams::WARP_SIZE;
  int blocks_by_warps = MACAOptParams::MAX_WARPS_PER_SM / warps_per_block;
  
  // Calculate blocks limited by registers
  int regs_per_block = registers_per_thread * threads_per_block;
  int blocks_by_regs = MACAOptParams::REGISTERS_PER_SM / regs_per_block;
  
  // Calculate blocks limited by shared memory
  int blocks_by_smem = (smem_per_block > 0) ? 
      MACAOptParams::SHARED_MEM_PER_SM / smem_per_block : 
      MACAOptParams::MAX_BLOCKS_PER_SM;
  
  // Take minimum
  int blocks_per_sm = std::min({blocks_by_warps, blocks_by_regs, 
                                blocks_by_smem, MACAOptParams::MAX_BLOCKS_PER_SM});
  
  // Calculate occupancy
  int active_warps = blocks_per_sm * warps_per_block;
  return static_cast<float>(active_warps) / MACAOptParams::MAX_WARPS_PER_SM;
}

/**
 * @brief Get launch configuration for reduction kernels
 * 
 * @param num_elements Number of elements to reduce
 * @return MACAReductionConfig with optimal settings
 */
MACAReductionConfig get_reduction_config(size_t num_elements) {
  MACAReductionConfig config;
  
  // For MACA's 64-thread warps, we need 6 iterations for warp reduction
  config.warp_shuffle_iterations = 6;
  
  if (num_elements < 1024) {
    config.block_size = 64;  // 1 warp
    config.elements_per_thread = 1;
    config.use_two_pass = false;
  } else if (num_elements < 65536) {
    config.block_size = 256;  // 4 warps
    config.elements_per_thread = 4;
    config.use_two_pass = false;
  } else {
    config.block_size = 1024;  // 16 warps
    config.elements_per_thread = 8;
    config.use_two_pass = (num_elements > config.two_pass_threshold);
  }
  
  return config;
}

} // namespace maca
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED

