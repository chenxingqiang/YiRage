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


#include "yirage/kernel/cuda/cuda_kernel_config.h"

#ifdef YIRAGE_BACKEND_CUDA_ENABLED

#include <algorithm>
#include <cmath>

namespace yirage {
namespace kernel {
namespace cuda {

int CUDAOptimizer::compute_optimal_warps(size_t problem_size,
                                        int compute_capability) {
  // Based on compute capability, determine optimal warp count
  int max_warps_per_sm = 0;

  if (compute_capability >= 80) {
    // Ampere and later: up to 64 warps per SM
    max_warps_per_sm = 64;
  } else if (compute_capability >= 70) {
    // Volta/Turing: up to 64 warps per SM
    max_warps_per_sm = 64;
  } else {
    // Pascal and earlier: up to 64 warps per SM
    max_warps_per_sm = 64;
  }

  // Compute based on problem size
  int warps_needed = static_cast<int>((problem_size + 1023) / 1024);

  // Choose power of 2 for better scheduling
  int optimal_warps = 1;
  while (optimal_warps < warps_needed && optimal_warps < max_warps_per_sm) {
    optimal_warps *= 2;
  }

  return std::min(optimal_warps, max_warps_per_sm);
}

size_t CUDAOptimizer::compute_optimal_smem(size_t data_size, SmemLayout layout,
                                          int padding) {
  size_t smem_size = data_size;

  // Add padding to avoid bank conflicts
  if (layout == SmemLayout::SWIZZLED) {
    // Add padding based on number of banks (32 for most GPUs)
    int num_banks = 32;
    int elements_per_row = static_cast<int>(std::sqrt(data_size / 4)); // Assume float32
    int padded_elements = ((elements_per_row + num_banks - 1) / num_banks) *
                          num_banks + padding;
    smem_size = data_size + padding * sizeof(float);
  } else {
    smem_size += padding;
  }

  // Align to cache line (128 bytes)
  smem_size = ((smem_size + 127) / 128) * 128;

  return smem_size;
}

bool CUDAOptimizer::has_bank_conflict(SmemLayout layout, int stride,
                                     int bank_size) {
  if (layout == SmemLayout::SWIZZLED) {
    return false; // Swizzled layout avoids conflicts
  }

  // Check if stride is a multiple of number of banks
  int num_banks = 32;
  int bytes_per_bank = bank_size;

  if (stride % (num_banks * bytes_per_bank) == 0) {
    return true; // Bank conflict
  }

  return false;
}

float CUDAOptimizer::estimate_occupancy(CUDAKernelConfig const &config,
                                       int registers_per_thread) {
  int threads_per_block = config.get_total_threads();
  int warps_per_block = (threads_per_block + 31) / 32;

  // Get SM limits based on compute capability
  int max_threads_per_sm = 2048;    // For Ampere
  int max_warps_per_sm = 64;
  int max_blocks_per_sm = 32;
  int max_registers_per_sm = 65536; // For Ampere
  size_t max_smem_per_sm = 164 * 1024; // For Ampere A100

  if (config.compute_capability == 75) {
    // Turing
    max_threads_per_sm = 1024;
    max_warps_per_sm = 32;
    max_blocks_per_sm = 16;
    max_registers_per_sm = 65536;
    max_smem_per_sm = 64 * 1024;
  }

  // Calculate occupancy based on various limits
  int blocks_limited_by_threads =
      max_threads_per_sm / threads_per_block;
  int blocks_limited_by_warps =
      max_warps_per_sm / warps_per_block;
  int blocks_limited_by_registers =
      max_registers_per_sm / (registers_per_thread * threads_per_block);
  int blocks_limited_by_smem =
      static_cast<int>(max_smem_per_sm / config.shared_memory_size);

  // Take minimum
  int blocks_per_sm = std::min({blocks_limited_by_threads,
                                blocks_limited_by_warps,
                                blocks_limited_by_registers,
                                blocks_limited_by_smem,
                                max_blocks_per_sm});

  blocks_per_sm = std::max(1, blocks_per_sm);

  // Calculate achieved occupancy
  int active_warps = blocks_per_sm * warps_per_block;
  float occupancy = static_cast<float>(active_warps) / max_warps_per_sm;

  return occupancy;
}

bool CUDAOptimizer::select_tensor_core_config(int m, int n, int k,
                                              int compute_capability,
                                              CUDAKernelConfig &config) {
  // Tensor Cores are available from Volta (SM 7.0) onwards
  if (compute_capability < 70) {
    return false;
  }

  // Check if dimensions are compatible with Tensor Core shapes
  // Volta/Turing: 16x16x16
  // Ampere: 16x8x16, 16x8x8 (FP16/BF16)
  // Hopper: Various shapes

  if (compute_capability >= 90) {
    // Hopper - use most flexible config
    config.mma_m = 16;
    config.mma_n = 8;
    config.mma_k = 16;
  } else if (compute_capability >= 80) {
    // Ampere
    config.mma_m = 16;
    config.mma_n = 8;
    config.mma_k = 16;
  } else {
    // Volta/Turing
    config.mma_m = 16;
    config.mma_n = 16;
    config.mma_k = 16;
  }

  // Check if problem size is large enough to benefit from Tensor Cores
  bool large_enough = (m >= config.mma_m * 4) && (n >= config.mma_n * 4) &&
                     (k >= config.mma_k * 4);

  config.use_tensor_core = large_enough;
  return large_enough;
}

void CUDAOptimizer::optimize_grid_block_dims(int problem_m, int problem_n,
                                             int problem_k,
                                             int compute_capability,
                                             CUDAKernelConfig &config) {
  // Determine if we should use Tensor Cores
  select_tensor_core_config(problem_m, problem_n, problem_k,
                           compute_capability, config);

  if (config.use_tensor_core) {
    // Tile sizes based on Tensor Core dimensions
    int tile_m = config.mma_m * 4; // 64 for Ampere
    int tile_n = config.mma_n * 4; // 32 for Ampere
    int tile_k = config.mma_k * 2; // 32 for Ampere

    config.grid_dim_x = (problem_n + tile_n - 1) / tile_n;
    config.grid_dim_y = (problem_m + tile_m - 1) / tile_m;
    config.grid_dim_z = 1;

    // Block dimensions for Tensor Core warps
    config.block_dim_x = 32; // Warp size
    config.block_dim_y = 4;  // 4 warps
    config.block_dim_z = 1;

    config.num_warps = 4;
  } else {
    // Traditional CUDA cores configuration
    int block_size = 256; // Common choice
    int tile_size = 32;

    config.block_dim_x = block_size;
    config.block_dim_y = 1;
    config.block_dim_z = 1;

    config.grid_dim_x = (problem_n + tile_size - 1) / tile_size;
    config.grid_dim_y = (problem_m + tile_size - 1) / tile_size;
    config.grid_dim_z = 1;

    config.num_warps = (block_size + 31) / 32;
  }
}

float CUDAOptimizer::estimate_memory_bandwidth(CUDAKernelConfig const &config,
                                               size_t bytes_accessed,
                                               float execution_time_ms) {
  if (execution_time_ms <= 0.0f) {
    return 0.0f;
  }

  // Convert to GB/s
  float seconds = execution_time_ms / 1000.0f;
  float gigabytes = bytes_accessed / (1024.0f * 1024.0f * 1024.0f);
  return gigabytes / seconds;
}

float CUDAOptimizer::estimate_compute_throughput(
    CUDAKernelConfig const &config, size_t num_operations,
    float execution_time_ms) {
  if (execution_time_ms <= 0.0f) {
    return 0.0f;
  }

  // Convert to TFLOPS
  float seconds = execution_time_ms / 1000.0f;
  float tflops = (num_operations / 1e12f) / seconds;
  return tflops;
}

} // namespace cuda
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_CUDA_ENABLED





