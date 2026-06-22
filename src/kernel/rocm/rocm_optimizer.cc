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
 * ROCm Kernel Optimizer
 * 
 * Optimization utilities for AMD ROCm/HIP kernels.
 * Provides heuristics for launch configuration and tuning.
 */

#include "kernel/rocm/rocm_kernel_config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace yirage {
namespace kernel {
namespace rocm {

// =============================================================================
// Architecture-specific parameters
// =============================================================================

struct ROCmArchParams {
  int wavefront_size;
  int max_threads_per_block;
  int max_wavefronts_per_cu;
  int max_blocks_per_cu;
  int vgprs_per_cu;
  int sgprs_per_cu;
  int lds_per_cu;
  int cu_count;
  int mfma_m, mfma_n, mfma_k;
  bool has_fp8;
  bool has_sparsity;
};

static ROCmArchParams get_arch_params(ROCmArch arch) {
  ROCmArchParams params;
  
  // Common for all CDNA
  params.wavefront_size = 64;
  params.max_threads_per_block = 1024;
  params.sgprs_per_cu = 102;
  
  switch (arch) {
    case ROCmArch::CDNA1:  // MI100
      params.max_wavefronts_per_cu = 40;
      params.max_blocks_per_cu = 16;
      params.vgprs_per_cu = 256;
      params.lds_per_cu = 65536;  // 64KB
      params.cu_count = 120;
      params.mfma_m = 32;
      params.mfma_n = 32;
      params.mfma_k = 8;
      params.has_fp8 = false;
      params.has_sparsity = false;
      break;
      
    case ROCmArch::CDNA2:  // MI200 series
      params.max_wavefronts_per_cu = 40;
      params.max_blocks_per_cu = 16;
      params.vgprs_per_cu = 256;
      params.lds_per_cu = 65536;
      params.cu_count = 220;  // MI250X
      params.mfma_m = 32;
      params.mfma_n = 32;
      params.mfma_k = 8;
      params.has_fp8 = false;
      params.has_sparsity = false;
      break;
      
    case ROCmArch::CDNA3:  // MI300 series
      params.max_wavefronts_per_cu = 40;
      params.max_blocks_per_cu = 16;
      params.vgprs_per_cu = 512;
      params.lds_per_cu = 65536;
      params.cu_count = 304;  // MI300X
      params.mfma_m = 32;
      params.mfma_n = 32;
      params.mfma_k = 16;
      params.has_fp8 = true;
      params.has_sparsity = true;
      break;
      
    case ROCmArch::RDNA2:
    case ROCmArch::RDNA3:
      params.wavefront_size = 32;  // RDNA uses 32-thread waves
      params.max_wavefronts_per_cu = 32;
      params.max_blocks_per_cu = 16;
      params.vgprs_per_cu = 256;
      params.lds_per_cu = 65536;
      params.cu_count = 80;
      params.mfma_m = 0;
      params.mfma_n = 0;
      params.mfma_k = 0;
      params.has_fp8 = false;
      params.has_sparsity = false;
      break;
      
    default:
      params = get_arch_params(ROCmArch::CDNA3);
      break;
  }
  
  return params;
}

// =============================================================================
// ROCmOptimizer Implementation
// =============================================================================

int ROCmOptimizer::compute_optimal_wavefronts(size_t problem_size, ROCmArch arch) {
  ROCmArchParams params = get_arch_params(arch);
  
  // Compute waves needed
  size_t elements_per_wave = params.wavefront_size;
  int waves_needed = static_cast<int>((problem_size + elements_per_wave - 1) / 
                                       elements_per_wave);
  
  // Choose power of 2 for better scheduling
  int optimal_waves = 1;
  while (optimal_waves < waves_needed && 
         optimal_waves < params.max_wavefronts_per_cu) {
    optimal_waves *= 2;
  }
  
  return std::min(optimal_waves, params.max_wavefronts_per_cu);
}

size_t ROCmOptimizer::compute_optimal_lds(size_t data_size, LDSLayout layout,
                                          int padding) {
  size_t lds_size = data_size;
  
  // Add padding to avoid bank conflicts
  // LDS has 32 banks, each 4 bytes wide
  constexpr int LDS_BANKS = 32;
  constexpr int LDS_BANK_WIDTH = 4;
  
  switch (layout) {
    case LDSLayout::SWIZZLED: {
      // Compute row size assuming square-ish layout
      int row_size = static_cast<int>(std::sqrt(data_size / sizeof(float)));
      int padded_row = ((row_size + LDS_BANKS - 1) / LDS_BANKS) * LDS_BANKS + padding;
      lds_size = padded_row * row_size * sizeof(float);
      break;
    }
    case LDSLayout::TILED:
      // Add padding for tile boundaries
      lds_size += padding * sizeof(float);
      break;
    default:
      lds_size += padding;
      break;
  }
  
  // Align to 256 bytes for optimal access
  lds_size = ((lds_size + 255) / 256) * 256;
  
  return lds_size;
}

bool ROCmOptimizer::has_bank_conflict(LDSLayout layout, int stride, int bank_size) {
  constexpr int LDS_BANKS = 32;
  
  if (layout == LDSLayout::SWIZZLED) {
    return false;  // Swizzled layout avoids conflicts
  }
  
  // Check if stride causes bank conflicts
  // Conflict occurs when stride is multiple of bank count
  int stride_in_banks = stride / bank_size;
  return (stride_in_banks % LDS_BANKS) == 0;
}

float ROCmOptimizer::estimate_occupancy(ROCmKernelConfig const &config,
                                        int vgprs_used, int sgprs_used) {
  ROCmArchParams params = get_arch_params(config.arch);
  
  // Compute waves limited by registers
  int waves_by_vgprs = params.vgprs_per_cu / std::max(vgprs_used, 1);
  int waves_by_sgprs = params.sgprs_per_cu / std::max(sgprs_used, 1);
  
  // Compute waves limited by LDS
  int lds_per_block = static_cast<int>(config.lds_size);
  int blocks_by_lds = params.lds_per_cu / std::max(lds_per_block, 1);
  int waves_per_block = config.num_wavefronts;
  int waves_by_lds = blocks_by_lds * waves_per_block;
  
  // Take minimum
  int achievable_waves = std::min({waves_by_vgprs, waves_by_sgprs, 
                                   waves_by_lds, params.max_wavefronts_per_cu});
  
  // Occupancy = achievable / max
  return static_cast<float>(achievable_waves) / params.max_wavefronts_per_cu;
}

bool ROCmOptimizer::select_matrix_core_config(int m, int n, int k,
                                              ROCmArch arch,
                                              ROCmKernelConfig &config) {
  if (!config.has_matrix_cores()) {
    return false;
  }
  
  ROCmArchParams params = get_arch_params(arch);
  
  // MFMA dimensions based on architecture
  config.use_matrix_core = true;
  config.mfma_m = params.mfma_m;
  config.mfma_n = params.mfma_n;
  config.mfma_k = params.mfma_k;
  
  // Select tile sizes based on problem dimensions
  if (m >= 256 && n >= 256) {
    // Large problem: 256x256 tiles
    config.forall_dim[0] = 256;
    config.forall_dim[1] = 256;
    config.forall_dim[2] = 64;
  } else if (m >= 128 && n >= 128) {
    // Medium problem: 128x128 tiles
    config.forall_dim[0] = 128;
    config.forall_dim[1] = 128;
    config.forall_dim[2] = 32;
  } else {
    // Small problem: 64x64 tiles
    config.forall_dim[0] = 64;
    config.forall_dim[1] = 64;
    config.forall_dim[2] = 32;
  }
  
  return true;
}

void ROCmOptimizer::optimize_grid_block_dims(int problem_m, int problem_n,
                                             int problem_k, ROCmArch arch,
                                             ROCmKernelConfig &config) {
  ROCmArchParams params = get_arch_params(arch);
  
  // Determine tile sizes
  int tile_m = 128, tile_n = 128;
  
  if (problem_m >= 4096 && problem_n >= 4096) {
    tile_m = tile_n = 256;
  } else if (problem_m < 256 || problem_n < 256) {
    tile_m = tile_n = 64;
  }
  
  // Compute grid dimensions
  int grid_m = (problem_m + tile_m - 1) / tile_m;
  int grid_n = (problem_n + tile_n - 1) / tile_n;
  
  // Set forall dimensions
  config.forall_dim[0] = tile_m;
  config.forall_dim[1] = tile_n;
  config.forall_dim[2] = 1;
  
  // Set block size (256 threads = 4 wavefronts for CDNA)
  config.num_threads = 256;
  config.num_wavefronts = config.num_threads / params.wavefront_size;
  
  // Store grid info
  config.imap_dim[0] = grid_m;
  config.imap_dim[1] = grid_n;
  config.imap_dim[2] = 1;
}

std::string ROCmOptimizer::get_arch_string(ROCmArch arch) {
  switch (arch) {
    case ROCmArch::CDNA1:
      return "gfx908";  // MI100
    case ROCmArch::CDNA2:
      return "gfx90a";  // MI200 series
    case ROCmArch::CDNA3:
      return "gfx942";  // MI300 series
    case ROCmArch::RDNA2:
      return "gfx1030"; // RX 6000 series
    case ROCmArch::RDNA3:
      return "gfx1100"; // RX 7000 series
    default:
      return "gfx942";
  }
}

// =============================================================================
// Additional Optimization Utilities
// =============================================================================

/**
 * @brief Compute optimal block size for ROCm
 */
int compute_optimal_block_size(size_t problem_size, float compute_intensity,
                               ROCmArch arch) {
  ROCmArchParams params = get_arch_params(arch);
  int wavefront_size = params.wavefront_size;
  
  // Start with default: 4 wavefronts
  int block_size = 4 * wavefront_size;
  
  // Adjust based on problem size
  if (problem_size < 1024) {
    block_size = wavefront_size;
  } else if (problem_size < 8192) {
    block_size = 2 * wavefront_size;
  } else if (problem_size < 65536) {
    block_size = 4 * wavefront_size;
  } else if (problem_size < 262144) {
    block_size = 8 * wavefront_size;
  } else {
    block_size = 16 * wavefront_size;
  }
  
  // Adjust for compute intensity
  if (compute_intensity > 10.0f) {
    block_size = std::min(block_size * 2, params.max_threads_per_block);
  } else if (compute_intensity < 1.0f) {
    block_size = std::max(block_size / 2, wavefront_size);
  }
  
  // Ensure multiple of wavefront size
  block_size = (block_size / wavefront_size) * wavefront_size;
  
  return block_size;
}

/**
 * @brief Compute optimal grid dimensions
 */
void compute_optimal_grid(size_t total_elements, int block_size,
                          int &grid_x, int &grid_y, int &grid_z,
                          ROCmArch arch) {
  ROCmArchParams params = get_arch_params(arch);
  
  int num_blocks = static_cast<int>((total_elements + block_size - 1) / block_size);
  int max_blocks = params.cu_count * params.max_blocks_per_cu;
  num_blocks = std::min(num_blocks, max_blocks);
  
  // Factorize for 2D grid if large
  if (num_blocks > 65535) {
    grid_x = 65535;
    grid_y = (num_blocks + 65534) / 65535;
    grid_z = 1;
  } else {
    grid_x = num_blocks;
    grid_y = 1;
    grid_z = 1;
  }
}

/**
 * @brief Estimate LDS usage for GEMM
 */
size_t estimate_gemm_lds_usage(int tile_m, int tile_n, int tile_k,
                               int num_stages, size_t element_size) {
  // Double buffer for A and B tiles
  size_t a_tile = tile_m * tile_k * element_size;
  size_t b_tile = tile_k * tile_n * element_size;
  
  // Add padding for bank conflict avoidance
  a_tile += tile_m * 8;  // 8 bytes padding per row
  b_tile += tile_k * 8;
  
  return (a_tile + b_tile) * num_stages;
}

/**
 * @brief Check if async copy is beneficial
 */
bool should_use_async_copy(ROCmArch arch, size_t transfer_size) {
  // Async copy available on CDNA2+
  if (arch == ROCmArch::CDNA1 || arch == ROCmArch::RDNA2 || 
      arch == ROCmArch::RDNA3) {
    return false;
  }
  
  // Beneficial for larger transfers
  return transfer_size >= 4096;
}

/**
 * @brief Get recommended number of pipeline stages
 */
int get_recommended_stages(ROCmArch arch, size_t lds_available,
                           size_t lds_per_stage) {
  int max_stages = static_cast<int>(lds_available / lds_per_stage);
  
  // Cap based on architecture
  switch (arch) {
    case ROCmArch::CDNA3:
      return std::min(max_stages, 4);
    case ROCmArch::CDNA2:
      return std::min(max_stages, 3);
    default:
      return std::min(max_stages, 2);
  }
}

}  // namespace rocm
}  // namespace kernel
}  // namespace yirage
