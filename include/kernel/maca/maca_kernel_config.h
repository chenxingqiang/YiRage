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
 * MACA Kernel Configuration
 * 
 * Configuration parameters for MetaX MACA GPU kernels.
 * MACA hardware architecture is similar to NVIDIA GPUs, so many
 * CUDA kernel configurations apply with minor adjustments.
 */

#pragma once

#ifdef YIRAGE_BACKEND_MACA_ENABLED

#include <cstddef>
#include <cstdint>

namespace yirage {
namespace kernel {
namespace maca {

/**
 * @brief MACA device architecture parameters
 * 
 * MetaX GPU architecture parameters based on actual MetaX C500 hardware.
 * Key differences from NVIDIA GPUs:
 * - warpSize = 64 (vs NVIDIA's 32)
 * - Wider memory bus (4096 bits)
 * - Different register file configuration
 */
struct MACAArchConfig {
  // Thread hierarchy - NOTE: warp_size is 64 on MetaX (not 32!)
  int max_threads_per_block = 1024;
  int warp_size = 64;                          // MetaX uses 64-wide warps!
  int max_warps_per_sm = 32;                   // 2048 threads / 64 warp_size
  int max_blocks_per_sm = 16;
  
  // Memory hierarchy (in bytes) - MetaX C500 values
  size_t shared_memory_per_block = 65536;      // 64 KB
  size_t shared_memory_per_sm = 65536;         // 64 KB (same as per block)
  size_t l1_cache_size = 65536;                // 64 KB
  size_t l2_cache_size = 8388608;              // 8 MB
  
  // Register file - MetaX C500 has larger register file
  int registers_per_block = 131072;            // 128K registers per block
  int registers_per_thread = 255;
  
  // Compute capability (MetaX reports major=10, minor=0)
  int compute_capability_major = 10;
  int compute_capability_minor = 0;
  
  // MetaX-specific parameters
  int memory_bus_width = 4096;                 // 4096-bit memory bus
  int multi_processor_count = 104;             // 104 SMs on C500
  int max_threads_per_multiprocessor = 2048;
};

/**
 * @brief MACA kernel launch configuration
 */
struct MACALaunchConfig {
  // Grid dimensions
  uint32_t grid_x = 1;
  uint32_t grid_y = 1;
  uint32_t grid_z = 1;
  
  // Block dimensions
  uint32_t block_x = 256;
  uint32_t block_y = 1;
  uint32_t block_z = 1;
  
  // Shared memory
  size_t shared_mem_bytes = 0;
  
  // Stream (nullptr for default)
  void *stream = nullptr;
  
  // Get total threads per block
  uint32_t threads_per_block() const {
    return block_x * block_y * block_z;
  }
  
  // Get total blocks
  uint32_t total_blocks() const {
    return grid_x * grid_y * grid_z;
  }
};

/**
 * @brief MACA matmul kernel configuration
 * 
 * Tile sizes and algorithm parameters for matrix multiplication
 * optimized for MetaX GPU architecture.
 * 
 * NOTE: MetaX uses 64-wide warps, so warp tile dimensions should
 * be adjusted accordingly for optimal occupancy.
 */
struct MACAMatmulConfig {
  // Tile dimensions
  int tile_m = 128;
  int tile_n = 128;
  int tile_k = 32;
  
  // Thread block dimensions
  int block_m = 16;
  int block_n = 16;
  
  // Warp tile dimensions - adjusted for 64-wide warps
  // With warpSize=64, we can cover larger tiles per warp
  int warp_m = 64;
  int warp_n = 64;
  
  // Pipeline stages for software pipelining
  int num_stages = 3;
  
  // Use tensor cores (if available on MetaX hardware)
  bool use_tensor_cores = true;
  
  // Data types for compute
  bool use_fp16_accumulate = false;
  
  // MetaX-specific optimizations
  bool use_vectorized_load = true;  // MetaX supports wide memory transactions
};

/**
 * @brief MACA reduction kernel configuration
 * 
 * NOTE: MetaX warpSize is 64, so warp reductions cover more threads
 * per warp-level operation, potentially improving efficiency.
 */
struct MACAReductionConfig {
  // Block size for reduction (should be multiple of warpSize=64)
  int block_size = 256;  // 4 warps of 64 threads each
  
  // Number of elements processed per thread
  int elements_per_thread = 4;
  
  // Use warp-level primitives (warp shuffle with 64 threads)
  bool use_warp_shuffle = true;
  
  // Warp shuffle iterations: log2(64) = 6 (vs log2(32) = 5 for NVIDIA)
  int warp_shuffle_iterations = 6;
  
  // Two-pass reduction for large arrays
  bool use_two_pass = true;
  int two_pass_threshold = 1048576;  // 1M elements
};

/**
 * @brief MACA attention kernel configuration
 * 
 * MetaX provides mcflashattn library (Flash Attention 2.0 compatible)
 * with optimized implementations for MACA hardware.
 */
struct MACAAttentionConfig {
  // Block sizes (adjusted for 64-thread warps)
  int block_m = 64;
  int block_n = 64;
  int block_k = 64;
  
  // Head dimension
  int head_dim = 128;
  
  // Softmax temperature
  float softmax_scale = 1.0f;
  
  // Flash attention parameters
  // MACA provides mcflashattn library with native Flash Attention support
  bool use_flash_attention = true;
  int flash_tile_size = 128;
  
  // Use MACA native mcflashattn library when available
  bool use_mcflashattn = true;
  
  // Causal masking
  bool causal = true;
  
  // Softcap for attention scores (mcflashattn extension)
  float softcap = 0.0f;
  
  // Page attention support
  bool use_paged_attention = false;
};

/**
 * @brief MACA normalization kernel configuration
 */
struct MACANormConfig {
  // Block size (multiple of warpSize=64)
  int block_size = 256;
  
  // Elements per thread
  int elements_per_thread = 4;
  
  // Epsilon for numerical stability
  float epsilon = 1e-5f;
  
  // RMS norm vs Layer norm
  bool use_rms_norm = true;
};

/**
 * @brief MACA BLAS configuration
 * 
 * MetaX provides mcblas and mcblasLt libraries for optimized BLAS operations.
 */
struct MACABlasConfig {
  // Use mcblasLt for advanced GEMM optimizations
  bool use_mcblas_lt = true;
  
  // Math mode for mixed precision
  enum class MathMode {
    DEFAULT,
    TENSOR_OP,      // Use tensor cores when available
    PEDANTIC,       // Strict IEEE compliance
    ALLOW_TF32      // Allow TF32 precision
  };
  MathMode math_mode = MathMode::TENSOR_OP;
  
  // Workspace size for mcblasLt algorithms
  size_t workspace_size = 32 * 1024 * 1024;  // 32 MB default
  
  // Enable algorithm heuristics
  bool enable_heuristics = true;
};

/**
 * @brief MACA CUTLASS-equivalent (mctlass) configuration
 * 
 * MetaX provides mctlass library for tile-based GEMM operations,
 * similar to NVIDIA CUTLASS but optimized for MACA hardware.
 */
struct MACATlassConfig {
  // Thread block tile dimensions
  int threadblock_m = 128;
  int threadblock_n = 128;
  int threadblock_k = 32;
  
  // Warp tile dimensions (adjusted for 64-thread warps)
  int warp_m = 64;
  int warp_n = 64;
  int warp_k = 32;
  
  // Instruction tile (for tensor ops)
  int instruction_m = 16;
  int instruction_n = 16;
  int instruction_k = 16;
  
  // Number of pipeline stages
  int stages = 3;
  
  // Swizzle mode for shared memory
  enum class SwizzleMode {
    NONE,
    SWIZZLE_32,
    SWIZZLE_64,
    SWIZZLE_128
  };
  SwizzleMode swizzle = SwizzleMode::SWIZZLE_64;
  
  // Epilogue configuration
  bool use_epilogue_visitor = false;
  bool split_k_mode = false;
  int split_k_slices = 1;
};

/**
 * @brief Get default MACA architecture configuration
 * @param compute_capability Compute capability (MetaX uses 100 for C500)
 * @return MACAArchConfig with appropriate defaults
 * 
 * MetaX compute capability mapping:
 * - 100 = MetaX C500 (current flagship)
 * - Future chips may have different values
 */
inline MACAArchConfig get_maca_arch_config(int compute_capability = 100) {
  MACAArchConfig config;
  
  // All MetaX GPUs use warpSize=64
  config.warp_size = 64;
  
  if (compute_capability >= 100) {
    // MetaX C500 generation
    config.shared_memory_per_block = 65536;   // 64 KB
    config.shared_memory_per_sm = 65536;      // 64 KB
    config.l2_cache_size = 8388608;           // 8 MB
    config.registers_per_block = 131072;      // 128K registers
    config.max_warps_per_sm = 32;             // 2048 / 64
    config.max_threads_per_multiprocessor = 2048;
    config.multi_processor_count = 104;
    config.memory_bus_width = 4096;
    config.compute_capability_major = 10;
    config.compute_capability_minor = 0;
  } else {
    // Fallback for potential older MetaX hardware
    config.shared_memory_per_block = 49152;   // 48 KB
    config.shared_memory_per_sm = 65536;      // 64 KB
    config.l2_cache_size = 6291456;           // 6 MB
    config.registers_per_block = 65536;
    config.max_warps_per_sm = 32;
    config.compute_capability_major = 7;
    config.compute_capability_minor = 5;
  }
  
  return config;
}

/**
 * @brief Get optimal matmul config for given problem size
 * @param M Matrix M dimension
 * @param N Matrix N dimension
 * @param K Matrix K dimension
 * @param arch Architecture configuration
 * @return Optimal MACAMatmulConfig
 */
inline MACAMatmulConfig get_optimal_matmul_config(
    int M, int N, int K, 
    MACAArchConfig const &arch = MACAArchConfig()) {
  
  MACAMatmulConfig config;
  
  // Adjust tile sizes based on problem dimensions
  if (M >= 4096 && N >= 4096) {
    // Large matrices: use larger tiles
    config.tile_m = 128;
    config.tile_n = 256;
    config.tile_k = 32;
    config.num_stages = 4;
  } else if (M >= 1024 && N >= 1024) {
    // Medium matrices
    config.tile_m = 128;
    config.tile_n = 128;
    config.tile_k = 32;
    config.num_stages = 3;
  } else {
    // Small matrices: use smaller tiles
    config.tile_m = 64;
    config.tile_n = 64;
    config.tile_k = 32;
    config.num_stages = 2;
  }
  
  // Check shared memory constraints
  size_t smem_required = 
      (config.tile_m * config.tile_k + config.tile_k * config.tile_n) * 
      sizeof(float) * config.num_stages;
  
  while (smem_required > arch.shared_memory_per_block && config.num_stages > 1) {
    config.num_stages--;
    smem_required = 
        (config.tile_m * config.tile_k + config.tile_k * config.tile_n) * 
        sizeof(float) * config.num_stages;
  }
  
  return config;
}

} // namespace maca
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED

