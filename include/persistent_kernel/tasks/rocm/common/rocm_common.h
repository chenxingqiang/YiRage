/* Copyright 2025 YiRage Team
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

/**
 * @file rocm_common.h
 * @brief Common definitions for AMD ROCm/HIP GPU kernels
 *
 * AMD GPU Architecture Overview:
 * - CDNA (Compute DNA): Data center GPUs (MI100, MI200, MI300)
 * - RDNA: Consumer/Gaming GPUs
 *
 * Key differences from CUDA:
 * - 64-thread wavefronts (not warps)
 * - LDS (Local Data Share) instead of shared memory
 * - MFMA (Matrix Fused Multiply-Add) instead of Tensor Cores
 * - Different memory hierarchy
 */

namespace yirage {
namespace persistent_kernel {
namespace rocm {

// =============================================================================
// AMD GPU Architecture Detection
// =============================================================================

enum class AMDArch {
    // CDNA (Data Center)
    MI100 = 100,        // gfx908, CDNA1 (2020)
    MI200 = 200,        // gfx90a, CDNA2 (2021) - MI210/MI250/MI250X
    MI250 = 250,        // gfx90a, CDNA2 (2021) - Dual-die
    MI300X = 300,       // gfx942, CDNA3 (2023) - 192GB HBM3
    MI300A = 301,       // gfx942, CDNA3 APU (2023) - Unified memory
    MI325X = 325,       // gfx942+, CDNA3+ (2024)
    MI350 = 350,        // gfx950?, CDNA4 (2025+)
    
    // RDNA (Consumer) - Optional support
    RDNA2 = 2000,       // gfx1030 (RX 6000 series)
    RDNA3 = 3000,       // gfx1100 (RX 7000 series)
    RDNA4 = 4000,       // gfx1200 (RX 8000 series)
    
    UNKNOWN = 0
};

enum class AMDVariant {
    STANDARD,
    X,          // Extended (more CUs)
    A,          // APU (unified memory)
    DUAL_DIE    // Multi-die (MI250X)
};

// =============================================================================
// Hardware Specifications by Architecture
// =============================================================================

struct AMDSpecs {
    int compute_units;          // Number of Compute Units (CUs)
    int wavefront_size;         // Threads per wavefront (64 for CDNA)
    int max_waves_per_cu;       // Max wavefronts per CU
    int lds_kb;                 // LDS per CU in KB
    int vgpr_per_cu;            // Vector GPRs per CU
    int sgpr_per_cu;            // Scalar GPRs per CU
    int hbm_gb;                 // HBM memory in GB
    int hbm_bw_gbps;            // HBM bandwidth in GB/s
    int fp16_tflops;            // FP16 TFLOPS
    int fp32_tflops;            // FP32 TFLOPS
    int mfma_m;                 // MFMA tile M dimension
    int mfma_n;                 // MFMA tile N dimension
    int mfma_k;                 // MFMA tile K dimension
    bool has_matrix_core;       // MFMA support
    bool has_fp8;               // FP8 support
    bool has_sparsity;          // Sparse MFMA
};

// MI100 (2020) - CDNA1
constexpr AMDSpecs MI100_SPECS = {
    .compute_units = 120,
    .wavefront_size = 64,
    .max_waves_per_cu = 32,
    .lds_kb = 64,
    .vgpr_per_cu = 262144,
    .sgpr_per_cu = 3200,
    .hbm_gb = 32,
    .hbm_bw_gbps = 1228,
    .fp16_tflops = 184,
    .fp32_tflops = 23,
    .mfma_m = 32,
    .mfma_n = 32,
    .mfma_k = 8,
    .has_matrix_core = true,
    .has_fp8 = false,
    .has_sparsity = false
};

// MI200 (2021) - CDNA2 (single die, MI210)
constexpr AMDSpecs MI200_SPECS = {
    .compute_units = 104,
    .wavefront_size = 64,
    .max_waves_per_cu = 32,
    .lds_kb = 64,
    .vgpr_per_cu = 262144,
    .sgpr_per_cu = 3200,
    .hbm_gb = 64,
    .hbm_bw_gbps = 1638,
    .fp16_tflops = 181,
    .fp32_tflops = 45,
    .mfma_m = 32,
    .mfma_n = 32,
    .mfma_k = 8,
    .has_matrix_core = true,
    .has_fp8 = false,
    .has_sparsity = false
};

// MI250X (2021) - CDNA2 dual-die
constexpr AMDSpecs MI250X_SPECS = {
    .compute_units = 220,       // 110 per die x 2
    .wavefront_size = 64,
    .max_waves_per_cu = 32,
    .lds_kb = 64,
    .vgpr_per_cu = 262144,
    .sgpr_per_cu = 3200,
    .hbm_gb = 128,              // 64GB x 2
    .hbm_bw_gbps = 3276,        // 1638 x 2
    .fp16_tflops = 383,         // 191.5 x 2
    .fp32_tflops = 95,
    .mfma_m = 32,
    .mfma_n = 32,
    .mfma_k = 8,
    .has_matrix_core = true,
    .has_fp8 = false,
    .has_sparsity = false
};

// MI300X (2023) - CDNA3
constexpr AMDSpecs MI300X_SPECS = {
    .compute_units = 304,
    .wavefront_size = 64,
    .max_waves_per_cu = 32,
    .lds_kb = 64,
    .vgpr_per_cu = 524288,      // Doubled
    .sgpr_per_cu = 3200,
    .hbm_gb = 192,
    .hbm_bw_gbps = 5300,
    .fp16_tflops = 1307,
    .fp32_tflops = 163,
    .mfma_m = 32,
    .mfma_n = 32,
    .mfma_k = 16,               // Doubled K
    .has_matrix_core = true,
    .has_fp8 = true,
    .has_sparsity = true
};

// MI300A (2023) - CDNA3 APU
constexpr AMDSpecs MI300A_SPECS = {
    .compute_units = 228,
    .wavefront_size = 64,
    .max_waves_per_cu = 32,
    .lds_kb = 64,
    .vgpr_per_cu = 524288,
    .sgpr_per_cu = 3200,
    .hbm_gb = 128,              // Unified with CPU
    .hbm_bw_gbps = 5300,
    .fp16_tflops = 980,
    .fp32_tflops = 122,
    .mfma_m = 32,
    .mfma_n = 32,
    .mfma_k = 16,
    .has_matrix_core = true,
    .has_fp8 = true,
    .has_sparsity = true
};

// MI325X (2024) - CDNA3+
constexpr AMDSpecs MI325X_SPECS = {
    .compute_units = 304,
    .wavefront_size = 64,
    .max_waves_per_cu = 32,
    .lds_kb = 64,
    .vgpr_per_cu = 524288,
    .sgpr_per_cu = 3200,
    .hbm_gb = 256,              // Increased
    .hbm_bw_gbps = 6000,
    .fp16_tflops = 1380,
    .fp32_tflops = 172,
    .mfma_m = 32,
    .mfma_n = 32,
    .mfma_k = 16,
    .has_matrix_core = true,
    .has_fp8 = true,
    .has_sparsity = true
};

// MI350 (2025+) - CDNA4
constexpr AMDSpecs MI350_SPECS = {
    .compute_units = 400,       // Projected
    .wavefront_size = 64,
    .max_waves_per_cu = 32,
    .lds_kb = 128,              // Projected double
    .vgpr_per_cu = 524288,
    .sgpr_per_cu = 3200,
    .hbm_gb = 288,
    .hbm_bw_gbps = 8000,
    .fp16_tflops = 2000,        // Projected
    .fp32_tflops = 250,
    .mfma_m = 64,               // Projected larger
    .mfma_n = 64,
    .mfma_k = 16,
    .has_matrix_core = true,
    .has_fp8 = true,
    .has_sparsity = true
};

// =============================================================================
// Kernel Configuration by Architecture
// =============================================================================

struct ROCmKernelConfig {
    int block_size;             // Threads per block
    int waves_per_block;        // Wavefronts per block
    int tile_m;                 // GEMM M tile
    int tile_n;                 // GEMM N tile
    int tile_k;                 // GEMM K tile
    int lds_tile_kb;            // LDS usage in KB
    bool use_mfma;              // Use Matrix Core
    bool use_fp8;               // Use FP8
    bool use_async_copy;        // Async global->LDS copy
};

constexpr ROCmKernelConfig MI100_KERNEL_CONFIG = {
    .block_size = 256,
    .waves_per_block = 4,
    .tile_m = 128,
    .tile_n = 128,
    .tile_k = 32,
    .lds_tile_kb = 32,
    .use_mfma = true,
    .use_fp8 = false,
    .use_async_copy = false
};

constexpr ROCmKernelConfig MI250_KERNEL_CONFIG = {
    .block_size = 256,
    .waves_per_block = 4,
    .tile_m = 256,
    .tile_n = 128,
    .tile_k = 32,
    .lds_tile_kb = 48,
    .use_mfma = true,
    .use_fp8 = false,
    .use_async_copy = true
};

constexpr ROCmKernelConfig MI300X_KERNEL_CONFIG = {
    .block_size = 256,
    .waves_per_block = 4,
    .tile_m = 256,
    .tile_n = 256,
    .tile_k = 64,
    .lds_tile_kb = 64,
    .use_mfma = true,
    .use_fp8 = true,
    .use_async_copy = true
};

constexpr ROCmKernelConfig MI350_KERNEL_CONFIG = {
    .block_size = 512,
    .waves_per_block = 8,
    .tile_m = 256,
    .tile_n = 256,
    .tile_k = 64,
    .lds_tile_kb = 96,
    .use_mfma = true,
    .use_fp8 = true,
    .use_async_copy = true
};

// =============================================================================
// Constants
// =============================================================================

// AMD uses 64-thread wavefronts
constexpr int ROCM_WAVEFRONT_SIZE = 64;
constexpr int ROCM_DEFAULT_BLOCK_SIZE = 256;
constexpr int ROCM_LDS_BANK_SIZE = 4;  // bytes
constexpr int ROCM_LDS_NUM_BANKS = 32;
constexpr int ROCM_MAX_LDS_KB = 64;

// MFMA configurations
constexpr int MFMA_32x32x8_FP16 = 0;
constexpr int MFMA_32x32x16_FP16 = 1;  // CDNA3+
constexpr int MFMA_16x16x16_FP16 = 2;
constexpr int MFMA_16x16x32_FP8 = 3;   // CDNA3+

// =============================================================================
// Runtime Detection Functions
// =============================================================================

/**
 * @brief Detect AMD GPU architecture at runtime
 * @return AMDArch enum value
 */
AMDArch detect_amd_arch();

/**
 * @brief Detect AMD GPU variant
 * @return AMDVariant enum value
 */
AMDVariant detect_amd_variant();

/**
 * @brief Get hardware specs for current device
 * @return AMDSpecs for the detected hardware
 */
const AMDSpecs& get_device_specs();

/**
 * @brief Get optimal kernel config for current device
 * @return ROCmKernelConfig for the detected hardware
 */
const ROCmKernelConfig& get_kernel_config();

}  // namespace rocm
}  // namespace persistent_kernel
}  // namespace yirage
