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
 * @file maca_common.h
 * @brief Common definitions for MACA (MetaX GPU) kernels
 *
 * MetaX MACA is an alternative GPU computing platform.
 * Key differences from CUDA:
 * - 64-thread warps (vs 32 in CUDA)
 * - Different memory hierarchy
 * - HBM-based memory system
 */

namespace yirage {
namespace persistent_kernel {
namespace maca {

// =============================================================================
// MetaX GPU Architecture Detection
// =============================================================================

enum class MetaXGen {
    C500 = 500,      // First generation (2023)
    C550 = 550,      // Enhanced C500 (2024)
    C600 = 600,      // Second generation (2024)
    C650 = 650,      // Enhanced C600 (2025)
    C700 = 700,      // Third generation (2025+)
    UNKNOWN = 0
};

enum class MetaXVariant {
    STANDARD,        // Standard configuration
    PRO,             // Pro variant (more SMs)
    MAX,             // Max variant (more memory)
    ULTRA            // Ultra variant (fully enabled)
};

// =============================================================================
// Hardware Specifications by Generation
// =============================================================================

struct MetaXSpecs {
    int sm_count;               // Number of Streaming Multiprocessors
    int warp_size;              // Threads per warp (64 for MetaX)
    int max_threads_per_block;  // Max threads per block
    int max_warps_per_sm;       // Max warps per SM
    int shared_memory_kb;       // Shared memory per block in KB
    int registers_per_sm;       // Registers per SM
    int hbm_memory_gb;          // HBM memory in GB
    int memory_bw_gbps;         // Memory bandwidth in GB/s
    int fp32_tflops;            // FP32 performance in TFLOPS
    int fp16_tflops;            // FP16 performance in TFLOPS
    bool has_tensor_core;       // Tensor core support
    bool has_sparsity;          // Sparsity acceleration
};

// C500 (2023) - First generation MetaX GPU
constexpr MetaXSpecs C500_STANDARD_SPECS = {
    .sm_count = 104,
    .warp_size = 64,
    .max_threads_per_block = 1024,
    .max_warps_per_sm = 32,
    .shared_memory_kb = 64,
    .registers_per_sm = 131072,
    .hbm_memory_gb = 64,
    .memory_bw_gbps = 1600,
    .fp32_tflops = 100,
    .fp16_tflops = 200,
    .has_tensor_core = true,
    .has_sparsity = false
};

constexpr MetaXSpecs C500_PRO_SPECS = {
    .sm_count = 128,
    .warp_size = 64,
    .max_threads_per_block = 1024,
    .max_warps_per_sm = 32,
    .shared_memory_kb = 64,
    .registers_per_sm = 131072,
    .hbm_memory_gb = 80,
    .memory_bw_gbps = 2000,
    .fp32_tflops = 125,
    .fp16_tflops = 250,
    .has_tensor_core = true,
    .has_sparsity = false
};

// C550 (2024) - Enhanced first generation
constexpr MetaXSpecs C550_STANDARD_SPECS = {
    .sm_count = 112,
    .warp_size = 64,
    .max_threads_per_block = 1024,
    .max_warps_per_sm = 32,
    .shared_memory_kb = 96,        // Increased
    .registers_per_sm = 131072,
    .hbm_memory_gb = 80,
    .memory_bw_gbps = 1800,
    .fp32_tflops = 120,
    .fp16_tflops = 240,
    .has_tensor_core = true,
    .has_sparsity = false
};

constexpr MetaXSpecs C550_PRO_SPECS = {
    .sm_count = 140,
    .warp_size = 64,
    .max_threads_per_block = 1024,
    .max_warps_per_sm = 32,
    .shared_memory_kb = 96,
    .registers_per_sm = 131072,
    .hbm_memory_gb = 96,
    .memory_bw_gbps = 2200,
    .fp32_tflops = 150,
    .fp16_tflops = 300,
    .has_tensor_core = true,
    .has_sparsity = false
};

// C600 (2024) - Second generation
constexpr MetaXSpecs C600_STANDARD_SPECS = {
    .sm_count = 128,
    .warp_size = 64,
    .max_threads_per_block = 1024,
    .max_warps_per_sm = 48,        // Increased occupancy
    .shared_memory_kb = 128,       // Doubled
    .registers_per_sm = 262144,    // Doubled
    .hbm_memory_gb = 96,
    .memory_bw_gbps = 2400,
    .fp32_tflops = 160,
    .fp16_tflops = 320,
    .has_tensor_core = true,
    .has_sparsity = true           // New!
};

constexpr MetaXSpecs C600_PRO_SPECS = {
    .sm_count = 160,
    .warp_size = 64,
    .max_threads_per_block = 1024,
    .max_warps_per_sm = 48,
    .shared_memory_kb = 128,
    .registers_per_sm = 262144,
    .hbm_memory_gb = 128,
    .memory_bw_gbps = 3000,
    .fp32_tflops = 200,
    .fp16_tflops = 400,
    .has_tensor_core = true,
    .has_sparsity = true
};

// C650 (2025) - Enhanced second generation
constexpr MetaXSpecs C650_STANDARD_SPECS = {
    .sm_count = 144,
    .warp_size = 64,
    .max_threads_per_block = 1024,
    .max_warps_per_sm = 48,
    .shared_memory_kb = 128,
    .registers_per_sm = 262144,
    .hbm_memory_gb = 128,
    .memory_bw_gbps = 2800,
    .fp32_tflops = 180,
    .fp16_tflops = 360,
    .has_tensor_core = true,
    .has_sparsity = true
};

constexpr MetaXSpecs C650_PRO_SPECS = {
    .sm_count = 180,
    .warp_size = 64,
    .max_threads_per_block = 1024,
    .max_warps_per_sm = 48,
    .shared_memory_kb = 128,
    .registers_per_sm = 262144,
    .hbm_memory_gb = 192,
    .memory_bw_gbps = 3600,
    .fp32_tflops = 225,
    .fp16_tflops = 450,
    .has_tensor_core = true,
    .has_sparsity = true
};

// C700 (2025+) - Third generation
constexpr MetaXSpecs C700_STANDARD_SPECS = {
    .sm_count = 160,
    .warp_size = 64,
    .max_threads_per_block = 2048,  // Increased!
    .max_warps_per_sm = 64,         // Doubled
    .shared_memory_kb = 192,        // Increased
    .registers_per_sm = 524288,     // Doubled
    .hbm_memory_gb = 192,
    .memory_bw_gbps = 4000,
    .fp32_tflops = 250,
    .fp16_tflops = 500,
    .has_tensor_core = true,
    .has_sparsity = true
};

constexpr MetaXSpecs C700_PRO_SPECS = {
    .sm_count = 200,
    .warp_size = 64,
    .max_threads_per_block = 2048,
    .max_warps_per_sm = 64,
    .shared_memory_kb = 192,
    .registers_per_sm = 524288,
    .hbm_memory_gb = 256,
    .memory_bw_gbps = 5000,
    .fp32_tflops = 320,
    .fp16_tflops = 640,
    .has_tensor_core = true,
    .has_sparsity = true
};

// =============================================================================
// Kernel Configuration by Generation
// =============================================================================

struct MacaKernelConfig {
    int block_size;             // Threads per block
    int warps_per_block;        // Warps per block
    int tile_size_m;            // GEMM M tile size
    int tile_size_n;            // GEMM N tile size
    int tile_size_k;            // GEMM K tile size
    int attention_tile_size;    // Attention tile size
    bool use_tensor_core;       // Use tensor cores
    bool use_sparsity;          // Use sparsity acceleration
};

constexpr MacaKernelConfig C500_KERNEL_CONFIG = {
    .block_size = 256,
    .warps_per_block = 4,       // 256 / 64
    .tile_size_m = 64,
    .tile_size_n = 64,
    .tile_size_k = 32,
    .attention_tile_size = 64,
    .use_tensor_core = true,
    .use_sparsity = false
};

constexpr MacaKernelConfig C550_KERNEL_CONFIG = {
    .block_size = 256,
    .warps_per_block = 4,
    .tile_size_m = 64,
    .tile_size_n = 128,
    .tile_size_k = 32,
    .attention_tile_size = 128,
    .use_tensor_core = true,
    .use_sparsity = false
};

constexpr MacaKernelConfig C600_KERNEL_CONFIG = {
    .block_size = 384,
    .warps_per_block = 6,
    .tile_size_m = 128,
    .tile_size_n = 128,
    .tile_size_k = 64,
    .attention_tile_size = 128,
    .use_tensor_core = true,
    .use_sparsity = true
};

constexpr MacaKernelConfig C650_KERNEL_CONFIG = {
    .block_size = 512,
    .warps_per_block = 8,
    .tile_size_m = 128,
    .tile_size_n = 256,
    .tile_size_k = 64,
    .attention_tile_size = 256,
    .use_tensor_core = true,
    .use_sparsity = true
};

constexpr MacaKernelConfig C700_KERNEL_CONFIG = {
    .block_size = 1024,
    .warps_per_block = 16,
    .tile_size_m = 256,
    .tile_size_n = 256,
    .tile_size_k = 64,
    .attention_tile_size = 512,
    .use_tensor_core = true,
    .use_sparsity = true
};

// =============================================================================
// Constants
// =============================================================================

// MACA uses 64-thread warps (NOT 32 like NVIDIA)
constexpr int MACA_WARP_SIZE = 64;
constexpr int MACA_MAX_THREADS_PER_BLOCK_C500 = 1024;
constexpr int MACA_MAX_THREADS_PER_BLOCK_C700 = 2048;
constexpr int MACA_DEFAULT_BLOCK_SIZE = 256;

// =============================================================================
// Runtime Detection Functions
// =============================================================================

/**
 * @brief Detect MetaX GPU generation at runtime
 * @return MetaXGen enum value
 */
MetaXGen detect_metax_gen();

/**
 * @brief Detect MetaX GPU variant (Standard/Pro/Max/Ultra)
 * @return MetaXVariant enum value
 */
MetaXVariant detect_metax_variant();

/**
 * @brief Get hardware specs for current device
 * @return MetaXSpecs for the detected hardware
 */
const MetaXSpecs& get_device_specs();

/**
 * @brief Get optimal kernel config for current device
 * @return MacaKernelConfig for the detected hardware
 */
const MacaKernelConfig& get_kernel_config();

}  // namespace maca
}  // namespace persistent_kernel
}  // namespace yirage
