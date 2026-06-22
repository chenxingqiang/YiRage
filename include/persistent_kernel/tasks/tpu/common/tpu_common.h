/* Copyright 2025 YiRage Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

/**
 * @file tpu_common.h
 * @brief Common definitions for TPU kernels across all TPU generations
 */

namespace yirage {
namespace persistent_kernel {
namespace tpu {

// =============================================================================
// TPU Version Detection
// =============================================================================

enum class TPUVersion {
    V2 = 2,      // TPU v2 (45 TFLOPS BF16)
    V3 = 3,      // TPU v3 (90 TFLOPS BF16)
    V4 = 4,      // TPU v4 (275 TFLOPS BF16)
    V5E = 5,     // TPU v5e (197 TFLOPS BF16, efficient)
    V5P = 6,     // TPU v5p (459 TFLOPS BF16, performance)
    UNKNOWN = 0
};

// =============================================================================
// Hardware Specifications by Generation
// =============================================================================

struct TPUSpecs {
    int mxu_size;               // MXU systolic array size (128x128)
    int num_cores;              // Number of TPU cores per chip
    size_t vmem_mb;             // Vector memory in MB per core
    size_t cmem_mb;             // Common memory in MB
    size_t hbm_gb;              // HBM capacity in GB
    int hbm_bw_gbps;            // HBM bandwidth in GB/s
    int bf16_tflops;            // Peak BF16 TFLOPS
    int int8_tops;              // Peak INT8 TOPS
    bool supports_sparsity;     // Sparse tensor support
    bool supports_int4;         // INT4 support
};

// TPU v2 (2017) - First cloud TPU
constexpr TPUSpecs TPU_V2_SPECS = {
    .mxu_size = 128,
    .num_cores = 2,
    .vmem_mb = 8,
    .cmem_mb = 2,
    .hbm_gb = 8,
    .hbm_bw_gbps = 600,
    .bf16_tflops = 45,
    .int8_tops = 90,
    .supports_sparsity = false,
    .supports_int4 = false
};

// TPU v3 (2018) - Liquid cooled
constexpr TPUSpecs TPU_V3_SPECS = {
    .mxu_size = 128,
    .num_cores = 2,
    .vmem_mb = 16,
    .cmem_mb = 4,
    .hbm_gb = 16,
    .hbm_bw_gbps = 900,
    .bf16_tflops = 90,
    .int8_tops = 180,
    .supports_sparsity = false,
    .supports_int4 = false
};

// TPU v4 (2021) - Major architecture update
constexpr TPUSpecs TPU_V4_SPECS = {
    .mxu_size = 128,
    .num_cores = 2,
    .vmem_mb = 32,
    .cmem_mb = 8,
    .hbm_gb = 32,
    .hbm_bw_gbps = 1200,
    .bf16_tflops = 275,
    .int8_tops = 550,
    .supports_sparsity = true,
    .supports_int4 = false
};

// TPU v5e (2023) - Efficient variant
constexpr TPUSpecs TPU_V5E_SPECS = {
    .mxu_size = 128,
    .num_cores = 1,
    .vmem_mb = 16,
    .cmem_mb = 4,
    .hbm_gb = 16,
    .hbm_bw_gbps = 800,
    .bf16_tflops = 197,
    .int8_tops = 394,
    .supports_sparsity = true,
    .supports_int4 = true
};

// TPU v5p (2023) - Performance variant
constexpr TPUSpecs TPU_V5P_SPECS = {
    .mxu_size = 128,
    .num_cores = 2,
    .vmem_mb = 48,
    .cmem_mb = 12,
    .hbm_gb = 95,
    .hbm_bw_gbps = 2760,
    .bf16_tflops = 459,
    .int8_tops = 918,
    .supports_sparsity = true,
    .supports_int4 = true
};

// =============================================================================
// Kernel Configuration by Generation
// =============================================================================

struct TPUKernelConfig {
    int tile_m;                 // GEMM M tile size
    int tile_n;                 // GEMM N tile size
    int tile_k;                 // GEMM K tile size
    int pipeline_depth;         // Pipeline stages
    bool use_double_buffering;  // Enable double buffering
    bool use_bf16;              // Use BF16 precision
    bool generate_xla;          // Generate XLA HLO
    bool generate_pallas;       // Generate Pallas kernel
};

// Default configurations per version
constexpr TPUKernelConfig TPU_V2_KERNEL_CONFIG = {
    .tile_m = 128,
    .tile_n = 128,
    .tile_k = 128,
    .pipeline_depth = 1,
    .use_double_buffering = false,
    .use_bf16 = true,
    .generate_xla = true,
    .generate_pallas = false
};

constexpr TPUKernelConfig TPU_V3_KERNEL_CONFIG = {
    .tile_m = 128,
    .tile_n = 256,
    .tile_k = 128,
    .pipeline_depth = 2,
    .use_double_buffering = true,
    .use_bf16 = true,
    .generate_xla = true,
    .generate_pallas = false
};

constexpr TPUKernelConfig TPU_V4_KERNEL_CONFIG = {
    .tile_m = 256,
    .tile_n = 256,
    .tile_k = 128,
    .pipeline_depth = 2,
    .use_double_buffering = true,
    .use_bf16 = true,
    .generate_xla = true,
    .generate_pallas = true
};

constexpr TPUKernelConfig TPU_V5E_KERNEL_CONFIG = {
    .tile_m = 128,
    .tile_n = 256,
    .tile_k = 128,
    .pipeline_depth = 2,
    .use_double_buffering = true,
    .use_bf16 = true,
    .generate_xla = true,
    .generate_pallas = true
};

constexpr TPUKernelConfig TPU_V5P_KERNEL_CONFIG = {
    .tile_m = 256,
    .tile_n = 512,
    .tile_k = 128,
    .pipeline_depth = 4,
    .use_double_buffering = true,
    .use_bf16 = true,
    .generate_xla = true,
    .generate_pallas = true
};

// =============================================================================
// Runtime Detection Functions
// =============================================================================

TPUVersion detect_tpu_version();
const TPUSpecs& get_tpu_specs(TPUVersion version);
const TPUKernelConfig& get_tpu_kernel_config(TPUVersion version);

}  // namespace tpu
}  // namespace persistent_kernel
}  // namespace yirage
