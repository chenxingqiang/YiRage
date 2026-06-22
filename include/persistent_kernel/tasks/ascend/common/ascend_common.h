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
 * @file ascend_common.h
 * @brief Common definitions for Huawei Ascend NPU kernels
 *
 * Huawei Ascend NPU Architecture:
 * - AI Cores: Specialized tensor processing units
 *   - Cube Unit: 16x16 matrix multiplication
 *   - Vector Unit: Element-wise operations
 *   - Scalar Unit: Control flow
 * - L1 Buffer: High-speed on-chip memory (128KB-1MB)
 * - L0 Buffer: Register file for Cube/Vector
 * - HBM: High-bandwidth memory (8GB-64GB)
 *
 * Programming Model:
 * - CANN (Compute Architecture for Neural Networks)
 * - TBE (Tensor Boost Engine) for custom operators
 * - AscendCL runtime API
 */

namespace yirage {
namespace persistent_kernel {
namespace ascend {

// =============================================================================
// Ascend NPU Model Detection
// =============================================================================

enum class AscendModel {
    ASCEND_310 = 310,       // Inference chip (2019)
    ASCEND_310P = 3101,     // Enhanced 310 (2021)
    ASCEND_910 = 910,       // Training chip (2019)
    ASCEND_910A = 9101,     // First revision
    ASCEND_910B = 9102,     // Second revision (2022)
    ASCEND_910C = 9103,     // Third revision (2024)
    UNKNOWN = 0
};

enum class AscendSeries {
    SERIES_300,     // Inference-focused (310, 310P)
    SERIES_900,     // Training-focused (910, 910B, 910C)
    UNKNOWN
};

// =============================================================================
// Hardware Specifications by Model
// =============================================================================

struct AscendSpecs {
    int ai_core_count;          // Number of AI Cores
    int cube_size;              // Cube unit matrix size (16 or 32)
    int vector_width;           // Vector unit width
    int l1_buffer_kb;           // L1 buffer per AI Core in KB
    int l0_buffer_kb;           // L0 buffer (register file) in KB
    int hbm_memory_gb;          // HBM memory in GB
    int memory_bw_gbps;         // Memory bandwidth in GB/s
    int fp16_tflops;            // FP16 performance in TFLOPS
    int int8_tops;              // INT8 performance in TOPS
    bool has_fp32_cube;         // FP32 Cube support
    bool has_bf16;              // BF16 support
    bool has_int4;              // INT4 quantization support
};

// Ascend 310 (2019) - Inference chip
constexpr AscendSpecs ASCEND_310_SPECS = {
    .ai_core_count = 2,
    .cube_size = 16,
    .vector_width = 128,
    .l1_buffer_kb = 128,
    .l0_buffer_kb = 64,
    .hbm_memory_gb = 8,
    .memory_bw_gbps = 60,
    .fp16_tflops = 8,
    .int8_tops = 16,
    .has_fp32_cube = false,
    .has_bf16 = false,
    .has_int4 = false
};

// Ascend 310P (2021) - Enhanced inference
constexpr AscendSpecs ASCEND_310P_SPECS = {
    .ai_core_count = 8,
    .cube_size = 16,
    .vector_width = 256,
    .l1_buffer_kb = 256,
    .l0_buffer_kb = 64,
    .hbm_memory_gb = 24,
    .memory_bw_gbps = 200,
    .fp16_tflops = 22,
    .int8_tops = 44,
    .has_fp32_cube = false,
    .has_bf16 = true,
    .has_int4 = false
};

// Ascend 910 (2019) - First training chip
constexpr AscendSpecs ASCEND_910_SPECS = {
    .ai_core_count = 32,
    .cube_size = 16,
    .vector_width = 256,
    .l1_buffer_kb = 256,
    .l0_buffer_kb = 64,
    .hbm_memory_gb = 32,
    .memory_bw_gbps = 1200,
    .fp16_tflops = 256,
    .int8_tops = 512,
    .has_fp32_cube = true,
    .has_bf16 = false,
    .has_int4 = false
};

// Ascend 910B (2022) - Enhanced training chip
constexpr AscendSpecs ASCEND_910B_SPECS = {
    .ai_core_count = 32,
    .cube_size = 16,
    .vector_width = 512,
    .l1_buffer_kb = 512,
    .l0_buffer_kb = 128,
    .hbm_memory_gb = 64,
    .memory_bw_gbps = 1600,
    .fp16_tflops = 320,
    .int8_tops = 640,
    .has_fp32_cube = true,
    .has_bf16 = true,
    .has_int4 = true
};

// Ascend 910C (2024) - Latest training chip
constexpr AscendSpecs ASCEND_910C_SPECS = {
    .ai_core_count = 48,
    .cube_size = 32,            // Enhanced Cube unit
    .vector_width = 512,
    .l1_buffer_kb = 1024,       // 1MB L1
    .l0_buffer_kb = 256,
    .hbm_memory_gb = 96,
    .memory_bw_gbps = 2400,
    .fp16_tflops = 500,
    .int8_tops = 1000,
    .has_fp32_cube = true,
    .has_bf16 = true,
    .has_int4 = true
};

// =============================================================================
// Kernel Configuration by Model
// =============================================================================

struct AscendKernelConfig {
    int ai_cores_per_block;     // AI Cores per execution block
    int cube_tile_m;            // Cube M dimension
    int cube_tile_n;            // Cube N dimension
    int cube_tile_k;            // Cube K dimension
    int vector_batch_size;      // Elements per vector operation
    int l1_tile_size_kb;        // L1 tile size in KB
    bool use_double_buffer;     // Enable L1 double buffering
    bool use_data_move_async;   // Async data movement
};

constexpr AscendKernelConfig ASCEND_310_KERNEL_CONFIG = {
    .ai_cores_per_block = 2,
    .cube_tile_m = 16,
    .cube_tile_n = 16,
    .cube_tile_k = 16,
    .vector_batch_size = 128,
    .l1_tile_size_kb = 32,
    .use_double_buffer = false,
    .use_data_move_async = false
};

constexpr AscendKernelConfig ASCEND_310P_KERNEL_CONFIG = {
    .ai_cores_per_block = 4,
    .cube_tile_m = 16,
    .cube_tile_n = 32,
    .cube_tile_k = 16,
    .vector_batch_size = 256,
    .l1_tile_size_kb = 64,
    .use_double_buffer = true,
    .use_data_move_async = true
};

constexpr AscendKernelConfig ASCEND_910_KERNEL_CONFIG = {
    .ai_cores_per_block = 8,
    .cube_tile_m = 16,
    .cube_tile_n = 16,
    .cube_tile_k = 16,
    .vector_batch_size = 256,
    .l1_tile_size_kb = 64,
    .use_double_buffer = true,
    .use_data_move_async = true
};

constexpr AscendKernelConfig ASCEND_910B_KERNEL_CONFIG = {
    .ai_cores_per_block = 8,
    .cube_tile_m = 32,
    .cube_tile_n = 32,
    .cube_tile_k = 16,
    .vector_batch_size = 512,
    .l1_tile_size_kb = 128,
    .use_double_buffer = true,
    .use_data_move_async = true
};

constexpr AscendKernelConfig ASCEND_910C_KERNEL_CONFIG = {
    .ai_cores_per_block = 12,
    .cube_tile_m = 32,
    .cube_tile_n = 64,
    .cube_tile_k = 32,
    .vector_batch_size = 512,
    .l1_tile_size_kb = 256,
    .use_double_buffer = true,
    .use_data_move_async = true
};

// =============================================================================
// Constants
// =============================================================================

// Cube unit native tile size
constexpr int ASCEND_CUBE_SIZE_16 = 16;
constexpr int ASCEND_CUBE_SIZE_32 = 32;

// Data alignment requirements
constexpr int ASCEND_DATA_ALIGN = 32;  // 32-byte alignment

// Memory copy block size
constexpr int ASCEND_DMA_BLOCK_SIZE = 512;

// =============================================================================
// Runtime Detection Functions
// =============================================================================

/**
 * @brief Detect Ascend NPU model at runtime
 * @return AscendModel enum value
 */
AscendModel detect_ascend_model();

/**
 * @brief Get Ascend series (300 or 900)
 * @return AscendSeries enum value
 */
AscendSeries detect_ascend_series();

/**
 * @brief Get hardware specs for current device
 * @return AscendSpecs for the detected hardware
 */
const AscendSpecs& get_device_specs();

/**
 * @brief Get optimal kernel config for current device
 * @return AscendKernelConfig for the detected hardware
 */
const AscendKernelConfig& get_kernel_config();

}  // namespace ascend
}  // namespace persistent_kernel
}  // namespace yirage
