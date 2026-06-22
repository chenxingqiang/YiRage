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
 * @file task_header.h
 * @brief Master header for Ascend NPU persistent kernel tasks
 *
 * This file provides the main dispatch mechanism for selecting
 * model-optimized kernels based on the detected Ascend NPU.
 *
 * Supported models:
 * - Ascend 310:  Edge inference (2 AI Cores, 128KB L1)
 * - Ascend 310P: Enhanced inference (8 AI Cores, 256KB L1, BF16)
 * - Ascend 910:  Training (32 AI Cores, 256KB L1, FP32 Cube)
 * - Ascend 910B: Enhanced training (32 AI Cores, 512KB L1, BF16, INT4)
 * - Ascend 910C: Latest training (48 AI Cores, 1MB L1, 32x32 Cube)
 *
 * Key concepts:
 * - AI Core: Tensor processing unit with Cube and Vector units
 * - Cube Unit: Native 16x16 (or 32x32 on 910C) matrix multiply
 * - L1 Buffer: High-speed on-chip memory per AI Core
 * - TBE: Tensor Boost Engine for custom operators
 */

// Common definitions
#include "common/ascend_common.h"
#include "common/ascend_detection.h"

// Model-specific kernels
#include "ascend310/task_header.h"
#include "ascend310p/task_header.h"
#include "ascend910/task_header.h"
#include "ascend910b/task_header.h"
#include "ascend910c/task_header.h"

namespace yirage {
namespace persistent_kernel {
namespace ascend {

// =============================================================================
// Dynamic Kernel Source Selection
// =============================================================================

/**
 * @brief Get optimized kernel source for the specified Ascend model
 * @param model Ascend NPU model
 * @return Pointer to kernel source string
 */
inline const char* get_optimized_kernel_source(AscendModel model) {
    switch (model) {
        case AscendModel::ASCEND_910C:
            return ascend910c::ASCEND910C_KERNEL_SOURCE;
        case AscendModel::ASCEND_910B:
            return ascend910b::ASCEND910B_KERNEL_SOURCE;
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
            return ascend910::ASCEND910_KERNEL_SOURCE;
        case AscendModel::ASCEND_310P:
            return ascend310p::ASCEND310P_KERNEL_SOURCE;
        case AscendModel::ASCEND_310:
        default:
            return ascend310::ASCEND310_KERNEL_SOURCE;
    }
}

/**
 * @brief Get optimized kernel source for current device
 * @return Pointer to kernel source string
 */
inline const char* get_optimized_kernel_source() {
    return get_optimized_kernel_source(detect_ascend_model());
}

// =============================================================================
// Kernel Configuration Selection
// =============================================================================

/**
 * @brief Get kernel configuration for specified model
 * @param model Ascend NPU model
 * @return Kernel configuration
 */
inline AscendKernelConfig get_kernel_config_for_model(AscendModel model) {
    switch (model) {
        case AscendModel::ASCEND_910C:
            return ASCEND_910C_KERNEL_CONFIG;
        case AscendModel::ASCEND_910B:
            return ASCEND_910B_KERNEL_CONFIG;
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
            return ASCEND_910_KERNEL_CONFIG;
        case AscendModel::ASCEND_310P:
            return ASCEND_310P_KERNEL_CONFIG;
        case AscendModel::ASCEND_310:
        default:
            return ASCEND_310_KERNEL_CONFIG;
    }
}

/**
 * @brief Get kernel configuration for current device
 * @return Kernel configuration
 */
inline AscendKernelConfig get_kernel_config() {
    return get_kernel_config_for_model(detect_ascend_model());
}

// =============================================================================
// Hardware Specs Selection
// =============================================================================

/**
 * @brief Get hardware specs for specified model
 * @param model Ascend NPU model
 * @return Hardware specifications
 */
inline AscendSpecs get_specs_for_model(AscendModel model) {
    switch (model) {
        case AscendModel::ASCEND_910C:
            return ASCEND_910C_SPECS;
        case AscendModel::ASCEND_910B:
            return ASCEND_910B_SPECS;
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
            return ASCEND_910_SPECS;
        case AscendModel::ASCEND_310P:
            return ASCEND_310P_SPECS;
        case AscendModel::ASCEND_310:
        default:
            return ASCEND_310_SPECS;
    }
}

/**
 * @brief Get hardware specs for current device
 * @return Hardware specifications
 */
inline AscendSpecs get_device_specs() {
    return get_specs_for_model(detect_ascend_model());
}

// =============================================================================
// Feature Query Functions
// =============================================================================

/**
 * @brief Check if BF16 is supported
 * @param model Ascend NPU model
 * @return true if BF16 is supported
 */
inline bool has_bf16(AscendModel model) {
    return (model == AscendModel::ASCEND_310P ||
            model == AscendModel::ASCEND_910B ||
            model == AscendModel::ASCEND_910C);
}

/**
 * @brief Check BF16 support for current device
 */
inline bool has_bf16() {
    return has_bf16(detect_ascend_model());
}

/**
 * @brief Check if INT4 quantization is supported
 * @param model Ascend NPU model
 * @return true if INT4 is supported
 */
inline bool has_int4(AscendModel model) {
    return (model == AscendModel::ASCEND_910B ||
            model == AscendModel::ASCEND_910C);
}

/**
 * @brief Check INT4 support for current device
 */
inline bool has_int4() {
    return has_int4(detect_ascend_model());
}

/**
 * @brief Check if 32x32 Cube is available
 * @param model Ascend NPU model
 * @return true if 32x32 Cube is supported
 */
inline bool has_large_cube(AscendModel model) {
    return (model == AscendModel::ASCEND_910C);
}

/**
 * @brief Check 32x32 Cube for current device
 */
inline bool has_large_cube() {
    return has_large_cube(detect_ascend_model());
}

/**
 * @brief Get L1 buffer size in KB
 * @param model Ascend NPU model
 * @return L1 buffer size in KB
 */
inline int get_l1_buffer_kb(AscendModel model) {
    switch (model) {
        case AscendModel::ASCEND_910C:
            return 1024;
        case AscendModel::ASCEND_910B:
            return 512;
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
        case AscendModel::ASCEND_310P:
            return 256;
        case AscendModel::ASCEND_310:
        default:
            return 128;
    }
}

/**
 * @brief Get L1 buffer for current device
 */
inline int get_l1_buffer_kb() {
    return get_l1_buffer_kb(detect_ascend_model());
}

/**
 * @brief Get AI Core count
 * @param model Ascend NPU model
 * @return Number of AI Cores
 */
inline int get_ai_cores(AscendModel model) {
    switch (model) {
        case AscendModel::ASCEND_910C:
            return 48;
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
        case AscendModel::ASCEND_910B:
            return 32;
        case AscendModel::ASCEND_310P:
            return 8;
        case AscendModel::ASCEND_310:
        default:
            return 2;
    }
}

/**
 * @brief Get AI Core count for current device
 */
inline int get_ai_cores() {
    return get_ai_cores(detect_ascend_model());
}

/**
 * @brief Get Cube unit size (16 or 32)
 * @param model Ascend NPU model
 * @return Cube unit dimension
 */
inline int get_cube_size(AscendModel model) {
    if (model == AscendModel::ASCEND_910C) {
        return 32;
    }
    return 16;
}

/**
 * @brief Get Cube size for current device
 */
inline int get_cube_size() {
    return get_cube_size(detect_ascend_model());
}

// =============================================================================
// Kernel Selection Based on Model
// =============================================================================

/**
 * @brief Select appropriate MatMul kernel name based on model
 * @param model Ascend NPU model
 * @return Kernel function name
 */
inline const char* select_matmul_kernel(AscendModel model) {
    switch (model) {
        case AscendModel::ASCEND_910C:
            return "matmul_910c";
        case AscendModel::ASCEND_910B:
            return "matmul_910b";
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
            return "matmul_910";
        case AscendModel::ASCEND_310P:
            return "matmul_310p";
        case AscendModel::ASCEND_310:
        default:
            return "matmul_310";
    }
}

/**
 * @brief Select appropriate RMSNorm kernel name
 * @param model Ascend NPU model
 * @return Kernel function name
 */
inline const char* select_rmsnorm_kernel(AscendModel model) {
    switch (model) {
        case AscendModel::ASCEND_910C:
            return "rms_norm_910c";
        case AscendModel::ASCEND_910B:
            return "rms_norm_910b";
        case AscendModel::ASCEND_910:
        case AscendModel::ASCEND_910A:
            return "rms_norm_910";
        case AscendModel::ASCEND_310P:
            return "rms_norm_310p";
        case AscendModel::ASCEND_310:
        default:
            return "rms_norm_310";
    }
}

/**
 * @brief Select appropriate Flash Attention kernel name
 * @param model Ascend NPU model
 * @return Kernel function name or nullptr if not supported
 */
inline const char* select_flash_attention_kernel(AscendModel model) {
    switch (model) {
        case AscendModel::ASCEND_910C:
            return "flash_attn_910c";
        case AscendModel::ASCEND_910B:
            return "flash_attn_910b";
        case AscendModel::ASCEND_310P:
            return "flash_attn_310p";
        default:
            return nullptr;  // Not supported on 310/910
    }
}

/**
 * @brief Select appropriate Fused MLP kernel name
 * @param model Ascend NPU model
 * @return Kernel function name or nullptr if not supported
 */
inline const char* select_fused_mlp_kernel(AscendModel model) {
    if (model == AscendModel::ASCEND_910C) {
        return "fused_mlp_910c";
    }
    return nullptr;  // Only supported on 910C
}

/**
 * @brief Check if model is training-focused (900 series)
 * @param model Ascend NPU model
 * @return true if training-focused
 */
inline bool is_training_model(AscendModel model) {
    return (model == AscendModel::ASCEND_910 ||
            model == AscendModel::ASCEND_910A ||
            model == AscendModel::ASCEND_910B ||
            model == AscendModel::ASCEND_910C);
}

/**
 * @brief Check if model is inference-focused (300 series)
 * @param model Ascend NPU model
 * @return true if inference-focused
 */
inline bool is_inference_model(AscendModel model) {
    return (model == AscendModel::ASCEND_310 ||
            model == AscendModel::ASCEND_310P);
}

}  // namespace ascend
}  // namespace persistent_kernel
}  // namespace yirage
