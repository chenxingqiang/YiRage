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
 * @brief Master header for MPS (Metal Performance Shaders) task kernels
 *
 * This file provides architecture-optimized kernels for all Apple Silicon generations.
 * Each generation has specific optimizations leveraging its hardware features.
 *
 * Architecture Support:
 * - M1 (2020): Base Apple Silicon, 32KB shared mem, no simdgroup_matrix
 * - M2 (2022): Better bandwidth, improved half-precision
 * - M3 (2023): Ray tracing, mesh shaders, simdgroup_matrix, dynamic caching
 * - M4 (2024): 48KB shared mem, 38 TOPS Neural Engine
 * - M5 (2025+): 64KB shared mem, enhanced capabilities
 *
 * Usage:
 *   auto gen = detect_apple_silicon_gen();
 *   const char* source = get_optimized_kernel_source(gen);
 *   // Compile with MTLDevice.newLibraryWithSource()
 */

// Common definitions
#include "common/mps_common.h"

// Architecture-specific kernels
#include "m1/task_header.h"
#include "m2/task_header.h"
#include "m3/task_header.h"
#include "m4/task_header.h"
#include "m5/task_header.h"

// Generic fallback kernels
#include "embedding.metal.h"
#include "rmsnorm.metal.h"
#include "silu_mul.metal.h"
#include "argmax.metal.h"
#include "linear.metal.h"
#include "attention.metal.h"
#include "rotary_embedding.metal.h"
#include "softmax.metal.h"

namespace yirage {
namespace persistent_kernel {
namespace mps {

// =============================================================================
// Kernel Source Dispatch
// =============================================================================

/**
 * @brief Get optimized kernel source for detected Apple Silicon generation
 * @param gen Apple Silicon generation
 * @return Pointer to Metal kernel source string
 */
inline const char* get_optimized_kernel_source(AppleSiliconGen gen) {
    switch (gen) {
        case AppleSiliconGen::M1:
            return m1::M1_KERNEL_SOURCE;
        case AppleSiliconGen::M2:
            return m2::M2_KERNEL_SOURCE;
        case AppleSiliconGen::M3:
            return m3::M3_KERNEL_SOURCE;
        case AppleSiliconGen::M4:
            return m4::M4_KERNEL_SOURCE;
        case AppleSiliconGen::M5:
            return m5::M5_KERNEL_SOURCE;
        default:
            // Fallback to M1 (most conservative)
            return m1::M1_KERNEL_SOURCE;
    }
}

/**
 * @brief Get kernel configuration for detected hardware
 * @param gen Apple Silicon generation
 * @return MpsKernelConfig for optimal kernel launch parameters
 */
inline MpsKernelConfig get_kernel_config_for_gen(AppleSiliconGen gen) {
    switch (gen) {
        case AppleSiliconGen::M1:
            return M1_KERNEL_CONFIG;
        case AppleSiliconGen::M2:
            return M2_KERNEL_CONFIG;
        case AppleSiliconGen::M3:
            return M3_KERNEL_CONFIG;
        case AppleSiliconGen::M4:
            return M4_KERNEL_CONFIG;
        case AppleSiliconGen::M5:
            return M5_KERNEL_CONFIG;
        default:
            return M1_KERNEL_CONFIG;
    }
}

/**
 * @brief Get hardware specs for detected hardware
 * @param gen Apple Silicon generation
 * @param variant Chip variant (Base/Pro/Max/Ultra)
 * @return AppleSiliconSpecs for the hardware
 */
inline AppleSiliconSpecs get_specs_for_gen(AppleSiliconGen gen, 
                                           AppleSiliconVariant variant = AppleSiliconVariant::BASE) {
    switch (gen) {
        case AppleSiliconGen::M1:
            switch (variant) {
                case AppleSiliconVariant::PRO:  return M1_PRO_SPECS;
                case AppleSiliconVariant::MAX:  return M1_MAX_SPECS;
                default:                        return M1_BASE_SPECS;
            }
        case AppleSiliconGen::M2:
            switch (variant) {
                case AppleSiliconVariant::PRO:  return M2_PRO_SPECS;
                case AppleSiliconVariant::MAX:  return M2_MAX_SPECS;
                default:                        return M2_BASE_SPECS;
            }
        case AppleSiliconGen::M3:
            switch (variant) {
                case AppleSiliconVariant::PRO:  return M3_PRO_SPECS;
                case AppleSiliconVariant::MAX:  return M3_MAX_SPECS;
                default:                        return M3_BASE_SPECS;
            }
        case AppleSiliconGen::M4:
            switch (variant) {
                case AppleSiliconVariant::PRO:  return M4_PRO_SPECS;
                case AppleSiliconVariant::MAX:  return M4_MAX_SPECS;
                default:                        return M4_BASE_SPECS;
            }
        case AppleSiliconGen::M5:
            switch (variant) {
                case AppleSiliconVariant::PRO:  return M5_PRO_SPECS;
                case AppleSiliconVariant::MAX:  return M5_MAX_SPECS;
                default:                        return M5_BASE_SPECS;
            }
        default:
            return M1_BASE_SPECS;
    }
}

/**
 * @brief Check if simdgroup_matrix is available
 * @param gen Apple Silicon generation
 * @return true if simdgroup_matrix operations are supported (M3+)
 */
inline bool has_simdgroup_matrix(AppleSiliconGen gen) {
    return gen >= AppleSiliconGen::M3;
}

/**
 * @brief Get maximum shared memory in KB
 * @param gen Apple Silicon generation
 * @return Shared memory size in KB
 */
inline int get_shared_memory_kb(AppleSiliconGen gen) {
    switch (gen) {
        case AppleSiliconGen::M5: return 64;
        case AppleSiliconGen::M4: return 48;
        default:                  return 32;
    }
}

/**
 * @brief Get optimal threadgroup size for generation
 * @param gen Apple Silicon generation
 * @return Recommended threadgroup size
 */
inline int get_optimal_threadgroup_size(AppleSiliconGen gen) {
    switch (gen) {
        case AppleSiliconGen::M5: return 512;
        case AppleSiliconGen::M4: return 384;
        default:                  return 256;
    }
}

// =============================================================================
// Combined Kernel Sources for Fallback
// =============================================================================

/**
 * @brief Get all generic MPS kernel source (not architecture-optimized)
 * @return Combined Metal kernel source string
 */
inline const char* get_generic_kernel_source() {
    static const char* combined = 
        EMBEDDING_KERNEL_SOURCE
        RMSNORM_KERNEL_SOURCE
        SILU_MUL_KERNEL_SOURCE
        ARGMAX_KERNEL_SOURCE
        LINEAR_KERNEL_SOURCE
        ATTENTION_KERNEL_SOURCE
        ROTARY_EMBEDDING_KERNEL_SOURCE
        SOFTMAX_KERNEL_SOURCE;
    return combined;
}

}  // namespace mps
}  // namespace persistent_kernel
}  // namespace yirage
