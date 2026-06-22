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
 * @brief Master header for AMD ROCm/HIP persistent kernel tasks
 *
 * This file provides the main dispatch mechanism for selecting
 * architecture-optimized kernels based on the detected AMD GPU.
 *
 * Supported architectures:
 * - MI100: CDNA1 (gfx908) - 120 CUs, 32GB HBM2
 * - MI200/MI210: CDNA2 single-die (gfx90a) - 104 CUs, 64GB HBM2e
 * - MI250/MI250X: CDNA2 dual-die (gfx90a) - 220 CUs, 128GB HBM2e
 * - MI300X: CDNA3 (gfx942) - 304 CUs, 192GB HBM3, FP8, Sparsity
 * - MI300A: CDNA3 APU (gfx942) - 228 CUs, unified memory
 * - MI325X: CDNA3+ (gfx942) - Enhanced MI300X
 * - MI350: CDNA4 (future)
 *
 * Key differences from CUDA:
 * - 64-thread wavefronts (not 32-thread warps)
 * - LDS (Local Data Share) instead of shared memory
 * - MFMA instead of Tensor Cores
 */

// Common definitions
#include "common/rocm_common.h"
#include "common/rocm_detection.h"

// Architecture-specific kernels
#include "mi100/task_header.h"
#include "mi200/task_header.h"
#include "mi250/task_header.h"
#include "mi300/task_header.h"

namespace yirage {
namespace persistent_kernel {
namespace rocm {

// =============================================================================
// Dynamic Kernel Source Selection
// =============================================================================

/**
 * @brief Get optimized kernel source for the specified AMD architecture
 * @param arch AMD GPU architecture
 * @return Pointer to kernel source string
 */
inline const char* get_optimized_kernel_source(AMDArch arch) {
    switch (arch) {
        case AMDArch::MI350:
        case AMDArch::MI325X:
        case AMDArch::MI300X:
        case AMDArch::MI300A:
            return mi300::MI300_KERNEL_SOURCE;
        case AMDArch::MI250:
            return mi250::MI250_KERNEL_SOURCE;
        case AMDArch::MI200:
            return mi200::MI200_KERNEL_SOURCE;
        case AMDArch::MI100:
        default:
            return mi100::MI100_KERNEL_SOURCE;
    }
}

/**
 * @brief Get optimized kernel source for current device
 * @return Pointer to kernel source string
 */
inline const char* get_optimized_kernel_source() {
    return get_optimized_kernel_source(detect_amd_arch());
}

// =============================================================================
// Kernel Configuration Selection
// =============================================================================

/**
 * @brief Get kernel configuration for specified architecture
 * @param arch AMD GPU architecture
 * @return Kernel configuration
 */
inline ROCmKernelConfig get_kernel_config_for_arch(AMDArch arch) {
    switch (arch) {
        case AMDArch::MI350:
            return MI350_KERNEL_CONFIG;
        case AMDArch::MI325X:
        case AMDArch::MI300X:
        case AMDArch::MI300A:
            return MI300X_KERNEL_CONFIG;
        case AMDArch::MI250:
            return MI250_KERNEL_CONFIG;
        case AMDArch::MI200:
        case AMDArch::MI100:
        default:
            return MI100_KERNEL_CONFIG;
    }
}

/**
 * @brief Get kernel configuration for current device
 * @return Kernel configuration
 */
inline ROCmKernelConfig get_kernel_config() {
    return get_kernel_config_for_arch(detect_amd_arch());
}

// =============================================================================
// Hardware Specs Selection
// =============================================================================

/**
 * @brief Get hardware specs for specified architecture
 * @param arch AMD GPU architecture
 * @return Hardware specifications
 */
inline AMDSpecs get_specs_for_arch(AMDArch arch) {
    switch (arch) {
        case AMDArch::MI350:
            return MI350_SPECS;
        case AMDArch::MI325X:
            return MI325X_SPECS;
        case AMDArch::MI300X:
            return MI300X_SPECS;
        case AMDArch::MI300A:
            return MI300A_SPECS;
        case AMDArch::MI250:
            return MI250X_SPECS;
        case AMDArch::MI200:
            return MI200_SPECS;
        case AMDArch::MI100:
        default:
            return MI100_SPECS;
    }
}

/**
 * @brief Get hardware specs for current device
 * @return Hardware specifications
 */
inline AMDSpecs get_device_specs() {
    return get_specs_for_arch(detect_amd_arch());
}

// =============================================================================
// Feature Query Functions
// =============================================================================

/**
 * @brief Check if MFMA (Matrix Core) is supported
 * @param arch AMD GPU architecture
 * @return true if MFMA is supported
 */
inline bool has_mfma(AMDArch arch) {
    return is_cdna(arch);
}

/**
 * @brief Check MFMA support for current device
 */
inline bool has_mfma() {
    return has_mfma(detect_amd_arch());
}

/**
 * @brief Get MFMA tile dimensions
 * @param arch AMD GPU architecture
 * @param[out] m M dimension
 * @param[out] n N dimension
 * @param[out] k K dimension
 */
inline void get_mfma_dims(AMDArch arch, int& m, int& n, int& k) {
    AMDSpecs specs = get_specs_for_arch(arch);
    m = specs.mfma_m;
    n = specs.mfma_n;
    k = specs.mfma_k;
}

/**
 * @brief Get LDS size in KB
 * @param arch AMD GPU architecture
 * @return LDS size in KB per CU
 */
inline int get_lds_kb(AMDArch arch) {
    switch (arch) {
        case AMDArch::MI350:
            return 128;
        default:
            return 64;
    }
}

/**
 * @brief Get LDS size for current device
 */
inline int get_lds_kb() {
    return get_lds_kb(detect_amd_arch());
}

/**
 * @brief Get wavefront size (always 64 for CDNA)
 * @return Wavefront size
 */
constexpr int get_wavefront_size() {
    return ROCM_WAVEFRONT_SIZE;
}

/**
 * @brief Get optimal block size
 * @param arch AMD GPU architecture
 * @return Optimal block size (threads)
 */
inline int get_optimal_block_size(AMDArch arch) {
    ROCmKernelConfig config = get_kernel_config_for_arch(arch);
    return config.block_size;
}

/**
 * @brief Get optimal block size for current device
 */
inline int get_optimal_block_size() {
    return get_optimal_block_size(detect_amd_arch());
}

// =============================================================================
// Kernel Selection
// =============================================================================

/**
 * @brief Select appropriate GEMM kernel name
 * @param arch AMD GPU architecture
 * @return Kernel function name
 */
inline const char* select_gemm_kernel(AMDArch arch) {
    switch (arch) {
        case AMDArch::MI350:
        case AMDArch::MI325X:
        case AMDArch::MI300X:
        case AMDArch::MI300A:
            return "gemm_mi300";
        case AMDArch::MI250:
            return "gemm_mi250";
        case AMDArch::MI200:
            return "gemm_mi200";
        case AMDArch::MI100:
        default:
            return "gemm_mi100";
    }
}

/**
 * @brief Select sparse GEMM kernel if supported
 * @param arch AMD GPU architecture
 * @return Kernel name or nullptr if not supported
 */
inline const char* select_sparse_gemm_kernel(AMDArch arch) {
    if (has_sparsity(arch)) {
        return "sparse_gemm_mi300";
    }
    return nullptr;
}

/**
 * @brief Select FP8 GEMM kernel if supported
 * @param arch AMD GPU architecture
 * @return Kernel name or nullptr if not supported
 */
inline const char* select_fp8_gemm_kernel(AMDArch arch) {
    if (has_fp8(arch)) {
        return "gemm_fp8_mi300";
    }
    return nullptr;
}

/**
 * @brief Select appropriate RMSNorm kernel
 * @param arch AMD GPU architecture
 * @param batched Use batched version
 * @return Kernel function name
 */
inline const char* select_rmsnorm_kernel(AMDArch arch, bool batched = false) {
    if (batched && (arch == AMDArch::MI300X || arch == AMDArch::MI300A ||
                    arch == AMDArch::MI325X || arch == AMDArch::MI350)) {
        return "rms_norm_batched_mi300";
    }
    
    switch (arch) {
        case AMDArch::MI250:
            return "rms_norm_mi250";
        case AMDArch::MI200:
            return "rms_norm_mi200";
        case AMDArch::MI100:
        default:
            return "rms_norm_mi100";
    }
}

/**
 * @brief Select appropriate Flash Attention kernel
 * @param arch AMD GPU architecture
 * @return Kernel name or nullptr if not optimized
 */
inline const char* select_flash_attention_kernel(AMDArch arch) {
    switch (arch) {
        case AMDArch::MI350:
        case AMDArch::MI325X:
        case AMDArch::MI300X:
        case AMDArch::MI300A:
            return "flash_attention_mi300";
        case AMDArch::MI250:
            return "flash_attention_mi250";
        default:
            return nullptr;  // Use generic attention
    }
}

/**
 * @brief Get GFX target for HIP compilation
 * @param arch AMD GPU architecture
 * @return GFX target string
 */
inline const char* get_gfx_target(AMDArch arch) {
    return arch_to_gfx(arch);
}

}  // namespace rocm
}  // namespace persistent_kernel
}  // namespace yirage
