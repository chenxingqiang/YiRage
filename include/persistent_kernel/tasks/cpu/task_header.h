/* Copyright 2025 YiRage Team */
#pragma once

#include "common/cpu_common.h"
#include "common/cpu_detection.h"

/**
 * @file task_header.h
 * @brief Main include file for CPU persistent kernel tasks
 * 
 * CPU kernels use:
 * - SIMD intrinsics (SSE/AVX/AVX-512/NEON/SVE)
 * - OpenMP for parallelization
 * - Cache-optimized tiling
 */

namespace yirage {
namespace persistent_kernel {
namespace cpu {

enum class CPUTaskType {
    GEMM,
    ATTENTION,
    SOFTMAX,
    RMS_NORM,
    EMBEDDING,
    ARGMAX,
    SILU_MUL,
    REDUCTION
};

struct CPUTaskDesc {
    CPUTaskType type;
    CPUArch target_arch;
    int m, n, k;
    int num_threads;
    bool use_bf16;
    int num_attention_heads;
    int head_dim;
};

inline CPUKernelConfig get_optimal_config(CPUTaskDesc const& task) {
    switch (task.target_arch) {
        case CPUArch::X86_AVX2:
        case CPUArch::X86_AVX:
            return AVX2_KERNEL_CONFIG;
        case CPUArch::X86_AVX512:
        case CPUArch::X86_AVX512_BF16:
            return AVX512_KERNEL_CONFIG;
        case CPUArch::X86_AMX:
            return AMX_KERNEL_CONFIG;
        case CPUArch::ARM_NEON:
            return NEON_KERNEL_CONFIG;
        case CPUArch::ARM_SVE:
        case CPUArch::ARM_SVE2:
        case CPUArch::ARM_SME:
            return SVE_KERNEL_CONFIG;
        default:
            return AVX2_KERNEL_CONFIG;
    }
}

/**
 * @brief Get optimal thread count for problem size
 */
inline int get_optimal_threads(size_t problem_size, CPUArch arch) {
    const CPUSpecs* specs = nullptr;
    switch (arch) {
        case CPUArch::X86_AVX512: specs = &X86_AVX512_SPECS; break;
        case CPUArch::X86_AMX: specs = &X86_AMX_SPECS; break;
        case CPUArch::ARM_SVE: specs = &ARM_SVE_SPECS; break;
        default: specs = &X86_AVX2_SPECS;
    }
    
    // Minimum work per thread
    size_t min_work = 4096;
    int max_threads = specs->num_cores * specs->threads_per_core;
    int optimal = static_cast<int>(problem_size / min_work);
    
    if (optimal < 1) return 1;
    if (optimal > max_threads) return max_threads;
    return optimal;
}

}  // namespace cpu
}  // namespace persistent_kernel
}  // namespace yirage
