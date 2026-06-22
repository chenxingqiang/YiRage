/* Copyright 2025 YiRage Team */
#pragma once

#include "common/xpu_common.h"
#include "common/xpu_detection.h"

/**
 * @file task_header.h
 * @brief Main include file for Intel XPU persistent kernel tasks
 * 
 * XPU kernels are written in:
 * - SYCL (oneAPI)
 * - Level Zero
 * - oneDNN primitives
 */

namespace yirage {
namespace persistent_kernel {
namespace xpu {

enum class XPUTaskType {
    GEMM,
    ATTENTION,
    FLASH_ATTENTION,
    SOFTMAX,
    RMS_NORM,
    EMBEDDING,
    ARGMAX
};

struct XPUTaskDesc {
    XPUTaskType type;
    XPUArch target_arch;
    int m, n, k;
    bool use_bf16;
    bool use_xmx;
    int num_attention_heads;
    int head_dim;
};

inline XPUKernelConfig get_optimal_config(XPUTaskDesc const& task) {
    switch (task.target_arch) {
        case XPUArch::PONTE_VECCHIO: return PVC_KERNEL_CONFIG;
        case XPUArch::ARC_A770: return ARC_A770_KERNEL_CONFIG;
        case XPUArch::ARC_A750: return ARC_A750_KERNEL_CONFIG;
        case XPUArch::FLEX_170: return FLEX_KERNEL_CONFIG;
        default: return PVC_KERNEL_CONFIG;
    }
}

}  // namespace xpu
}  // namespace persistent_kernel
}  // namespace yirage
