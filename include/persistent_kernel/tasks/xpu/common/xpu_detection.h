/* Copyright 2025 YiRage Team */
#pragma once

#include "xpu_common.h"

namespace yirage {
namespace persistent_kernel {
namespace xpu {

/**
 * @brief Detect Intel XPU architecture via SYCL/Level Zero
 */
inline XPUArch detect_xpu_arch() {
    // Would use sycl::device or ze_driver APIs
    return XPUArch::PONTE_VECCHIO;  // Default to PVC
}

/**
 * @brief Check if XPU is available
 */
inline bool is_xpu_available() {
    // Would query SYCL runtime
    return false;
}

/**
 * @brief Get device name string
 */
inline const char* get_xpu_name(XPUArch arch) {
    switch (arch) {
        case XPUArch::PONTE_VECCHIO: return "Intel Data Center GPU Max";
        case XPUArch::ARC_A770: return "Intel Arc A770";
        case XPUArch::ARC_A750: return "Intel Arc A750";
        case XPUArch::FLEX_170: return "Intel Flex 170";
        default: return "Unknown Intel GPU";
    }
}

}  // namespace xpu
}  // namespace persistent_kernel
}  // namespace yirage
