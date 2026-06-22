/* Copyright 2025 YiRage Team */
#pragma once

#include "fpga_common.h"

namespace yirage {
namespace persistent_kernel {
namespace fpga {

/**
 * @brief Detect FPGA device via XRT/OpenCL
 */
inline FPGADevice detect_fpga_device() {
    // Would use xrt::device or cl::Platform APIs
    return FPGADevice::ALVEO_U280;  // Default
}

/**
 * @brief Check if FPGA is available
 */
inline bool is_fpga_available() {
    // Would query XRT runtime
    return false;
}

/**
 * @brief Get FPGA vendor
 */
inline FPGAVendor get_fpga_vendor(FPGADevice device) {
    switch (device) {
        case FPGADevice::AGILEX_F:
        case FPGADevice::STRATIX_10:
            return FPGAVendor::INTEL;
        default:
            return FPGAVendor::XILINX;
    }
}

}  // namespace fpga
}  // namespace persistent_kernel
}  // namespace yirage
