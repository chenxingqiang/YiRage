/* Copyright 2025 YiRage Team */
#pragma once

#include "common/fpga_common.h"
#include "common/fpga_detection.h"

/**
 * @file task_header.h
 * @brief Main include file for FPGA persistent kernel tasks
 * 
 * FPGA kernels are written in:
 * - Vitis HLS (C++)
 * - OpenCL
 * - RTL (Verilog/VHDL)
 */

namespace yirage {
namespace persistent_kernel {
namespace fpga {

enum class FPGATaskType {
    GEMM,
    ATTENTION,
    SOFTMAX,
    LAYER_NORM,
    EMBEDDING,
    CONV2D,
    POOLING
};

struct FPGATaskDesc {
    FPGATaskType type;
    FPGADevice target_device;
    int m, n, k;
    bool use_int8;
    bool use_streaming;
};

inline FPGAKernelConfig get_optimal_config(FPGATaskDesc const& task) {
    switch (task.target_device) {
        case FPGADevice::ALVEO_U200: return ALVEO_U200_KERNEL_CONFIG;
        case FPGADevice::ALVEO_U250: return ALVEO_U250_KERNEL_CONFIG;
        case FPGADevice::ALVEO_U280: return ALVEO_U280_KERNEL_CONFIG;
        case FPGADevice::AGILEX_F:
        case FPGADevice::STRATIX_10: return AGILEX_KERNEL_CONFIG;
        case FPGADevice::AWS_F1: return AWS_F1_KERNEL_CONFIG;
        default: return ALVEO_U280_KERNEL_CONFIG;
    }
}

}  // namespace fpga
}  // namespace persistent_kernel
}  // namespace yirage
