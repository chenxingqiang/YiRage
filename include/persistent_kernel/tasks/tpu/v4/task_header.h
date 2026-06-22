/* Copyright 2025 YiRage Team */
#pragma once

#include "../common/tpu_common.h"

namespace yirage {
namespace persistent_kernel {
namespace tpu {
namespace v4 {

// TPU v4 specific configuration
constexpr auto SPECS = TPU_V4_SPECS;
constexpr auto KERNEL_CONFIG = TPU_V4_KERNEL_CONFIG;

// v4-specific optimizations
constexpr int OPTIMAL_BATCH_SIZE = 256;
constexpr int MAX_SEQUENCE_LENGTH = 8192;
constexpr bool ENABLE_SPARSITY = true;

}  // namespace v4
}  // namespace tpu
}  // namespace persistent_kernel
}  // namespace yirage
