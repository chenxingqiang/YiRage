/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/xpu_common.h"

namespace yirage { namespace persistent_kernel { namespace xpu { namespace arc_a770 {
constexpr auto SPECS = ARC_A770_SPECS;
constexpr auto KERNEL_CONFIG = ARC_A770_KERNEL_CONFIG;
constexpr int OPTIMAL_BATCH_SIZE = 128;
constexpr int MAX_SEQUENCE_LENGTH = 8192;
}}}}
