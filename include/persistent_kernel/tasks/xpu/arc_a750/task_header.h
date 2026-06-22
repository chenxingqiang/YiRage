/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/xpu_common.h"

namespace yirage { namespace persistent_kernel { namespace xpu { namespace arc_a750 {
constexpr auto SPECS = ARC_A750_SPECS;
constexpr auto KERNEL_CONFIG = ARC_A750_KERNEL_CONFIG;
constexpr int OPTIMAL_BATCH_SIZE = 128;
constexpr int MAX_SEQUENCE_LENGTH = 8192;
}}}}
