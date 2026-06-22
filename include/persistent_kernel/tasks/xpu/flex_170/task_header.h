/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/xpu_common.h"

namespace yirage { namespace persistent_kernel { namespace xpu { namespace flex_170 {
constexpr auto SPECS = FLEX_170_SPECS;
constexpr auto KERNEL_CONFIG = FLEX_KERNEL_CONFIG;
constexpr int OPTIMAL_BATCH_SIZE = 64;
constexpr int MAX_SEQUENCE_LENGTH = 4096;
}}}}
