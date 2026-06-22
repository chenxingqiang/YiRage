/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/xpu_common.h"

namespace yirage { namespace persistent_kernel { namespace xpu { namespace ponte_vecchio {
constexpr auto SPECS = PONTE_VECCHIO_SPECS;
constexpr auto KERNEL_CONFIG = PVC_KERNEL_CONFIG;
constexpr int OPTIMAL_BATCH_SIZE = 512;
constexpr int MAX_SEQUENCE_LENGTH = 32768;
constexpr bool ENABLE_MULTI_TILE = true;
}}}}
