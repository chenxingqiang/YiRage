/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/fpga_common.h"

namespace yirage { namespace persistent_kernel { namespace fpga { namespace alveo_u280 {
constexpr auto SPECS = ALVEO_U280_SPECS;
constexpr auto KERNEL_CONFIG = ALVEO_U280_KERNEL_CONFIG;
constexpr int HBM_CHANNELS = 32;
constexpr int HBM_BW_GBPS = 460;  // Total HBM bandwidth
}}}}
