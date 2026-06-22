/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/tpu_common.h"

namespace yirage { namespace persistent_kernel { namespace tpu { namespace v2 {
constexpr auto SPECS = TPU_V2_SPECS;
constexpr auto KERNEL_CONFIG = TPU_V2_KERNEL_CONFIG;
constexpr int OPTIMAL_BATCH_SIZE = 64;
constexpr int MAX_SEQUENCE_LENGTH = 2048;
}}}}
