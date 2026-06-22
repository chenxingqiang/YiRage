/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/tpu_common.h"

namespace yirage { namespace persistent_kernel { namespace tpu { namespace v3 {
constexpr auto SPECS = TPU_V3_SPECS;
constexpr auto KERNEL_CONFIG = TPU_V3_KERNEL_CONFIG;
constexpr int OPTIMAL_BATCH_SIZE = 128;
constexpr int MAX_SEQUENCE_LENGTH = 4096;
}}}}
