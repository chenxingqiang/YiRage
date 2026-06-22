/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/tpu_common.h"

namespace yirage { namespace persistent_kernel { namespace tpu { namespace v5e {
constexpr auto SPECS = TPU_V5E_SPECS;
constexpr auto KERNEL_CONFIG = TPU_V5E_KERNEL_CONFIG;
constexpr int OPTIMAL_BATCH_SIZE = 256;
constexpr int MAX_SEQUENCE_LENGTH = 8192;
constexpr bool ENABLE_INT4 = true;
}}}}
