/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/tpu_common.h"

namespace yirage { namespace persistent_kernel { namespace tpu { namespace v5p {
constexpr auto SPECS = TPU_V5P_SPECS;
constexpr auto KERNEL_CONFIG = TPU_V5P_KERNEL_CONFIG;
constexpr int OPTIMAL_BATCH_SIZE = 512;
constexpr int MAX_SEQUENCE_LENGTH = 32768;
constexpr bool ENABLE_INT4 = true;
constexpr bool ENABLE_SPARSITY = true;
}}}}
