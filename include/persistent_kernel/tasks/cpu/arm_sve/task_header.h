/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/cpu_common.h"

namespace yirage { namespace persistent_kernel { namespace cpu { namespace arm_sve {

constexpr auto SPECS = ARM_SVE_SPECS;
constexpr auto KERNEL_CONFIG = SVE_KERNEL_CONFIG;

// SVE: Scalable vector length (128-2048 bits)
// At runtime, use svcntw() to get actual vector length
constexpr int MAX_SIMD_WIDTH_F32 = 64;  // Max 64 floats (2048-bit)

// SVE specific features
constexpr bool HAS_PREDICATION = true;
constexpr bool HAS_GATHER_SCATTER = true;
constexpr bool HAS_FIRST_FAULTING = true;

// Fujitsu A64FX specific (512-bit SVE)
constexpr int A64FX_SIMD_WIDTH_F32 = 16;
constexpr int A64FX_L2_CACHE_MB = 8;  // Per CMG

}}}}
