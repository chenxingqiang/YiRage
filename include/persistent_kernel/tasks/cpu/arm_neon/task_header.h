/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/cpu_common.h"

namespace yirage { namespace persistent_kernel { namespace cpu { namespace arm_neon {

constexpr auto SPECS = ARM_NEON_SPECS;
constexpr auto KERNEL_CONFIG = NEON_KERNEL_CONFIG;

// NEON intrinsics: 128-bit registers
constexpr int SIMD_WIDTH_F32 = 4;   // 4 floats per register
constexpr int SIMD_WIDTH_F64 = 2;   // 2 doubles per register

// Optimal micro-kernel dimensions for GEMM
constexpr int MR = 8;   // Registers for A
constexpr int NR = 8;   // Registers for B

// NEON specific features
constexpr bool HAS_FMA = true;      // vfmaq_f32
constexpr bool HAS_DOT_PROD = true; // vdotq_s32 (ARMv8.2+)

}}}}
