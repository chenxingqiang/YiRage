/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/cpu_common.h"

namespace yirage { namespace persistent_kernel { namespace cpu { namespace x86_avx2 {

constexpr auto SPECS = X86_AVX2_SPECS;
constexpr auto KERNEL_CONFIG = AVX2_KERNEL_CONFIG;

// AVX2 intrinsics: 256-bit registers
constexpr int SIMD_WIDTH_F32 = 8;   // 8 floats per register
constexpr int SIMD_WIDTH_F64 = 4;   // 4 doubles per register

// Optimal micro-kernel dimensions for GEMM
constexpr int MR = 6;   // Registers for A
constexpr int NR = 16;  // Registers for B (2 AVX registers)

}}}}
