/* Copyright 2025 YiRage Team */
#pragma once
#include "../common/cpu_common.h"

namespace yirage { namespace persistent_kernel { namespace cpu { namespace x86_avx512 {

constexpr auto SPECS = X86_AVX512_SPECS;
constexpr auto KERNEL_CONFIG = AVX512_KERNEL_CONFIG;

// AVX-512 intrinsics: 512-bit registers
constexpr int SIMD_WIDTH_F32 = 16;  // 16 floats per register
constexpr int SIMD_WIDTH_F64 = 8;   // 8 doubles per register

// Optimal micro-kernel dimensions for GEMM
constexpr int MR = 6;   // Registers for A
constexpr int NR = 32;  // Registers for B (2 ZMM registers)

// AVX-512 specific features
constexpr bool HAS_MASK_REGISTERS = true;
constexpr bool HAS_GATHER_SCATTER = true;
constexpr int NUM_ZMM_REGISTERS = 32;

}}}}
