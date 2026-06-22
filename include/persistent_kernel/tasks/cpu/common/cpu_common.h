/* Copyright 2025 YiRage Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

/**
 * @file cpu_common.h
 * @brief Common definitions for CPU kernels (x86/ARM SIMD)
 */

#include <cstddef>

namespace yirage {
namespace persistent_kernel {
namespace cpu {

// =============================================================================
// CPU Architecture Detection
// =============================================================================

enum class CPUArch {
    X86_SSE,        // SSE4.2
    X86_AVX,        // AVX
    X86_AVX2,       // AVX2 + FMA
    X86_AVX512,     // AVX-512
    X86_AVX512_BF16,// AVX-512 with BF16
    X86_AMX,        // Intel AMX (Sapphire Rapids+)
    ARM_NEON,       // ARM NEON
    ARM_SVE,        // ARM SVE (Scalable Vector Extension)
    ARM_SVE2,       // ARM SVE2
    ARM_SME,        // ARM SME (Scalable Matrix Extension)
    UNKNOWN
};

enum class CPUVendor {
    INTEL,
    AMD,
    APPLE,
    ARM,
    QUALCOMM,
    UNKNOWN
};

// =============================================================================
// Hardware Specifications by Architecture
// =============================================================================

struct CPUSpecs {
    CPUVendor vendor;
    int simd_width_bytes;       // SIMD register width in bytes
    int simd_width_f32;         // Number of F32 per SIMD register
    int num_cores;              // Physical cores
    int threads_per_core;       // SMT/HT threads
    size_t l1_cache_kb;         // L1 cache per core
    size_t l2_cache_kb;         // L2 cache per core
    size_t l3_cache_mb;         // L3 cache total
    int mem_bw_gbps;            // Memory bandwidth
    bool has_fma;               // FMA support
    bool has_bf16;              // BF16 support
    bool has_amx;               // AMX support (Intel)
    bool has_sme;               // SME support (ARM)
};

// x86 AVX2 (Haswell+)
constexpr CPUSpecs X86_AVX2_SPECS = {
    .vendor = CPUVendor::INTEL,
    .simd_width_bytes = 32,
    .simd_width_f32 = 8,
    .num_cores = 8,
    .threads_per_core = 2,
    .l1_cache_kb = 32,
    .l2_cache_kb = 256,
    .l3_cache_mb = 20,
    .mem_bw_gbps = 50,
    .has_fma = true,
    .has_bf16 = false,
    .has_amx = false,
    .has_sme = false
};

// x86 AVX-512 (Skylake-X+)
constexpr CPUSpecs X86_AVX512_SPECS = {
    .vendor = CPUVendor::INTEL,
    .simd_width_bytes = 64,
    .simd_width_f32 = 16,
    .num_cores = 16,
    .threads_per_core = 2,
    .l1_cache_kb = 32,
    .l2_cache_kb = 1024,
    .l3_cache_mb = 30,
    .mem_bw_gbps = 100,
    .has_fma = true,
    .has_bf16 = false,
    .has_amx = false,
    .has_sme = false
};

// x86 AVX-512 BF16 (Cooper Lake / Sapphire Rapids)
constexpr CPUSpecs X86_AVX512_BF16_SPECS = {
    .vendor = CPUVendor::INTEL,
    .simd_width_bytes = 64,
    .simd_width_f32 = 16,
    .num_cores = 32,
    .threads_per_core = 2,
    .l1_cache_kb = 48,
    .l2_cache_kb = 2048,
    .l3_cache_mb = 60,
    .mem_bw_gbps = 200,
    .has_fma = true,
    .has_bf16 = true,
    .has_amx = false,
    .has_sme = false
};

// x86 AMX (Sapphire Rapids+)
constexpr CPUSpecs X86_AMX_SPECS = {
    .vendor = CPUVendor::INTEL,
    .simd_width_bytes = 64,
    .simd_width_f32 = 16,
    .num_cores = 56,
    .threads_per_core = 2,
    .l1_cache_kb = 48,
    .l2_cache_kb = 2048,
    .l3_cache_mb = 105,
    .mem_bw_gbps = 300,
    .has_fma = true,
    .has_bf16 = true,
    .has_amx = true,
    .has_sme = false
};

// ARM NEON (Cortex-A series, Apple M1+)
constexpr CPUSpecs ARM_NEON_SPECS = {
    .vendor = CPUVendor::ARM,
    .simd_width_bytes = 16,
    .simd_width_f32 = 4,
    .num_cores = 8,
    .threads_per_core = 1,
    .l1_cache_kb = 64,
    .l2_cache_kb = 512,
    .l3_cache_mb = 8,
    .mem_bw_gbps = 70,
    .has_fma = true,
    .has_bf16 = false,
    .has_amx = false,
    .has_sme = false
};

// ARM SVE (Neoverse V1/V2, Fujitsu A64FX)
constexpr CPUSpecs ARM_SVE_SPECS = {
    .vendor = CPUVendor::ARM,
    .simd_width_bytes = 64,     // 512-bit SVE
    .simd_width_f32 = 16,
    .num_cores = 48,
    .threads_per_core = 1,
    .l1_cache_kb = 64,
    .l2_cache_kb = 1024,
    .l3_cache_mb = 32,
    .mem_bw_gbps = 200,
    .has_fma = true,
    .has_bf16 = true,
    .has_amx = false,
    .has_sme = false
};

// ARM SME (Neoverse V2+)
constexpr CPUSpecs ARM_SME_SPECS = {
    .vendor = CPUVendor::ARM,
    .simd_width_bytes = 64,
    .simd_width_f32 = 16,
    .num_cores = 64,
    .threads_per_core = 1,
    .l1_cache_kb = 64,
    .l2_cache_kb = 2048,
    .l3_cache_mb = 64,
    .mem_bw_gbps = 400,
    .has_fma = true,
    .has_bf16 = true,
    .has_amx = false,
    .has_sme = true
};

// =============================================================================
// Kernel Configuration by Architecture
// =============================================================================

struct CPUKernelConfig {
    int tile_m;                 // GEMM M tile size
    int tile_n;                 // GEMM N tile size
    int tile_k;                 // GEMM K tile size
    int num_threads;            // Thread count
    int cache_line_bytes;       // Cache line size
    bool use_openmp;            // Use OpenMP
    bool use_tbb;               // Use Intel TBB
    bool prefetch;              // Enable prefetching
    int unroll_factor;          // Loop unroll factor
};

constexpr CPUKernelConfig AVX2_KERNEL_CONFIG = {
    .tile_m = 6,
    .tile_n = 16,
    .tile_k = 256,
    .num_threads = 8,
    .cache_line_bytes = 64,
    .use_openmp = true,
    .use_tbb = false,
    .prefetch = true,
    .unroll_factor = 4
};

constexpr CPUKernelConfig AVX512_KERNEL_CONFIG = {
    .tile_m = 6,
    .tile_n = 32,
    .tile_k = 256,
    .num_threads = 16,
    .cache_line_bytes = 64,
    .use_openmp = true,
    .use_tbb = false,
    .prefetch = true,
    .unroll_factor = 4
};

constexpr CPUKernelConfig AMX_KERNEL_CONFIG = {
    .tile_m = 16,
    .tile_n = 16,
    .tile_k = 64,
    .num_threads = 32,
    .cache_line_bytes = 64,
    .use_openmp = true,
    .use_tbb = false,
    .prefetch = true,
    .unroll_factor = 1
};

constexpr CPUKernelConfig NEON_KERNEL_CONFIG = {
    .tile_m = 8,
    .tile_n = 8,
    .tile_k = 256,
    .num_threads = 8,
    .cache_line_bytes = 64,
    .use_openmp = true,
    .use_tbb = false,
    .prefetch = true,
    .unroll_factor = 4
};

constexpr CPUKernelConfig SVE_KERNEL_CONFIG = {
    .tile_m = 8,
    .tile_n = 32,
    .tile_k = 256,
    .num_threads = 48,
    .cache_line_bytes = 64,
    .use_openmp = true,
    .use_tbb = false,
    .prefetch = true,
    .unroll_factor = 4
};

// =============================================================================
// Detection Functions
// =============================================================================

CPUArch detect_cpu_arch();
CPUVendor detect_cpu_vendor();
const CPUSpecs& get_cpu_specs(CPUArch arch);
const CPUKernelConfig& get_cpu_kernel_config(CPUArch arch);

}  // namespace cpu
}  // namespace persistent_kernel
}  // namespace yirage
