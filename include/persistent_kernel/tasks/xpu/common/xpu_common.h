/* Copyright 2025 YiRage Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

/**
 * @file xpu_common.h
 * @brief Common definitions for Intel XPU kernels (Data Center GPU Max, Arc)
 */

namespace yirage {
namespace persistent_kernel {
namespace xpu {

// =============================================================================
// Intel XPU Architecture Detection
// =============================================================================

enum class XPUArch {
    PONTE_VECCHIO,  // Data Center GPU Max (PVC)
    ARC_A770,       // Arc A-series consumer
    ARC_A750,
    FLEX_170,       // Data Center GPU Flex
    UNKNOWN
};

// =============================================================================
// Hardware Specifications by Architecture
// =============================================================================

struct XPUSpecs {
    int total_eus;              // Total Execution Units
    int subslices;              // Number of subslices
    int eu_per_subslice;        // EUs per subslice
    int threads_per_eu;         // Threads per EU
    int simd_width;             // SIMD width (8, 16, or 32)
    size_t slm_kb;              // Shared Local Memory in KB
    size_t l3_cache_mb;         // L3 cache in MB
    int hbm_gb;                 // HBM capacity (0 for GDDR)
    int mem_bw_gbps;            // Memory bandwidth
    int bf16_tflops;            // Peak BF16 TFLOPS
    int num_tiles;              // Number of tiles
    bool has_xmx;               // XMX (Xe Matrix eXtensions)
    bool has_dpas;              // DPAS support
};

// Intel Data Center GPU Max (Ponte Vecchio)
constexpr XPUSpecs PONTE_VECCHIO_SPECS = {
    .total_eus = 512,           // 128 per tile * 4 stacks
    .subslices = 64,
    .eu_per_subslice = 8,
    .threads_per_eu = 8,
    .simd_width = 16,
    .slm_kb = 128,
    .l3_cache_mb = 204,
    .hbm_gb = 128,
    .mem_bw_gbps = 3200,
    .bf16_tflops = 840,         // Per GPU
    .num_tiles = 2,
    .has_xmx = true,
    .has_dpas = true
};

// Intel Arc A770
constexpr XPUSpecs ARC_A770_SPECS = {
    .total_eus = 512,
    .subslices = 32,
    .eu_per_subslice = 16,
    .threads_per_eu = 8,
    .simd_width = 16,
    .slm_kb = 64,
    .l3_cache_mb = 16,
    .hbm_gb = 0,
    .mem_bw_gbps = 560,
    .bf16_tflops = 35,
    .num_tiles = 1,
    .has_xmx = true,
    .has_dpas = true
};

// Intel Arc A750
constexpr XPUSpecs ARC_A750_SPECS = {
    .total_eus = 448,
    .subslices = 28,
    .eu_per_subslice = 16,
    .threads_per_eu = 8,
    .simd_width = 16,
    .slm_kb = 64,
    .l3_cache_mb = 16,
    .hbm_gb = 0,
    .mem_bw_gbps = 512,
    .bf16_tflops = 30,
    .num_tiles = 1,
    .has_xmx = true,
    .has_dpas = true
};

// Intel Flex 170
constexpr XPUSpecs FLEX_170_SPECS = {
    .total_eus = 256,
    .subslices = 16,
    .eu_per_subslice = 16,
    .threads_per_eu = 8,
    .simd_width = 16,
    .slm_kb = 64,
    .l3_cache_mb = 8,
    .hbm_gb = 0,
    .mem_bw_gbps = 450,
    .bf16_tflops = 20,
    .num_tiles = 1,
    .has_xmx = true,
    .has_dpas = true
};

// =============================================================================
// Kernel Configuration by Architecture
// =============================================================================

struct XPUKernelConfig {
    int simd_width;             // Sub-group SIMD width
    int num_sub_groups;         // Sub-groups per work-group
    int xmx_m;                  // XMX tile M
    int xmx_n;                  // XMX tile N
    int xmx_k;                  // XMX tile K
    int dpas_depth;             // DPAS depth
    bool use_xmx;               // Enable XMX
    bool use_dpas;              // Enable DPAS
    bool use_bf16;              // Use BF16
    bool use_multi_tile;        // Multi-tile for PVC
};

constexpr XPUKernelConfig PVC_KERNEL_CONFIG = {
    .simd_width = 16,
    .num_sub_groups = 8,
    .xmx_m = 8,
    .xmx_n = 16,
    .xmx_k = 16,
    .dpas_depth = 8,
    .use_xmx = true,
    .use_dpas = true,
    .use_bf16 = true,
    .use_multi_tile = true
};

constexpr XPUKernelConfig ARC_A770_KERNEL_CONFIG = {
    .simd_width = 16,
    .num_sub_groups = 4,
    .xmx_m = 8,
    .xmx_n = 16,
    .xmx_k = 16,
    .dpas_depth = 8,
    .use_xmx = true,
    .use_dpas = true,
    .use_bf16 = true,
    .use_multi_tile = false
};

constexpr XPUKernelConfig ARC_A750_KERNEL_CONFIG = {
    .simd_width = 16,
    .num_sub_groups = 4,
    .xmx_m = 8,
    .xmx_n = 16,
    .xmx_k = 16,
    .dpas_depth = 8,
    .use_xmx = true,
    .use_dpas = true,
    .use_bf16 = true,
    .use_multi_tile = false
};

constexpr XPUKernelConfig FLEX_KERNEL_CONFIG = {
    .simd_width = 16,
    .num_sub_groups = 2,
    .xmx_m = 8,
    .xmx_n = 16,
    .xmx_k = 16,
    .dpas_depth = 8,
    .use_xmx = true,
    .use_dpas = true,
    .use_bf16 = true,
    .use_multi_tile = false
};

// =============================================================================
// Detection Functions
// =============================================================================

XPUArch detect_xpu_arch();
const XPUSpecs& get_xpu_specs(XPUArch arch);
const XPUKernelConfig& get_xpu_kernel_config(XPUArch arch);

}  // namespace xpu
}  // namespace persistent_kernel
}  // namespace yirage
