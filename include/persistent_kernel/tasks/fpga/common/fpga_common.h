/* Copyright 2025 YiRage Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#pragma once

/**
 * @file fpga_common.h
 * @brief Common definitions for FPGA kernels (Xilinx, Intel, AWS F1)
 */

namespace yirage {
namespace persistent_kernel {
namespace fpga {

// =============================================================================
// FPGA Device Detection
// =============================================================================

enum class FPGAVendor {
    XILINX,
    INTEL,
    LATTICE,
    UNKNOWN
};

enum class FPGADevice {
    // Xilinx Alveo
    ALVEO_U200,
    ALVEO_U250,
    ALVEO_U280,
    VERSAL_VCK5000,
    
    // Intel
    AGILEX_F,
    STRATIX_10,
    
    // AWS
    AWS_F1,
    
    UNKNOWN
};

// =============================================================================
// Hardware Specifications by Device
// =============================================================================

struct FPGASpecs {
    FPGAVendor vendor;
    int dsp_slices;             // DSP48 or DSP blocks
    int bram_kb;                // Block RAM in KB
    int uram_kb;                // UltraRAM in KB (Xilinx)
    int lut_count;              // Logic LUTs (approx)
    int hbm_channels;           // HBM channels (0 if DDR only)
    int ddr_channels;           // DDR channels
    int ddr_bw_gbps;            // DDR bandwidth per channel
    int target_freq_mhz;        // Target clock frequency
};

// Xilinx Alveo U200
constexpr FPGASpecs ALVEO_U200_SPECS = {
    .vendor = FPGAVendor::XILINX,
    .dsp_slices = 6840,
    .bram_kb = 2160,
    .uram_kb = 960,
    .lut_count = 1182000,
    .hbm_channels = 0,
    .ddr_channels = 4,
    .ddr_bw_gbps = 19,
    .target_freq_mhz = 300
};

// Xilinx Alveo U250
constexpr FPGASpecs ALVEO_U250_SPECS = {
    .vendor = FPGAVendor::XILINX,
    .dsp_slices = 12288,
    .bram_kb = 5376,
    .uram_kb = 1280,
    .lut_count = 1728000,
    .hbm_channels = 0,
    .ddr_channels = 4,
    .ddr_bw_gbps = 19,
    .target_freq_mhz = 300
};

// Xilinx Alveo U280 (with HBM)
constexpr FPGASpecs ALVEO_U280_SPECS = {
    .vendor = FPGAVendor::XILINX,
    .dsp_slices = 9024,
    .bram_kb = 4032,
    .uram_kb = 1280,
    .lut_count = 1303680,
    .hbm_channels = 32,
    .ddr_channels = 2,
    .ddr_bw_gbps = 19,
    .target_freq_mhz = 300
};

// Intel Agilex F-Series
constexpr FPGASpecs AGILEX_F_SPECS = {
    .vendor = FPGAVendor::INTEL,
    .dsp_slices = 5760,
    .bram_kb = 2880,
    .uram_kb = 0,
    .lut_count = 1400000,
    .hbm_channels = 0,
    .ddr_channels = 4,
    .ddr_bw_gbps = 25,
    .target_freq_mhz = 400
};

// AWS F1 (based on Xilinx VU9P)
constexpr FPGASpecs AWS_F1_SPECS = {
    .vendor = FPGAVendor::XILINX,
    .dsp_slices = 6840,
    .bram_kb = 2160,
    .uram_kb = 960,
    .lut_count = 1182000,
    .hbm_channels = 0,
    .ddr_channels = 4,
    .ddr_bw_gbps = 17,
    .target_freq_mhz = 250
};

// =============================================================================
// HLS Configuration
// =============================================================================

struct FPGAKernelConfig {
    int pipeline_ii;            // Initiation interval
    int unroll_factor;          // Loop unroll factor
    int array_partition;        // Array partition factor
    int num_compute_units;      // Kernel replicas
    int num_parallel_ops;       // Parallel MACs per CU
    bool enable_dataflow;       // Dataflow optimization
    bool use_int8;              // INT8 precision
    bool use_half;              // FP16 precision
};

constexpr FPGAKernelConfig ALVEO_U200_KERNEL_CONFIG = {
    .pipeline_ii = 1,
    .unroll_factor = 8,
    .array_partition = 8,
    .num_compute_units = 2,
    .num_parallel_ops = 16,
    .enable_dataflow = true,
    .use_int8 = true,
    .use_half = true
};

constexpr FPGAKernelConfig ALVEO_U250_KERNEL_CONFIG = {
    .pipeline_ii = 1,
    .unroll_factor = 16,
    .array_partition = 16,
    .num_compute_units = 4,
    .num_parallel_ops = 32,
    .enable_dataflow = true,
    .use_int8 = true,
    .use_half = true
};

constexpr FPGAKernelConfig ALVEO_U280_KERNEL_CONFIG = {
    .pipeline_ii = 1,
    .unroll_factor = 16,
    .array_partition = 16,
    .num_compute_units = 4,
    .num_parallel_ops = 32,
    .enable_dataflow = true,
    .use_int8 = true,
    .use_half = true
};

constexpr FPGAKernelConfig AGILEX_KERNEL_CONFIG = {
    .pipeline_ii = 1,
    .unroll_factor = 8,
    .array_partition = 8,
    .num_compute_units = 2,
    .num_parallel_ops = 16,
    .enable_dataflow = true,
    .use_int8 = true,
    .use_half = true
};

constexpr FPGAKernelConfig AWS_F1_KERNEL_CONFIG = {
    .pipeline_ii = 1,
    .unroll_factor = 8,
    .array_partition = 8,
    .num_compute_units = 2,
    .num_parallel_ops = 16,
    .enable_dataflow = true,
    .use_int8 = true,
    .use_half = true
};

// =============================================================================
// Detection Functions
// =============================================================================

FPGADevice detect_fpga_device();
const FPGASpecs& get_fpga_specs(FPGADevice device);
const FPGAKernelConfig& get_fpga_kernel_config(FPGADevice device);

}  // namespace fpga
}  // namespace persistent_kernel
}  // namespace yirage
