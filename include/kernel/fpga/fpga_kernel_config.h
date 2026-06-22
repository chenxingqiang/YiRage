/* Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * FPGA Kernel Configuration
 * 
 * Supports:
 * - Xilinx Alveo (U200, U250, U280)
 * - Intel Agilex/Stratix
 * - AWS F1 instances
 */

#pragma once

#include "kernel/common/kernel_interface.h"

namespace yirage {
namespace kernel {
namespace fpga {

/**
 * @brief FPGA vendor types
 */
enum class FPGAVendor {
  XILINX,
  INTEL,
  LATTICE,
  UNKNOWN
};

/**
 * @brief FPGA device types
 */
enum class FPGADevice {
  // Xilinx
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

/**
 * @brief HLS (High-Level Synthesis) optimization directives
 */
struct HLSDirectives {
  int pipeline_ii = 1;           // Initiation interval
  bool enable_dataflow = true;   // Dataflow optimization
  int array_partition_factor = 8;
  bool enable_loop_unroll = true;
  int unroll_factor = 4;
  bool enable_inline = true;
};

/**
 * @brief FPGA memory configuration
 */
struct FPGAMemoryConfig {
  // On-chip memory
  size_t bram_kb = 0;            // Block RAM
  size_t uram_kb = 0;            // Ultra RAM
  size_t lutram_kb = 0;          // LUT RAM
  
  // External memory
  int hbm_channels = 0;          // HBM channels (U280, etc.)
  int ddr_channels = 4;          // DDR channels
  size_t ddr_bandwidth_gbps = 77.0;  // Per channel
  
  // Memory ports
  int memory_ports = 4;
};

/**
 * @brief FPGA-specific kernel configuration
 */
struct FPGAKernelConfig : public KernelConfig {
  // Device configuration
  FPGAVendor vendor = FPGAVendor::XILINX;
  FPGADevice device = FPGADevice::ALVEO_U280;
  
  // Clock configuration
  int target_frequency_mhz = 300;
  int achieved_frequency_mhz = 0;
  
  // Resource utilization targets
  float target_lut_utilization = 0.7;
  float target_ff_utilization = 0.7;
  float target_bram_utilization = 0.8;
  float target_dsp_utilization = 0.9;
  
  // DSP configuration
  int dsp_slices = 0;
  bool use_dsp_for_mul = true;
  bool use_dsp_for_add = false;
  
  // HLS directives
  HLSDirectives hls;
  
  // Memory configuration
  FPGAMemoryConfig memory;
  
  // Precision
  int fixed_point_width = 16;
  int fixed_point_frac = 8;
  bool use_float = false;
  bool use_half = true;
  bool use_int8 = true;
  bool use_int4 = true;         // FPGA excels at low precision
  
  // Kernel configuration
  int num_compute_units = 4;
  int num_parallel_ops = 8;
  
  // OpenCL/XRT configuration
  std::string platform = "xilinx_u280_xdma_201920_3";
  std::string kernel_name = "krnl_compute";
  bool enable_streaming = true;
  
  // Dataflow configuration
  int stream_depth = 32;
  bool enable_ping_pong = true;
  
  FPGAKernelConfig() {
    backend_type = type::BT_FPGA;
  }
  
  // Get estimated resource utilization
  float get_estimated_lut_util() const;
  float get_estimated_ff_util() const;
  float get_estimated_bram_util() const;
  float get_estimated_dsp_util() const;
  
  // Check if design fits
  bool fits_on_device() const {
    return get_estimated_lut_util() <= target_lut_utilization &&
           get_estimated_ff_util() <= target_ff_utilization &&
           get_estimated_bram_util() <= target_bram_utilization &&
           get_estimated_dsp_util() <= target_dsp_utilization;
  }
};

/**
 * @brief FPGA kernel optimizer
 */
class FPGAOptimizer {
public:
  /**
   * @brief Compute optimal parallelization factor
   */
  static int compute_optimal_parallelism(size_t problem_size,
                                         FPGADevice device,
                                         int available_dsp);
  
  /**
   * @brief Compute optimal tiling for on-chip memory
   */
  static void compute_optimal_tiling(int m, int n, int k,
                                     FPGAKernelConfig &config);
  
  /**
   * @brief Estimate resource utilization
   */
  static void estimate_resources(FPGAKernelConfig const &config,
                                 float &lut_util, float &ff_util,
                                 float &bram_util, float &dsp_util);
  
  /**
   * @brief Estimate latency
   */
  static float estimate_latency_us(FPGAKernelConfig const &config,
                                   int m, int n, int k);
  
  /**
   * @brief Estimate throughput
   */
  static float estimate_throughput_gops(FPGAKernelConfig const &config,
                                        int m, int n, int k);
  
  /**
   * @brief Generate HLS C++ code
   */
  static std::string generate_hls_code(std::string const &op_name,
                                       FPGAKernelConfig const &config);
  
  /**
   * @brief Generate OpenCL kernel
   */
  static std::string generate_opencl_kernel(std::string const &op_name,
                                            FPGAKernelConfig const &config);
  
  /**
   * @brief Generate Vitis HLS directives
   */
  static std::string generate_hls_directives(FPGAKernelConfig const &config);
  
  /**
   * @brief Optimize memory access pattern
   */
  static void optimize_memory_access(FPGAKernelConfig &config,
                                     int data_width, int access_pattern);
  
  /**
   * @brief Generate connectivity configuration for multi-port memory
   */
  static std::string generate_connectivity_cfg(FPGAKernelConfig const &config);
};

} // namespace fpga
} // namespace kernel
} // namespace yirage
