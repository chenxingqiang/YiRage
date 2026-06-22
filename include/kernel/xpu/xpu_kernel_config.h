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
 * Intel XPU (Data Center GPU Max / Arc) Kernel Configuration
 * 
 * Intel XPU kernels use oneAPI/SYCL for programming.
 */

#pragma once

#include "kernel/common/kernel_interface.h"

namespace yirage {
namespace kernel {
namespace xpu {

/**
 * @brief Intel XPU architecture types
 */
enum class XPUArch {
  PONTE_VECCHIO,  // Data Center GPU Max (PVC)
  ARC_A770,       // Arc A-series consumer
  ARC_A750,
  FLEX_170,       // Data Center GPU Flex
  UNKNOWN
};

/**
 * @brief SLM (Shared Local Memory) layout
 */
enum class SLMLayout {
  ROW_MAJOR,
  COLUMN_MAJOR,
  BLOCKED,
  VNNI_PACKED    // For INT8/BF16 optimized layout
};

/**
 * @brief Intel XPU-specific kernel configuration
 */
struct XPUKernelConfig : public KernelConfig {
  // Architecture
  XPUArch arch = XPUArch::PONTE_VECCHIO;
  
  // Execution unit configuration
  int subslice_count = 8;       // Number of subslices
  int eu_per_subslice = 16;     // Execution units per subslice
  int threads_per_eu = 8;       // Threads per EU
  
  // Sub-group (SIMD width) configuration
  int simd_width = 16;          // 8, 16, or 32
  int num_sub_groups = 4;
  
  // SLM (Shared Local Memory) configuration
  SLMLayout slm_layout = SLMLayout::BLOCKED;
  size_t slm_size = 131072;     // 128KB default for PVC
  int slm_bank_count = 32;
  
  // XMX (Xe Matrix eXtensions) configuration
  bool use_xmx = true;
  int xmx_m = 8;
  int xmx_n = 16;
  int xmx_k = 16;
  
  // DPAS (Dot Product Accumulate Systolic) configuration
  bool use_dpas = true;
  int dpas_depth = 8;
  
  // Memory configuration
  int l3_cache_size_mb = 204;   // L3 cache for PVC
  bool enable_prefetch = true;
  int prefetch_distance = 2;
  
  // Precision support
  bool use_bf16 = true;
  bool use_tf32 = false;
  bool use_int8 = false;
  
  // SYCL configuration
  std::string sycl_backend = "level_zero";
  bool enable_kernel_fusion = true;
  
  // Multi-tile configuration
  int num_tiles = 2;            // PVC has 2 tiles
  bool enable_multi_tile = true;
  
  XPUKernelConfig() {
    backend_type = type::BT_XPU;
  }
  
  // Get total EUs
  int get_total_eus() const {
    return subslice_count * eu_per_subslice;
  }
  
  // Get total hardware threads
  int get_hw_threads() const {
    return get_total_eus() * threads_per_eu;
  }
  
  // Check XMX availability
  bool has_xmx() const {
    return arch == XPUArch::PONTE_VECCHIO || 
           arch == XPUArch::ARC_A770 ||
           arch == XPUArch::ARC_A750;
  }
};

/**
 * @brief Intel XPU kernel optimizer
 */
class XPUOptimizer {
public:
  /**
   * @brief Compute optimal sub-group configuration
   */
  static int compute_optimal_simd_width(size_t problem_size, XPUArch arch);
  
  /**
   * @brief Compute optimal SLM configuration
   */
  static size_t compute_optimal_slm(size_t data_size, SLMLayout layout);
  
  /**
   * @brief Check for SLM bank conflicts
   */
  static bool has_bank_conflict(SLMLayout layout, int stride);
  
  /**
   * @brief Estimate occupancy
   */
  static float estimate_occupancy(XPUKernelConfig const &config,
                                  int registers_used);
  
  /**
   * @brief Select optimal XMX configuration
   */
  static bool select_xmx_config(int m, int n, int k,
                                XPUArch arch,
                                XPUKernelConfig &config);
  
  /**
   * @brief Generate SYCL kernel
   */
  static std::string generate_sycl_kernel(std::string const &op_name,
                                          XPUKernelConfig const &config);
  
  /**
   * @brief Generate oneDNN primitive configuration
   */
  static std::string generate_onednn_config(std::string const &op_name,
                                            XPUKernelConfig const &config);
  
  /**
   * @brief Optimize work-group dimensions
   */
  static void optimize_work_groups(int problem_m, int problem_n,
                                   int problem_k, XPUArch arch,
                                   XPUKernelConfig &config);
};

} // namespace xpu
} // namespace kernel
} // namespace yirage
