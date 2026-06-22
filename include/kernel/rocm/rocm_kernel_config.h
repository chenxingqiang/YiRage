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
 * ROCm (AMD GPU) Kernel Configuration
 */

#pragma once

#include "kernel/common/kernel_interface.h"

namespace yirage {
namespace kernel {
namespace rocm {

/**
 * @brief ROCm architecture types
 */
enum class ROCmArch {
  CDNA1,    // MI100
  CDNA2,    // MI200 series
  CDNA3,    // MI300 series
  RDNA2,    // Consumer GPUs
  RDNA3,    // Consumer GPUs
  UNKNOWN
};

/**
 * @brief LDS (Local Data Share) layout strategies
 */
enum class LDSLayout {
  ROW_MAJOR,
  COLUMN_MAJOR,
  SWIZZLED,
  TILED
};

/**
 * @brief ROCm-specific kernel configuration
 */
struct ROCmKernelConfig : public KernelConfig {
  // Wavefront configuration (AMD's equivalent of warp)
  int wavefront_size = 64;  // 64 for CDNA, can be 32 for RDNA
  int num_wavefronts = 4;
  int num_threads = 256;    // Total threads per workgroup
  
  // Grid/block dimensions for kernel launch
  int forall_dim[3] = {256, 256, 64};  // Tiling dimensions
  int imap_dim[3] = {1, 1, 1};         // Input mapping dimensions
  
  // LDS (Local Data Share) configuration - equivalent to CUDA shared memory
  LDSLayout lds_layout = LDSLayout::SWIZZLED;
  size_t lds_size = 65536;  // 64KB default
  int lds_bank_size = 4;    // bytes per bank
  int lds_padding = 8;
  
  // Matrix Core configuration (CDNA)
  bool use_matrix_core = false;
  int mfma_m = 16;  // Matrix FMA dimensions
  int mfma_n = 16;
  int mfma_k = 16;
  
  // WMMA (Wavefront Matrix Multiply Accumulate)
  bool use_wmma = false;
  int wmma_m = 16;
  int wmma_n = 16;
  int wmma_k = 16;
  
  // Register usage
  int max_vgprs = 256;      // Vector GPRs
  int max_sgprs = 102;      // Scalar GPRs
  int vgpr_spill_threshold = 200;
  
  // Architecture
  ROCmArch arch = ROCmArch::CDNA3;
  std::string gfx_arch = "gfx942";  // MI300X default
  
  // Memory configuration
  int coalesce_width = 128;
  bool enable_flat_memory = true;
  
  // Kernel fusion
  bool enable_fusion = true;
  int max_fusion_depth = 3;
  
  ROCmKernelConfig() { 
    backend_type = type::BT_UNKNOWN;  // Use a generic type, add BT_ROCM later
  }
  
  // Get number of wavefronts
  int get_num_wavefronts() const {
    return (get_total_threads() + wavefront_size - 1) / wavefront_size;
  }
  
  // Get LDS size per wavefront
  size_t get_lds_per_wavefront() const {
    int num_wf = get_num_wavefronts();
    return num_wf > 0 ? lds_size / num_wf : 0;
  }
  
  // Check if Matrix Cores are available
  bool has_matrix_cores() const {
    return arch == ROCmArch::CDNA1 || arch == ROCmArch::CDNA2 || 
           arch == ROCmArch::CDNA3;
  }
};

/**
 * @brief ROCm kernel optimizer
 */
class ROCmOptimizer {
public:
  /**
   * @brief Compute optimal wavefront configuration
   */
  static int compute_optimal_wavefronts(size_t problem_size, ROCmArch arch);
  
  /**
   * @brief Compute optimal LDS configuration
   */
  static size_t compute_optimal_lds(size_t data_size, LDSLayout layout,
                                    int padding = 8);
  
  /**
   * @brief Check for LDS bank conflicts
   */
  static bool has_bank_conflict(LDSLayout layout, int stride, int bank_size);
  
  /**
   * @brief Estimate occupancy
   */
  static float estimate_occupancy(ROCmKernelConfig const &config,
                                  int vgprs_used, int sgprs_used);
  
  /**
   * @brief Select optimal Matrix Core configuration
   */
  static bool select_matrix_core_config(int m, int n, int k,
                                        ROCmArch arch,
                                        ROCmKernelConfig &config);
  
  /**
   * @brief Optimize grid and block dimensions
   */
  static void optimize_grid_block_dims(int problem_m, int problem_n,
                                       int problem_k, ROCmArch arch,
                                       ROCmKernelConfig &config);
  
  /**
   * @brief Get architecture string for compilation
   */
  static std::string get_arch_string(ROCmArch arch);
};

} // namespace rocm
} // namespace kernel
} // namespace yirage
