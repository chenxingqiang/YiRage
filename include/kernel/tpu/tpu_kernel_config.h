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
 * TPU (Google Tensor Processing Unit) Kernel Configuration
 * 
 * TPU kernels are typically written using:
 * - XLA (Accelerated Linear Algebra) HLO
 * - Pallas (JAX-based kernel language)
 */

#pragma once

#include "kernel/common/kernel_interface.h"

namespace yirage {
namespace kernel {
namespace tpu {

/**
 * @brief TPU version types
 */
enum class TPUVersion {
  V2,     // TPU v2 (45 TFLOPS BF16)
  V3,     // TPU v3 (90 TFLOPS BF16)
  V4,     // TPU v4 (275 TFLOPS BF16)
  V5E,    // TPU v5e (197 TFLOPS BF16)
  V5P,    // TPU v5p (High performance)
  UNKNOWN
};

/**
 * @brief Memory layout for TPU
 */
enum class TPUMemoryLayout {
  XY,       // Standard 2D layout
  YX,       // Transposed 2D layout
  XYZ,      // 3D layout
  TILED,    // Tiled layout for MXU
  PACKED    // Packed for low-precision
};

/**
 * @brief TPU-specific kernel configuration
 */
struct TPUKernelConfig : public KernelConfig {
  // TPU version
  TPUVersion version = TPUVersion::V4;
  
  // MXU (Matrix Multiply Unit) configuration
  int mxu_size = 128;         // MXU is 128x128 on v4
  bool use_mxu = true;
  
  // Vector unit configuration
  int vector_width = 128;     // VPU width
  bool use_vector_unit = true;
  
  // Memory configuration
  TPUMemoryLayout input_layout = TPUMemoryLayout::TILED;
  TPUMemoryLayout output_layout = TPUMemoryLayout::TILED;
  size_t vmem_size = 16 * 1024 * 1024;  // 16MB VMEM per core
  size_t cmem_size = 4 * 1024 * 1024;   // 4MB CMEM
  
  // Tiling configuration
  int tile_m = 128;
  int tile_n = 128;
  int tile_k = 128;
  
  // Pipeline configuration
  int pipeline_depth = 2;
  bool enable_double_buffering = true;
  
  // Precision
  bool use_bf16 = true;
  bool use_int8 = false;
  
  // XLA/Pallas configuration
  bool generate_xla = true;
  bool generate_pallas = false;
  std::string xla_backend = "tpu";
  
  // ICI (Inter-Chip Interconnect) for multi-TPU
  bool enable_ici = true;
  int ici_mesh_dim_x = 1;
  int ici_mesh_dim_y = 1;
  int ici_mesh_dim_z = 1;
  
  TPUKernelConfig() {
    backend_type = type::BT_TPU;
  }
  
  // Get MXU size based on version
  int get_mxu_size() const {
    switch (version) {
      case TPUVersion::V2:
      case TPUVersion::V3:
        return 128;
      case TPUVersion::V4:
      case TPUVersion::V5E:
      case TPUVersion::V5P:
        return 128;
      default:
        return 128;
    }
  }
  
  // Get VMEM size based on version
  size_t get_vmem_size() const {
    switch (version) {
      case TPUVersion::V2:
        return 8 * 1024 * 1024;   // 8MB
      case TPUVersion::V3:
        return 16 * 1024 * 1024;  // 16MB
      case TPUVersion::V4:
      case TPUVersion::V5E:
      case TPUVersion::V5P:
        return 32 * 1024 * 1024;  // 32MB
      default:
        return 16 * 1024 * 1024;
    }
  }
};

/**
 * @brief TPU kernel optimizer
 */
class TPUOptimizer {
public:
  /**
   * @brief Compute optimal tiling for MXU
   */
  static void compute_optimal_tiling(int m, int n, int k,
                                     TPUVersion version,
                                     TPUKernelConfig &config);
  
  /**
   * @brief Estimate memory usage
   */
  static size_t estimate_memory_usage(TPUKernelConfig const &config,
                                      int m, int n, int k);
  
  /**
   * @brief Check if problem fits in VMEM
   */
  static bool fits_in_vmem(TPUKernelConfig const &config,
                           size_t required_bytes);
  
  /**
   * @brief Optimize pipeline configuration
   */
  static void optimize_pipeline(TPUKernelConfig &config,
                                int m, int n, int k);
  
  /**
   * @brief Generate XLA HLO for matmul
   */
  static std::string generate_matmul_xla(int m, int n, int k,
                                         TPUKernelConfig const &config);
  
  /**
   * @brief Generate Pallas kernel for custom operation
   */
  static std::string generate_pallas_kernel(std::string const &op_name,
                                            TPUKernelConfig const &config);
  
  /**
   * @brief Get optimal mesh configuration for multi-TPU
   */
  static void get_optimal_mesh(int num_tpus, int &mesh_x, int &mesh_y);
};

} // namespace tpu
} // namespace kernel
} // namespace yirage
