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
 * This file is part of YiRage (Yi Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */


#pragma once

#include "yirage/kernel/cuda/cuda_kernel_config.h"

#ifdef YIRAGE_BACKEND_CUDNN_ENABLED

namespace yirage {
namespace kernel {
namespace cudnn {

/**
 * @brief cuDNN-specific kernel configuration
 * 
 * Extends CUDA configuration with cuDNN library options
 */
struct CUDNNKernelConfig : public cuda::CUDAKernelConfig {
  // cuDNN algorithm selection
  enum class Algorithm {
    AUTO,           // Let cuDNN auto-select
    IMPLICIT_GEMM,  // Implicit GEMM
    WINOGRAD,       // Winograd transform
    FFT,            // FFT-based
    DIRECT          // Direct convolution
  };
  Algorithm algorithm = Algorithm::AUTO;
  
  // Tensor Core options
  enum class MathType {
    DEFAULT,
    TENSOR_OP,      // Use Tensor Cores
    TENSOR_OP_FP16, // FP16 Tensor Cores
    TENSOR_OP_TF32  // TF32 Tensor Cores (Ampere+)
  };
  MathType math_type = MathType::TENSOR_OP;
  
  // Workspace configuration
  size_t workspace_size = 256 * 1024 * 1024; // 256 MB default
  bool allow_workspace = true;
  
  // Convolution mode
  enum class ConvMode {
    CONVOLUTION,
    CROSS_CORRELATION
  };
  ConvMode conv_mode = ConvMode::CROSS_CORRELATION;
  
  // Determinism
  bool deterministic = false;
  
  // cuDNN version requirement
  int min_cudnn_version = 8000; // cuDNN 8.0+

  CUDNNKernelConfig() { backend_type = type::BT_CUDNN; }
};

/**
 * @brief cuDNN kernel optimizer
 */
class CUDNNOptimizer {
public:
  /**
   * @brief Check if cuDNN is available
   * @return true if cuDNN library is loaded
   */
  static bool is_cudnn_available();

  /**
   * @brief Get cuDNN version
   * @return cuDNN version number
   */
  static int get_cudnn_version();

  /**
   * @brief Select optimal algorithm
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param compute_capability CUDA compute capability
   * @return Optimal algorithm
   */
  static CUDNNKernelConfig::Algorithm
  select_algorithm(int m, int n, int k, int compute_capability);

  /**
   * @brief Select optimal math type
   * @param compute_capability CUDA compute capability
   * @param data_type Data type
   * @return Optimal math type
   */
  static CUDNNKernelConfig::MathType
  select_math_type(int compute_capability, type::DataType data_type);

  /**
   * @brief Estimate workspace size needed
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param algorithm Algorithm choice
   * @return Estimated workspace size in bytes
   */
  static size_t estimate_workspace_size(int m, int n, int k,
                                       CUDNNKernelConfig::Algorithm algorithm);

  /**
   * @brief Optimize configuration for cuDNN
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param config Configuration to optimize
   */
  static void optimize_for_cudnn(int m, int n, int k,
                                CUDNNKernelConfig &config);
};

} // namespace cudnn
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_CUDNN_ENABLED





