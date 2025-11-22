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


#include "yirage/kernel/cudnn/cudnn_kernel_config.h"

#ifdef YIRAGE_BACKEND_CUDNN_ENABLED

#include <iostream>

// Try to include cuDNN header if available
#ifdef __has_include
#if __has_include(<cudnn.h>)
#include <cudnn.h>
#define HAS_CUDNN_HEADER 1
#endif
#endif

namespace yirage {
namespace kernel {
namespace cudnn {

bool CUDNNOptimizer::is_cudnn_available() {
#ifdef HAS_CUDNN_HEADER
  return true;
#else
  return false;
#endif
}

int CUDNNOptimizer::get_cudnn_version() {
#ifdef HAS_CUDNN_HEADER
  return static_cast<int>(cudnnGetVersion());
#else
  return 0;
#endif
}

CUDNNKernelConfig::Algorithm
CUDNNOptimizer::select_algorithm(int m, int n, int k,
                                int compute_capability) {
  // For small problems, use direct convolution
  if (m < 64 && n < 64) {
    return CUDNNKernelConfig::Algorithm::DIRECT;
  }
  
  // For medium problems on newer GPUs, use Winograd
  if (compute_capability >= 70 && m >= 64 && m <= 512) {
    return CUDNNKernelConfig::Algorithm::WINOGRAD;
  }
  
  // For large problems, use implicit GEMM (best on most hardware)
  if (m >= 512 || n >= 512) {
    return CUDNNKernelConfig::Algorithm::IMPLICIT_GEMM;
  }
  
  // Default: let cuDNN auto-select
  return CUDNNKernelConfig::Algorithm::AUTO;
}

CUDNNKernelConfig::MathType
CUDNNOptimizer::select_math_type(int compute_capability,
                                type::DataType data_type) {
  // Tensor Cores available from Volta (SM 7.0+)
  if (compute_capability < 70) {
    return CUDNNKernelConfig::MathType::DEFAULT;
  }
  
  // Select based on data type and hardware
  if (data_type == type::DT_FLOAT16) {
    return CUDNNKernelConfig::MathType::TENSOR_OP_FP16;
  }
  
  // Ampere and later support TF32
  if (compute_capability >= 80 && data_type == type::DT_FLOAT32) {
    return CUDNNKernelConfig::MathType::TENSOR_OP_TF32;
  }
  
  // Default to Tensor Op for other cases
  return CUDNNKernelConfig::MathType::TENSOR_OP;
}

size_t CUDNNOptimizer::estimate_workspace_size(
    int m, int n, int k, CUDNNKernelConfig::Algorithm algorithm) {
  // Rough estimates based on algorithm
  size_t base_size = static_cast<size_t>(m) * n * k * sizeof(float);
  
  switch (algorithm) {
  case CUDNNKernelConfig::Algorithm::WINOGRAD:
    // Winograd needs more workspace
    return base_size * 4;
    
  case CUDNNKernelConfig::Algorithm::FFT:
    // FFT needs workspace for transforms
    return base_size * 2;
    
  case CUDNNKernelConfig::Algorithm::IMPLICIT_GEMM:
    // Implicit GEMM moderate workspace
    return base_size;
    
  case CUDNNKernelConfig::Algorithm::DIRECT:
    // Direct needs minimal workspace
    return base_size / 2;
    
  default:
    // AUTO or unknown
    return base_size * 2;
  }
}

void CUDNNOptimizer::optimize_for_cudnn(int m, int n, int k,
                                       CUDNNKernelConfig &config) {
  // Use CUDA optimizer as base
  cuda::CUDAOptimizer::optimize_grid_block_dims(m, n, k,
                                                config.compute_capability,
                                                config);
  
  // Select cuDNN-specific options
  config.algorithm = select_algorithm(m, n, k, config.compute_capability);
  
  // Select math type (Tensor Core usage)
  config.math_type = select_math_type(config.compute_capability,
                                     type::DT_FLOAT32);
  
  // Estimate and set workspace
  config.workspace_size = estimate_workspace_size(m, n, k, config.algorithm);
  
  // Enable Tensor Cores if available
  if (config.compute_capability >= 70) {
    config.use_tensor_core = true;
  }
}

void CUDNNOptimizer::set_mkl_env(CUDNNKernelConfig const &config) {
#ifdef HAS_CUDNN_HEADER
  // cuDNN environment setup would go here
  // Most cuDNN configuration is done through API calls, not env vars
#else
  std::cerr << "cuDNN not available" << std::endl;
#endif
}

} // namespace cudnn
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_CUDNN_ENABLED





