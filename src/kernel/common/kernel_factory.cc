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


#include "yirage/kernel/common/kernel_interface.h"
#include <iostream>

// Include backend-specific implementations
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
// TODO: Include CUDA kernel executors when implemented
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
// TODO: Include CPU kernel executors when implemented
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
// TODO: Include MPS kernel executors when implemented
#endif

namespace yirage {
namespace kernel {

std::unique_ptr<KernelExecutor>
KernelExecutorFactory::create_matmul_executor(type::BackendType backend) {
  switch (backend) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  case type::BT_CUDA:
    // TODO: Return CUDA matmul executor
    std::cerr << "CUDA MatMul executor not yet implemented" << std::endl;
    return nullptr;
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
  case type::BT_CPU:
    // TODO: Return CPU matmul executor
    std::cerr << "CPU MatMul executor not yet implemented" << std::endl;
    return nullptr;
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
  case type::BT_MPS:
    // TODO: Return MPS matmul executor
    std::cerr << "MPS MatMul executor not yet implemented" << std::endl;
    return nullptr;
#endif

  default:
    std::cerr << "Backend not supported for MatMul: "
              << static_cast<int>(backend) << std::endl;
    return nullptr;
  }
}

std::unique_ptr<KernelExecutor>
KernelExecutorFactory::create_rmsnorm_executor(type::BackendType backend) {
  switch (backend) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  case type::BT_CUDA:
    std::cerr << "CUDA RMSNorm executor not yet implemented" << std::endl;
    return nullptr;
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
  case type::BT_CPU:
    std::cerr << "CPU RMSNorm executor not yet implemented" << std::endl;
    return nullptr;
#endif

  default:
    std::cerr << "Backend not supported for RMSNorm: "
              << static_cast<int>(backend) << std::endl;
    return nullptr;
  }
}

std::unique_ptr<KernelExecutor>
KernelExecutorFactory::create_reduction_executor(type::BackendType backend) {
  switch (backend) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  case type::BT_CUDA:
    std::cerr << "CUDA Reduction executor not yet implemented" << std::endl;
    return nullptr;
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
  case type::BT_CPU:
    std::cerr << "CPU Reduction executor not yet implemented" << std::endl;
    return nullptr;
#endif

  default:
    std::cerr << "Backend not supported for Reduction: "
              << static_cast<int>(backend) << std::endl;
    return nullptr;
  }
}

std::unique_ptr<KernelExecutor>
KernelExecutorFactory::create_element_unary_executor(
    type::BackendType backend, type::KNOperatorType op_type) {
  switch (backend) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  case type::BT_CUDA:
    std::cerr << "CUDA ElementUnary executor not yet implemented" << std::endl;
    return nullptr;
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
  case type::BT_CPU:
    std::cerr << "CPU ElementUnary executor not yet implemented" << std::endl;
    return nullptr;
#endif

  default:
    std::cerr << "Backend not supported for ElementUnary: "
              << static_cast<int>(backend) << std::endl;
    return nullptr;
  }
}

std::unique_ptr<KernelExecutor>
KernelExecutorFactory::create_element_binary_executor(
    type::BackendType backend, type::KNOperatorType op_type) {
  switch (backend) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  case type::BT_CUDA:
    std::cerr << "CUDA ElementBinary executor not yet implemented"
              << std::endl;
    return nullptr;
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
  case type::BT_CPU:
    std::cerr << "CPU ElementBinary executor not yet implemented" << std::endl;
    return nullptr;
#endif

  default:
    std::cerr << "Backend not supported for ElementBinary: "
              << static_cast<int>(backend) << std::endl;
    return nullptr;
  }
}

} // namespace kernel
} // namespace yirage





