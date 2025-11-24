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


#include "yirage/search/common/search_strategy.h"
#include <iostream>

// Include backend-specific strategies
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#include "yirage/search/backend_strategies/cuda_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
#include "yirage/search/backend_strategies/cpu_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
#include "yirage/search/backend_strategies/mps_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_TRITON_ENABLED
#include "yirage/search/backend_strategies/triton_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
#include "yirage/search/backend_strategies/nki_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
#include "yirage/search/backend_strategies/ascend_strategy.h"
#endif

// Note: CUDNN and MKL can reuse CUDA and CPU strategies respectively
// If needed, dedicated strategies can be implemented later

namespace yirage {
namespace search {

std::unique_ptr<SearchStrategy>
SearchStrategyFactory::create_strategy(type::BackendType backend,
                                      SearchConfig const &config) {
  std::unique_ptr<SearchStrategy> strategy;

  switch (backend) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  case type::BT_CUDA: {
    strategy = std::make_unique<CUDASearchStrategy>();
    break;
  }
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
  case type::BT_CPU: {
    strategy = std::make_unique<CPUSearchStrategy>();
    break;
  }
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
  case type::BT_MPS: {
    strategy = std::make_unique<MPSSearchStrategy>();
    break;
  }
#endif

#ifdef YIRAGE_BACKEND_TRITON_ENABLED
  case type::BT_TRITON: {
    strategy = std::make_unique<TritonSearchStrategy>();
    break;
  }
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
  case type::BT_NKI: {
    strategy = std::make_unique<NKISearchStrategy>();
    break;
  }
#endif

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
  case type::BT_ASCEND: {
    strategy = std::make_unique<AscendSearchStrategy>();
    break;
  }
#endif

  default:
    std::cerr << "No search strategy available for backend: "
              << static_cast<int>(backend) << std::endl;
    return nullptr;
  }

  if (strategy) {
    if (!strategy->initialize(config)) {
      std::cerr << "Failed to initialize search strategy" << std::endl;
      return nullptr;
    }
  }

  return strategy;
}

bool SearchStrategyFactory::has_strategy(type::BackendType backend) {
  switch (backend) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  case type::BT_CUDA:
    return true;
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
  case type::BT_CPU:
    return true;
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
  case type::BT_MPS:
    return true;
#endif

#ifdef YIRAGE_BACKEND_TRITON_ENABLED
  case type::BT_TRITON:
    return true;
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
  case type::BT_NKI:
    return true;
#endif

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
  case type::BT_ASCEND:
    return true;
#endif

  default:
    return false;
  }
}

} // namespace search
} // namespace yirage

