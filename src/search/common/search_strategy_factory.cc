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


#include "search/common/search_strategy.h"
#include "type.h"
#include <iostream>

// Include backend-specific strategies
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
#include "search/backend_strategies/cuda_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
#include "search/backend_strategies/cpu_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
#include "search/backend_strategies/mps_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_TRITON_ENABLED
#include "search/backend_strategies/triton_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
#include "search/backend_strategies/nki_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
#include "search/backend_strategies/ascend_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_MACA_ENABLED
#include "search/backend_strategies/maca_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
#include "search/backend_strategies/rocm_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_TPU_ENABLED
#include "search/backend_strategies/tpu_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
#include "search/backend_strategies/fpga_strategy.h"
#endif

#ifdef YIRAGE_BACKEND_XPU_ENABLED
#include "search/backend_strategies/xpu_strategy.h"
#endif

// MLIR ecosystem strategies
#ifdef YIRAGE_BACKEND_MLIR_ENABLED
#include "search/backend_strategies/mlir_strategy.h"
#endif

// StableHLO/TOSA/Linalg can also use MLIR strategy when MLIR is enabled
#if defined(YIRAGE_BACKEND_STABLEHLO_ENABLED) || \
    defined(YIRAGE_BACKEND_TVM_ENABLED) || \
    defined(YIRAGE_BACKEND_IREE_ENABLED)
#include "search/backend_strategies/mlir_strategy.h"
#endif

// Note: Library backends (CUDNN, MKL, etc.) use hardware backend strategies
// MLIR backends provide universal IR compilation to multiple targets

namespace yirage {
namespace search {

// =============================================================================
// Internal: Create strategy for hardware backend
// =============================================================================
static std::unique_ptr<SearchStrategy>
create_hardware_strategy(type::BackendType backend) {
  switch (backend) {
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  case type::BT_CUDA:
    return std::make_unique<CUDASearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
  case type::BT_CPU:
    return std::make_unique<CPUSearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
  case type::BT_MPS:
    return std::make_unique<MPSSearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_TRITON_ENABLED
  case type::BT_TRITON:
    return std::make_unique<TritonSearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
  case type::BT_NKI:
    return std::make_unique<NKISearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
  case type::BT_ASCEND:
    return std::make_unique<AscendSearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_MACA_ENABLED
  case type::BT_MACA:
    return std::make_unique<MACASearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
  case type::BT_ROCM:
    return std::make_unique<ROCmSearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_TPU_ENABLED
  case type::BT_TPU:
    return std::make_unique<TPUSearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
  case type::BT_FPGA:
    return std::make_unique<FPGASearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_XPU_ENABLED
  case type::BT_XPU:
    return std::make_unique<XPUSearchStrategy>();
#endif

  // ==========================================================================
  // MLIR Ecosystem Backends
  // ==========================================================================
#ifdef YIRAGE_BACKEND_MLIR_ENABLED
  case type::BT_MLIR:
  case type::BT_MLIR_LLVM:
  case type::BT_MLIR_GPU:
    return std::make_unique<MLIRSearchStrategy>();
  
  case type::BT_MLIR_NVVM:
    {
      auto strategy = std::make_unique<MLIRSearchStrategy>();
      strategy->set_target_backend(type::BT_CUDA);
      return strategy;
    }
  
  case type::BT_MLIR_ROCDL:
    {
      auto strategy = std::make_unique<MLIRSearchStrategy>();
      strategy->set_target_backend(type::BT_ROCM);
      return strategy;
    }
  
  case type::BT_MLIR_SPIRV:
    {
      auto strategy = std::make_unique<MLIRSearchStrategy>();
      strategy->set_target_backend(type::BT_XPU);
      return strategy;
    }
  
  case type::BT_LINALG:
    return std::make_unique<LinalgSearchStrategy>();
#endif

#if defined(YIRAGE_BACKEND_STABLEHLO_ENABLED) || defined(YIRAGE_BACKEND_MLIR_ENABLED)
  case type::BT_STABLEHLO:
  case type::BT_MHLO:
  case type::BT_XLA:
    return std::make_unique<StableHLOSearchStrategy>();
  
  case type::BT_TOSA:
    return std::make_unique<TOSASearchStrategy>();
#endif

#ifdef YIRAGE_BACKEND_TVM_ENABLED
  case type::BT_TVM:
    {
      auto strategy = std::make_unique<MLIRSearchStrategy>();
      // TVM can target multiple backends
      return strategy;
    }
#endif

#ifdef YIRAGE_BACKEND_IREE_ENABLED
  case type::BT_IREE:
    {
      auto strategy = std::make_unique<MLIRSearchStrategy>();
      // IREE runtime handles target selection
      return strategy;
    }
#endif

  default:
    return nullptr;
  }
}

// =============================================================================
// Public API: Create strategy with fallback support
// =============================================================================
// 
// Strategy resolution order:
//   1. For hardware backends: Create directly
//   2. For DSL backends (TRITON, NKI): Try dedicated strategy first, fallback if unavailable
//   3. For library backends (CUDNN, MKL, etc.): Always fallback to hardware backend
//
std::unique_ptr<SearchStrategy>
SearchStrategyFactory::create_strategy(type::BackendType backend,
                                      SearchConfig const &config) {
  std::unique_ptr<SearchStrategy> strategy;
  type::BackendType effective_backend = backend;
  bool used_fallback = false;
  
  // Case 1: Hardware backend - create directly
  if (type::is_hardware_backend(backend)) {
    strategy = create_hardware_strategy(backend);
    effective_backend = backend;
  }
  // Case 2: DSL/Compiler backend - try dedicated strategy first
  else if (type::is_dsl_backend(backend)) {
    // First try the dedicated DSL strategy
    strategy = create_hardware_strategy(backend);
    
    if (!strategy) {
      // DSL strategy not available, fallback to hardware
      type::BackendType fallback = type::get_fallback_backend(backend);
      const char* lib_name = type::get_software_library_name(backend);
      
      std::cout << "[SearchStrategy] DSL backend '" 
                << (lib_name ? lib_name : "unknown")
                << "' strategy not available, fallback to "
                << type::backend_type_to_string(fallback) << std::endl;
      
      strategy = create_hardware_strategy(fallback);
      effective_backend = fallback;
      used_fallback = true;
    }
  }
  // Case 3: Library backend - always fallback to hardware
  else if (type::is_library_backend(backend)) {
    type::BackendType fallback = type::get_fallback_backend(backend);
    const char* lib_name = type::get_software_library_name(backend);
    
    std::cout << "[SearchStrategy] Library backend '" 
              << (lib_name ? lib_name : "unknown")
              << "' -> using " << type::backend_type_to_string(fallback) 
              << " hardware strategy" << std::endl;
    
    strategy = create_hardware_strategy(fallback);
    effective_backend = fallback;
    used_fallback = true;
  }
  // Case 4: Unknown backend - try fallback
  else {
    type::BackendType fallback = type::get_fallback_backend(backend);
    if (fallback != type::BT_UNKNOWN) {
      std::cerr << "[SearchStrategy] Unknown backend " << static_cast<int>(backend)
                << ", trying fallback to " << static_cast<int>(fallback) << std::endl;
      strategy = create_hardware_strategy(fallback);
      effective_backend = fallback;
      used_fallback = true;
    }
  }
  
  // Final fallback attempt if still no strategy
  if (!strategy && effective_backend != type::BT_UNKNOWN) {
    type::BackendType final_fallback = type::get_fallback_backend(effective_backend);
    if (final_fallback != effective_backend && final_fallback != type::BT_UNKNOWN) {
      std::cerr << "[SearchStrategy] Backend " << static_cast<int>(effective_backend)
                << " not available, trying final fallback to " 
                << static_cast<int>(final_fallback) << std::endl;
      strategy = create_hardware_strategy(final_fallback);
      effective_backend = final_fallback;
      used_fallback = true;
    }
  }
  
  if (!strategy) {
    std::cerr << "[SearchStrategy] No search strategy available for backend: "
              << static_cast<int>(backend) 
              << " (effective: " << static_cast<int>(effective_backend) << ")"
              << std::endl;
    return nullptr;
  }

  if (!strategy->initialize(config)) {
    std::cerr << "[SearchStrategy] Failed to initialize search strategy" << std::endl;
    return nullptr;
  }
  
  // Log successful creation with fallback info
  if (used_fallback) {
    const char* lib_name = type::get_software_library_name(backend);
    std::cout << "[SearchStrategy] Using " 
              << type::backend_type_to_string(effective_backend)
              << " strategy for " << (lib_name ? lib_name : "backend")
              << " optimization" << std::endl;
  }

  return strategy;
}

// =============================================================================
// Internal: Check if hardware strategy is available
// =============================================================================
static bool has_hardware_strategy(type::BackendType backend) {
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

#ifdef YIRAGE_BACKEND_MACA_ENABLED
  case type::BT_MACA:
    return true;
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
  case type::BT_ROCM:
    return true;
#endif

#ifdef YIRAGE_BACKEND_TPU_ENABLED
  case type::BT_TPU:
    return true;
#endif

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
  case type::BT_FPGA:
    return true;
#endif

#ifdef YIRAGE_BACKEND_XPU_ENABLED
  case type::BT_XPU:
    return true;
#endif

  // MLIR ecosystem
#ifdef YIRAGE_BACKEND_MLIR_ENABLED
  case type::BT_MLIR:
  case type::BT_MLIR_LLVM:
  case type::BT_MLIR_NVVM:
  case type::BT_MLIR_ROCDL:
  case type::BT_MLIR_SPIRV:
  case type::BT_MLIR_GPU:
  case type::BT_LINALG:
    return true;
#endif

#if defined(YIRAGE_BACKEND_STABLEHLO_ENABLED) || defined(YIRAGE_BACKEND_MLIR_ENABLED)
  case type::BT_STABLEHLO:
  case type::BT_MHLO:
  case type::BT_TOSA:
  case type::BT_XLA:
    return true;
#endif

#ifdef YIRAGE_BACKEND_TVM_ENABLED
  case type::BT_TVM:
    return true;
#endif

#ifdef YIRAGE_BACKEND_IREE_ENABLED
  case type::BT_IREE:
    return true;
#endif

  default:
    return false;
  }
}

// =============================================================================
// Public API: Check strategy availability with fallback
// =============================================================================
bool SearchStrategyFactory::has_strategy(type::BackendType backend) {
  // Direct check
  if (has_hardware_strategy(backend)) {
    return true;
  }
  
  // Check via fallback for software backends
  if (type::is_software_backend(backend)) {
    type::BackendType fallback = type::get_fallback_backend(backend);
    if (fallback != type::BT_UNKNOWN) {
      return has_hardware_strategy(fallback);
    }
  }
  
  return false;
}

// =============================================================================
// Get the effective backend (after fallback resolution)
// =============================================================================
type::BackendType 
SearchStrategyFactory::get_effective_backend(type::BackendType backend) {
  if (type::is_software_backend(backend)) {
    return type::get_fallback_backend(backend);
  }
  return backend;
}

// =============================================================================
// Get all supported backends (including software backends via fallback)
// =============================================================================
std::vector<type::BackendType>
SearchStrategyFactory::get_supported_backends() {
  std::vector<type::BackendType> backends;
  
  // Hardware backends
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  backends.push_back(type::BT_CUDA);
  // Software backends that fallback to CUDA
  backends.push_back(type::BT_CUDNN);
  backends.push_back(type::BT_CUSPARSELT);
  backends.push_back(type::BT_MHA);
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
  backends.push_back(type::BT_CPU);
  // Software backends that fallback to CPU
  backends.push_back(type::BT_MKL);
  backends.push_back(type::BT_MKLDNN);
  backends.push_back(type::BT_OPENMP);
  backends.push_back(type::BT_XEON);
  backends.push_back(type::BT_NNPACK);
  backends.push_back(type::BT_OPT_EINSUM);
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
  backends.push_back(type::BT_MPS);
#endif

#ifdef YIRAGE_BACKEND_TRITON_ENABLED
  backends.push_back(type::BT_TRITON);
#endif

#ifdef YIRAGE_BACKEND_NKI_ENABLED
  backends.push_back(type::BT_NKI);
#endif

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED
  backends.push_back(type::BT_ASCEND);
#endif

#ifdef YIRAGE_BACKEND_MACA_ENABLED
  backends.push_back(type::BT_MACA);
#endif

#ifdef YIRAGE_BACKEND_ROCM_ENABLED
  backends.push_back(type::BT_ROCM);
#endif

#ifdef YIRAGE_BACKEND_TPU_ENABLED
  backends.push_back(type::BT_TPU);
#endif

#ifdef YIRAGE_BACKEND_FPGA_ENABLED
  backends.push_back(type::BT_FPGA);
#endif

#ifdef YIRAGE_BACKEND_XPU_ENABLED
  backends.push_back(type::BT_XPU);
#endif

  return backends;
}

} // namespace search
} // namespace yirage

