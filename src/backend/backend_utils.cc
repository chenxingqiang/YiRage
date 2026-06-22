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


#include "type.h"
#include <unordered_map>

namespace yirage {
namespace type {

std::string backend_type_to_string(BackendType type) {
  static std::unordered_map<BackendType, std::string> const type_to_string = {
      // ==========================================================================
      // Hardware Backends (Physical devices)
      // ==========================================================================
      // GPU Backends
      {BT_CUDA, "cuda"},
      {BT_MPS, "mps"},
      {BT_ROCM, "rocm"},
      {BT_ASCEND, "ascend"},
      {BT_MACA, "maca"},
      
      // CPU Backends
      {BT_CPU, "cpu"},
      
      // Accelerator Backends
      {BT_TPU, "tpu"},
      {BT_FPGA, "fpga"},
      {BT_XPU, "xpu"},
      
      // ==========================================================================
      // Library Backends (Software optimization libraries)
      // ==========================================================================
      // CUDA Libraries
      {BT_CUDNN, "cudnn"},
      {BT_CUSPARSELT, "cusparselt"},
      {BT_CUTLASS, "cutlass"},
      {BT_MHA, "mha"},
      
      // CPU Libraries
      {BT_MKL, "mkl"},
      {BT_MKLDNN, "mkldnn"},
      {BT_OPENMP, "openmp"},
      {BT_XEON, "xeon"},
      {BT_NNPACK, "nnpack"},
      {BT_OPT_EINSUM, "opt_einsum"},
      
      // ==========================================================================
      // DSL/Compiler Backends
      // ==========================================================================
      {BT_TRITON, "triton"},
      {BT_NKI, "nki"},
      
      // ==========================================================================
      // MLIR Ecosystem
      // ==========================================================================
      // Core MLIR
      {BT_MLIR, "mlir"},
      {BT_MLIR_LLVM, "mlir_llvm"},
      {BT_MLIR_NVVM, "mlir_nvvm"},
      {BT_MLIR_ROCDL, "mlir_rocdl"},
      {BT_MLIR_SPIRV, "mlir_spirv"},
      {BT_MLIR_GPU, "mlir_gpu"},
      
      // High-level MLIR Dialects
      {BT_STABLEHLO, "stablehlo"},
      {BT_MHLO, "mhlo"},
      {BT_TOSA, "tosa"},
      {BT_LINALG, "linalg"},
      {BT_TCP, "tcp"},
      
      // Vendor-specific MLIR
      {BT_IREE, "iree"},
      {BT_TVM, "tvm"},
      {BT_XLA, "xla"},
      
      {BT_UNKNOWN, "unknown"},
  };

  auto it = type_to_string.find(type);
  if (it != type_to_string.end()) {
    return it->second;
  }
  return "unknown";
}

BackendType string_to_backend_type(std::string const &name) {
  static std::unordered_map<std::string, BackendType> const string_to_type = {
      // ==========================================================================
      // Hardware Backends (Physical devices)
      // ==========================================================================
      // GPU Backends
      {"cuda", BT_CUDA},
      {"mps", BT_MPS},
      {"rocm", BT_ROCM},
      {"hip", BT_ROCM},  // Alias
      {"ascend", BT_ASCEND},
      {"npu", BT_ASCEND},  // Alias
      {"maca", BT_MACA},
      {"metax", BT_MACA},  // Alias
      
      // CPU Backends
      {"cpu", BT_CPU},
      
      // Accelerator Backends
      {"tpu", BT_TPU},
      {"fpga", BT_FPGA},
      {"xpu", BT_XPU},
      {"intel_gpu", BT_XPU},  // Alias
      
      // ==========================================================================
      // Library Backends (Software optimization libraries)
      // ==========================================================================
      // CUDA Libraries
      {"cudnn", BT_CUDNN},
      {"cusparselt", BT_CUSPARSELT},
      {"cutlass", BT_CUTLASS},
      {"mha", BT_MHA},
      
      // CPU Libraries
      {"mkl", BT_MKL},
      {"mkldnn", BT_MKLDNN},
      {"onednn", BT_MKLDNN},  // Alias
      {"openmp", BT_OPENMP},
      {"xeon", BT_XEON},
      {"nnpack", BT_NNPACK},
      {"opt_einsum", BT_OPT_EINSUM},
      
      // ==========================================================================
      // DSL/Compiler Backends
      // ==========================================================================
      {"triton", BT_TRITON},
      {"nki", BT_NKI},
      {"neuron", BT_NKI},  // Alias
      
      // ==========================================================================
      // MLIR Ecosystem
      // ==========================================================================
      // Core MLIR
      {"mlir", BT_MLIR},
      {"mlir_llvm", BT_MLIR_LLVM},
      {"mlir_nvvm", BT_MLIR_NVVM},
      {"mlir_rocdl", BT_MLIR_ROCDL},
      {"mlir_spirv", BT_MLIR_SPIRV},
      {"mlir_gpu", BT_MLIR_GPU},
      
      // High-level MLIR Dialects
      {"stablehlo", BT_STABLEHLO},
      {"mhlo", BT_MHLO},
      {"tosa", BT_TOSA},
      {"linalg", BT_LINALG},
      {"tcp", BT_TCP},
      
      // Vendor-specific MLIR
      {"iree", BT_IREE},
      {"tvm", BT_TVM},
      {"xla", BT_XLA},
      
      {"unknown", BT_UNKNOWN},
  };

  auto it = string_to_type.find(name);
  if (it != string_to_type.end()) {
    return it->second;
  }
  return BT_UNKNOWN;
}

} // namespace type
} // namespace yirage

