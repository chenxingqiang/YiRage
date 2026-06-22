/* Copyright 2023-2024 CMU
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
 */

#pragma once

#include "utils/json_utils.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace yirage {
namespace type {

typedef uint16_t FPType;
typedef int64_t GuidType;

// only to be used in create_op in search.cc
inline std::unordered_map<std::string, float> CLAMP_MIN_MAX;

enum BackendType {
  // ==========================================================================
  // Hardware Backends (Physical devices)
  // ==========================================================================
  // GPU Backends
  BT_CUDA = 0,
  BT_MPS = 1,
  BT_ROCM = 6,           // AMD ROCm/HIP GPU
  BT_ASCEND = 4,         // Huawei Ascend NPU/GPU
  BT_MACA = 5,           // MetaX MACA GPU (CUDA-compatible)
  
  // CPU Backends
  BT_CPU = 10,
  
  // Accelerator Backends
  BT_TPU = 30,           // Google TPU (v2-v5)
  BT_FPGA = 31,          // FPGA (Xilinx/Intel)
  BT_XPU = 32,           // Intel XPU (Data Center GPU Max)
  
  // ==========================================================================
  // Library Backends (Software optimization libraries → Hardware)
  // ==========================================================================
  // CUDA Libraries
  BT_CUDNN = 2,          // cuDNN → CUDA
  BT_CUSPARSELT = 3,     // cuSPARSELt → CUDA
  BT_CUTLASS = 7,        // CUTLASS → CUDA
  BT_MHA = 22,           // Multi-Head Attention → CUDA
  
  // CPU Libraries
  BT_MKL = 11,           // Intel MKL → CPU
  BT_MKLDNN = 12,        // oneDNN (MKL-DNN) → CPU
  BT_OPENMP = 13,        // OpenMP → CPU
  BT_XEON = 14,          // Intel Xeon Optimized → CPU
  BT_NNPACK = 23,        // NNPACK → CPU
  BT_OPT_EINSUM = 24,    // opt_einsum → CPU
  
  // ==========================================================================
  // DSL/Compiler Backends (Domain-specific languages → MLIR → Hardware)
  // ==========================================================================
  BT_TRITON = 21,        // OpenAI Triton → MLIR/LLVM → CUDA/ROCm
  BT_NKI = 20,           // Neuron Kernel Interface → Trainium/Inferentia
  
  // ==========================================================================
  // MLIR Ecosystem (Multi-Level IR → Multiple targets)
  // ==========================================================================
  // Core MLIR
  BT_MLIR = 40,          // Generic MLIR (auto-select lowering)
  BT_MLIR_LLVM = 41,     // MLIR → LLVM IR → CPU/GPU
  
  // MLIR Dialects for specific targets
  BT_MLIR_NVVM = 42,     // MLIR → NVVM → CUDA
  BT_MLIR_ROCDL = 43,    // MLIR → ROCDL → ROCm
  BT_MLIR_SPIRV = 44,    // MLIR → SPIR-V → Intel GPU/Vulkan
  BT_MLIR_GPU = 45,      // MLIR GPU dialect (generic)
  
  // High-level MLIR Dialects
  BT_STABLEHLO = 50,     // StableHLO → TPU/GPU/CPU
  BT_MHLO = 51,          // MHLO (XLA HLO in MLIR) → TPU/GPU
  BT_TOSA = 52,          // TOSA (Tensor Operator Set) → various accelerators
  BT_LINALG = 53,        // Linalg dialect → various targets
  BT_TCP = 54,           // Torch-MLIR TCP dialect
  
  // Vendor-specific MLIR
  BT_IREE = 55,          // IREE runtime → CPU/GPU/TPU
  BT_TVM = 56,           // Apache TVM → various targets
  BT_XLA = 57,           // XLA (via MLIR) → TPU/GPU/CPU
  
  BT_UNKNOWN = 999,
};

// =============================================================================
// MLIR Dialect Types
// =============================================================================
enum MLIRDialect {
  MLIR_DIALECT_UNKNOWN = 0,
  
  // Core dialects
  MLIR_DIALECT_BUILTIN = 1,
  MLIR_DIALECT_ARITH = 2,
  MLIR_DIALECT_FUNC = 3,
  MLIR_DIALECT_SCF = 4,       // Structured Control Flow
  MLIR_DIALECT_AFFINE = 5,
  MLIR_DIALECT_MEMREF = 6,
  MLIR_DIALECT_TENSOR = 7,
  MLIR_DIALECT_VECTOR = 8,
  
  // Computation dialects
  MLIR_DIALECT_LINALG = 10,
  MLIR_DIALECT_TOSA = 11,
  MLIR_DIALECT_STABLEHLO = 12,
  MLIR_DIALECT_MHLO = 13,
  
  // Target dialects
  MLIR_DIALECT_LLVM = 20,
  MLIR_DIALECT_NVVM = 21,
  MLIR_DIALECT_ROCDL = 22,
  MLIR_DIALECT_SPIRV = 23,
  MLIR_DIALECT_GPU = 24,
  MLIR_DIALECT_AMX = 25,      // Intel AMX
  MLIR_DIALECT_X86VECTOR = 26,
  MLIR_DIALECT_ARM_NEON = 27,
  MLIR_DIALECT_ARM_SVE = 28,
  
  // DSL dialects
  MLIR_DIALECT_TRITON = 30,
  MLIR_DIALECT_TCP = 31,      // Torch-MLIR
};

// MLIR lowering target
struct MLIRLoweringTarget {
  BackendType target_backend;
  MLIRDialect intermediate_dialect;
  MLIRDialect target_dialect;
  std::string pipeline;  // MLIR pass pipeline
};

// Backend metadata structure
struct BackendInfo {
  BackendType type;
  std::string name;
  std::string display_name;
  bool requires_gpu;
  std::vector<std::string> required_libs;
  
  BackendInfo() 
    : type(BT_UNKNOWN), requires_gpu(false) {}
  
  BackendInfo(BackendType t, std::string const &n, 
              std::string const &dn, bool gpu,
              std::vector<std::string> const &libs = {})
    : type(t), name(n), display_name(dn), 
      requires_gpu(gpu), required_libs(libs) {}
};

// Convert backend type to string
std::string backend_type_to_string(BackendType type);

// Convert string to backend type
BackendType string_to_backend_type(std::string const &name);

// =============================================================================
// Backend Classification & Fallback Mapping
// =============================================================================
// 
// Backend hierarchy:
//   1. Hardware backends: Direct hardware (CUDA, CPU, MPS, ROCm, etc.)
//   2. MLIR backends: Compiler IR that lowers to hardware (MLIR, StableHLO, etc.)
//   3. DSL backends: Domain-specific languages (TRITON, NKI)
//   4. Library backends: Pure software optimization (CUDNN, MKL, etc.)
//
// Lowering chain:
//   DSL → MLIR → Hardware
//   Library → Hardware (direct)
//
// Example: TRITON → MLIR (Triton dialect) → NVVM → CUDA
//          StableHLO → MHLO → TPU/GPU/CPU

// =============================================================================
// MLIR Backend Mapping
// =============================================================================

// Check if backend is an MLIR-based backend
inline bool is_mlir_backend(BackendType type) {
  switch (type) {
    case BT_MLIR:
    case BT_MLIR_LLVM:
    case BT_MLIR_NVVM:
    case BT_MLIR_ROCDL:
    case BT_MLIR_SPIRV:
    case BT_MLIR_GPU:
    case BT_STABLEHLO:
    case BT_MHLO:
    case BT_TOSA:
    case BT_LINALG:
    case BT_TCP:
    case BT_IREE:
    case BT_TVM:
    case BT_XLA:
      return true;
    default:
      return false;
  }
}

// Get the target hardware for MLIR-based backends
inline BackendType get_mlir_target_backend(BackendType mlir_backend, 
                                           BackendType preferred = BT_UNKNOWN) {
  switch (mlir_backend) {
    // Target-specific MLIR dialects
    case BT_MLIR_NVVM:
      return BT_CUDA;
    case BT_MLIR_ROCDL:
      return BT_ROCM;
    case BT_MLIR_SPIRV:
      return (preferred == BT_XPU) ? BT_XPU : BT_CUDA;  // SPIR-V can target Intel or via Vulkan
    
    // Generic MLIR - select based on availability
    case BT_MLIR:
    case BT_MLIR_LLVM:
    case BT_MLIR_GPU:
      if (preferred != BT_UNKNOWN) return preferred;
      return BT_CUDA;  // Default to CUDA if available
    
    // High-level dialects - can target multiple backends
    case BT_STABLEHLO:
    case BT_MHLO:
      if (preferred != BT_UNKNOWN) return preferred;
      return BT_TPU;  // StableHLO/MHLO originated from XLA for TPU
    
    case BT_TOSA:
    case BT_LINALG:
      if (preferred != BT_UNKNOWN) return preferred;
      return BT_CPU;  // TOSA/Linalg often used for CPU/edge
    
    case BT_TCP:
      if (preferred != BT_UNKNOWN) return preferred;
      return BT_CUDA;  // Torch-MLIR typically targets GPU
    
    // Runtime backends
    case BT_IREE:
      if (preferred != BT_UNKNOWN) return preferred;
      return BT_CPU;  // IREE has good CPU support
    
    case BT_TVM:
      if (preferred != BT_UNKNOWN) return preferred;
      return BT_CUDA;  // TVM commonly targets GPU
    
    case BT_XLA:
      if (preferred != BT_UNKNOWN) return preferred;
      return BT_TPU;  // XLA is TPU-focused
    
    default:
      return BT_UNKNOWN;
  }
}

// Get the recommended MLIR lowering path for a hardware target
inline BackendType get_mlir_dialect_for_target(BackendType hw_backend) {
  switch (hw_backend) {
    case BT_CUDA:
      return BT_MLIR_NVVM;
    case BT_ROCM:
      return BT_MLIR_ROCDL;
    case BT_XPU:
      return BT_MLIR_SPIRV;
    case BT_TPU:
      return BT_STABLEHLO;
    case BT_CPU:
      return BT_MLIR_LLVM;
    default:
      return BT_MLIR;
  }
}

// =============================================================================
// General Backend Fallback
// =============================================================================

// Get the hardware backend that a software/MLIR backend falls back to
inline BackendType get_fallback_backend(BackendType type) {
  // Check MLIR backends first
  if (is_mlir_backend(type)) {
    return get_mlir_target_backend(type);
  }
  
  switch (type) {
    // NVIDIA CUDA ecosystem - library backends
    case BT_CUDNN:
    case BT_CUSPARSELT:
    case BT_CUTLASS:
    case BT_MHA:
      return BT_CUDA;
    
    // DSL/Compiler backends - can go through MLIR or direct
    case BT_TRITON:
      return BT_CUDA;  // Triton → (MLIR) → CUDA
    case BT_NKI:
      return BT_CUDA;  // NKI fallback (native target is Trainium)
    
    // CPU ecosystem - library backends
    case BT_MKL:
    case BT_MKLDNN:
    case BT_OPENMP:
    case BT_XEON:
    case BT_NNPACK:
    case BT_OPT_EINSUM:
      return BT_CPU;
    
    // Hardware backends return themselves
    case BT_CUDA:
    case BT_CPU:
    case BT_MPS:
    case BT_ASCEND:
    case BT_MACA:
    case BT_ROCM:
    case BT_TPU:
    case BT_FPGA:
    case BT_XPU:
      return type;
    
    default:
      return BT_UNKNOWN;
  }
}

// =============================================================================
// Backend Classification Functions
// =============================================================================

// Check if backend is a pure library backend (always needs fallback)
inline bool is_library_backend(BackendType type) {
  switch (type) {
    case BT_CUDNN:
    case BT_CUSPARSELT:
    case BT_CUTLASS:
    case BT_MKL:
    case BT_MKLDNN:
    case BT_OPENMP:
    case BT_XEON:
    case BT_MHA:
    case BT_NNPACK:
    case BT_OPT_EINSUM:
      return true;
    default:
      return false;
  }
}

// Check if backend is a DSL/compiler backend (has own strategy but can fallback)
inline bool is_dsl_backend(BackendType type) {
  switch (type) {
    case BT_TRITON:
    case BT_NKI:
    // MLIR-based DSLs/compilers also count as DSL backends
    case BT_MLIR:
    case BT_MLIR_LLVM:
    case BT_MLIR_NVVM:
    case BT_MLIR_ROCDL:
    case BT_MLIR_SPIRV:
    case BT_MLIR_GPU:
    case BT_STABLEHLO:
    case BT_MHLO:
    case BT_TOSA:
    case BT_LINALG:
    case BT_TCP:
    case BT_IREE:
    case BT_TVM:
    case BT_XLA:
      return true;
    default:
      return false;
  }
}

// Check if backend is a software/library backend (vs hardware)
// Includes both library backends and DSL backends
inline bool is_software_backend(BackendType type) {
  return is_library_backend(type) || is_dsl_backend(type);
}

// Check if backend is a hardware backend
inline bool is_hardware_backend(BackendType type) {
  switch (type) {
    case BT_CUDA:
    case BT_CPU:
    case BT_MPS:
    case BT_ASCEND:
    case BT_MACA:
    case BT_ROCM:
    case BT_TPU:
    case BT_FPGA:
    case BT_XPU:
      return true;
    default:
      return false;
  }
}

// Get the software library/DSL name for display
inline const char* get_software_library_name(BackendType type) {
  switch (type) {
    // Library backends
    case BT_CUDNN:      return "cuDNN";
    case BT_CUSPARSELT: return "cuSPARSELt";
    case BT_CUTLASS:    return "CUTLASS";
    case BT_MKL:        return "Intel MKL";
    case BT_MKLDNN:     return "oneDNN (MKL-DNN)";
    case BT_OPENMP:     return "OpenMP";
    case BT_XEON:       return "Intel Xeon Optimized";
    case BT_MHA:        return "Multi-Head Attention";
    case BT_NNPACK:     return "NNPACK";
    case BT_OPT_EINSUM: return "opt_einsum";
    
    // DSL backends
    case BT_TRITON:     return "OpenAI Triton";
    case BT_NKI:        return "Neuron Kernel Interface";
    
    // MLIR backends
    case BT_MLIR:       return "MLIR";
    case BT_MLIR_LLVM:  return "MLIR-LLVM";
    case BT_MLIR_NVVM:  return "MLIR-NVVM";
    case BT_MLIR_ROCDL: return "MLIR-ROCDL";
    case BT_MLIR_SPIRV: return "MLIR-SPIR-V";
    case BT_MLIR_GPU:   return "MLIR-GPU";
    case BT_STABLEHLO:  return "StableHLO";
    case BT_MHLO:       return "MHLO (XLA)";
    case BT_TOSA:       return "TOSA";
    case BT_LINALG:     return "Linalg";
    case BT_TCP:        return "Torch-MLIR TCP";
    case BT_IREE:       return "IREE";
    case BT_TVM:        return "Apache TVM";
    case BT_XLA:        return "XLA";
    
    default:            return nullptr;
  }
}

// Backend category for grouping
enum BackendCategory {
  BC_NVIDIA_GPU = 0,      // CUDA, CUDNN, CUSPARSELT, TRITON
  BC_AMD_GPU = 1,         // ROCm
  BC_INTEL_GPU = 2,       // XPU, SPIR-V
  BC_APPLE_GPU = 3,       // MPS
  BC_HUAWEI_NPU = 4,      // Ascend
  BC_METAX_GPU = 5,       // MACA
  BC_GOOGLE_TPU = 6,      // TPU, StableHLO, MHLO, XLA
  BC_FPGA = 7,            // FPGA
  BC_CPU = 8,             // CPU, MKL, MKLDNN, OPENMP, XEON, NNPACK, LLVM
  BC_MLIR = 9,            // Generic MLIR (can target multiple)
  BC_UNKNOWN = 99,
};

inline BackendCategory get_backend_category(BackendType type) {
  switch (type) {
    // NVIDIA GPU ecosystem
    case BT_CUDA:
    case BT_CUDNN:
    case BT_CUSPARSELT:
    case BT_CUTLASS:
    case BT_TRITON:
    case BT_MHA:
    case BT_MLIR_NVVM:
      return BC_NVIDIA_GPU;
    
    // AMD GPU ecosystem
    case BT_ROCM:
    case BT_MLIR_ROCDL:
      return BC_AMD_GPU;
    
    // Intel GPU ecosystem
    case BT_XPU:
    case BT_MLIR_SPIRV:
      return BC_INTEL_GPU;
    
    // Apple GPU
    case BT_MPS:
      return BC_APPLE_GPU;
    
    // Huawei NPU
    case BT_ASCEND:
      return BC_HUAWEI_NPU;
    
    // MetaX GPU
    case BT_MACA:
      return BC_METAX_GPU;
    
    // Google TPU ecosystem
    case BT_TPU:
    case BT_STABLEHLO:
    case BT_MHLO:
    case BT_XLA:
      return BC_GOOGLE_TPU;
    
    // FPGA
    case BT_FPGA:
      return BC_FPGA;
    
    // CPU ecosystem
    case BT_CPU:
    case BT_MKL:
    case BT_MKLDNN:
    case BT_OPENMP:
    case BT_XEON:
    case BT_NNPACK:
    case BT_OPT_EINSUM:
    case BT_MLIR_LLVM:
      return BC_CPU;
    
    // Generic MLIR (can target multiple backends)
    case BT_MLIR:
    case BT_MLIR_GPU:
    case BT_TOSA:
    case BT_LINALG:
    case BT_TCP:
    case BT_IREE:
    case BT_TVM:
    case BT_NKI:  // NKI targets Trainium, which doesn't fit other categories
      return BC_MLIR;
    
    default:
      return BC_UNKNOWN;
  }
}

// =============================================================================
// MLIR Lowering Path Utilities
// =============================================================================

// Get the MLIR dialect name as string
inline const char* get_mlir_dialect_name(MLIRDialect dialect) {
  switch (dialect) {
    case MLIR_DIALECT_BUILTIN:   return "builtin";
    case MLIR_DIALECT_ARITH:     return "arith";
    case MLIR_DIALECT_FUNC:      return "func";
    case MLIR_DIALECT_SCF:       return "scf";
    case MLIR_DIALECT_AFFINE:    return "affine";
    case MLIR_DIALECT_MEMREF:    return "memref";
    case MLIR_DIALECT_TENSOR:    return "tensor";
    case MLIR_DIALECT_VECTOR:    return "vector";
    case MLIR_DIALECT_LINALG:    return "linalg";
    case MLIR_DIALECT_TOSA:      return "tosa";
    case MLIR_DIALECT_STABLEHLO: return "stablehlo";
    case MLIR_DIALECT_MHLO:      return "mhlo";
    case MLIR_DIALECT_LLVM:      return "llvm";
    case MLIR_DIALECT_NVVM:      return "nvvm";
    case MLIR_DIALECT_ROCDL:     return "rocdl";
    case MLIR_DIALECT_SPIRV:     return "spirv";
    case MLIR_DIALECT_GPU:       return "gpu";
    case MLIR_DIALECT_AMX:       return "amx";
    case MLIR_DIALECT_X86VECTOR: return "x86vector";
    case MLIR_DIALECT_ARM_NEON:  return "arm_neon";
    case MLIR_DIALECT_ARM_SVE:   return "arm_sve";
    case MLIR_DIALECT_TRITON:    return "triton";
    case MLIR_DIALECT_TCP:       return "tcp";
    default:                     return "unknown";
  }
}

// Get recommended lowering pass pipeline for a target
inline std::string get_mlir_lowering_pipeline(BackendType source, BackendType target) {
  // StableHLO/MHLO → various targets
  if (source == BT_STABLEHLO || source == BT_MHLO) {
    switch (target) {
      case BT_TPU:
        return "stablehlo-legalize-to-hlo,xla-legalize-to-linalg";
      case BT_CUDA:
        return "stablehlo-legalize-to-linalg,linalg-to-gpu,gpu-to-nvvm";
      case BT_ROCM:
        return "stablehlo-legalize-to-linalg,linalg-to-gpu,gpu-to-rocdl";
      case BT_CPU:
        return "stablehlo-legalize-to-linalg,linalg-to-loops,convert-to-llvm";
      default:
        return "stablehlo-legalize-to-linalg";
    }
  }
  
  // TOSA → various targets
  if (source == BT_TOSA) {
    switch (target) {
      case BT_CUDA:
        return "tosa-to-linalg,linalg-to-gpu,gpu-to-nvvm";
      case BT_CPU:
        return "tosa-to-linalg,linalg-to-loops,convert-to-llvm";
      default:
        return "tosa-to-linalg";
    }
  }
  
  // Linalg → various targets
  if (source == BT_LINALG) {
    switch (target) {
      case BT_CUDA:
        return "linalg-tile-and-fuse,linalg-to-gpu,gpu-to-nvvm";
      case BT_ROCM:
        return "linalg-tile-and-fuse,linalg-to-gpu,gpu-to-rocdl";
      case BT_CPU:
        return "linalg-tile-and-fuse,linalg-to-loops,convert-to-llvm";
      default:
        return "linalg-to-loops";
    }
  }
  
  // Generic MLIR
  switch (target) {
    case BT_CUDA:
      return "convert-to-gpu,gpu-to-nvvm";
    case BT_ROCM:
      return "convert-to-gpu,gpu-to-rocdl";
    case BT_XPU:
      return "convert-to-gpu,gpu-to-spirv";
    case BT_CPU:
      return "convert-to-llvm";
    default:
      return "";
  }
}

// Get all supported target backends for an MLIR dialect
inline std::vector<BackendType> get_mlir_supported_targets(BackendType mlir_backend) {
  std::vector<BackendType> targets;
  
  switch (mlir_backend) {
    case BT_MLIR:
    case BT_LINALG:
      // Generic MLIR can target everything
      targets = {BT_CUDA, BT_ROCM, BT_CPU, BT_XPU, BT_TPU};
      break;
    
    case BT_STABLEHLO:
    case BT_MHLO:
    case BT_XLA:
      // XLA ecosystem targets
      targets = {BT_TPU, BT_CUDA, BT_CPU};
      break;
    
    case BT_TOSA:
      // TOSA is designed for portability
      targets = {BT_CPU, BT_CUDA, BT_ROCM, BT_FPGA};
      break;
    
    case BT_MLIR_NVVM:
      targets = {BT_CUDA};
      break;
    
    case BT_MLIR_ROCDL:
      targets = {BT_ROCM};
      break;
    
    case BT_MLIR_SPIRV:
      targets = {BT_XPU, BT_CUDA};  // SPIR-V can go to Vulkan too
      break;
    
    case BT_MLIR_LLVM:
      targets = {BT_CPU};
      break;
    
    case BT_IREE:
      targets = {BT_CPU, BT_CUDA, BT_ROCM};
      break;
    
    case BT_TVM:
      targets = {BT_CUDA, BT_ROCM, BT_CPU, BT_XPU};
      break;
    
    default:
      break;
  }
  
  return targets;
}

enum DataType {
  // 1-bit types
  // range: 900-909
  // 2-bit types
  // range: 910-919
  // 4-bit types
  // range: 920-929
  DT_FLOAT4 = 920,
  DT_INT4 = 925,
  DT_UINT4 = 926,
  // 8-bit types
  // range(float types): 930-934
  // range(int types): 935-939
  DT_FLOAT8 = 930,
  DT_INT8 = 935,
  DT_UINT8 = 936,
  // 16-bit types
  // range(float types): 940-944
  // range(int types): 945-949
  DT_FLOAT16 = 940,
  DT_BFLOAT16 = 941,
  DT_INT16 = 945,
  DT_UINT16 = 946,
  // 32-bit types
  // range(float type): 950-954
  // range(int type): 955-959
  DT_FLOAT32 = 950,
  DT_INT32 = 955,
  DT_UINT32 = 956,
  // 64-bit types
  // range(float types): 960-964
  // range(int type): 965-969
  DT_DOUBLE = 960,
  DT_INT64 = 965,
  DT_UINT64 = 966,
  DT_UNKNOWN = 999,
};

size_t get_datatype_size(DataType type);
std::string get_datatype_str(DataType dtype);

enum KNOperatorType {
  KN_UNKOWN = 1000,
  KN_INPUT_OP = 1001,
  KN_OUTPUT_OP = 1002,
  KN_MATMUL_OP = 1003,
  // ElementUnary
  KN_EXP_OP = 1100,
  KN_SQUARE_OP = 1101,
  KN_SQRT_OP = 1102,
  KN_MUL_SCALAR_OP = 1103,
  KN_SILU_OP = 1104,
  KN_SIGMOID_OP = 1105,
  KN_GELU_OP = 1106,
  // non-lax elementunary ops
  KN_RELU_OP = 1150,
  KN_CLAMP_OP = 1151,
  KN_LOG_OP = 1160,
  // ElementBinary
  KN_ADD_OP = 1200,
  KN_MUL_OP = 1201,
  KN_DIV_OP = 1202,
  KN_POW_OP = 1203,
  // Reduction & Normalization
  KN_REDUCTION_0_OP = 1300,
  KN_REDUCTION_1_OP = 1301,
  KN_REDUCTION_2_OP = 1302,
  KN_RMS_NORM_OP = 1350,
  // Concat & Split
  KN_CONCAT_FIRST_OP_ID = 1400,
  KN_CONCAT_0_OP = 1400,
  KN_CONCAT_1_OP = 1401,
  KN_CONCAT_2_OP = 1402,
  KN_CONCAT_LAST_OP_ID = 1409,
  KN_CONCAT_THEN_MATMUL_OP = 1410,
  KN_SPLIT_FIRST_OP_ID = 1420,
  KN_SPLIT_0_OP = 1420,
  KN_SPLIT_1_OP = 1421,
  KN_SPLIT_2_OP = 1422,
  KN_CHUNK_0_OP = 1423,
  KN_CHUNK_1_OP = 1424,
  KN_CHUNK_2_OP = 1425,
  KN_SPLIT_LAST_OP_ID = 1429,
  // Communication / Collective Operations (COMET)
  KN_ALLREDUCE_OP = 1900,
  KN_ALLGATHER_OP = 1901,       // All-Gather collective
  KN_REDUCE_SCATTER_OP = 1902,  // Reduce-Scatter collective
  KN_BROADCAST_OP = 1903,       // Broadcast collective
  KN_P2P_SEND_OP = 1904,        // Point-to-point send
  KN_P2P_RECV_OP = 1905,        // Point-to-point receive
  KN_CUSTOMIZED_OP = 1999,
};

NLOHMANN_JSON_SERIALIZE_ENUM(KNOperatorType,
                             {
                                 {KN_UNKOWN, "kn_unkown"},
                                 {KN_INPUT_OP, "kn_input_op"},
                                 {KN_OUTPUT_OP, "kn_output_op"},
                                 {KN_MATMUL_OP, "kn_matmul_op"},
                                 {KN_EXP_OP, "kn_exp_op"},
                                 {KN_SQUARE_OP, "kn_square_op"},
                                 {KN_SQRT_OP, "kn_sqrt_op"},
                                 {KN_MUL_SCALAR_OP, "kn_mul_scalar_op"},
                                 {KN_SILU_OP, "kn_silu_op"},
                                 {KN_SIGMOID_OP, "kn_sigmoid_op"},
                                 {KN_GELU_OP, "kn_gelu_op"},
                                 {KN_RELU_OP, "kn_relu_op"},
                                 {KN_CLAMP_OP, "kn_clamp_op"},
                                 {KN_LOG_OP, "kn_log_op"},
                                 {KN_ADD_OP, "kn_add_op"},
                                 {KN_MUL_OP, "kn_mul_op"},
                                 {KN_DIV_OP, "kn_div_op"},
                                 {KN_POW_OP, "kn_pow_op"},
                                 {KN_REDUCTION_0_OP, "kn_reduction_0_op"},
                                 {KN_REDUCTION_1_OP, "kn_reduction_1_op"},
                                 {KN_REDUCTION_2_OP, "kn_reduction_2_op"},
                                 {KN_RMS_NORM_OP, "kn_rms_norm_op"},
                                 {KN_CONCAT_FIRST_OP_ID,
                                  "kn_concat_first_op_id"},
                                 {KN_CONCAT_0_OP, "kn_concat_0_op"},
                                 {KN_CONCAT_1_OP, "kn_concat_1_op"},
                                 {KN_CONCAT_2_OP, "kn_concat_2_op"},
                                 {KN_CONCAT_LAST_OP_ID, "kn_concat_last_op_id"},
                                 {KN_CONCAT_THEN_MATMUL_OP,
                                  "kn_concat_then_matmul_op"},
                                 {KN_SPLIT_FIRST_OP_ID, "kn_split_first_op_id"},
                                 {KN_SPLIT_0_OP, "kn_split_0_op"},
                                 {KN_SPLIT_1_OP, "kn_split_1_op"},
                                 {KN_SPLIT_2_OP, "kn_split_2_op"},
                                 {KN_CHUNK_0_OP, "kn_chunk_0_op"},
                                 {KN_CHUNK_1_OP, "kn_chunk_1_op"},
                                 {KN_CHUNK_2_OP, "kn_chunk_2_op"},
                                 {KN_SPLIT_LAST_OP_ID, "kn_split_last_op_id"},
                                 {KN_ALLREDUCE_OP, "kn_allreduce_op"},
                                 {KN_ALLGATHER_OP, "kn_allgather_op"},
                                 {KN_REDUCE_SCATTER_OP, "kn_reduce_scatter_op"},
                                 {KN_BROADCAST_OP, "kn_broadcast_op"},
                                 {KN_P2P_SEND_OP, "kn_p2p_send_op"},
                                 {KN_P2P_RECV_OP, "kn_p2p_recv_op"},
                                 {KN_CUSTOMIZED_OP, "kn_customized_op"},
                             })

enum TBOperatorType {
  TB_UNKOWN = 2000,
  TB_INPUT_OP = 2001,
  TB_OUTPUT_OP = 2002,
  TB_MATMUL_OP = 2003,
  // ElementUnary
  TB_EXP_OP = 2100,
  TB_SQUARE_OP = 2101,
  TB_SQRT_OP = 2102,
  TB_MUL_SCALAR_OP = 2103,
  TB_SILU_OP = 2104,
  TB_SIGMOID_OP = 2105,
  TB_GELU_OP = 2106,
  // non-lax elementunary ops
  TB_RELU_OP = 2150,
  TB_CLAMP_OP = 2151,
  TB_LOG_OP = 2160,
  // ElementBinary
  TB_ADD_OP = 2200,
  TB_MUL_OP = 2201,
  TB_DIV_OP = 2202,
  TB_SUB_OP = 2203,
  TB_POW_OP = 2204,
  // Reduction and Normalization
  TB_REDUCTION_FIRST_OP_ID = 2300,
  TB_REDUCTION_0_OP = 2301,
  TB_REDUCTION_1_OP = 2302,
  TB_REDUCTION_2_OP = 2303,
  TB_REDUCTION_0_TO_DIMX_OP = 2304,
  TB_REDUCTION_1_TO_DIMX_OP = 2305,
  TB_REDUCTION_2_TO_DIMX_OP = 2306,
  TB_REDUCTION_0_MAX_OP = 2307,
  TB_REDUCTION_1_MAX_OP = 2308,
  TB_REDUCTION_2_MAX_OP = 2309,
  TB_REDUCTION_LAST_OP_ID = 2349,
  TB_RMS_NORM_OP = 2350,
  // Concat & Split
  TB_CONCAT_FIRST_OP_ID = 2400,
  TB_CONCAT_0_OP = 2400,
  TB_CONCAT_1_OP = 2401,
  TB_CONCAT_2_OP = 2402,
  TB_CONCAT_LAST_OP_ID = 2409,
  TB_CONCAT_THEN_MATMUL_OP = 2411,
  TB_SPLIT_FIRST_OP_ID = 2420,
  TB_SPLIT_0_OP = 2420,
  TB_SPLIT_1_OP = 2421,
  TB_SPLIT_2_OP = 2422,
  TB_CHUNK_0_OP = 2423,
  TB_CHUNK_1_OP = 2424,
  TB_CHUNK_2_OP = 2425,
  TB_SPLIT_LAST_OP_ID = 2429,
  // Forloop Accum
  // LD indicates last dimension
  TB_FORLOOP_ACCUM_FIRST_OP = 2500,
  TB_FORLOOP_ACCUM_NO_RED_OP = 2500,
  TB_FORLOOP_ACCUM_RED_LD_SUM_OP = 2501,
  TB_FORLOOP_ACCUM_RED_LD_MEAN_OP = 2502,
  TB_FORLOOP_ACCUM_RED_LD_RMS_OP = 2503,
  TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP = 2504,
  TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP = 2505,
  TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP = 2506,
  TB_FORLOOP_ACCUM_MAX_OP = 2507,
  TB_FORLOOP_ACCUM_LAST_OP = 2599,
  TB_CUSTOMIZED_OP = 2999
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    TBOperatorType,
    {
        {TB_UNKOWN, "tb_unkown"},
        {TB_INPUT_OP, "tb_input_op"},
        {TB_OUTPUT_OP, "tb_output_op"},
        {TB_MATMUL_OP, "tb_matmul_op"},
        {TB_EXP_OP, "tb_exp_op"},
        {TB_SQUARE_OP, "tb_square_op"},
        {TB_SQRT_OP, "tb_sqrt_op"},
        {TB_MUL_SCALAR_OP, "tb_mul_scalar_op"},
        {TB_SILU_OP, "tb_silu_op"},
        {TB_SIGMOID_OP, "tb_sigmoid_op"},
        {TB_GELU_OP, "tb_gelu_op"},
        {TB_RELU_OP, "tb_relu_op"},
        {TB_CLAMP_OP, "tb_clamp_op"},
        {TB_LOG_OP, "tb_log_op"},
        {TB_ADD_OP, "tb_add_op"},
        {TB_MUL_OP, "tb_mul_op"},
        {TB_DIV_OP, "tb_div_op"},
        {TB_SUB_OP, "tb_sub_op"},
        {TB_POW_OP, "tb_pow_op"},
        {TB_REDUCTION_FIRST_OP_ID, "tb_reduction_first_op_id"},
        {TB_REDUCTION_0_OP, "tb_reduction_0_op"},
        {TB_REDUCTION_1_OP, "tb_reduction_1_op"},
        {TB_REDUCTION_2_OP, "tb_reduction_2_op"},
        {TB_REDUCTION_0_TO_DIMX_OP, "tb_reduction_0_to_dimx_op"},
        {TB_REDUCTION_1_TO_DIMX_OP, "tb_reduction_1_to_dimx_op"},
        {TB_REDUCTION_2_TO_DIMX_OP, "tb_reduction_2_to_dimx_op"},
        {TB_REDUCTION_0_MAX_OP, "tb_reduction_0_max_op"},
        {TB_REDUCTION_1_MAX_OP, "tb_reduction_1_max_op"},
        {TB_REDUCTION_2_MAX_OP, "tb_reduction_2_max_op"},
        {TB_REDUCTION_LAST_OP_ID, "tb_reduction_last_op_id"},
        {TB_RMS_NORM_OP, "tb_rms_norm_op"},
        {TB_CONCAT_FIRST_OP_ID, "tb_concat_first_op_id"},
        {TB_CONCAT_0_OP, "tb_concat_0_op"},
        {TB_CONCAT_1_OP, "tb_concat_1_op"},
        {TB_CONCAT_2_OP, "tb_concat_2_op"},
        {TB_CONCAT_LAST_OP_ID, "tb_concat_last_op_id"},
        {TB_CONCAT_THEN_MATMUL_OP, "tb_concat_then_matmul_op"},
        {TB_SPLIT_FIRST_OP_ID, "tb_split_first_op_id"},
        {TB_SPLIT_0_OP, "tb_split_0_op"},
        {TB_SPLIT_1_OP, "tb_split_1_op"},
        {TB_SPLIT_2_OP, "tb_split_2_op"},
        {TB_CHUNK_0_OP, "tb_chunk_0_op"},
        {TB_CHUNK_1_OP, "tb_chunk_1_op"},
        {TB_CHUNK_2_OP, "tb_chunk_2_op"},
        {TB_SPLIT_LAST_OP_ID, "tb_split_last_op_id"},
        {TB_FORLOOP_ACCUM_NO_RED_OP, "tb_forloop_accum_nored_op"},
        {TB_FORLOOP_ACCUM_RED_LD_SUM_OP, "tb_forloop_accum_red_ld_sum_op"},
        {TB_FORLOOP_ACCUM_RED_LD_MEAN_OP, "tb_forloop_accum_red_ld_mean_op"},
        {TB_FORLOOP_ACCUM_RED_LD_RMS_OP, "tb_forloop_accum_red_ld_rms_op"},
        {TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP,
         "tb_forloop_accum_redtox_ld_sum_op"},
        {TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP,
         "tb_forloop_accum_nored_rescale_op"},
        {TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP,
         "tb_forloop_accum_red_ld_sum_rescale_op"},
        {TB_FORLOOP_ACCUM_MAX_OP, "tb_forloop_accum_max_op"},
        {TB_FORLOOP_ACCUM_LAST_OP, "tb_forloop_accum_last_op"},
        {TB_CUSTOMIZED_OP, "tb_customized_op"},
    })

bool is_threadblock_element_unary(TBOperatorType op_type);

enum ActivationType {
  ACT_UNKOWN = 3000,
  ACT_EXP = 3001,
  ACT_RELU = 3002,
  ACT_GELU = 3003,
  ACT_SILU = 3004,
  ACT_NONE = 3099,
};

enum TBEpilogueType {
  TB_EPILOGUE_NONE = 3100,
  TB_EPILOGUE_ALLREDUCE = 3101,
  TB_EPILOGUE_ALLTOALL = 3102,
  TB_EPILOGUE_INVALID = 3199,
};

NLOHMANN_JSON_SERIALIZE_ENUM(TBEpilogueType,
                             {
                                 {TB_EPILOGUE_NONE, "tb_epilogue_none"},
                                 {TB_EPILOGUE_ALLREDUCE,
                                  "tb_epilogue_allreduce"},
                                 {TB_EPILOGUE_ALLTOALL, "tb_epilogue_alltoall"},
                                 {TB_EPILOGUE_INVALID, "tb_epilogue_invalid"},
                             })

// =============================================================================
// COMET: Compound Operation with Explicit Collectives (COMET paper)
// =============================================================================

// Collective operation types (COMET Fig. 1b)
// Explicit representation for dataflow modeling
enum CollectiveOpType {
  COLL_ALL_REDUCE = 4000,      // Sum/Avg across all participants
  COLL_ALL_GATHER = 4001,      // Gather data from all to all
  COLL_REDUCE_SCATTER = 4002,  // Reduce then scatter result
  COLL_BROADCAST = 4003,       // One-to-all
  COLL_P2P = 4004,             // Point-to-point
  COLL_NONE = 4099,
};

NLOHMANN_JSON_SERIALIZE_ENUM(CollectiveOpType,
                             {
                                 {COLL_ALL_REDUCE, "coll_all_reduce"},
                                 {COLL_ALL_GATHER, "coll_all_gather"},
                                 {COLL_REDUCE_SCATTER, "coll_reduce_scatter"},
                                 {COLL_BROADCAST, "coll_broadcast"},
                                 {COLL_P2P, "coll_p2p"},
                                 {COLL_NONE, "coll_none"},
                             })

// Reduction operation types for collectives
enum CollectiveReduceOp {
  REDUCE_SUM = 4100,
  REDUCE_AVG = 4101,
  REDUCE_MAX = 4102,
  REDUCE_MIN = 4103,
  REDUCE_PROD = 4104,
};

NLOHMANN_JSON_SERIALIZE_ENUM(CollectiveReduceOp,
                             {
                                 {REDUCE_SUM, "reduce_sum"},
                                 {REDUCE_AVG, "reduce_avg"},
                                 {REDUCE_MAX, "reduce_max"},
                                 {REDUCE_MIN, "reduce_min"},
                                 {REDUCE_PROD, "reduce_prod"},
                             })

// Memory hierarchy levels (COMET Fig. 2b)
// DRAM -> Global Buffer (GB) -> Input/Weight/Output Buffer -> Compute
enum MemoryLevel {
  MEM_DRAM = 4200,            // Off-chip DRAM
  MEM_GLOBAL_BUFFER = 4201,   // On-chip global buffer (GB)
  MEM_INPUT_BUFFER = 4202,    // Input buffer (IB)
  MEM_WEIGHT_BUFFER = 4203,   // Weight buffer (WB)
  MEM_OUTPUT_BUFFER = 4204,   // Output buffer (OB)
  MEM_REGISTER = 4205,        // Register file
  MEM_SHARED = 4206,          // Shared memory (GPU)
  MEM_L1_CACHE = 4207,        // L1 cache
  MEM_L2_CACHE = 4208,        // L2 cache
};

NLOHMANN_JSON_SERIALIZE_ENUM(MemoryLevel,
                             {
                                 {MEM_DRAM, "mem_dram"},
                                 {MEM_GLOBAL_BUFFER, "mem_global_buffer"},
                                 {MEM_INPUT_BUFFER, "mem_input_buffer"},
                                 {MEM_WEIGHT_BUFFER, "mem_weight_buffer"},
                                 {MEM_OUTPUT_BUFFER, "mem_output_buffer"},
                                 {MEM_REGISTER, "mem_register"},
                                 {MEM_SHARED, "mem_shared"},
                                 {MEM_L1_CACHE, "mem_l1_cache"},
                                 {MEM_L2_CACHE, "mem_l2_cache"},
                             })

// Scheduling strategies for compound operations (COMET Fig. 1d)
enum SchedulingStrategy {
  SCHED_SEQUENTIAL = 4300,  // Operations execute one after another
  SCHED_PIPELINED = 4301,   // Operations overlap in pipeline stages
  SCHED_PARALLEL = 4302,    // Operations execute concurrently
};

NLOHMANN_JSON_SERIALIZE_ENUM(SchedulingStrategy,
                             {
                                 {SCHED_SEQUENTIAL, "sched_sequential"},
                                 {SCHED_PIPELINED, "sched_pipelined"},
                                 {SCHED_PARALLEL, "sched_parallel"},
                             })

// Compound operation types (COMET Section V)
enum CompoundOpType {
  COMP_GEMM_SOFTMAX = 4400,     // GEMM followed by Softmax
  COMP_GEMM_LAYERNORM = 4401,   // GEMM followed by LayerNorm
  COMP_SELF_ATTENTION = 4402,   // Q@K^T -> Softmax -> @V
  COMP_GEMM_GELU = 4403,        // GEMM followed by GELU
  COMP_GEMM_SILU = 4404,        // GEMM followed by SiLU
  COMP_GATED_MLP = 4405,        // gate * up projection pattern
  COMP_RMS_NORM_LINEAR = 4406,  // RMSNorm followed by Linear
  COMP_CUSTOM = 4499,
};

NLOHMANN_JSON_SERIALIZE_ENUM(CompoundOpType,
                             {
                                 {COMP_GEMM_SOFTMAX, "comp_gemm_softmax"},
                                 {COMP_GEMM_LAYERNORM, "comp_gemm_layernorm"},
                                 {COMP_SELF_ATTENTION, "comp_self_attention"},
                                 {COMP_GEMM_GELU, "comp_gemm_gelu"},
                                 {COMP_GEMM_SILU, "comp_gemm_silu"},
                                 {COMP_GATED_MLP, "comp_gated_mlp"},
                                 {COMP_RMS_NORM_LINEAR, "comp_rms_norm_linear"},
                                 {COMP_CUSTOM, "comp_custom"},
                             })

// Data staging states for COMET cost model (Section IV-B)
// Used to model ramp-up/ramp-down phases
enum DataStagingState {
  STAGE_IDLE = 4500,       // No data movement
  STAGE_RAMP_UP = 4501,    // Filling buffer (compute waiting)
  STAGE_STEADY = 4502,     // Steady state (compute and memory overlap)
  STAGE_RAMP_DOWN = 4503,  // Draining buffer (memory waiting)
};

NLOHMANN_JSON_SERIALIZE_ENUM(DataStagingState,
                             {
                                 {STAGE_IDLE, "stage_idle"},
                                 {STAGE_RAMP_UP, "stage_ramp_up"},
                                 {STAGE_STEADY, "stage_steady"},
                                 {STAGE_RAMP_DOWN, "stage_ramp_down"},
                             })

} // namespace type
} // namespace yirage
