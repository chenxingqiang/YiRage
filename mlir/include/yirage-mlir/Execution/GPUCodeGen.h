//===- GPUCodeGen.h - GPU Code Generation Interface -------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header defines the GPU code generation interface for YiRage MLIR.
// Supports all major accelerator backends:
//   - NVIDIA CUDA (PTX/cubin)
//   - AMD ROCm (GCN/HSACO)
//   - Intel XPU (SPIR-V/Level Zero)
//   - Google TPU (StableHLO/XLA)
//   - Huawei Ascend (CCE)
//   - Apple Metal (MSL/metallib)
//   - MetaX MACA
//   - FPGA (OpenCL/SPIR-V)
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_MLIR_EXECUTION_GPUCODEGEN_H
#define YIRAGE_MLIR_EXECUTION_GPUCODEGEN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace yirage {

//===----------------------------------------------------------------------===//
// Backend Enumeration
//===----------------------------------------------------------------------===//

/// Supported accelerator backends
enum class GPUBackend {
  // GPU Backends
  CUDA,       // NVIDIA CUDA (PTX/cubin)
  ROCm,       // AMD ROCm (GCN/HSACO)
  SPIRV,      // SPIR-V (Intel XPU, Vulkan)
  Metal,      // Apple Metal (MSL/metallib)
  MACA,       // MetaX MACA
  
  // NPU/TPU Backends
  TPU,        // Google TPU (StableHLO → XLA)
  Ascend,     // Huawei Ascend (CCE)
  
  // FPGA Backends
  FPGA_OpenCL,    // FPGA via OpenCL
  FPGA_HLS,       // FPGA via HLS
  
  // CPU Backends (for completeness)
  CPU_LLVM,       // CPU via LLVM
  CPU_OpenMP,     // CPU with OpenMP
};

//===----------------------------------------------------------------------===//
// Target Configuration
//===----------------------------------------------------------------------===//

/// Configuration for code generation
struct GPUTargetConfig {
  GPUBackend backend = GPUBackend::CUDA;
  std::string arch;          // e.g., "sm_90", "gfx942", "spirv64"
  std::string triple;        // Target triple
  std::string features;      // Additional features
  
  // Optimization settings
  int optLevel = 3;          // 0-3
  bool useFastMath = true;
  bool useFMA = true;
  bool debug = false;
  
  // Backend-specific options
  std::string sdkPath;       // Path to SDK (CUDA, ROCm, etc.)
  std::string toolchain;     // Preferred toolchain
  
  /// Create configuration for NVIDIA CUDA
  static GPUTargetConfig forCUDA(int computeCapability = 80);
  
  /// Create configuration for AMD ROCm
  static GPUTargetConfig forROCm(llvm::StringRef gpuArch = "gfx942");
  
  /// Create configuration for SPIR-V (Intel XPU, Vulkan)
  static GPUTargetConfig forSPIRV();
  
  /// Create configuration for Apple Metal
  static GPUTargetConfig forMetal(llvm::StringRef arch = "m3");
  
  /// Create configuration for Google TPU
  static GPUTargetConfig forTPU(llvm::StringRef tpuVersion = "v5e");
  
  /// Create configuration for Huawei Ascend
  static GPUTargetConfig forAscend(llvm::StringRef chipType = "910B");
  
  /// Create configuration for MetaX MACA
  static GPUTargetConfig forMACA(llvm::StringRef gpuArch = "mxc500");
  
  /// Create configuration for FPGA
  static GPUTargetConfig forFPGA(llvm::StringRef vendor = "xilinx");
  
  /// Create configuration for CPU
  static GPUTargetConfig forCPU(llvm::StringRef cpuArch = "x86-64-v3");
};

//===----------------------------------------------------------------------===//
// Binary Output Types
//===----------------------------------------------------------------------===//

/// Binary output format
enum class BinaryFormat {
  PTX,          // NVIDIA PTX text
  CUBIN,        // NVIDIA cubin binary
  HSACO,        // AMD HSACO binary
  SPIRV,        // SPIR-V binary
  SPIRV_ASM,    // SPIR-V assembly text
  METALLIB,     // Apple metallib
  MSL,          // Metal Shading Language text
  LLVM_IR,      // LLVM IR text
  LLVM_BC,      // LLVM bitcode
  XLA_HLO,      // XLA HLO text (for TPU)
  STABLEHLO,    // StableHLO text
  ASCEND_CCE,   // Ascend CCE binary
  OBJECT,       // Native object file
};

/// Compilation result
struct CompilationResult {
  bool success = false;
  std::string errorMessage;
  
  // Generated artifacts
  std::string textCode;              // PTX, MSL, SPIR-V asm, etc.
  std::vector<uint8_t> binaryCode;   // cubin, HSACO, SPIR-V binary, etc.
  
  // Metadata
  std::string targetArch;
  std::vector<std::string> kernelNames;
  size_t registerUsage = 0;
  size_t sharedMemUsage = 0;
  
  operator bool() const { return success; }
};

//===----------------------------------------------------------------------===//
// GPU Code Generator
//===----------------------------------------------------------------------===//

class GPUCodeGenImpl;

/// GPU Code Generator for compiling to accelerator-specific binaries.
///
/// Example usage:
/// \code
///   MLIRContext context;
///   // ... lower module to GPU/target dialect ...
///
///   auto codegen = createCUDACodeGen(&context, 90);
///   auto result = codegen->compile(module);
///
///   if (result) {
///     std::string ptx = codegen->getPTX(module);
///     std::vector<uint8_t> cubin = codegen->getCubin(module);
///   }
/// \endcode
class GPUCodeGen {
public:
  GPUCodeGen(mlir::MLIRContext *context, const GPUTargetConfig &config);
  ~GPUCodeGen();

  // Disable copy
  GPUCodeGen(const GPUCodeGen &) = delete;
  GPUCodeGen &operator=(const GPUCodeGen &) = delete;

  //==========================================================================
  // Compilation
  //==========================================================================
  
  /// Compile the module to target-specific IR.
  mlir::LogicalResult compile(mlir::ModuleOp module);
  
  /// Compile and return full result with metadata
  CompilationResult compileWithResult(mlir::ModuleOp module);

  //==========================================================================
  // Code Export
  //==========================================================================
  
  /// Export generated code as text (PTX, GCN assembly, SPIR-V, MSL).
  std::string exportCode(mlir::ModuleOp module);
  
  /// Get PTX text (CUDA only)
  std::string getPTX(mlir::ModuleOp module);
  
  /// Get Metal Shading Language text (Metal only)
  std::string getMSL(mlir::ModuleOp module);
  
  /// Get SPIR-V assembly text
  std::string getSPIRVAsm(mlir::ModuleOp module);
  
  /// Get StableHLO text (TPU only)
  std::string getStableHLO(mlir::ModuleOp module);
  
  //==========================================================================
  // Binary Generation
  //==========================================================================
  
  /// Get the compiled binary blob.
  std::vector<uint8_t> getBinary(mlir::ModuleOp module);
  
  /// Get cubin binary (CUDA only)
  std::vector<uint8_t> getCubin(mlir::ModuleOp module);
  
  /// Get HSACO binary (ROCm only)
  std::vector<uint8_t> getHSACO(mlir::ModuleOp module);
  
  /// Get SPIR-V binary
  std::vector<uint8_t> getSPIRV(mlir::ModuleOp module);

  //==========================================================================
  // Configuration
  //==========================================================================
  
  /// Get the target configuration.
  const GPUTargetConfig &getConfig() const { return config; }
  
  /// Get backend type
  GPUBackend getBackend() const { return config.backend; }
  
  /// Check if backend is available on this system
  static bool isBackendAvailable(GPUBackend backend);

private:
  std::unique_ptr<GPUCodeGenImpl> impl;
  mlir::MLIRContext *context;
  GPUTargetConfig config;
};

//===----------------------------------------------------------------------===//
// Factory Functions
//===----------------------------------------------------------------------===//

/// Create a CUDA code generator
std::unique_ptr<GPUCodeGen> createCUDACodeGen(mlir::MLIRContext *context,
                                               int computeCapability = 80);

/// Create a ROCm code generator
std::unique_ptr<GPUCodeGen> createROCmCodeGen(mlir::MLIRContext *context,
                                               llvm::StringRef arch = "gfx942");

/// Create a SPIR-V code generator (Intel XPU, Vulkan)
std::unique_ptr<GPUCodeGen> createSPIRVCodeGen(mlir::MLIRContext *context);

/// Create a Metal code generator (Apple Silicon)
std::unique_ptr<GPUCodeGen> createMetalCodeGen(mlir::MLIRContext *context,
                                               llvm::StringRef arch = "m3");

/// Create a TPU code generator (Google Cloud)
std::unique_ptr<GPUCodeGen> createTPUCodeGen(mlir::MLIRContext *context,
                                              llvm::StringRef version = "v5e");

/// Create an Ascend code generator (Huawei NPU)
std::unique_ptr<GPUCodeGen> createAscendCodeGen(mlir::MLIRContext *context,
                                                 llvm::StringRef chip = "910B");

/// Create a MACA code generator (MetaX GPU)
std::unique_ptr<GPUCodeGen> createMACACodeGen(mlir::MLIRContext *context,
                                               llvm::StringRef arch = "mxc500");

/// Create an FPGA code generator
std::unique_ptr<GPUCodeGen> createFPGACodeGen(mlir::MLIRContext *context,
                                               llvm::StringRef vendor = "xilinx");

/// Create a CPU code generator
std::unique_ptr<GPUCodeGen> createCPUCodeGen(mlir::MLIRContext *context,
                                              llvm::StringRef arch = "x86-64-v3");

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Get list of available backends on this system
std::vector<GPUBackend> getAvailableBackends();

/// Convert backend enum to string
llvm::StringRef backendToString(GPUBackend backend);

/// Convert string to backend enum
GPUBackend stringToBackend(llvm::StringRef name);

} // namespace yirage

#endif // YIRAGE_MLIR_EXECUTION_GPUCODEGEN_H
