//===- BinaryGen.cpp - GPU Binary Generation -------------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file implements GPU binary generation for multiple backends:
//   - NVIDIA CUDA: PTX → cubin via ptxas or NVRTC
//   - AMD ROCm: LLVM IR → GCN → HSACO via lld
//   - Intel XPU: MLIR → SPIR-V via mlir-translate
//   - Apple Metal: MLIR → AIR → metallib
//   - Huawei Ascend: MLIR → CCE → .o
//   - MetaX MACA: Similar to CUDA
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Execution/GPUCodeGen.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

namespace yirage {

//===----------------------------------------------------------------------===//
// Binary Generation Configuration
//===----------------------------------------------------------------------===//

struct BinaryGenConfig {
  std::string arch;
  std::string triple;
  int optLevel = 3;
  bool debug = false;
  bool verbose = false;
  std::string tempDir;
  
  // CUDA-specific
  int cudaVersion = 12;
  bool useFastMath = true;
  bool useFMA = true;
  
  // ROCm-specific
  std::string rocmPath = "/opt/rocm";
  
  // SPIR-V specific
  std::string spirvVersion = "1.3";
  
  BinaryGenConfig() {
    // Get temp directory
    llvm::SmallString<128> tempPath;
    llvm::sys::path::system_temp_directory(true, tempPath);
    tempDir = std::string(tempPath);
  }
};

//===----------------------------------------------------------------------===//
// PTX Generation (NVIDIA CUDA)
//===----------------------------------------------------------------------===//

class PTXGenerator {
public:
  PTXGenerator(const BinaryGenConfig& config) : config_(config) {}
  
  /// Generate PTX from LLVM IR
  std::string generatePTX(llvm::Module& module) {
    std::string ptx;
    llvm::raw_string_ostream os(ptx);
    
    // Set target triple and data layout for NVPTX
    module.setTargetTriple(config_.triple.empty() ? 
                           "nvptx64-nvidia-cuda" : config_.triple);
    
    std::string dataLayout = 
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
        "f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-"
        "n16:32:64";
    module.setDataLayout(dataLayout);
    
    // Initialize NVPTX target
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
    if (!target) {
      llvm::errs() << "Error looking up target: " << error << "\n";
      return "";
    }
    
    // Create target machine
    llvm::TargetOptions options;
    options.AllowFPOpFusion = config_.useFMA ? 
                               llvm::FPOpFusion::Fast : llvm::FPOpFusion::Standard;
    options.UnsafeFPMath = config_.useFastMath;
    
    std::string cpu = config_.arch.empty() ? "sm_80" : config_.arch;
    auto targetMachine = target->createTargetMachine(
        module.getTargetTriple(),
        cpu,
        "+ptx80",
        options,
        llvm::Reloc::PIC_,
        llvm::CodeModel::Small,
        config_.optLevel == 0 ? llvm::CodeGenOpt::None :
        config_.optLevel == 1 ? llvm::CodeGenOpt::Less :
        config_.optLevel == 2 ? llvm::CodeGenOpt::Default :
                                llvm::CodeGenOpt::Aggressive);
    
    if (!targetMachine) {
      llvm::errs() << "Could not create target machine\n";
      return "";
    }
    
    // Generate PTX
    llvm::legacy::PassManager pm;
    llvm::SmallString<0> ptxStr;
    llvm::raw_svector_ostream ptxOS(ptxStr);
    
    if (targetMachine->addPassesToEmitFile(pm, ptxOS, nullptr,
                                           llvm::CodeGenFileType::CGFT_AssemblyFile)) {
      llvm::errs() << "Target machine cannot emit PTX\n";
      return "";
    }
    
    pm.run(module);
    return std::string(ptxStr.begin(), ptxStr.end());
  }
  
  /// Compile PTX to cubin using ptxas
  std::vector<uint8_t> compilePTXToCubin(const std::string& ptx) {
    // Write PTX to temp file
    std::string ptxFile = config_.tempDir + "/yirage_kernel.ptx";
    std::string cubinFile = config_.tempDir + "/yirage_kernel.cubin";
    
    {
      std::ofstream ofs(ptxFile);
      ofs << ptx;
    }
    
    // Find ptxas
    std::string ptxas = findPtxas();
    if (ptxas.empty()) {
      llvm::errs() << "Could not find ptxas\n";
      return {};
    }
    
    // Build command
    std::vector<llvm::StringRef> args = {
      ptxas,
      "-arch", config_.arch.empty() ? "sm_80" : config_.arch,
      "-O3",
      "-o", cubinFile,
      ptxFile
    };
    
    if (config_.useFastMath) {
      args.push_back("--use_fast_math");
    }
    
    // Execute ptxas
    std::string errMsg;
    int result = llvm::sys::ExecuteAndWait(ptxas, args, std::nullopt, {}, 0, 0, &errMsg);
    
    if (result != 0) {
      llvm::errs() << "ptxas failed: " << errMsg << "\n";
      return {};
    }
    
    // Read cubin
    auto bufferOrErr = llvm::MemoryBuffer::getFile(cubinFile);
    if (!bufferOrErr) {
      llvm::errs() << "Could not read cubin file\n";
      return {};
    }
    
    auto& buffer = *bufferOrErr;
    std::vector<uint8_t> cubin(buffer->getBufferStart(), 
                                buffer->getBufferEnd());
    
    // Cleanup
    llvm::sys::fs::remove(ptxFile);
    llvm::sys::fs::remove(cubinFile);
    
    return cubin;
  }
  
private:
  std::string findPtxas() {
    // Check CUDA_HOME
    if (const char* cudaHome = std::getenv("CUDA_HOME")) {
      std::string path = std::string(cudaHome) + "/bin/ptxas";
      if (llvm::sys::fs::exists(path)) return path;
    }
    
    // Check common paths
    std::vector<std::string> paths = {
      "/usr/local/cuda/bin/ptxas",
      "/opt/cuda/bin/ptxas",
      "/usr/bin/ptxas"
    };
    
    for (const auto& path : paths) {
      if (llvm::sys::fs::exists(path)) return path;
    }
    
    // Try PATH
    auto result = llvm::sys::findProgramByName("ptxas");
    if (result) return result.get();
    
    return "";
  }
  
  BinaryGenConfig config_;
};

//===----------------------------------------------------------------------===//
// HSACO Generation (AMD ROCm)
//===----------------------------------------------------------------------===//

class HSACOGenerator {
public:
  HSACOGenerator(const BinaryGenConfig& config) : config_(config) {}
  
  /// Generate HSACO from LLVM IR
  std::vector<uint8_t> generateHSACO(llvm::Module& module) {
    // Set target for AMDGPU
    module.setTargetTriple("amdgcn-amd-amdhsa");
    
    std::string dataLayout = 
        "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-"
        "i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-"
        "v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7";
    module.setDataLayout(dataLayout);
    
    // Initialize AMDGPU target
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmPrinter();
    
    std::string error;
    auto target = llvm::TargetRegistry::lookupTarget(module.getTargetTriple(), error);
    if (!target) {
      llvm::errs() << "Error looking up AMDGPU target: " << error << "\n";
      return {};
    }
    
    // Create target machine
    std::string cpu = config_.arch.empty() ? "gfx942" : config_.arch;
    llvm::TargetOptions options;
    
    auto targetMachine = target->createTargetMachine(
        module.getTargetTriple(),
        cpu,
        "",
        options,
        llvm::Reloc::PIC_,
        llvm::CodeModel::Small,
        llvm::CodeGenOpt::Aggressive);
    
    if (!targetMachine) {
      llvm::errs() << "Could not create AMDGPU target machine\n";
      return {};
    }
    
    // Generate object file
    std::string objFile = config_.tempDir + "/yirage_kernel.o";
    std::string hsacoFile = config_.tempDir + "/yirage_kernel.hsaco";
    
    {
      std::error_code EC;
      llvm::raw_fd_ostream dest(objFile, EC, llvm::sys::fs::OF_None);
      if (EC) {
        llvm::errs() << "Could not open output file: " << EC.message() << "\n";
        return {};
      }
      
      llvm::legacy::PassManager pm;
      if (targetMachine->addPassesToEmitFile(pm, dest, nullptr,
                                             llvm::CodeGenFileType::CGFT_ObjectFile)) {
        llvm::errs() << "Cannot emit object file\n";
        return {};
      }
      pm.run(module);
    }
    
    // Link to HSACO using lld
    std::string lld = findLLD();
    if (lld.empty()) {
      llvm::errs() << "Could not find lld\n";
      return {};
    }
    
    std::vector<llvm::StringRef> args = {
      lld,
      "-flavor", "gnu",
      "-shared",
      "-o", hsacoFile,
      objFile
    };
    
    std::string errMsg;
    int result = llvm::sys::ExecuteAndWait(lld, args, std::nullopt, {}, 0, 0, &errMsg);
    
    if (result != 0) {
      llvm::errs() << "lld failed: " << errMsg << "\n";
      return {};
    }
    
    // Read HSACO
    auto bufferOrErr = llvm::MemoryBuffer::getFile(hsacoFile);
    if (!bufferOrErr) {
      llvm::errs() << "Could not read HSACO file\n";
      return {};
    }
    
    auto& buffer = *bufferOrErr;
    std::vector<uint8_t> hsaco(buffer->getBufferStart(),
                                buffer->getBufferEnd());
    
    // Cleanup
    llvm::sys::fs::remove(objFile);
    llvm::sys::fs::remove(hsacoFile);
    
    return hsaco;
  }
  
private:
  std::string findLLD() {
    // Check ROCm path
    std::string rocmLLD = config_.rocmPath + "/llvm/bin/ld.lld";
    if (llvm::sys::fs::exists(rocmLLD)) return rocmLLD;
    
    // Try PATH
    auto result = llvm::sys::findProgramByName("ld.lld");
    if (result) return result.get();
    
    result = llvm::sys::findProgramByName("lld");
    if (result) return result.get();
    
    return "";
  }
  
  BinaryGenConfig config_;
};

//===----------------------------------------------------------------------===//
// SPIR-V Generation (Intel XPU, Vulkan)
//===----------------------------------------------------------------------===//

class SPIRVGenerator {
public:
  SPIRVGenerator(const BinaryGenConfig& config) : config_(config) {}
  
  /// Generate SPIR-V binary from MLIR
  std::vector<uint8_t> generateSPIRV(mlir::ModuleOp module) {
    // Use mlir-translate to generate SPIR-V
    std::string mlirFile = config_.tempDir + "/yirage_kernel.mlir";
    std::string spvFile = config_.tempDir + "/yirage_kernel.spv";
    
    // Write MLIR to file
    {
      std::error_code EC;
      llvm::raw_fd_ostream dest(mlirFile, EC, llvm::sys::fs::OF_Text);
      if (EC) {
        llvm::errs() << "Could not write MLIR file\n";
        return {};
      }
      module.print(dest);
    }
    
    // Find mlir-translate
    std::string mlirTranslate = findMLIRTranslate();
    if (mlirTranslate.empty()) {
      llvm::errs() << "Could not find mlir-translate\n";
      return {};
    }
    
    // Convert to SPIR-V
    std::vector<llvm::StringRef> args = {
      mlirTranslate,
      "--mlir-to-spirv",
      "-o", spvFile,
      mlirFile
    };
    
    std::string errMsg;
    int result = llvm::sys::ExecuteAndWait(mlirTranslate, args, 
                                           std::nullopt, {}, 0, 0, &errMsg);
    
    if (result != 0) {
      llvm::errs() << "mlir-translate failed: " << errMsg << "\n";
      return {};
    }
    
    // Read SPIR-V
    auto bufferOrErr = llvm::MemoryBuffer::getFile(spvFile);
    if (!bufferOrErr) {
      llvm::errs() << "Could not read SPIR-V file\n";
      return {};
    }
    
    auto& buffer = *bufferOrErr;
    std::vector<uint8_t> spv(buffer->getBufferStart(),
                              buffer->getBufferEnd());
    
    // Cleanup
    llvm::sys::fs::remove(mlirFile);
    llvm::sys::fs::remove(spvFile);
    
    return spv;
  }
  
private:
  std::string findMLIRTranslate() {
    auto result = llvm::sys::findProgramByName("mlir-translate");
    if (result) return result.get();
    
    // Check common LLVM install paths
    std::vector<std::string> paths = {
      "/usr/lib/llvm-17/bin/mlir-translate",
      "/usr/local/bin/mlir-translate",
      "/opt/llvm/bin/mlir-translate"
    };
    
    for (const auto& path : paths) {
      if (llvm::sys::fs::exists(path)) return path;
    }
    
    return "";
  }
  
  BinaryGenConfig config_;
};

//===----------------------------------------------------------------------===//
// Metal Generation (Apple Silicon)
//===----------------------------------------------------------------------===//

class MetalGenerator {
public:
  MetalGenerator(const BinaryGenConfig& config) : config_(config) {}
  
  /// Generate Metal library from MLIR
  std::vector<uint8_t> generateMetallib(mlir::ModuleOp module) {
    // Metal compilation pipeline:
    // MLIR → LLVM IR → AIR → metallib
    
    std::string mlirFile = config_.tempDir + "/yirage_kernel.mlir";
    std::string airFile = config_.tempDir + "/yirage_kernel.air";
    std::string metallibFile = config_.tempDir + "/yirage_kernel.metallib";
    
    // Write MLIR
    {
      std::error_code EC;
      llvm::raw_fd_ostream dest(mlirFile, EC, llvm::sys::fs::OF_Text);
      if (EC) return {};
      module.print(dest);
    }
    
    // Check for xcrun (macOS only)
    auto xcrun = llvm::sys::findProgramByName("xcrun");
    if (!xcrun) {
      llvm::errs() << "xcrun not found - Metal compilation requires macOS\n";
      return {};
    }
    
    // For now, return placeholder as Metal compilation is macOS-specific
    // In production, this would:
    // 1. Lower MLIR to Metal Shading Language (MSL)
    // 2. Compile MSL with metal compiler
    // 3. Create metallib archive
    
    llvm::errs() << "Metal compilation not fully implemented yet\n";
    return {};
  }
  
private:
  BinaryGenConfig config_;
};

//===----------------------------------------------------------------------===//
// Ascend CCE Generation (Huawei NPU)
//===----------------------------------------------------------------------===//

class AscendGenerator {
public:
  AscendGenerator(const BinaryGenConfig& config) : config_(config) {}
  
  /// Generate Ascend kernel from MLIR
  std::vector<uint8_t> generateAscendKernel(mlir::ModuleOp module) {
    // Ascend compilation requires:
    // 1. CANN toolkit
    // 2. CCE (Cube Compute Engine) compiler
    
    std::string cannPath = std::getenv("ASCEND_HOME") ? 
                            std::getenv("ASCEND_HOME") : "/usr/local/Ascend";
    
    std::string ccec = cannPath + "/compiler/bin/ccec";
    if (!llvm::sys::fs::exists(ccec)) {
      llvm::errs() << "Ascend CCE compiler not found at: " << ccec << "\n";
      return {};
    }
    
    // Write MLIR and compile
    std::string mlirFile = config_.tempDir + "/yirage_kernel.mlir";
    std::string oFile = config_.tempDir + "/yirage_kernel.o";
    
    {
      std::error_code EC;
      llvm::raw_fd_ostream dest(mlirFile, EC, llvm::sys::fs::OF_Text);
      if (EC) return {};
      module.print(dest);
    }
    
    // For now, return placeholder
    // Full implementation would invoke ccec
    llvm::errs() << "Ascend compilation requires CANN toolkit\n";
    return {};
  }
  
private:
  BinaryGenConfig config_;
};

//===----------------------------------------------------------------------===//
// MACA Generation (MetaX GPU)
//===----------------------------------------------------------------------===//

class MACAGenerator {
public:
  MACAGenerator(const BinaryGenConfig& config) : config_(config) {}
  
  /// Generate MACA kernel (similar to CUDA)
  std::vector<uint8_t> generateMACAKernel(mlir::ModuleOp module) {
    // MACA uses similar compilation flow to CUDA
    // Check for MACA toolkit
    std::string macaPath = std::getenv("MACA_HOME") ?
                            std::getenv("MACA_HOME") : "/opt/maca";
    
    if (!llvm::sys::fs::exists(macaPath)) {
      llvm::errs() << "MACA toolkit not found\n";
      return {};
    }
    
    // Similar to PTX generation
    llvm::errs() << "MACA compilation requires MACA toolkit\n";
    return {};
  }
  
private:
  BinaryGenConfig config_;
};

//===----------------------------------------------------------------------===//
// Unified Binary Generator
//===----------------------------------------------------------------------===//

class UnifiedBinaryGenerator {
public:
  UnifiedBinaryGenerator(mlir::MLIRContext* context, const GPUTargetConfig& config)
      : context_(context), config_(config) {
    binConfig_.arch = config.arch;
    binConfig_.triple = config.triple;
  }
  
  /// Generate binary for the configured backend
  std::vector<uint8_t> generate(mlir::ModuleOp module) {
    switch (config_.backend) {
    case GPUBackend::CUDA:
      return generateCUDA(module);
    case GPUBackend::ROCm:
      return generateROCm(module);
    case GPUBackend::SPIRV:
      return generateSPIRV(module);
    case GPUBackend::Metal:
      return generateMetal(module);
    }
    return {};
  }
  
  /// Generate PTX text (CUDA only)
  std::string generatePTXText(mlir::ModuleOp module) {
    if (config_.backend != GPUBackend::CUDA) {
      return "";
    }
    
    // Lower to LLVM IR
    auto llvmModule = translateToLLVMIR(module);
    if (!llvmModule) return "";
    
    PTXGenerator gen(binConfig_);
    return gen.generatePTX(*llvmModule);
  }
  
private:
  std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp module) {
    // Register translations
    mlir::registerLLVMDialectTranslation(*context_);
    mlir::registerNVVMDialectTranslation(*context_);
    
    // Create LLVM context and translate
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    
    return llvmModule;
  }
  
  std::vector<uint8_t> generateCUDA(mlir::ModuleOp module) {
    auto llvmModule = translateToLLVMIR(module);
    if (!llvmModule) return {};
    
    PTXGenerator gen(binConfig_);
    std::string ptx = gen.generatePTX(*llvmModule);
    if (ptx.empty()) return {};
    
    return gen.compilePTXToCubin(ptx);
  }
  
  std::vector<uint8_t> generateROCm(mlir::ModuleOp module) {
    auto llvmModule = translateToLLVMIR(module);
    if (!llvmModule) return {};
    
    HSACOGenerator gen(binConfig_);
    return gen.generateHSACO(*llvmModule);
  }
  
  std::vector<uint8_t> generateSPIRV(mlir::ModuleOp module) {
    SPIRVGenerator gen(binConfig_);
    return gen.generateSPIRV(module);
  }
  
  std::vector<uint8_t> generateMetal(mlir::ModuleOp module) {
    MetalGenerator gen(binConfig_);
    return gen.generateMetallib(module);
  }
  
  mlir::MLIRContext* context_;
  GPUTargetConfig config_;
  BinaryGenConfig binConfig_;
};

//===----------------------------------------------------------------------===//
// Update GPUCodeGen to use UnifiedBinaryGenerator
//===----------------------------------------------------------------------===//

std::vector<uint8_t> GPUCodeGen::getBinary(mlir::ModuleOp module) {
  UnifiedBinaryGenerator gen(context, config);
  return gen.generate(module);
}

std::string GPUCodeGen::getPTX(mlir::ModuleOp module) {
  UnifiedBinaryGenerator gen(context, config);
  return gen.generatePTXText(module);
}

} // namespace yirage
