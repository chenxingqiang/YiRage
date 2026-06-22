//===- GPUCodeGen.cpp - GPU Code Generation Backend --------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file implements GPU code generation for different GPU backends:
//   - NVIDIA CUDA (PTX/cubin)
//   - AMD ROCm (GCN/hsaco)
//   - Intel oneAPI (SPIR-V)
//   - Apple Metal (MSL/metallib)
//
// Pipeline:
//   GPU dialect → Target dialect (NVVM/ROCDL/SPIRV) → Binary
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Execution/GPUCodeGen.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRVPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

using namespace mlir;

namespace yirage {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

namespace {

/// Find CUDA toolkit path
std::string findCUDAPath() {
  // Check environment variable
  if (auto cudaHome = std::getenv("CUDA_HOME")) {
    return std::string(cudaHome);
  }
  if (auto cudaPath = std::getenv("CUDA_PATH")) {
    return std::string(cudaPath);
  }
  
  // Common paths
  const char *paths[] = {
    "/usr/local/cuda",
    "/opt/cuda",
    "/usr/lib/cuda",
  };
  
  for (const auto &path : paths) {
    if (llvm::sys::fs::exists(std::string(path) + "/bin/nvcc")) {
      return std::string(path);
    }
  }
  
  return "";
}

/// Find ROCm path
std::string findROCmPath() {
  if (auto rocmPath = std::getenv("ROCM_PATH")) {
    return std::string(rocmPath);
  }
  
  const char *paths[] = {
    "/opt/rocm",
    "/opt/rocm-6.0.0",
    "/opt/rocm-5.7.0",
  };
  
  for (const auto &path : paths) {
    if (llvm::sys::fs::exists(path)) {
      return std::string(path);
    }
  }
  
  return "";
}

/// Run external compiler command
bool runExternalCommand(const std::string &program,
                        const std::vector<std::string> &args,
                        std::string &output) {
  // Build argument list
  llvm::SmallVector<llvm::StringRef, 16> argRefs;
  argRefs.push_back(program);
  for (const auto &arg : args) {
    argRefs.push_back(arg);
  }
  
  // Find program
  auto programPath = llvm::sys::findProgramByName(program);
  if (!programPath) {
    output = "Program not found: " + program;
    return false;
  }
  
  // Run command
  int result = llvm::sys::ExecuteAndWait(*programPath, argRefs);
  return result == 0;
}

} // namespace

//===----------------------------------------------------------------------===//
// GPU Code Generator Implementation
//===----------------------------------------------------------------------===//

class GPUCodeGenImpl {
public:
  GPUCodeGenImpl(MLIRContext *context, const GPUTargetConfig &config)
      : context(context), config(config) {
    // Initialize LLVM targets
    initializeLLVMTargets();
  }
  
  /// Initialize required LLVM targets
  void initializeLLVMTargets() {
    static bool initialized = false;
    if (!initialized) {
      // Initialize NVPTX target for CUDA
      LLVMInitializeNVPTXTargetInfo();
      LLVMInitializeNVPTXTarget();
      LLVMInitializeNVPTXTargetMC();
      LLVMInitializeNVPTXAsmPrinter();
      
      // Initialize AMDGPU target for ROCm
      LLVMInitializeAMDGPUTargetInfo();
      LLVMInitializeAMDGPUTarget();
      LLVMInitializeAMDGPUTargetMC();
      LLVMInitializeAMDGPUAsmPrinter();
      
      initialized = true;
    }
  }
  
  /// Generate target-specific IR from GPU dialect
  LogicalResult generateTargetIR(ModuleOp module) {
    PassManager pm(context);
    
    // Enable pass instrumentation for debugging
    pm.enableVerifier(true);
    
    switch (config.backend) {
    case GPUBackend::CUDA:
      return generateNVVM(pm, module);
    case GPUBackend::ROCm:
      return generateROCDL(pm, module);
    case GPUBackend::SPIRV:
      return generateSPIRV(pm, module);
    case GPUBackend::Metal:
      return generateMetal(pm, module);
    default:
      return failure();
    }
  }
  
  /// Export generated code as text (PTX, GCN assembly, SPIR-V text)
  std::string exportAsText(ModuleOp module) {
    std::string output;
    llvm::raw_string_ostream os(output);
    
    switch (config.backend) {
    case GPUBackend::CUDA:
      output = exportPTX(module);
      break;
    case GPUBackend::ROCm:
      output = exportGCN(module);
      break;
    case GPUBackend::SPIRV:
      output = exportSPIRVText(module);
      break;
    case GPUBackend::Metal:
      output = exportMSL(module);
      break;
    default:
      module.print(os);
      break;
    }
    
    return output;
  }
  
  /// Get the compiled binary (cubin, HSACO, SPIR-V binary)
  std::vector<uint8_t> getBinary(ModuleOp module) {
    switch (config.backend) {
    case GPUBackend::CUDA:
      return generateCubin(module);
    case GPUBackend::ROCm:
      return generateHSACO(module);
    case GPUBackend::SPIRV:
      return generateSPIRVBinary(module);
    default:
      return {};
    }
  }

private:
  //===--------------------------------------------------------------------===//
  // NVVM (CUDA) Code Generation
  //===--------------------------------------------------------------------===//
  
  LogicalResult generateNVVM(PassManager &pm, ModuleOp module) {
    // Step 1: Canonicalize and optimize
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    
    // Step 2: Lower GPU dialect to NVVM dialect
    // The GPU to NVVM conversion handles:
    // - gpu.launch_func → CUDA kernel launch
    // - gpu.block_id → nvvm.read.ptx.sreg.ctaid.*
    // - gpu.thread_id → nvvm.read.ptx.sreg.tid.*
    // - gpu.barrier → nvvm.barrier0
    // - Shared memory allocation
    pm.addNestedPass<gpu::GPUModuleOp>(createLowerGpuOpsToNVVMOpsPass());
    
    // Step 3: Lower remaining arith/memref ops to LLVM
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    
    // Step 4: Cleanup
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    
    return pm.run(module);
  }
  
  /// Export PTX assembly text
  std::string exportPTX(ModuleOp module) {
    // Clone module to avoid modifying the original
    auto clonedModule = module.clone();
    
    // Apply NVVM lowering
    PassManager pm(context);
    if (failed(generateNVVM(pm, clonedModule))) {
      return "// Failed to lower to NVVM\n";
    }
    
    // Translate to LLVM IR
    llvm::LLVMContext llvmContext;
    registerLLVMDialectTranslation(*context);
    registerNVVMDialectTranslation(*context);
    
    auto llvmModule = translateModuleToLLVMIR(clonedModule, llvmContext);
    if (!llvmModule) {
      return "// Failed to translate to LLVM IR\n";
    }
    
    // Set target triple and data layout for NVPTX
    llvmModule->setTargetTriple("nvptx64-nvidia-cuda");
    llvmModule->setDataLayout(
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-"
        "f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-"
        "n16:32:64");
    
    // Generate PTX using LLVM backend
    llvm::SmallVector<char, 0> ptxBuffer;
    llvm::raw_svector_ostream ptxStream(ptxBuffer);
    
    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
        "nvptx64-nvidia-cuda", error);
    
    if (!target) {
      return "// NVPTX target not available: " + error + "\n";
    }
    
    llvm::TargetOptions options;
    auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(
            "nvptx64-nvidia-cuda",
            config.arch.empty() ? "sm_80" : config.arch,
            config.features,
            options,
            std::nullopt));
    
    if (!targetMachine) {
      return "// Failed to create target machine\n";
    }
    
    llvm::legacy::PassManager passManager;
    if (targetMachine->addPassesToEmitFile(passManager, ptxStream, nullptr,
                                            llvm::CodeGenFileType::CGFT_AssemblyFile)) {
      return "// Failed to add passes for PTX emission\n";
    }
    
    passManager.run(*llvmModule);
    return std::string(ptxBuffer.begin(), ptxBuffer.end());
  }
  
  /// Generate cubin binary
  std::vector<uint8_t> generateCubin(ModuleOp module) {
    // First generate PTX
    std::string ptx = exportPTX(module);
    if (ptx.empty() || ptx.find("// Failed") == 0) {
      return {};
    }
    
    // Find CUDA toolkit
    std::string cudaPath = findCUDAPath();
    if (cudaPath.empty()) {
      return {};
    }
    
    // Write PTX to temporary file
    llvm::SmallString<128> ptxPath;
    llvm::sys::fs::createTemporaryFile("yirage", "ptx", ptxPath);
    {
      std::ofstream file(ptxPath.str().str());
      file << ptx;
    }
    
    // Prepare output file
    llvm::SmallString<128> cubinPath;
    llvm::sys::fs::createTemporaryFile("yirage", "cubin", cubinPath);
    
    // Run ptxas to compile PTX to cubin
    std::string ptxas = cudaPath + "/bin/ptxas";
    std::vector<std::string> args = {
        "-arch=" + (config.arch.empty() ? "sm_80" : config.arch),
        "-o", cubinPath.str().str(),
        ptxPath.str().str()
    };
    
    std::string output;
    if (!runExternalCommand(ptxas, args, output)) {
      return {};
    }
    
    // Read cubin file
    auto buffer = llvm::MemoryBuffer::getFile(cubinPath);
    if (!buffer) {
      return {};
    }
    
    auto data = (*buffer)->getBuffer();
    return std::vector<uint8_t>(data.begin(), data.end());
  }
  
  //===--------------------------------------------------------------------===//
  // ROCDL (ROCm) Code Generation
  //===--------------------------------------------------------------------===//
  
  LogicalResult generateROCDL(PassManager &pm, ModuleOp module) {
    // Step 1: Canonicalize and optimize
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    
    // Step 2: Lower GPU dialect to ROCDL dialect
    // ROCDL (ROCm Device Library) uses 64-thread wavefronts
    pm.addNestedPass<gpu::GPUModuleOp>(
        createLowerGpuOpsToROCDLOpsPass(config.arch, /*indexBitwidth=*/64));
    
    // Step 3: Lower to LLVM
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(createConvertFuncToLLVMPass());
    
    // Step 4: Cleanup
    pm.addPass(createCanonicalizerPass());
    
    return pm.run(module);
  }
  
  /// Export GCN assembly text
  std::string exportGCN(ModuleOp module) {
    auto clonedModule = module.clone();
    
    PassManager pm(context);
    if (failed(generateROCDL(pm, clonedModule))) {
      return "// Failed to lower to ROCDL\n";
    }
    
    // Translate to LLVM IR
    llvm::LLVMContext llvmContext;
    registerLLVMDialectTranslation(*context);
    registerROCDLDialectTranslation(*context);
    
    auto llvmModule = translateModuleToLLVMIR(clonedModule, llvmContext);
    if (!llvmModule) {
      return "// Failed to translate to LLVM IR\n";
    }
    
    // Set target triple for AMDGPU
    llvmModule->setTargetTriple("amdgcn-amd-amdhsa");
    
    // Generate assembly using LLVM backend
    llvm::SmallVector<char, 0> gcnBuffer;
    llvm::raw_svector_ostream asmStream(gcnBuffer);
    
    std::string error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
        "amdgcn-amd-amdhsa", error);
    
    if (!target) {
      return "// AMDGPU target not available: " + error + "\n";
    }
    
    llvm::TargetOptions options;
    std::string gpuArch = config.arch.empty() ? "gfx908" : config.arch;
    
    auto targetMachine = std::unique_ptr<llvm::TargetMachine>(
        target->createTargetMachine(
            "amdgcn-amd-amdhsa",
            gpuArch,
            config.features,
            options,
            std::nullopt));
    
    if (!targetMachine) {
      return "// Failed to create target machine\n";
    }
    
    llvm::legacy::PassManager passManager;
    if (targetMachine->addPassesToEmitFile(passManager, asmStream, nullptr,
                                            llvm::CodeGenFileType::CGFT_AssemblyFile)) {
      return "// Failed to add passes for GCN emission\n";
    }
    
    passManager.run(*llvmModule);
    return std::string(gcnBuffer.begin(), gcnBuffer.end());
  }
  
  /// Generate HSACO binary
  std::vector<uint8_t> generateHSACO(ModuleOp module) {
    // First generate GCN assembly
    std::string gcnAsm = exportGCN(module);
    if (gcnAsm.empty() || gcnAsm.find("// Failed") == 0) {
      return {};
    }
    
    // Find ROCm toolkit
    std::string rocmPath = findROCmPath();
    if (rocmPath.empty()) {
      return {};
    }
    
    // Write assembly to temporary file
    llvm::SmallString<128> asmPath;
    llvm::sys::fs::createTemporaryFile("yirage", "s", asmPath);
    {
      std::ofstream file(asmPath.str().str());
      file << gcnAsm;
    }
    
    // Prepare output file
    llvm::SmallString<128> hsacoPath;
    llvm::sys::fs::createTemporaryFile("yirage", "hsaco", hsacoPath);
    
    // Run lld to link assembly to HSACO
    std::string lld = rocmPath + "/llvm/bin/ld.lld";
    std::vector<std::string> args = {
        "-shared",
        "-o", hsacoPath.str().str(),
        asmPath.str().str()
    };
    
    std::string output;
    if (!runExternalCommand(lld, args, output)) {
      return {};
    }
    
    // Read HSACO file
    auto buffer = llvm::MemoryBuffer::getFile(hsacoPath);
    if (!buffer) {
      return {};
    }
    
    auto data = (*buffer)->getBuffer();
    return std::vector<uint8_t>(data.begin(), data.end());
  }
  
  //===--------------------------------------------------------------------===//
  // SPIR-V Code Generation
  //===--------------------------------------------------------------------===//
  
  LogicalResult generateSPIRV(PassManager &pm, ModuleOp module) {
    // Step 1: Canonicalize and optimize
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    
    // Step 2: Lower GPU dialect to SPIR-V dialect
    pm.addPass(createConvertGPUToSPIRVPass());
    
    // Step 3: Apply SPIR-V-specific optimizations
    pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
    pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVUpdateVCEPass());
    
    // Step 4: Cleanup
    pm.addPass(createCanonicalizerPass());
    
    return pm.run(module);
  }
  
  /// Export SPIR-V assembly text
  std::string exportSPIRVText(ModuleOp module) {
    auto clonedModule = module.clone();
    
    PassManager pm(context);
    if (failed(generateSPIRV(pm, clonedModule))) {
      return "; Failed to lower to SPIR-V\n";
    }
    
    // Print the SPIR-V module
    std::string output;
    llvm::raw_string_ostream os(output);
    clonedModule.print(os);
    
    return output;
  }
  
  /// Generate SPIR-V binary
  std::vector<uint8_t> generateSPIRVBinary(ModuleOp module) {
    auto clonedModule = module.clone();
    
    PassManager pm(context);
    if (failed(generateSPIRV(pm, clonedModule))) {
      return {};
    }
    
    // Find the SPIR-V module
    spirv::ModuleOp spirvModule;
    clonedModule.walk([&](spirv::ModuleOp op) {
      spirvModule = op;
      return WalkResult::interrupt();
    });
    
    if (!spirvModule) {
      return {};
    }
    
    // Serialize to binary
    llvm::SmallVector<uint32_t, 1024> binary;
    if (failed(spirv::serialize(spirvModule, binary))) {
      return {};
    }
    
    // Convert to byte vector
    std::vector<uint8_t> result;
    result.reserve(binary.size() * sizeof(uint32_t));
    for (uint32_t word : binary) {
      result.push_back(word & 0xFF);
      result.push_back((word >> 8) & 0xFF);
      result.push_back((word >> 16) & 0xFF);
      result.push_back((word >> 24) & 0xFF);
    }
    
    return result;
  }
  
  //===--------------------------------------------------------------------===//
  // Metal Code Generation
  //===--------------------------------------------------------------------===//
  
  LogicalResult generateMetal(PassManager &pm, ModuleOp module) {
    // Metal codegen strategy:
    // 1. First lower to SPIR-V
    // 2. Then use SPIRV-Cross to convert to MSL
    
    // Step 1: Apply standard optimizations
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    
    // Step 2: Lower to SPIR-V (as intermediate)
    pm.addPass(createConvertGPUToSPIRVPass());
    pm.addNestedPass<spirv::ModuleOp>(spirv::createSPIRVLowerABIAttributesPass());
    
    return pm.run(module);
  }
  
  /// Export Metal Shading Language (MSL) text
  std::string exportMSL(ModuleOp module) {
    // First generate SPIR-V binary
    std::vector<uint8_t> spirvBinary = generateSPIRVBinary(module);
    if (spirvBinary.empty()) {
      return "// Failed to generate SPIR-V for Metal conversion\n";
    }
    
    // Write SPIR-V to temporary file
    llvm::SmallString<128> spirvPath;
    llvm::sys::fs::createTemporaryFile("yirage", "spv", spirvPath);
    {
      std::ofstream file(spirvPath.str().str(), std::ios::binary);
      file.write(reinterpret_cast<const char*>(spirvBinary.data()),
                 spirvBinary.size());
    }
    
    // Prepare output file
    llvm::SmallString<128> mslPath;
    llvm::sys::fs::createTemporaryFile("yirage", "metal", mslPath);
    
    // Run SPIRV-Cross to convert to MSL
    // MSL version encoding: major * 10000 + minor * 100
    // 20000 = Metal 2.0, 20100 = Metal 2.1, 30000 = Metal 3.0
    std::vector<std::string> args = {
        "--msl",
        "--msl-version", "20000",  // Metal 2.0 (major=2, minor=0)
        "--output", mslPath.str().str(),
        spirvPath.str().str()
    };
    
    std::string output;
    if (!runExternalCommand("spirv-cross", args, output)) {
      return "// SPIRV-Cross not found. Install with: brew install spirv-cross\n"
             "// Or manually convert the SPIR-V binary.\n";
    }
    
    // Read MSL file
    auto buffer = llvm::MemoryBuffer::getFile(mslPath);
    if (!buffer) {
      return "// Failed to read generated MSL file\n";
    }
    
    return (*buffer)->getBuffer().str();
  }

  MLIRContext *context;
  GPUTargetConfig config;
};

//===----------------------------------------------------------------------===//
// GPU Code Generator Public Interface
//===----------------------------------------------------------------------===//

GPUCodeGen::GPUCodeGen(MLIRContext *context, const GPUTargetConfig &config)
    : impl(std::make_unique<GPUCodeGenImpl>(context, config)),
      context(context), config(config) {}

GPUCodeGen::~GPUCodeGen() = default;

LogicalResult GPUCodeGen::compile(ModuleOp module) {
  return impl->generateTargetIR(module);
}

std::string GPUCodeGen::exportCode(ModuleOp module) {
  return impl->exportAsText(module);
}

std::vector<uint8_t> GPUCodeGen::getBinary(ModuleOp module) {
  return impl->getBinary(module);
}

} // namespace yirage
