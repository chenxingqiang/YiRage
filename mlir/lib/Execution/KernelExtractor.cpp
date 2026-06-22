//===- KernelExtractor.cpp - GPU Kernel Extraction Utility ------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Execution/KernelExtractor.h"
#include "yirage-mlir/Execution/GPUCodeGen.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <fstream>
#include <mutex>

using namespace mlir;

namespace yirage {

//===----------------------------------------------------------------------===//
// Kernel Extractor Implementation
//===----------------------------------------------------------------------===//

class KernelExtractorImpl {
public:
  KernelExtractorImpl(MLIRContext *ctx) : context(ctx) {}
  
  /// Extract kernel info from a function
  KernelInfo extractKernelInfo(func::FuncOp func) {
    KernelInfo info;
    info.name = func.getSymName().str();
    info.mangledName = info.name;
    
    // Extract argument types
    auto funcType = func.getFunctionType();
    for (auto inputType : funcType.getInputs()) {
      std::string typeStr;
      llvm::raw_string_ostream os(typeStr);
      inputType.print(os);
      info.argTypes.push_back(os.str());
    }
    
    // Extract return type
    if (funcType.getNumResults() > 0) {
      std::string typeStr;
      llvm::raw_string_ostream os(typeStr);
      funcType.getResult(0).print(os);
      info.returnType = os.str();
    }
    
    // Check for GPU kernel attributes
    if (auto gpuAttr = func->getAttrOfType<gpu::KernelDim3Attr>("gpu.grid_size")) {
      info.gridSize = {gpuAttr.getX(), gpuAttr.getY(), gpuAttr.getZ()};
    }
    
    if (auto gpuAttr = func->getAttrOfType<gpu::KernelDim3Attr>("gpu.block_size")) {
      info.blockSize = {gpuAttr.getX(), gpuAttr.getY(), gpuAttr.getZ()};
    }
    
    // Extract original MLIR
    std::string mlirStr;
    llvm::raw_string_ostream os(mlirStr);
    func.print(os);
    info.originalMLIR = os.str();
    
    // Get source location
    if (auto loc = func.getLoc().dyn_cast<FileLineColLoc>()) {
      info.sourceLocation = loc.getFilename().str() + ":" + 
                            std::to_string(loc.getLine()) + ":" +
                            std::to_string(loc.getColumn());
    }
    
    return info;
  }
  
  /// Check if function is a kernel
  bool isKernel(func::FuncOp func) {
    // Check for explicit kernel marker
    if (func->hasAttr("gpu.kernel")) return true;
    if (func->hasAttr("nvvm.kernel")) return true;
    if (func->hasAttr("rocdl.kernel")) return true;
    
    // Check for naming conventions
    std::string name = func.getSymName().str();
    if (name.find("kernel") != std::string::npos) return true;
    if (name.find("_gpu") != std::string::npos) return true;
    
    // Check if it contains GPU operations
    bool hasGPUOps = false;
    func.walk([&](Operation *op) {
      if (op->getDialect()->getNamespace() == "gpu") {
        hasGPUOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    
    return hasGPUOps;
  }
  
  /// Analyze resource usage
  void analyzeResources(KernelInfo &info, func::FuncOp func) {
    // Count operations
    int64_t loads = 0, stores = 0;
    int64_t fmas = 0;
    int64_t barriers = 0;
    
    func.walk([&](Operation *op) {
      if (isa<memref::LoadOp>(op)) loads++;
      else if (isa<memref::StoreOp>(op)) stores++;
      else if (isa<gpu::BarrierOp>(op)) barriers++;
    });
    
    // Estimate register usage (simplified)
    // Real analysis would use LLVM's register allocator
    int64_t liveRanges = 0;
    func.walk([&](Operation *op) {
      liveRanges += op->getNumResults();
    });
    
    info.registersUsed = std::min(liveRanges * 2, (int64_t)255);
    
    // Estimate shared memory (from alloc operations)
    func.walk([&](memref::AllocaOp alloc) {
      auto type = alloc.getType();
      if (auto addrSpace = type.getMemorySpace()) {
        if (auto intAttr = addrSpace.dyn_cast<IntegerAttr>()) {
          // Address space 3 is shared memory in CUDA
          if (intAttr.getInt() == 3) {
            info.staticSharedMemory += type.getNumElements() * 
                                       type.getElementTypeBitWidth() / 8;
          }
        }
      }
    });
  }
  
  /// Estimate occupancy
  double estimateOccupancy(const KernelInfo &info, llvm::StringRef target) {
    // Simplified occupancy calculator
    // Real implementation would use CUDA/ROCm occupancy APIs
    
    int maxThreadsPerSM = 2048;  // Ampere
    int maxBlocksPerSM = 32;
    int maxRegistersPerSM = 65536;
    int maxSharedMemPerSM = 164 * 1024;  // A100
    
    // Parse target
    if (target.contains("sm_80") || target.contains("sm_90")) {
      maxSharedMemPerSM = 164 * 1024;
      maxRegistersPerSM = 65536;
    } else if (target.contains("gfx9")) {
      // AMD MI250/MI300
      maxSharedMemPerSM = 64 * 1024;
      maxThreadsPerSM = 2048;
    }
    
    // Calculate limits
    int threadsPerBlock = info.blockSize[0] * info.blockSize[1] * info.blockSize[2];
    int registersPerThread = info.registersUsed;
    int sharedMemPerBlock = info.staticSharedMemory + info.dynamicSharedMemory;
    
    // Register limit
    int blocksByRegisters = maxRegistersPerSM / (registersPerThread * threadsPerBlock);
    
    // Shared memory limit
    int blocksBySharedMem = sharedMemPerBlock > 0 ? 
                            maxSharedMemPerSM / sharedMemPerBlock : maxBlocksPerSM;
    
    // Thread limit
    int blocksByThreads = maxThreadsPerSM / threadsPerBlock;
    
    // Take minimum
    int activeBlocks = std::min({blocksByRegisters, blocksBySharedMem, 
                                  blocksByThreads, maxBlocksPerSM});
    
    return (double)(activeBlocks * threadsPerBlock) / maxThreadsPerSM;
  }
  
private:
  MLIRContext *context;
};

//===----------------------------------------------------------------------===//
// Kernel Extractor Public Interface
//===----------------------------------------------------------------------===//

KernelExtractor::KernelExtractor(MLIRContext *context)
    : impl(std::make_unique<KernelExtractorImpl>(context)), context(context) {}

KernelExtractor::~KernelExtractor() = default;

std::vector<KernelInfo> KernelExtractor::extractKernels(ModuleOp module) {
  std::vector<KernelInfo> kernels;
  
  module.walk([&](func::FuncOp func) {
    if (impl->isKernel(func)) {
      auto info = impl->extractKernelInfo(func);
      impl->analyzeResources(info, func);
      kernels.push_back(info);
    }
  });
  
  return kernels;
}

std::optional<KernelInfo> KernelExtractor::getKernel(ModuleOp module,
                                                      StringRef name) {
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.getSymName() == name && impl->isKernel(func)) {
      auto info = impl->extractKernelInfo(func);
      impl->analyzeResources(info, func);
      return info;
    }
  }
  return std::nullopt;
}

bool KernelExtractor::isKernel(func::FuncOp func) {
  return impl->isKernel(func);
}

OwningOpRef<ModuleOp> KernelExtractor::isolateKernel(ModuleOp module,
                                                       StringRef kernelName) {
  // Create new module
  auto newModule = ModuleOp::create(module.getLoc());
  OpBuilder builder(newModule.getBodyRegion());
  
  // Find and clone the kernel
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.getSymName() == kernelName) {
      IRMapping mapping;
      builder.clone(*func, mapping);
      
      // Also clone any helper functions it calls
      func.walk([&](func::CallOp call) {
        if (auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee())) {
          if (!newModule.lookupSymbol(callee.getSymName())) {
            builder.clone(*callee, mapping);
          }
        }
      });
      
      break;
    }
  }
  
  return newModule;
}

OwningOpRef<ModuleOp> KernelExtractor::isolateKernels(
    ModuleOp module,
    ArrayRef<StringRef> kernelNames) {
  auto newModule = ModuleOp::create(module.getLoc());
  OpBuilder builder(newModule.getBodyRegion());
  IRMapping mapping;
  
  llvm::DenseSet<StringRef> requested(kernelNames.begin(), kernelNames.end());
  llvm::DenseSet<StringRef> cloned;
  
  for (auto func : module.getOps<func::FuncOp>()) {
    if (requested.contains(func.getSymName())) {
      builder.clone(*func, mapping);
      cloned.insert(func.getSymName());
    }
  }
  
  return newModule;
}

CompiledKernel KernelExtractor::compile(const KernelInfo &kernel,
                                          ModuleOp module,
                                          StringRef target) {
  CompiledKernel result;
  result.info = kernel;
  result.targetArch = target.str();
  
  auto start = std::chrono::high_resolution_clock::now();
  
  // Isolate the kernel
  auto isolatedModule = isolateKernel(module, kernel.name);
  if (!isolatedModule) {
    return result;
  }
  
  // Parse target
  GPUTargetConfig config;
  if (target.contains("cuda") || target.contains("sm_")) {
    config.backend = GPUBackend::CUDA;
    config.arch = target.str();
  } else if (target.contains("rocm") || target.contains("gfx")) {
    config.backend = GPUBackend::ROCm;
    config.arch = target.str();
  } else if (target.contains("spirv") || target.contains("xpu")) {
    config.backend = GPUBackend::SPIRV;
  } else if (target.contains("metal")) {
    config.backend = GPUBackend::Metal;
  }
  
  // Create code generator
  auto codegen = std::make_unique<GPUCodeGen>(context, config);
  
  // Compile
  if (succeeded(codegen->compile(*isolatedModule))) {
    result.textCode = codegen->exportCode(*isolatedModule);
    result.binary = codegen->getBinary(*isolatedModule);
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  result.compilationTime = std::chrono::duration<double>(end - start).count();
  
  return result;
}

std::vector<CompiledKernel> KernelExtractor::compileAll(ModuleOp module,
                                                          StringRef target) {
  auto kernels = extractKernels(module);
  std::vector<CompiledKernel> results;
  
  for (const auto &kernel : kernels) {
    results.push_back(compile(kernel, module, target));
  }
  
  return results;
}

void KernelExtractor::analyzeResources(KernelInfo &kernel, func::FuncOp func) {
  impl->analyzeResources(kernel, func);
}

double KernelExtractor::estimateOccupancy(const KernelInfo &kernel,
                                           StringRef target) {
  return impl->estimateOccupancy(kernel, target);
}

std::array<int64_t, 3> KernelExtractor::recommendBlockSize(
    const KernelInfo &kernel,
    StringRef target) {
  // Simple heuristic - optimize for occupancy
  int registersPerThread = kernel.registersUsed;
  int sharedMemPerBlock = kernel.staticSharedMemory;
  
  // Start with default
  std::array<int64_t, 3> blockSize = {256, 1, 1};
  
  // If high register usage, reduce block size
  if (registersPerThread > 64) {
    blockSize[0] = 128;
  } else if (registersPerThread > 128) {
    blockSize[0] = 64;
  }
  
  // For matmul-like kernels, use 2D blocks
  if (kernel.name.find("matmul") != std::string::npos ||
      kernel.name.find("gemm") != std::string::npos) {
    blockSize = {16, 16, 1};
  }
  
  return blockSize;
}

//===----------------------------------------------------------------------===//
// Kernel Cache Implementation
//===----------------------------------------------------------------------===//

struct KernelCache::Impl {
  std::string cacheDir;
  std::unordered_map<std::string, CompiledKernel> memoryCache;
  mutable std::mutex mutex;
  CacheStats stats;
  
  std::string makeKey(StringRef name, StringRef target) {
    return name.str() + "_" + target.str();
  }
  
  std::string makePath(StringRef name, StringRef target) {
    return cacheDir + "/" + makeKey(name, target) + ".bin";
  }
};

KernelCache::KernelCache() : impl(std::make_unique<Impl>()) {
  // Use system temp directory
  llvm::SmallString<128> path;
  llvm::sys::path::system_temp_directory(true, path);
  llvm::sys::path::append(path, "yirage_kernel_cache");
  impl->cacheDir = std::string(path);
  
  // Create directory if needed
  llvm::sys::fs::create_directories(impl->cacheDir);
}

KernelCache::KernelCache(StringRef cacheDir) : impl(std::make_unique<Impl>()) {
  impl->cacheDir = cacheDir.str();
  llvm::sys::fs::create_directories(impl->cacheDir);
}

KernelCache::~KernelCache() = default;

void KernelCache::store(const CompiledKernel &kernel) {
  std::lock_guard<std::mutex> lock(impl->mutex);
  
  std::string key = impl->makeKey(kernel.info.name, kernel.targetArch);
  impl->memoryCache[key] = kernel;
  impl->stats.totalKernels = impl->memoryCache.size();
  
  // Persist to disk
  std::string path = impl->makePath(kernel.info.name, kernel.targetArch);
  std::ofstream file(path, std::ios::binary);
  if (file) {
    // Write binary size and data
    size_t size = kernel.binary.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(kernel.binary.data()), size);
    
    // Write text code size and data
    size = kernel.textCode.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(kernel.textCode.data(), size);
  }
}

std::optional<CompiledKernel> KernelCache::lookup(StringRef name,
                                                    StringRef target) {
  std::lock_guard<std::mutex> lock(impl->mutex);
  
  std::string key = impl->makeKey(name, target);
  
  // Check memory cache first
  auto it = impl->memoryCache.find(key);
  if (it != impl->memoryCache.end()) {
    impl->stats.memoryHits++;
    return it->second;
  }
  
  // Check disk cache
  std::string path = impl->makePath(name, target);
  if (llvm::sys::fs::exists(path)) {
    std::ifstream file(path, std::ios::binary);
    if (file) {
      CompiledKernel kernel;
      kernel.info.name = name.str();
      kernel.targetArch = target.str();
      
      // Read binary
      size_t size;
      file.read(reinterpret_cast<char*>(&size), sizeof(size));
      kernel.binary.resize(size);
      file.read(reinterpret_cast<char*>(kernel.binary.data()), size);
      
      // Read text code
      file.read(reinterpret_cast<char*>(&size), sizeof(size));
      kernel.textCode.resize(size);
      file.read(&kernel.textCode[0], size);
      
      impl->memoryCache[key] = kernel;
      impl->stats.diskHits++;
      return kernel;
    }
  }
  
  impl->stats.misses++;
  return std::nullopt;
}

bool KernelCache::contains(StringRef name, StringRef target) {
  std::lock_guard<std::mutex> lock(impl->mutex);
  
  std::string key = impl->makeKey(name, target);
  if (impl->memoryCache.count(key)) return true;
  
  std::string path = impl->makePath(name, target);
  return llvm::sys::fs::exists(path);
}

void KernelCache::clearMemory() {
  std::lock_guard<std::mutex> lock(impl->mutex);
  impl->memoryCache.clear();
  impl->stats.totalKernels = 0;
  impl->stats.memorySizeBytes = 0;
}

void KernelCache::clearAll() {
  clearMemory();
  std::error_code EC;
  llvm::sys::fs::remove_directories(impl->cacheDir);
  llvm::sys::fs::create_directories(impl->cacheDir);
}

KernelCache::CacheStats KernelCache::getStats() const {
  std::lock_guard<std::mutex> lock(impl->mutex);
  
  CacheStats stats = impl->stats;
  
  // Calculate memory size
  stats.memorySizeBytes = 0;
  for (const auto &pair : impl->memoryCache) {
    stats.memorySizeBytes += pair.second.binary.size();
    stats.memorySizeBytes += pair.second.textCode.size();
  }
  
  return stats;
}

//===----------------------------------------------------------------------===//
// Factory Functions
//===----------------------------------------------------------------------===//

std::unique_ptr<KernelExtractor> createKernelExtractor(MLIRContext *context) {
  return std::make_unique<KernelExtractor>(context);
}

std::unique_ptr<KernelCache> createKernelCache() {
  return std::make_unique<KernelCache>();
}

std::unique_ptr<KernelCache> createKernelCache(StringRef cacheDir) {
  return std::make_unique<KernelCache>(cacheDir);
}

} // namespace yirage
