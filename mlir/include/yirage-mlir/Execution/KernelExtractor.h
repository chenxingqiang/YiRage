//===- KernelExtractor.h - GPU Kernel Extraction Utility --------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header defines utilities for extracting and compiling individual GPU
// kernels from MLIR modules. This is essential for:
//
//   1. AOT compilation of specific kernels
//   2. Kernel caching and reuse
//   3. Profile-guided optimization
//   4. Debugging individual kernel performance
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_MLIR_EXECUTION_KERNELEXTRACTOR_H
#define YIRAGE_MLIR_EXECUTION_KERNELEXTRACTOR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace yirage {

//===----------------------------------------------------------------------===//
// Kernel Metadata
//===----------------------------------------------------------------------===//

/// Information about an extracted kernel
struct KernelInfo {
  std::string name;
  std::string mangledName;
  
  // Signature
  std::vector<std::string> argTypes;
  std::string returnType;
  
  // Execution configuration
  std::array<int64_t, 3> gridSize = {1, 1, 1};
  std::array<int64_t, 3> blockSize = {1, 1, 1};
  int64_t sharedMemory = 0;
  
  // Resource usage (after compilation)
  int64_t registersUsed = 0;
  int64_t staticSharedMemory = 0;
  int64_t dynamicSharedMemory = 0;
  int64_t spillLoads = 0;
  int64_t spillStores = 0;
  
  // Occupancy estimation
  double theoreticalOccupancy = 0.0;
  int maxActiveBlocksPerSM = 0;
  
  // Source info
  std::string sourceLocation;
  std::string originalMLIR;
};

/// Compiled kernel binary
struct CompiledKernel {
  KernelInfo info;
  
  // Binary data
  std::vector<uint8_t> binary;
  std::string textCode;  // PTX, GCN assembly, etc.
  
  // Target info
  std::string targetArch;
  std::string targetTriple;
  
  // Compilation stats
  double compilationTime = 0.0;
  std::string compiler;
  std::string compilerVersion;
};

//===----------------------------------------------------------------------===//
// Kernel Extractor
//===----------------------------------------------------------------------===//

class KernelExtractorImpl;

/// Utility for extracting and compiling individual kernels.
///
/// Example usage:
/// \code
///   MLIRContext context;
///   KernelExtractor extractor(&context);
///   
///   auto kernels = extractor.extractKernels(module);
///   for (auto& kernel : kernels) {
///     auto compiled = extractor.compile(kernel, Target::CUDA_H100);
///     cache.store(compiled);
///   }
/// \endcode
class KernelExtractor {
public:
  explicit KernelExtractor(mlir::MLIRContext *context);
  ~KernelExtractor();
  
  // Disable copy
  KernelExtractor(const KernelExtractor &) = delete;
  KernelExtractor &operator=(const KernelExtractor &) = delete;
  
  //==========================================================================
  // Kernel Discovery
  //==========================================================================
  
  /// Extract all kernels from a module.
  /// Returns a list of kernel info structures.
  std::vector<KernelInfo> extractKernels(mlir::ModuleOp module);
  
  /// Get a specific kernel by name.
  std::optional<KernelInfo> getKernel(mlir::ModuleOp module, 
                                       llvm::StringRef name);
  
  /// Check if a function is a GPU kernel.
  bool isKernel(mlir::func::FuncOp func);
  
  //==========================================================================
  // Kernel Isolation
  //==========================================================================
  
  /// Create a new module containing only the specified kernel.
  /// This is useful for isolating kernels for compilation or analysis.
  mlir::OwningOpRef<mlir::ModuleOp> isolateKernel(mlir::ModuleOp module,
                                                    llvm::StringRef kernelName);
  
  /// Create a module containing multiple kernels.
  mlir::OwningOpRef<mlir::ModuleOp> isolateKernels(
      mlir::ModuleOp module,
      llvm::ArrayRef<llvm::StringRef> kernelNames);
  
  //==========================================================================
  // Kernel Compilation
  //==========================================================================
  
  /// Compile a single kernel for a target.
  CompiledKernel compile(const KernelInfo &kernel, 
                          mlir::ModuleOp module,
                          llvm::StringRef target);
  
  /// Compile all kernels in a module.
  std::vector<CompiledKernel> compileAll(mlir::ModuleOp module,
                                          llvm::StringRef target);
  
  //==========================================================================
  // Analysis
  //==========================================================================
  
  /// Analyze kernel resource usage.
  void analyzeResources(KernelInfo &kernel, mlir::func::FuncOp func);
  
  /// Estimate occupancy for a target.
  double estimateOccupancy(const KernelInfo &kernel, 
                            llvm::StringRef target);
  
  /// Get recommended block size for a kernel.
  std::array<int64_t, 3> recommendBlockSize(const KernelInfo &kernel,
                                             llvm::StringRef target);

private:
  std::unique_ptr<KernelExtractorImpl> impl;
  mlir::MLIRContext *context;
};

//===----------------------------------------------------------------------===//
// Kernel Cache
//===----------------------------------------------------------------------===//

/// Cache for compiled kernels.
///
/// Supports:
///   - In-memory caching
///   - Disk persistence
///   - LRU eviction
///   - Version-aware invalidation
class KernelCache {
public:
  KernelCache();
  explicit KernelCache(llvm::StringRef cacheDir);
  ~KernelCache();
  
  /// Store a compiled kernel.
  void store(const CompiledKernel &kernel);
  
  /// Look up a kernel by name and target.
  std::optional<CompiledKernel> lookup(llvm::StringRef name,
                                        llvm::StringRef target);
  
  /// Check if a kernel is cached.
  bool contains(llvm::StringRef name, llvm::StringRef target);
  
  /// Clear in-memory cache.
  void clearMemory();
  
  /// Clear all caches including disk.
  void clearAll();
  
  /// Get cache statistics.
  struct CacheStats {
    size_t memoryHits = 0;
    size_t diskHits = 0;
    size_t misses = 0;
    size_t totalKernels = 0;
    size_t memorySizeBytes = 0;
    size_t diskSizeBytes = 0;
  };
  CacheStats getStats() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

//===----------------------------------------------------------------------===//
// Factory Functions
//===----------------------------------------------------------------------===//

/// Create a kernel extractor.
std::unique_ptr<KernelExtractor> createKernelExtractor(
    mlir::MLIRContext *context);

/// Create a kernel cache with default settings.
std::unique_ptr<KernelCache> createKernelCache();

/// Create a kernel cache with specific directory.
std::unique_ptr<KernelCache> createKernelCache(llvm::StringRef cacheDir);

} // namespace yirage

#endif // YIRAGE_MLIR_EXECUTION_KERNELEXTRACTOR_H
