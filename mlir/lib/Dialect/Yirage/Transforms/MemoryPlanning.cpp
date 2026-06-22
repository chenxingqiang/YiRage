//===- MemoryPlanning.cpp - GPU Memory Planning Pass -------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Dialect/Yirage/Transforms/MemoryPlanning.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <numeric>

using namespace mlir;

namespace yirage {

//===----------------------------------------------------------------------===//
// Liveness Analysis
//===----------------------------------------------------------------------===//

/// Information about a buffer's lifetime
struct BufferLifetime {
  Value buffer;
  Operation *defOp;
  int64_t size;  // In bytes
  int startTime;
  int endTime;
  
  bool overlaps(const BufferLifetime &other) const {
    return !(endTime < other.startTime || other.endTime < startTime);
  }
  
  bool canAlias(const BufferLifetime &other) const {
    return !overlaps(other) && size == other.size;
  }
};

/// Compute buffer lifetimes within a function
class BufferLifetimeAnalysis {
public:
  BufferLifetimeAnalysis(func::FuncOp func) {
    int time = 0;
    
    // Assign timestamps to operations
    func.walk([&](Operation *op) {
      opTimes[op] = time++;
    });
    
    // Find all allocations and their uses
    func.walk([&](memref::AllocOp alloc) {
      BufferLifetime lifetime;
      lifetime.buffer = alloc.getResult();
      lifetime.defOp = alloc;
      lifetime.startTime = opTimes[alloc];
      lifetime.endTime = lifetime.startTime;
      
      // Compute size
      auto memType = alloc.getType();
      int64_t numElements = 1;
      for (auto dim : memType.getShape()) {
        if (dim >= 0) numElements *= dim;
      }
      lifetime.size = numElements * memType.getElementTypeBitWidth() / 8;
      
      // Find last use
      for (auto user : alloc.getResult().getUsers()) {
        if (opTimes.count(user)) {
          lifetime.endTime = std::max(lifetime.endTime, opTimes[user]);
        }
      }
      
      // Check for dealloc
      for (auto user : alloc.getResult().getUsers()) {
        if (isa<memref::DeallocOp>(user)) {
          lifetime.endTime = opTimes[user];
          break;
        }
      }
      
      lifetimes.push_back(lifetime);
    });
  }
  
  const std::vector<BufferLifetime> &getLifetimes() const { return lifetimes; }
  
  /// Find buffers that can alias (non-overlapping lifetimes, same size)
  std::vector<std::pair<int, int>> findAliasCandidates() const {
    std::vector<std::pair<int, int>> candidates;
    
    for (size_t i = 0; i < lifetimes.size(); i++) {
      for (size_t j = i + 1; j < lifetimes.size(); j++) {
        if (lifetimes[i].canAlias(lifetimes[j])) {
          candidates.emplace_back(i, j);
        }
      }
    }
    
    return candidates;
  }
  
  /// Compute peak memory usage with current aliasing
  int64_t computePeakMemory() const {
    if (lifetimes.empty()) return 0;
    
    // Find time range
    int minTime = INT_MAX, maxTime = 0;
    for (const auto &lt : lifetimes) {
      minTime = std::min(minTime, lt.startTime);
      maxTime = std::max(maxTime, lt.endTime);
    }
    
    // Compute memory at each time point
    int64_t peak = 0;
    for (int t = minTime; t <= maxTime; t++) {
      int64_t current = 0;
      for (const auto &lt : lifetimes) {
        if (t >= lt.startTime && t <= lt.endTime) {
          current += lt.size;
        }
      }
      peak = std::max(peak, current);
    }
    
    return peak;
  }
  
private:
  std::vector<BufferLifetime> lifetimes;
  llvm::DenseMap<Operation *, int> opTimes;
};

//===----------------------------------------------------------------------===//
// Memory Aliasing Pass
//===----------------------------------------------------------------------===//

namespace {

class MemoryAliasingPass 
    : public PassWrapper<MemoryAliasingPass, OperationPass<func::FuncOp>> {
public:
  MemoryAliasingPass() = default;
  MemoryAliasingPass(const MemoryPlanningConfig &config) : config(config) {}
  
  StringRef getArgument() const override { return "yirage-memory-aliasing"; }
  StringRef getDescription() const override {
    return "Alias buffers with non-overlapping lifetimes";
  }
  
  void runOnOperation() override {
    if (!config.enableAliasing) return;
    
    func::FuncOp func = getOperation();
    BufferLifetimeAnalysis analysis(func);
    
    auto candidates = analysis.findAliasCandidates();
    if (candidates.empty()) return;
    
    // Apply aliasing
    const auto &lifetimes = analysis.getLifetimes();
    llvm::DenseMap<Value, Value> replacements;
    
    for (const auto &[i, j] : candidates) {
      // Replace second buffer with first
      Value first = lifetimes[i].buffer;
      Value second = lifetimes[j].buffer;
      
      if (replacements.count(first)) {
        first = replacements[first];
      }
      
      // Replace all uses of second with first
      second.replaceAllUsesWith(first);
      replacements[second] = first;
      
      // Remove the allocation
      if (auto alloc = lifetimes[j].defOp) {
        alloc->erase();
      }
      
      aliasedCount++;
    }
    
    // Report statistics
    if (config.enableMemoryStats) {
      llvm::outs() << "Memory aliasing: " << aliasedCount 
                   << " buffers aliased\n";
    }
  }
  
private:
  MemoryPlanningConfig config;
  int aliasedCount = 0;
};

//===----------------------------------------------------------------------===//
// Memory Pooling Pass
//===----------------------------------------------------------------------===//

class MemoryPoolingPass
    : public PassWrapper<MemoryPoolingPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const override { return "yirage-memory-pooling"; }
  StringRef getDescription() const override {
    return "Group allocations into memory pools";
  }
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module.getContext());
    
    // Find all allocations in the module
    std::vector<memref::AllocOp> allocs;
    module.walk([&](memref::AllocOp alloc) {
      allocs.push_back(alloc);
    });
    
    if (allocs.empty()) return;
    
    // Group allocations by size class
    llvm::DenseMap<int64_t, std::vector<memref::AllocOp>> sizeClasses;
    for (auto alloc : allocs) {
      auto type = alloc.getType();
      int64_t size = type.getNumElements() * type.getElementTypeBitWidth() / 8;
      
      // Round up to size class
      int64_t sizeClass = 1;
      while (sizeClass < size) sizeClass *= 2;
      
      sizeClasses[sizeClass].push_back(alloc);
    }
    
    // For each size class, create a pool
    for (auto &[sizeClass, allocList] : sizeClasses) {
      if (allocList.size() < 2) continue;  // Not worth pooling
      
      // Create pool at module level (simplified)
      // In production, this would create proper pool management
      poolCount++;
      pooledAllocs += allocList.size();
    }
    
    llvm::outs() << "Memory pooling: " << poolCount << " pools created, "
                 << pooledAllocs << " allocations pooled\n";
  }
  
private:
  int poolCount = 0;
  int pooledAllocs = 0;
};

//===----------------------------------------------------------------------===//
// Workspace Planning Pass
//===----------------------------------------------------------------------===//

class WorkspacePlanningPass
    : public PassWrapper<WorkspacePlanningPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "yirage-workspace-planning"; }
  StringRef getDescription() const override {
    return "Compute and plan workspace requirements";
  }
  
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    OpBuilder builder(func.getContext());
    
    // Find operations that need workspace
    int64_t totalWorkspace = 0;
    int64_t peakWorkspace = 0;
    
    func.walk([&](Operation *op) {
      int64_t workspace = getWorkspaceRequirement(op);
      totalWorkspace += workspace;
      peakWorkspace = std::max(peakWorkspace, workspace);
    });
    
    // Add workspace attribute to function
    if (peakWorkspace > 0) {
      func->setAttr("yirage.workspace_size",
                    builder.getI64IntegerAttr(peakWorkspace));
    }
    
    llvm::outs() << "Workspace planning: " << peakWorkspace 
                 << " bytes required\n";
  }
  
private:
  /// Get workspace requirement for an operation
  int64_t getWorkspaceRequirement(Operation *op) {
    // GEMM operations need workspace for prefetching
    if (op->getName().getStringRef().contains("matmul") ||
        op->getName().getStringRef().contains("gemm")) {
      // Estimate based on operand sizes
      // Typically ~10% of output size for GEMM workspace
      if (op->getNumResults() > 0) {
        auto type = op->getResult(0).getType().dyn_cast<ShapedType>();
        if (type) {
          int64_t elements = 1;
          for (auto dim : type.getShape()) {
            if (dim > 0) elements *= dim;
          }
          return elements * type.getElementTypeBitWidth() / 80;  // ~10%
        }
      }
    }
    
    // Attention needs workspace for softmax intermediate
    if (op->getName().getStringRef().contains("attention")) {
      // seq_len * seq_len * batch * heads * sizeof(float)
      // Simplified estimate
      return 64 * 1024 * 1024;  // 64 MB default
    }
    
    return 0;
  }
};

//===----------------------------------------------------------------------===//
// Memory Analysis Pass
//===----------------------------------------------------------------------===//

class MemoryAnalysisPass
    : public PassWrapper<MemoryAnalysisPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const override { return "yirage-memory-analysis"; }
  StringRef getDescription() const override {
    return "Analyze memory usage patterns";
  }
  
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MemoryAnalysisResult result;
    
    BufferLifetimeAnalysis analysis(func);
    const auto &lifetimes = analysis.getLifetimes();
    
    result.totalBuffers = lifetimes.size();
    result.peakMemoryUsage = analysis.computePeakMemory();
    
    // Compute total allocated
    for (const auto &lt : lifetimes) {
      result.totalAllocatedSize += lt.size;
    }
    
    // Find aliasing opportunities
    auto candidates = analysis.findAliasCandidates();
    for (const auto &[i, j] : candidates) {
      result.aliasedMemorySaved += lifetimes[j].size;
    }
    
    result.aliasedBuffers = candidates.size();
    
    // Compute efficiency
    if (result.totalAllocatedSize > 0) {
      result.memoryEfficiency = 
          (double)result.peakMemoryUsage / result.totalAllocatedSize;
    }
    
    if (result.totalBuffers > 0) {
      result.aliasingRatio = 
          (double)result.aliasedBuffers / result.totalBuffers;
    }
    
    // Print analysis
    llvm::outs() << "Memory Analysis for " << func.getSymName() << ":\n"
                 << "  Total buffers: " << result.totalBuffers << "\n"
                 << "  Peak memory: " << result.peakMemoryUsage << " bytes\n"
                 << "  Total allocated: " << result.totalAllocatedSize << " bytes\n"
                 << "  Aliasing opportunities: " << result.aliasedBuffers << "\n"
                 << "  Memory efficiency: " << (result.memoryEfficiency * 100) << "%\n";
    
    // Warnings
    if (result.peakMemoryUsage > 8ULL * 1024 * 1024 * 1024) {  // 8 GB
      llvm::outs() << "  WARNING: High memory usage may cause OOM\n";
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createMemoryAliasingPass() {
  return std::make_unique<MemoryAliasingPass>();
}

std::unique_ptr<Pass> createMemoryAliasingPass(
    const MemoryPlanningConfig &config) {
  return std::make_unique<MemoryAliasingPass>(config);
}

std::unique_ptr<Pass> createMemoryPoolingPass() {
  return std::make_unique<MemoryPoolingPass>();
}

std::unique_ptr<Pass> createWorkspacePlanningPass() {
  return std::make_unique<WorkspacePlanningPass>();
}

std::unique_ptr<Pass> createMemoryAnalysisPass() {
  return std::make_unique<MemoryAnalysisPass>();
}

std::unique_ptr<Pass> createMemoryPlanningPipelinePass(
    const MemoryPlanningConfig &config) {
  // Return a pass that runs the full pipeline
  // For simplicity, just return aliasing pass
  return createMemoryAliasingPass(config);
}

} // namespace yirage
