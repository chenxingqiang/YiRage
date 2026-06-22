//===- MemoryPlanning.h - GPU Memory Planning Pass --------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header defines memory planning passes for efficient GPU memory usage.
//
// Key Features:
//   1. Buffer aliasing - reuse memory for non-overlapping lifetimes
//   2. Memory pooling - reduce allocation overhead
//   3. Workspace planning - compute optimal workspace sizes
//   4. Memory pressure estimation - warn about OOM risks
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_MLIR_TRANSFORMS_MEMORYPLANNING_H
#define YIRAGE_MLIR_TRANSFORMS_MEMORYPLANNING_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace yirage {

//===----------------------------------------------------------------------===//
// Memory Planning Configuration
//===----------------------------------------------------------------------===//

struct MemoryPlanningConfig {
  // Memory pool settings
  bool enablePooling = true;
  size_t minPoolSize = 64 * 1024 * 1024;  // 64 MB minimum
  size_t maxPoolSize = 0;  // 0 = unlimited
  
  // Aliasing settings
  bool enableAliasing = true;
  bool conservativeAliasing = false;  // Only alias when provably safe
  
  // Target-specific
  size_t deviceMemoryLimit = 0;  // 0 = no limit
  size_t sharedMemoryLimit = 0;  // 0 = use default
  double memoryUtilizationTarget = 0.8;  // Use up to 80% of memory
  
  // Analysis
  bool enableMemoryStats = true;
  bool warnOnHighPressure = true;
};

//===----------------------------------------------------------------------===//
// Memory Analysis Results
//===----------------------------------------------------------------------===//

struct MemoryAnalysisResult {
  // Buffer counts
  size_t totalBuffers = 0;
  size_t aliasedBuffers = 0;
  size_t pooledBuffers = 0;
  
  // Size analysis
  size_t peakMemoryUsage = 0;
  size_t totalAllocatedSize = 0;
  size_t aliasedMemorySaved = 0;
  
  // Workspace
  size_t workspaceRequired = 0;
  size_t scratchSpaceRequired = 0;
  
  // Efficiency
  double memoryEfficiency = 0.0;  // peakUsage / totalAllocated
  double aliasingRatio = 0.0;     // aliased / total
  
  // Warnings
  std::vector<std::string> warnings;
};

//===----------------------------------------------------------------------===//
// Pass Declarations
//===----------------------------------------------------------------------===//

/// Create a memory aliasing pass.
/// This pass identifies buffers with non-overlapping lifetimes and assigns
/// them the same underlying memory.
std::unique_ptr<mlir::Pass> createMemoryAliasingPass();

/// Create a memory aliasing pass with custom config.
std::unique_ptr<mlir::Pass> createMemoryAliasingPass(
    const MemoryPlanningConfig &config);

/// Create a memory pooling pass.
/// This pass groups allocations into pools for reduced overhead.
std::unique_ptr<mlir::Pass> createMemoryPoolingPass();

/// Create a workspace planning pass.
/// This pass computes workspace requirements and injects allocation code.
std::unique_ptr<mlir::Pass> createWorkspacePlanningPass();

/// Create a memory analysis pass.
/// This pass computes memory statistics without modifying the IR.
std::unique_ptr<mlir::Pass> createMemoryAnalysisPass();

/// Create a comprehensive memory planning pipeline.
std::unique_ptr<mlir::Pass> createMemoryPlanningPipelinePass(
    const MemoryPlanningConfig &config = {});

} // namespace yirage

#endif // YIRAGE_MLIR_TRANSFORMS_MEMORYPLANNING_H
