//===- LinalgToCPU.cpp - Lower Linalg to CPU/LLVM dialects --------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering from Linalg dialect to CPU-optimized code,
// including vectorization, loop optimizations, and LLVM lowering.
//
// Pipeline:
//   Linalg (tensor) → Vectorization → Bufferization → Loops → LLVM
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Dialect/Yirage/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// CPU Optimization Configuration
//===----------------------------------------------------------------------===//

struct CPUOptConfig {
  // SIMD vector width based on target
  int64_t vectorWidth = 8;  // Default for AVX2 (256-bit / 32-bit float)
  
  // Cache blocking sizes
  int64_t l1CacheBlockSize = 32 * 1024;   // 32KB L1
  int64_t l2CacheBlockSize = 256 * 1024;  // 256KB L2
  int64_t l3CacheBlockSize = 8 * 1024 * 1024;  // 8MB L3
  
  // Loop unroll factor
  int64_t unrollFactor = 4;
  
  // Enable prefetching
  bool enablePrefetch = true;
  
  static CPUOptConfig forAVX2() {
    CPUOptConfig config;
    config.vectorWidth = 8;  // 256-bit / 32-bit
    return config;
  }
  
  static CPUOptConfig forAVX512() {
    CPUOptConfig config;
    config.vectorWidth = 16;  // 512-bit / 32-bit
    return config;
  }
  
  static CPUOptConfig forNEON() {
    CPUOptConfig config;
    config.vectorWidth = 4;  // 128-bit / 32-bit
    return config;
  }
  
  static CPUOptConfig forAppleSilicon() {
    CPUOptConfig config;
    config.vectorWidth = 4;  // NEON 128-bit
    config.l1CacheBlockSize = 64 * 1024;  // Larger L1 on M-series
    return config;
  }
};

//===----------------------------------------------------------------------===//
// Vectorization Pass
//===----------------------------------------------------------------------===//

struct LinalgVectorizationPass
    : public PassWrapper<LinalgVectorizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgVectorizationPass)

  StringRef getArgument() const final { return "yirage-linalg-vectorize"; }
  StringRef getDescription() const final {
    return "Vectorize Linalg operations for SIMD execution";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();
    
    CPUOptConfig config = CPUOptConfig::forAVX2();
    
    // Apply vectorization to linalg operations
    // This would use linalg::vectorize() or transform dialect
    
    func.walk([&](linalg::LinalgOp op) {
      // Vectorize the operation based on config.vectorWidth
      // Full implementation would:
      // 1. Analyze loop structure
      // 2. Determine vectorizable dimensions
      // 3. Apply vector.transfer_read/write
      // 4. Create vector.contract for matmul
    });
  }
};

//===----------------------------------------------------------------------===//
// Loop Optimization Pass
//===----------------------------------------------------------------------===//

struct LoopOptimizationPass
    : public PassWrapper<LoopOptimizationPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopOptimizationPass)

  StringRef getArgument() const final { return "yirage-loop-opt"; }
  StringRef getDescription() const final {
    return "Optimize loops for CPU cache hierarchy";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
    registry.insert<affine::AffineDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    
    CPUOptConfig config = CPUOptConfig::forAVX2();
    
    // Apply loop optimizations:
    // 1. Loop tiling for cache blocking
    // 2. Loop interchange for memory access patterns
    // 3. Loop unrolling
    // 4. Loop fusion where beneficial
    
    // These would use affine analysis and transformation utilities
  }
};

//===----------------------------------------------------------------------===//
// Full CPU Pipeline Pass
//===----------------------------------------------------------------------===//

struct LinalgToCPUPipelinePass
    : public PassWrapper<LinalgToCPUPipelinePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToCPUPipelinePass)

  LinalgToCPUPipelinePass() = default;
  LinalgToCPUPipelinePass(StringRef arch) : targetArch(arch.str()) {}

  StringRef getArgument() const final { return "yirage-linalg-to-cpu"; }
  StringRef getDescription() const final {
    return "Full pipeline: Linalg → CPU (vectorization, loop opts, LLVM)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<affine::AffineDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Full CPU lowering pipeline:
    // 1. Vectorization (linalg → vector)
    // 2. Loop tiling for cache
    // 3. Bufferization (tensor → memref)
    // 4. Lower to SCF loops
    // 5. Lower to LLVM dialect
    
    // Implementation would compose the sub-passes
  }

private:
  std::string targetArch = "x86-64";
};

//===----------------------------------------------------------------------===//
// CPU JIT ABI: (x, w) -> out tensor  =>  (x, w, out) void
//===----------------------------------------------------------------------===//

struct CpuJitOutParamPass
    : public PassWrapper<CpuJitOutParamPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CpuJitOutParamPass)

  StringRef getArgument() const final { return "yirage-cpu-jit-out-param"; }
  StringRef getDescription() const final {
    return "Add memref out-parameter for CPU JIT bare-ptr ABI";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.getNumResults() != 1)
      return;

    Type resultType = func.getResultTypes()[0];
    auto resultMemRef = llvm::dyn_cast<MemRefType>(resultType);
    if (!resultMemRef || !resultMemRef.hasStaticShape())
      return;

    func::ReturnOp returnOp;
    func.walk([&](func::ReturnOp op) { returnOp = op; });
    if (!returnOp || returnOp.getNumOperands() != 1)
      return signalPassFailure();

    Value retVal = returnOp.getOperand(0);
    Value bufferVal = retVal;
    memref::CastOp retCast;
    if (auto castOp = retVal.getDefiningOp<memref::CastOp>()) {
      retCast = castOp;
      bufferVal = castOp.getSource();
    }

    auto bufferType = llvm::dyn_cast<MemRefType>(bufferVal.getType());
    if (!bufferType || !bufferType.hasStaticShape())
      return signalPassFailure();

    Location loc = func.getLoc();
    MemRefType outType =
        MemRefType::get(bufferType.getShape(), bufferType.getElementType());
    func.insertArgument(func.getNumArguments(), outType, {}, loc);
    Value outArg = func.getArgument(func.getNumArguments() - 1);

    // Drop trailing copies; matmul / loops will write directly into outArg.
    SmallVector<Operation *> toErase;
    func.walk([&](memref::CopyOp copy) {
      if (copy.getSource() == retVal || copy.getSource() == bufferVal ||
          copy.getTarget() == outArg)
        toErase.push_back(copy.getOperation());
    });
    for (Operation *op : toErase)
      op->erase();

    bufferVal.replaceUsesWithIf(outArg, [](OpOperand &use) {
      return !isa<memref::DeallocOp>(use.getOwner());
    });

    if (auto *allocOp = bufferVal.getDefiningOp())
      if (allocOp->use_empty())
        allocOp->erase();
    if (retCast)
      retCast.erase();

    returnOp->setOperands({});
    SmallVector<Type> inputTypes(func.getArgumentTypes());
    func.setType(FunctionType::get(&getContext(), inputTypes, {}));
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

namespace yirage {

std::unique_ptr<mlir::Pass> createLinalgVectorizationPass() {
  return std::make_unique<LinalgVectorizationPass>();
}

std::unique_ptr<mlir::Pass> createCpuJitOutParamPass() {
  return std::make_unique<CpuJitOutParamPass>();
}

std::unique_ptr<mlir::Pass> createLoopOptimizationPass() {
  return std::make_unique<LoopOptimizationPass>();
}

std::unique_ptr<mlir::Pass> createLinalgToCPUPipelinePass(llvm::StringRef arch) {
  return std::make_unique<LinalgToCPUPipelinePass>(arch);
}

} // namespace yirage
