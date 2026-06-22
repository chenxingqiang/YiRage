//===- CPUJITKernel.cpp - CPU JIT kernel session -----------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Execution/CPUJITKernel.h"
#include "yirage-mlir/Execution/JITRunner.h"
#include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.h"
#include "yirage-mlir/Dialect/Yirage/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

#include <cstring>

using namespace mlir;

namespace yirage {

namespace {

void loadDialects(MLIRContext &ctx, bool loadYirage = true) {
  DialectRegistry registry;
  if (loadYirage)
    registry.insert<yirage::ir::YirageDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<memref::MemRefDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<math::MathDialect>();
  registry.insert<bufferization::BufferizationDialect>();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  memref::registerBufferizableOpInterfaceExternalModels(registry);
  scf::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();
}

bool needsYirageCpuJitPipeline(ModuleOp module) {
  bool need = false;
  module.walk([&](func::FuncOp func) {
    FunctionType fType = func.getFunctionType();
    for (Type t : fType.getInputs()) {
      if (llvm::isa<RankedTensorType>(t))
        need = true;
    }
    for (Type t : fType.getResults()) {
      if (llvm::isa<RankedTensorType>(t))
        need = true;
    }
  });
  if (need)
    return true;
  module.walk([&](Operation *op) {
    if (op->getDialect()->getNamespace() == "yirage")
      need = true;
  });
  return need;
}

} // namespace

CPUJITKernel::CPUJITKernel() = default;
CPUJITKernel::~CPUJITKernel() = default;

bool CPUJITKernel::isReady() const {
  return runner_ && runner_->isReady();
}

LogicalResult CPUJITKernel::runCpuJitPipeline(ModuleOp module) {
  const bool useDialectPipeline = needsYirageCpuJitPipeline(module);
  PassManager pm(context_.get());
  if (useDialectPipeline) {
    pm.addPass(createYirageToLinalgPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
    {
      bufferization::OneShotBufferizationOptions opts;
      opts.bufferizeFunctionBoundaries = true;
      opts.allowReturnAllocs = true;
      opts.setFunctionBoundaryTypeConversion(
          bufferization::LayoutMapOption::IdentityLayoutMap);
      pm.addPass(bufferization::createOneShotBufferizePass(opts));
    }
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    pm.addNestedPass<func::FuncOp>(createCpuJitOutParamPass());
    pm.addPass(memref::createExpandStridedMetadataPass());
    pm.addPass(createCanonicalizerPass());
  } else {
    // Hand-written memref + linalg from cpu_mlir_jit.py fast path.
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
  if (failed(pm.run(module))) {
    lastError_ = useDialectPipeline ? "yirage-cpu-jit-pipeline run failed"
                                    : "memref linalg-to-loops pipeline failed";
    return failure();
  }
  return success();
}

LogicalResult CPUJITKernel::compileFromText(const std::string &mlirText,
                                            const std::string &entry) {
  lastError_.clear();
  entry_ = entry;

  const bool hasYirageOps =
      mlirText.find("yirage.") != std::string::npos ||
      mlirText.find("yirage::") != std::string::npos;

  context_ = std::make_unique<MLIRContext>();
  loadDialects(*context_, hasYirageOps);
  if (hasYirageOps) {
    registerYiragePasses();
    registerYiragePassPipelines();
  }

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(mlirText, context_.get());
  if (!module) {
    lastError_ = "failed to parse MLIR module";
    return failure();
  }

  if (failed(runCpuJitPipeline(*module))) {
    if (lastError_.empty())
      lastError_ = "cpu jit pipeline failed";
    return failure();
  }

  runner_ = std::make_unique<JITRunner>(context_.get());
  if (failed(runner_->compile(*module))) {
    lastError_ = "JITRunner::compile failed";
    runner_.reset();
    return failure();
  }

  return success();
}

LogicalResult CPUJITKernel::invokeRmsMatmulF16(void *x, void *w, void *out,
                                             int64_t m, int64_t k, int64_t n) {
  if (!isReady()) {
    lastError_ = "JIT kernel not compiled";
    return failure();
  }

  // invokePacked expects pointers to each argument (bare memref ptr ABI).
  void *args[] = {&x, &w, &out};
  if (failed(runner_->invoke(entry_, llvm::MutableArrayRef<void *>(args)))) {
    lastError_ = "JIT invoke failed for entry " + entry_;
    return failure();
  }
  return success();
}

} // namespace yirage
