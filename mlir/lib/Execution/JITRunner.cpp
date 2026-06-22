//===- JITRunner.cpp - JIT Compilation and Execution Engine ------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT compilation engine for YiRage MLIR.
// It provides the final step in the compilation pipeline:
//
//   LLVM IR → JIT Compilation → Native Code Execution
//
// Two modes are supported:
//   1. JIT (Just-In-Time): Compile and execute immediately
//   2. AOT (Ahead-Of-Time): Generate object files for later linking
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Execution/JITRunner.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace yirage {

//===----------------------------------------------------------------------===//
// JIT Runner Implementation
//===----------------------------------------------------------------------===//

class JITRunnerImpl {
public:
  JITRunnerImpl(MLIRContext *context) : context(context) {
    // Initialize LLVM targets
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    
    // Register LLVM IR translations
    registerBuiltinDialectTranslation(*context);
    registerLLVMDialectTranslation(*context);
  }
  
  ~JITRunnerImpl() = default;
  
  /// Lower module to LLVM dialect
  LogicalResult lowerToLLVM(ModuleOp module) {
    PassManager pm(context);
    
    // Standard lowering to LLVM
    pm.addPass(createConvertSCFToCFPass());
    pm.addPass(createArithToLLVMConversionPass());
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createConvertControlFlowToLLVMPass());
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());
    ConvertFuncToLLVMPassOptions funcOpts;
    funcOpts.useBarePtrCallConv = true;
    pm.addPass(createConvertFuncToLLVMPass(funcOpts));
    pm.addPass(createReconcileUnrealizedCastsPass());
    
    // Cleanup
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    
    return pm.run(module);
  }
  
  /// Create execution engine from LLVM module
  llvm::Expected<std::unique_ptr<ExecutionEngine>>
  createEngine(ModuleOp module, unsigned optLevel = 3) {
    // Configure the execution engine
    ExecutionEngineOptions options;
    options.transformer = makeOptimizingTransformer(optLevel, 0, nullptr);
    options.enableObjectDump = true;
    
    return ExecutionEngine::create(module, options);
  }
  
  /// Export LLVM IR as text
  std::string exportLLVMIR(ModuleOp module) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
      return "// Error: Failed to translate to LLVM IR\n";
    }
    
    std::string output;
    llvm::raw_string_ostream os(output);
    llvmModule->print(os, nullptr);
    return output;
  }

private:
  MLIRContext *context;
};

//===----------------------------------------------------------------------===//
// JIT Runner Public Interface
//===----------------------------------------------------------------------===//

JITRunner::JITRunner(MLIRContext *context)
    : impl(std::make_unique<JITRunnerImpl>(context)), context(context) {}

JITRunner::~JITRunner() = default;

LogicalResult JITRunner::compile(ModuleOp module) {
  // Lower to LLVM dialect
  if (failed(impl->lowerToLLVM(module))) {
    return failure();
  }
  
  // Create execution engine
  auto engineOrErr = impl->createEngine(module, optimizationLevel);
  if (!engineOrErr) {
    llvm::errs() << "Failed to create execution engine: "
                 << llvm::toString(engineOrErr.takeError()) << "\n";
    return failure();
  }
  
  engine = std::move(*engineOrErr);
  return success();
}

LogicalResult JITRunner::invoke(StringRef funcName,
                                MutableArrayRef<void *> args) {
  if (!engine) {
    llvm::errs() << "Execution engine not initialized. Call compile() first.\n";
    return failure();
  }
  
  auto invocationResult = engine->invokePacked(funcName, args);
  if (invocationResult) {
    llvm::errs() << "Invocation failed: " << llvm::toString(std::move(invocationResult)) << "\n";
    return failure();
  }
  
  return success();
}

void *JITRunner::lookup(StringRef funcName) {
  if (!engine) {
    return nullptr;
  }
  
  auto symbol = engine->lookup(funcName);
  if (!symbol) {
    llvm::consumeError(symbol.takeError());
    return nullptr;
  }
  
  return reinterpret_cast<void *>(*symbol);
}

std::string JITRunner::dumpLLVMIR(ModuleOp module) {
  return impl->exportLLVMIR(module);
}

//===----------------------------------------------------------------------===//
// Convenience Functions
//===----------------------------------------------------------------------===//

std::unique_ptr<JITRunner> createJITRunner(MLIRContext *context) {
  return std::make_unique<JITRunner>(context);
}

} // namespace yirage
