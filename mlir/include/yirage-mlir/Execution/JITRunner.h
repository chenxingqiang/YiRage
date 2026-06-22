//===- JITRunner.h - JIT Compilation Engine Interface ------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This header defines the JIT compilation engine interface for YiRage MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_MLIR_EXECUTION_JITRUNNER_H
#define YIRAGE_MLIR_EXECUTION_JITRUNNER_H

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include <memory>
#include <string>

namespace mlir {
class ExecutionEngine;
} // namespace mlir

namespace yirage {

class JITRunnerImpl;

/// JIT Runner for compiling and executing MLIR modules.
///
/// This class provides the final step in the YiRage compilation pipeline,
/// taking LLVM dialect IR and producing executable native code.
///
/// Example usage:
/// \code
///   MLIRContext context;
///   // ... build or load module ...
///   
///   JITRunner runner(&context);
///   runner.compile(module);
///   
///   // Get function pointer and call
///   auto *fn = runner.lookup("kernel");
///   fn(args...);
/// \endcode
class JITRunner {
public:
  explicit JITRunner(mlir::MLIRContext *context);
  ~JITRunner();
  
  // Disable copy
  JITRunner(const JITRunner &) = delete;
  JITRunner &operator=(const JITRunner &) = delete;
  
  // Enable move
  JITRunner(JITRunner &&) = default;
  JITRunner &operator=(JITRunner &&) = default;
  
  /// Compile the module to native code.
  /// This lowers to LLVM dialect and JIT compiles.
  mlir::LogicalResult compile(mlir::ModuleOp module);
  
  /// Invoke a function by name with packed arguments.
  /// Arguments should be pointers to the actual values.
  mlir::LogicalResult invoke(llvm::StringRef funcName,
                             llvm::MutableArrayRef<void *> args);
  
  /// Look up a compiled function by name.
  /// Returns a function pointer that can be cast and called directly.
  void *lookup(llvm::StringRef funcName);
  
  /// Export the module as LLVM IR text (for debugging).
  std::string dumpLLVMIR(mlir::ModuleOp module);
  
  /// Set optimization level (0-3, default 3).
  void setOptLevel(unsigned level) { optimizationLevel = level; }
  
  /// Check if the engine is ready for execution.
  bool isReady() const { return engine != nullptr; }

private:
  std::unique_ptr<JITRunnerImpl> impl;
  std::unique_ptr<mlir::ExecutionEngine> engine;
  mlir::MLIRContext *context;
  unsigned optimizationLevel = 3;
};

/// Create a JIT runner instance.
std::unique_ptr<JITRunner> createJITRunner(mlir::MLIRContext *context);

} // namespace yirage

#endif // YIRAGE_MLIR_EXECUTION_JITRUNNER_H
