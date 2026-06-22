//===- yirage-opt.cpp - Yirage MLIR Optimizer Tool ----------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file implements the yirage-opt tool for optimizing and transforming
// MLIR modules using the Yirage dialect.
//
// Usage:
//   yirage-opt input.mlir -yirage-to-linalg -o output.mlir
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.h"
#include "yirage-mlir/Dialect/Yirage/IR/YirageOps.h"
#include "yirage-mlir/Dialect/Yirage/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

int main(int argc, char **argv) {
  // Register dialects
  mlir::DialectRegistry registry;
  
  // Core MLIR dialects
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::linalg::LinalgDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  
  // YiRage dialect
  registry.insert<yirage::ir::YirageDialect>();

  // Register Yirage passes
  yirage::registerYiragePasses();
  yirage::registerYiragePassPipelines();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Yirage MLIR Optimizer\n", registry));
}
