//===- YirageDialect.cpp - Yirage Dialect Implementation --------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.h"
#include "yirage-mlir/Dialect/Yirage/IR/YirageOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace yirage::ir;

//===----------------------------------------------------------------------===//
// Include TableGen-generated definitions
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// YirageDialect Implementation
//===----------------------------------------------------------------------===//

void YirageDialect::initialize() {
  // Register operations
  addOperations<
#define GET_OP_LIST
#include "yirage-mlir/Dialect/Yirage/IR/YirageOps.cpp.inc"
      >();
}
