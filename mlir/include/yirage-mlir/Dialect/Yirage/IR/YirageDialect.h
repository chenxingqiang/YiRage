//===- YirageDialect.h - Yirage Dialect Definition -------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_MLIR_DIALECT_YIRAGE_IR_YIRAGEDIALECT_H
#define YIRAGE_MLIR_DIALECT_YIRAGE_IR_YIRAGEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include TableGen-generated dialect declaration
#include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.h.inc"

#endif // YIRAGE_MLIR_DIALECT_YIRAGE_IR_YIRAGEDIALECT_H
