//===- YirageOps.h - Yirage Operations Definition ---------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_MLIR_DIALECT_YIRAGE_IR_YIRAGEOPS_H
#define YIRAGE_MLIR_DIALECT_YIRAGE_IR_YIRAGEOPS_H

#include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include TableGen-generated operation declarations
#define GET_OP_CLASSES
#include "yirage-mlir/Dialect/Yirage/IR/YirageOps.h.inc"

namespace yirage {
namespace ir {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

/// Check if a tensor type is compatible with quantization.
bool isQuantizationCompatible(mlir::RankedTensorType type);

/// Get the scale for attention based on head dimension.
float getAttentionScale(int64_t headDim);

/// Infer the output shape for matmul operation.
mlir::SmallVector<int64_t> inferMatmulShape(mlir::RankedTensorType lhs,
                                             mlir::RankedTensorType rhs,
                                             bool transposeLhs,
                                             bool transposeRhs);

/// Infer the output shape for attention operation.
mlir::SmallVector<int64_t> inferAttentionShape(mlir::RankedTensorType query,
                                                int64_t numKVHeads);

} // namespace ir
} // namespace yirage

#endif // YIRAGE_MLIR_DIALECT_YIRAGE_IR_YIRAGEOPS_H
