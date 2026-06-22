//===- YirageOps.cpp - Yirage Operations Implementation ---------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Dialect/Yirage/IR/YirageOps.h"
#include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include <cmath>

using namespace mlir;
using namespace yirage::ir;

//===----------------------------------------------------------------------===//
// Include TableGen-generated definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "yirage-mlir/Dialect/Yirage/IR/YirageOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Matrix Operations - Verifiers
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::verify() {
  auto lhsType = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::cast<RankedTensorType>(getRhs().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return emitOpError("expected 2D tensors for matmul");

  int64_t lhsK = lhsType.getDimSize(1);
  int64_t rhsK = getTransposeRhs() ? rhsType.getDimSize(1) : rhsType.getDimSize(0);

  if (lhsK != ShapedType::kDynamic && rhsK != ShapedType::kDynamic && lhsK != rhsK)
    return emitOpError("matmul dimension mismatch: lhs K=")
           << lhsK << ", rhs K=" << rhsK;

  if (resultType.getRank() != 2)
    return emitOpError("expected 2D result tensor");

  return success();
}

LogicalResult BatchMatmulOp::verify() {
  auto lhsType = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::cast<RankedTensorType>(getRhs().getType());

  if (lhsType.getRank() < 3 || rhsType.getRank() < 3)
    return emitOpError("expected at least 3D tensors for batch matmul");

  if (lhsType.getRank() != rhsType.getRank())
    return emitOpError("lhs and rhs must have same rank");

  return success();
}

LogicalResult QMatmulOp::verify() {
  auto lhsType = llvm::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = llvm::cast<RankedTensorType>(getRhsQuantized().getType());

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return emitOpError("expected 2D tensors for qmatmul");

  if (!isa<IntegerType>(rhsType.getElementType()))
    return emitOpError("rhs must be integer type for quantized matmul");

  return success();
}

void MatmulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {}

//===----------------------------------------------------------------------===//
// Attention Operations - Verifiers
//===----------------------------------------------------------------------===//

LogicalResult AttentionOp::verify() {
  auto qType = llvm::cast<RankedTensorType>(getQuery().getType());
  auto kType = llvm::cast<RankedTensorType>(getKey().getType());
  auto vType = llvm::cast<RankedTensorType>(getValue().getType());

  if (qType.getRank() < 3 || kType.getRank() < 3 || vType.getRank() < 3)
    return emitOpError("expected at least 3D tensors for attention (batch, seq, dim)");

  if (qType.getRank() != kType.getRank() || kType.getRank() != vType.getRank())
    return emitOpError("query, key, value must have same rank");

  // Check head dimension matches between Q, K, V
  int64_t qHeadDim = qType.getDimSize(qType.getRank() - 1);
  int64_t kHeadDim = kType.getDimSize(kType.getRank() - 1);

  if (qHeadDim != ShapedType::kDynamic && kHeadDim != ShapedType::kDynamic &&
      qHeadDim != kHeadDim)
    return emitOpError("query and key head dimensions must match");

  return success();
}

LogicalResult PagedAttentionOp::verify() {
  auto qType = llvm::cast<RankedTensorType>(getQuery().getType());

  if (qType.getRank() < 2)
    return emitOpError("expected at least 2D query tensor");

  auto blockTableType = llvm::cast<RankedTensorType>(getBlockTables().getType());
  if (!isa<IntegerType>(blockTableType.getElementType()))
    return emitOpError("block_tables must be integer type");

  return success();
}

LogicalResult KVCacheUpdateOp::verify() {
  auto keyCacheType = llvm::cast<RankedTensorType>(getKeyCache().getType());
  auto newKeysType = llvm::cast<RankedTensorType>(getNewKeys().getType());

  if (keyCacheType.getRank() < 3)
    return emitOpError("key_cache must be at least 3D (batch, seq, dim)");

  if (newKeysType.getRank() < 2)
    return emitOpError("new_keys must be at least 2D");

  return success();
}

void AttentionOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {}

//===----------------------------------------------------------------------===//
// Normalization Operations - Verifiers
//===----------------------------------------------------------------------===//

LogicalResult RMSNormOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto gammaType = llvm::cast<RankedTensorType>(getGamma().getType());

  if (inputType.getRank() < 1)
    return emitOpError("input must be at least 1D");

  if (gammaType.getRank() != 1)
    return emitOpError("gamma must be 1D");

  int64_t inputLastDim = inputType.getDimSize(inputType.getRank() - 1);
  int64_t gammaDim = gammaType.getDimSize(0);

  if (inputLastDim != ShapedType::kDynamic && gammaDim != ShapedType::kDynamic &&
      inputLastDim != gammaDim)
    return emitOpError("gamma dimension must match input's last dimension");

  return success();
}

LogicalResult LayerNormOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto gammaType = llvm::cast<RankedTensorType>(getGamma().getType());
  auto betaType = llvm::cast<RankedTensorType>(getBeta().getType());

  if (inputType.getRank() < 1)
    return emitOpError("input must be at least 1D");

  if (gammaType.getRank() != 1 || betaType.getRank() != 1)
    return emitOpError("gamma and beta must be 1D");

  int64_t gammaDim = gammaType.getDimSize(0);
  int64_t betaDim = betaType.getDimSize(0);

  if (gammaDim != ShapedType::kDynamic && betaDim != ShapedType::kDynamic &&
      gammaDim != betaDim)
    return emitOpError("gamma and beta dimensions must match");

  return success();
}

void RMSNormOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {}

//===----------------------------------------------------------------------===//
// MLP Operations - Verifiers
//===----------------------------------------------------------------------===//

LogicalResult GatedMLPOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto gateWeightType = llvm::cast<RankedTensorType>(getGateWeight().getType());
  auto upWeightType = llvm::cast<RankedTensorType>(getUpWeight().getType());
  auto downWeightType = llvm::cast<RankedTensorType>(getDownWeight().getType());

  if (inputType.getRank() < 2)
    return emitOpError("input must be at least 2D");

  if (gateWeightType.getRank() != 2 || upWeightType.getRank() != 2 ||
      downWeightType.getRank() != 2)
    return emitOpError("all weights must be 2D");

  return success();
}

LogicalResult LinearOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto weightType = llvm::cast<RankedTensorType>(getWeight().getType());

  if (inputType.getRank() < 2)
    return emitOpError("input must be at least 2D");

  if (weightType.getRank() != 2)
    return emitOpError("weight must be 2D");

  int64_t inputK = inputType.getDimSize(inputType.getRank() - 1);
  int64_t weightK = weightType.getDimSize(0);

  if (inputK != ShapedType::kDynamic && weightK != ShapedType::kDynamic &&
      inputK != weightK)
    return emitOpError("input last dim must match weight first dim");

  return success();
}

void GatedMLPOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {}

//===----------------------------------------------------------------------===//
// Activation Operations - Verifiers
//===----------------------------------------------------------------------===//

LogicalResult SoftmaxOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  int64_t axis = getAxis();

  if (axis < 0)
    axis += inputType.getRank();

  if (axis < 0 || axis >= inputType.getRank())
    return emitOpError("axis out of range");

  return success();
}

//===----------------------------------------------------------------------===//
// Activation Operations - Folders
//===----------------------------------------------------------------------===//

OpFoldResult SiLUOp::fold(FoldAdaptor adaptor) { return nullptr; }
OpFoldResult GELUOp::fold(FoldAdaptor adaptor) { return nullptr; }
OpFoldResult ReLUOp::fold(FoldAdaptor adaptor) { return nullptr; }

//===----------------------------------------------------------------------===//
// Embedding Operations - Verifiers
//===----------------------------------------------------------------------===//

LogicalResult EmbeddingOp::verify() {
  auto indicesType = llvm::cast<RankedTensorType>(getIndices().getType());
  auto tableType = llvm::cast<RankedTensorType>(getTable().getType());

  if (!isa<IntegerType>(indicesType.getElementType()))
    return emitOpError("indices must be integer type");

  if (tableType.getRank() != 2)
    return emitOpError("embedding table must be 2D (vocab_size, embed_dim)");

  return success();
}

LogicalResult RoPEOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto cosType = llvm::cast<RankedTensorType>(getCosCache().getType());
  auto sinType = llvm::cast<RankedTensorType>(getSinCache().getType());

  if (inputType.getRank() < 2)
    return emitOpError("input must be at least 2D");

  if (cosType.getRank() != sinType.getRank())
    return emitOpError("cos and sin must have same rank");

  return success();
}

//===----------------------------------------------------------------------===//
// Reduction Operations - Verifiers
//===----------------------------------------------------------------------===//

LogicalResult ReduceSumOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  int64_t axis = getAxis();

  if (axis < 0)
    axis += inputType.getRank();

  if (axis < 0 || axis >= inputType.getRank())
    return emitOpError("axis out of range");

  return success();
}

LogicalResult ReduceMaxOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  int64_t axis = getAxis();

  if (axis < 0)
    axis += inputType.getRank();

  if (axis < 0 || axis >= inputType.getRank())
    return emitOpError("axis out of range");

  return success();
}

LogicalResult TopKOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  int64_t k = getK();
  int64_t axis = getAxis();

  if (axis < 0)
    axis += inputType.getRank();

  if (axis < 0 || axis >= inputType.getRank())
    return emitOpError("axis out of range");

  int64_t axisDim = inputType.getDimSize(axis);
  if (k <= 0)
    return emitOpError("k must be positive");

  if (axisDim != ShapedType::kDynamic && k > axisDim)
    return emitOpError("k cannot be larger than axis dimension");

  return success();
}

LogicalResult ArgMaxOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  int64_t axis = getAxis();

  if (axis < 0)
    axis += inputType.getRank();

  if (axis < 0 || axis >= inputType.getRank())
    return emitOpError("axis out of range");

  return success();
}

//===----------------------------------------------------------------------===//
// Tensor Operations - Verifiers and Folders
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  // Check that total elements match (if statically known)
  int64_t inputElements = 1;
  int64_t resultElements = 1;
  bool inputDynamic = false;
  bool resultDynamic = false;

  for (int64_t dim : inputType.getShape()) {
    if (dim == ShapedType::kDynamic)
      inputDynamic = true;
    else
      inputElements *= dim;
  }

  for (int64_t dim : resultType.getShape()) {
    if (dim == ShapedType::kDynamic)
      resultDynamic = true;
    else
      resultElements *= dim;
  }

  if (!inputDynamic && !resultDynamic && inputElements != resultElements)
    return emitOpError("reshape changes number of elements from ")
           << inputElements << " to " << resultElements;

  return success();
}

LogicalResult TransposeOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto perm = getPermutation();

  if (perm.size() != static_cast<size_t>(inputType.getRank()))
    return emitOpError("permutation size must match input rank");

  llvm::SmallVector<bool> seen(perm.size(), false);
  for (auto p : perm) {
    int64_t idx = llvm::cast<IntegerAttr>(p).getInt();
    if (idx < 0 || idx >= static_cast<int64_t>(perm.size()))
      return emitOpError("permutation index out of range");
    if (seen[idx])
      return emitOpError("duplicate index in permutation");
    seen[idx] = true;
  }

  return success();
}

LogicalResult ConcatOp::verify() {
  if (getInputs().empty())
    return emitOpError("concat requires at least one input");

  auto firstType = llvm::cast<RankedTensorType>(getInputs()[0].getType());
  int64_t axis = getAxis();

  if (axis < 0)
    axis += firstType.getRank();

  if (axis < 0 || axis >= firstType.getRank())
    return emitOpError("axis out of range");

  for (auto input : getInputs().drop_front()) {
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    if (inputType.getRank() != firstType.getRank())
      return emitOpError("all inputs must have same rank");
  }

  return success();
}

LogicalResult SplitOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  int64_t axis = getAxis();

  if (axis < 0)
    axis += inputType.getRank();

  if (axis < 0 || axis >= inputType.getRank())
    return emitOpError("axis out of range");

  int64_t numSplits = getNumSplits();
  if (numSplits <= 0)
    return emitOpError("num_splits must be positive");

  int64_t axisDim = inputType.getDimSize(axis);
  if (axisDim != ShapedType::kDynamic && axisDim % numSplits != 0)
    return emitOpError("axis dimension must be divisible by num_splits");

  return success();
}

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) { return nullptr; }
OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) { return nullptr; }

//===----------------------------------------------------------------------===//
// Convolution Operations - Verifiers
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto kernelType = llvm::cast<RankedTensorType>(getKernel().getType());

  if (inputType.getRank() != 4)
    return emitOpError("input must be 4D (NCHW or NHWC)");

  if (kernelType.getRank() != 4)
    return emitOpError("kernel must be 4D (OIHW or OHWI)");

  auto strides = getStrides();
  auto padding = getPadding();
  auto dilations = getDilations();

  if (strides.size() != 2)
    return emitOpError("strides must have 2 elements");

  if (padding.size() != 4)
    return emitOpError("padding must have 4 elements (top, bottom, left, right)");

  if (dilations.size() != 2)
    return emitOpError("dilations must have 2 elements");

  return success();
}

LogicalResult MaxPool2DOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());

  if (inputType.getRank() != 4)
    return emitOpError("input must be 4D (NCHW or NHWC)");

  auto kernelSize = getKernelSize();
  auto strides = getStrides();
  auto padding = getPadding();

  if (kernelSize.size() != 2)
    return emitOpError("kernel_size must have 2 elements");

  if (strides.size() != 2)
    return emitOpError("strides must have 2 elements");

  if (padding.size() != 4)
    return emitOpError("padding must have 4 elements");

  return success();
}

//===----------------------------------------------------------------------===//
// Quantization Operations - Verifiers
//===----------------------------------------------------------------------===//

LogicalResult QuantizeOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  if (!inputType.getElementType().isa<FloatType>())
    return emitOpError("input must be floating-point type");

  if (!isa<IntegerType>(resultType.getElementType()))
    return emitOpError("result must be integer type");

  if (inputType.getShape() != resultType.getShape())
    return emitOpError("input and result shapes must match");

  return success();
}

LogicalResult DequantizeOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto resultType = llvm::cast<RankedTensorType>(getResult().getType());

  if (!isa<IntegerType>(inputType.getElementType()))
    return emitOpError("input must be integer type");

  if (!resultType.getElementType().isa<FloatType>())
    return emitOpError("result must be floating-point type");

  if (inputType.getShape() != resultType.getShape())
    return emitOpError("input and result shapes must match");

  return success();
}

//===----------------------------------------------------------------------===//
// Stub verifiers (ops declare hasVerifier in TableGen; full checks TBD)
//===----------------------------------------------------------------------===//

#define YIRAGE_STUB_VERIFY(Op)                                                 \
  LogicalResult Op::verify() { return success(); }

YIRAGE_STUB_VERIFY(MoERouterOp)
YIRAGE_STUB_VERIFY(MoEDispatchOp)
YIRAGE_STUB_VERIFY(MoECombineOp)
YIRAGE_STUB_VERIFY(MoEExpertOp)
YIRAGE_STUB_VERIFY(MoELayerOp)
YIRAGE_STUB_VERIFY(SpecDraftOp)
YIRAGE_STUB_VERIFY(SpecVerifyOp)
YIRAGE_STUB_VERIFY(LookaheadDecodeOp)
YIRAGE_STUB_VERIFY(MLACompressOp)
YIRAGE_STUB_VERIFY(MLADecompressOp)
YIRAGE_STUB_VERIFY(MLAttentionOp)
YIRAGE_STUB_VERIFY(SlidingWindowAttentionOp)
YIRAGE_STUB_VERIFY(CrossAttentionOp)
YIRAGE_STUB_VERIFY(PrefixCacheLookupOp)
YIRAGE_STUB_VERIFY(SampleTopPOp)
YIRAGE_STUB_VERIFY(SampleTopKOp)

#undef YIRAGE_STUB_VERIFY

void MoELayerOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {}

void MLAttentionOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {}
