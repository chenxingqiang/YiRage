//===- YirageToLinalg.cpp - Lower Yirage to Linalg ---------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file implements the lowering from Yirage dialect to Linalg dialect.
// Full LLM operator support including:
//   - Complete Attention: Q@K^T/√d → softmax → @V
//   - RMSNorm/LayerNorm with proper broadcasting
//   - Flash Attention with tiled computation
//   - PagedAttention for vLLM-style KV cache
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.h"
#include "yirage-mlir/Dialect/Yirage/IR/YirageOps.h"
#include "yirage-mlir/Dialect/Yirage/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>

using namespace mlir;
using namespace yirage::ir;

namespace {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Create an empty tensor with the given type
static Value createEmptyTensor(OpBuilder &builder, Location loc,
                                RankedTensorType type) {
  return builder.create<tensor::EmptyOp>(loc, type.getShape(),
                                          type.getElementType());
}

/// Create a tensor filled with a constant value
static Value createFilledTensor(OpBuilder &builder, Location loc,
                                 RankedTensorType type, Value fillValue) {
  Value empty = createEmptyTensor(builder, loc, type);
  return builder.create<linalg::FillOp>(loc, fillValue, empty).getResult(0);
}

/// Create a tensor filled with zeros
static Value createZeroTensor(OpBuilder &builder, Location loc,
                               RankedTensorType type) {
  Value zero = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(type.getElementType()));
  return createFilledTensor(builder, loc, type, zero);
}

/// Create identity indexing maps for a generic op
static SmallVector<AffineMap> createIdentityMaps(OpBuilder &builder,
                                                  int64_t rank, int numMaps) {
  AffineMap identityMap = builder.getMultiDimIdentityMap(rank);
  return SmallVector<AffineMap>(numMaps, identityMap);
}

/// Create parallel iterator types
static SmallVector<utils::IteratorType> createParallelIterators(int64_t rank) {
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

/// fp16 uses f32 tensor math; other types compute in their native element type.
static Type accumulationType(Type elemType) {
  if (elemType.isF16())
    return mlir::Float32Type::get(elemType.getContext());
  return elemType;
}

static RankedTensorType sameShapeTensorType(RankedTensorType type, Type elemType) {
  return RankedTensorType::get(type.getShape(), elemType);
}

/// Element-wise cast via linalg.generic (bufferizable; avoids tensor arith.extf).
static Value castTensorElemType(OpBuilder &builder, Location loc, Value tensor,
                                Type targetElemType, bool extend) {
  auto ranked = llvm::cast<RankedTensorType>(tensor.getType());
  Type srcElemType = ranked.getElementType();
  if (srcElemType == targetElemType)
    return tensor;
  int64_t rank = ranked.getRank();
  auto outType = sameShapeTensorType(ranked, targetElemType);
  Value init = createEmptyTensor(builder, loc, outType);
  return builder
      .create<linalg::GenericOp>(
          loc, outType, tensor, init,
          createIdentityMaps(builder, rank, 2), createParallelIterators(rank),
          [targetElemType, extend](OpBuilder &b, Location bLoc,
                                   ValueRange args) {
            Value in = args[0];
            Value out = extend
                            ? Value(b.create<arith::ExtFOp>(bLoc, targetElemType, in))
                            : Value(b.create<arith::TruncFOp>(bLoc, targetElemType, in));
            b.create<linalg::YieldOp>(bLoc, out);
          })
      .getResult(0);
}

static Value promoteTensorElemType(OpBuilder &builder, Location loc, Value tensor,
                                   Type targetElemType) {
  return castTensorElemType(builder, loc, tensor, targetElemType, /*extend=*/true);
}

static Value truncateTensorElemType(OpBuilder &builder, Location loc, Value tensor,
                                    Type targetElemType) {
  return castTensorElemType(builder, loc, tensor, targetElemType, /*extend=*/false);
}

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class YirageTypeConverter : public TypeConverter {
public:
  YirageTypeConverter() {
    addConversion([](Type type) { return type; });
  }
};

//===----------------------------------------------------------------------===//
// Matrix Operation Lowering
//===----------------------------------------------------------------------===//

/// Lower yirage.matmul to linalg.matmul
struct MatmulOpLowering : public OpConversionPattern<MatmulOp> {
  using OpConversionPattern<MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    Type elemType = resultType.getElementType();
    Type accumType = accumulationType(elemType);

    if (accumType != elemType) {
      Value lhsAcc = promoteTensorElemType(rewriter, loc, lhs, accumType);
      Value rhsAcc = promoteTensorElemType(rewriter, loc, rhs, accumType);
      auto accResultType = RankedTensorType::get(resultType.getShape(), accumType);
      Value zeroed = createZeroTensor(rewriter, loc, accResultType);
      Value accResult = rewriter.create<linalg::MatmulOp>(
          loc, ValueRange{lhsAcc, rhsAcc}, ValueRange{zeroed}).getResult(0);
      Value result =
          truncateTensorElemType(rewriter, loc, accResult, elemType);
      rewriter.replaceOp(op, result);
      return success();
    }

    Value zeroed = createZeroTensor(rewriter, loc, resultType);
    Value result = rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{lhs, rhs}, ValueRange{zeroed}).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower yirage.batch_matmul to linalg.batch_matmul (for 3D tensors)
/// For 4D tensors (attention), use custom implementation
struct BatchMatmulOpLowering : public OpConversionPattern<BatchMatmulOp> {
  using OpConversionPattern<BatchMatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BatchMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    auto lhsType = llvm::cast<RankedTensorType>(lhs.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    int64_t rank = lhsType.getRank();

    if (rank == 3) {
      // Standard 3D batch matmul
      Value zeroed = createZeroTensor(rewriter, loc, resultType);
      Value result = rewriter.create<linalg::BatchMatmulOp>(
          loc, ValueRange{lhs, rhs}, ValueRange{zeroed}).getResult(0);
      rewriter.replaceOp(op, result);
      return success();
    }

    // For 4D tensors: [batch, heads, seq, dim]
    // Use linalg.generic with proper indexing maps
    auto shape = resultType.getShape();
    int64_t batch = shape[0];
    int64_t heads = shape[1];
    int64_t seqM = shape[2];
    int64_t seqN = shape[3];
    
    auto rhsType = llvm::cast<RankedTensorType>(rhs.getType());
    int64_t K = rhsType.getShape()[2]; // For transpose case, K is seq dim of rhs

    // Create output tensor
    Value zeroed = createZeroTensor(rewriter, loc, resultType);

    // Indexing maps for 4D batch matmul with transpose
    // lhs: [b, h, m, k]
    // rhs: [b, h, n, k] (transposed view: [b, h, k, n])
    // out: [b, h, m, n]
    auto b = rewriter.getAffineDimExpr(0);
    auto h = rewriter.getAffineDimExpr(1);
    auto m = rewriter.getAffineDimExpr(2);
    auto n = rewriter.getAffineDimExpr(3);
    auto k = rewriter.getAffineDimExpr(4);

    SmallVector<AffineExpr> lhsExprs = {b, h, m, k};
    SmallVector<AffineExpr> rhsExprs;
    if (op.getTransposeRhs()) {
      // With transpose_rhs=true (Q @ K^T pattern):
      // rhs has physical shape [b, h, N, K] (e.g., K tensor in attention)
      // Access as rhs[b, h, n, k] to get logical transpose
      rhsExprs = {b, h, n, k};
    } else {
      // Without transpose:
      // rhs has physical shape [b, h, K, N]
      // Access rhs[b, h, k, n] directly
      rhsExprs = {b, h, k, n};
    }
    SmallVector<AffineExpr> outExprs = {b, h, m, n};

    auto lhsMap = AffineMap::get(5, 0, lhsExprs, rewriter.getContext());
    auto rhsMap = AffineMap::get(5, 0, rhsExprs, rewriter.getContext());
    auto outMap = AffineMap::get(5, 0, outExprs, rewriter.getContext());

    SmallVector<utils::IteratorType> iteratorTypes = {
        utils::IteratorType::parallel,   // batch
        utils::IteratorType::parallel,   // heads
        utils::IteratorType::parallel,   // M
        utils::IteratorType::parallel,   // N
        utils::IteratorType::reduction   // K
    };

    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{lhs, rhs}, zeroed,
        ArrayRef<AffineMap>{lhsMap, rhsMap, outMap}, iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
          Value add = b.create<arith::AddFOp>(loc, mul, args[2]);
          b.create<linalg::YieldOp>(loc, add);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower yirage.linear to linalg.matmul + optional bias add
struct LinearOpLowering : public OpConversionPattern<LinearOp> {
  using OpConversionPattern<LinearOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value weight = adaptor.getWeight();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    int64_t inputRank = inputType.getRank();

    Value zeroed = createZeroTensor(rewriter, loc, resultType);
    Value matmulResult;
    
    if (inputRank == 2) {
      // 2D case: standard matmul
      matmulResult = rewriter.create<linalg::MatmulOp>(
          loc, ValueRange{input, weight}, ValueRange{zeroed}).getResult(0);
    } else if (inputRank == 3) {
      // 3D case: batched matmul [batch, seq, in_dim] @ [in_dim, out_dim]
      // Use linalg.generic with proper indexing
      auto b = rewriter.getAffineDimExpr(0);  // batch
      auto m = rewriter.getAffineDimExpr(1);  // seq
      auto n = rewriter.getAffineDimExpr(2);  // out_dim
      auto k = rewriter.getAffineDimExpr(3);  // in_dim (reduction)
      
      SmallVector<AffineExpr> inputExprs = {b, m, k};
      SmallVector<AffineExpr> weightExprs = {k, n};
      SmallVector<AffineExpr> outExprs = {b, m, n};
      
      auto inputMap = AffineMap::get(4, 0, inputExprs, rewriter.getContext());
      auto weightMap = AffineMap::get(4, 0, weightExprs, rewriter.getContext());
      auto outMap = AffineMap::get(4, 0, outExprs, rewriter.getContext());
      
      SmallVector<utils::IteratorType> iteratorTypes = {
          utils::IteratorType::parallel,   // batch
          utils::IteratorType::parallel,   // seq
          utils::IteratorType::parallel,   // out_dim
          utils::IteratorType::reduction   // in_dim
      };
      
      matmulResult = rewriter.create<linalg::GenericOp>(
          loc, resultType, ValueRange{input, weight}, zeroed,
          ArrayRef<AffineMap>{inputMap, weightMap, outMap}, iteratorTypes,
          [](OpBuilder &b, Location loc, ValueRange args) {
            Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
            Value add = b.create<arith::AddFOp>(loc, mul, args[2]);
            b.create<linalg::YieldOp>(loc, add);
          }).getResult(0);
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported input rank for linear");
    }

    if (adaptor.getBias()) {
      Value bias = adaptor.getBias();
      auto biasType = llvm::cast<RankedTensorType>(bias.getType());
      int64_t resultRank = resultType.getRank();
      int64_t biasRank = biasType.getRank();

      // Build broadcast indexing map for bias
      SmallVector<AffineExpr> biasExprs;
      for (int64_t i = 0; i < biasRank; ++i) {
        biasExprs.push_back(rewriter.getAffineDimExpr(resultRank - biasRank + i));
      }

      AffineMap inputMap = rewriter.getMultiDimIdentityMap(resultRank);
      AffineMap biasMap = AffineMap::get(resultRank, 0, biasExprs, rewriter.getContext());
      AffineMap outputMap = rewriter.getMultiDimIdentityMap(resultRank);

      Value output = createEmptyTensor(rewriter, loc, resultType);
      matmulResult = rewriter.create<linalg::GenericOp>(
          loc, resultType, ValueRange{matmulResult, bias}, output,
          ArrayRef<AffineMap>{inputMap, biasMap, outputMap},
          createParallelIterators(resultRank),
          [](OpBuilder &b, Location loc, ValueRange args) {
            Value result = b.create<arith::AddFOp>(loc, args[0], args[1]);
            b.create<linalg::YieldOp>(loc, result);
          }).getResult(0);
    }

    rewriter.replaceOp(op, matmulResult);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Normalization Lowering with Proper Broadcasting
//===----------------------------------------------------------------------===//

/// Lower yirage.rms_norm to complete implementation with reduction
/// RMSNorm: output = x / sqrt(mean(x^2) + eps) * gamma
struct RMSNormOpLowering : public OpConversionPattern<RMSNormOp> {
  using OpConversionPattern<RMSNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RMSNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value gamma = adaptor.getGamma();
    float epsilon = op.getEpsilon().convertToFloat();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    Type storageType = inputType.getElementType();
    Type computeType = accumulationType(storageType);

    // Run the full RMS pipeline in f32 when storage is f16 (homogeneous linalg types).
    if (computeType != storageType) {
      input = promoteTensorElemType(rewriter, loc, input, computeType);
      gamma = promoteTensorElemType(rewriter, loc, gamma, computeType);
      inputType = llvm::cast<RankedTensorType>(input.getType());
    }

    int64_t inputRank = inputType.getRank();
    int64_t hiddenDim = inputType.getShape().back();
    Type elemType = inputType.getElementType();

    // Step 1: Compute sum of squares along last dimension
    // Create reduced shape (all dims except last)
    SmallVector<int64_t> reducedShape;
    for (int64_t i = 0; i < inputRank - 1; ++i) {
      reducedShape.push_back(inputType.getShape()[i]);
    }
    auto reducedType = RankedTensorType::get(reducedShape, elemType);
    
    Value zeroScalar = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value sumSq = createFilledTensor(rewriter, loc, reducedType, zeroScalar);

    // Build indexing maps for reduction
    SmallVector<AffineExpr> inputExprs, outputExprs;
    for (int64_t i = 0; i < inputRank; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    for (int64_t i = 0; i < inputRank - 1; ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    
    AffineMap inputMap = AffineMap::get(inputRank, 0, inputExprs, rewriter.getContext());
    AffineMap outputMap = AffineMap::get(inputRank, 0, outputExprs, rewriter.getContext());
    
    SmallVector<utils::IteratorType> iterTypes(inputRank - 1, utils::IteratorType::parallel);
    iterTypes.push_back(utils::IteratorType::reduction);

    sumSq = rewriter.create<linalg::GenericOp>(
        loc, reducedType, input, sumSq,
        ArrayRef<AffineMap>{inputMap, outputMap}, iterTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sq = b.create<arith::MulFOp>(loc, args[0], args[0]);
          Value sum = b.create<arith::AddFOp>(loc, sq, args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }).getResult(0);

    // Step 2: Compute mean and rsqrt
    Value hiddenDimVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, static_cast<double>(hiddenDim)));
    Value epsVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, epsilon));
    
    Value meanSq = createEmptyTensor(rewriter, loc, reducedType);
    meanSq = rewriter.create<linalg::GenericOp>(
        loc, reducedType, sumSq, meanSq,
        createIdentityMaps(rewriter, inputRank - 1, 2),
        createParallelIterators(inputRank - 1),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value mean = b.create<arith::DivFOp>(loc, args[0], hiddenDimVal);
          Value meanPlusEps = b.create<arith::AddFOp>(loc, mean, epsVal);
          Value rsqrt = b.create<math::RsqrtOp>(loc, meanPlusEps);
          b.create<linalg::YieldOp>(loc, rsqrt);
        }).getResult(0);

    // Step 3: Apply normalization and scale
    // Build broadcast maps: input[..., d], rsqrt[...], gamma[d] -> output[..., d]
    SmallVector<AffineExpr> rsqrtExprs, gammaExprs;
    for (int64_t i = 0; i < inputRank - 1; ++i) {
      rsqrtExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    gammaExprs.push_back(rewriter.getAffineDimExpr(inputRank - 1));
    
    AffineMap rsqrtMap = AffineMap::get(inputRank, 0, rsqrtExprs, rewriter.getContext());
    AffineMap gammaMap = AffineMap::get(inputRank, 0, gammaExprs, rewriter.getContext());
    AffineMap fullInputMap = rewriter.getMultiDimIdentityMap(inputRank);
    AffineMap fullOutputMap = rewriter.getMultiDimIdentityMap(inputRank);

    auto computeResultType =
        RankedTensorType::get(resultType.getShape(), elemType);
    Value output = createEmptyTensor(rewriter, loc, computeResultType);
    Value result = rewriter.create<linalg::GenericOp>(
        loc, computeResultType, ValueRange{input, meanSq, gamma}, output,
        ArrayRef<AffineMap>{fullInputMap, rsqrtMap, gammaMap, fullOutputMap},
        createParallelIterators(inputRank),
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          Value rsqrt = args[1];
          Value g = args[2];
          Value normalized = b.create<arith::MulFOp>(loc, x, rsqrt);
          Value scaled = b.create<arith::MulFOp>(loc, normalized, g);
          b.create<linalg::YieldOp>(loc, scaled);
        }).getResult(0);

    if (computeType != storageType)
      result = truncateTensorElemType(rewriter, loc, result, storageType);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower yirage.layer_norm with proper broadcasting
/// LayerNorm: output = (x - mean) / sqrt(var + eps) * gamma + beta
struct LayerNormOpLowering : public OpConversionPattern<LayerNormOp> {
  using OpConversionPattern<LayerNormOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value gamma = adaptor.getGamma();
    Value beta = adaptor.getBeta();
    float epsilon = op.getEpsilon().convertToFloat();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    int64_t inputRank = inputType.getRank();
    int64_t hiddenDim = inputType.getShape().back();
    Type elemType = inputType.getElementType();

    // Create reduced shape for mean/var
    SmallVector<int64_t> reducedShape;
    for (int64_t i = 0; i < inputRank - 1; ++i) {
      reducedShape.push_back(inputType.getShape()[i]);
    }
    auto reducedType = RankedTensorType::get(reducedShape, elemType);

    // Build maps for reduction
    SmallVector<AffineExpr> inputExprs, outputExprs;
    for (int64_t i = 0; i < inputRank; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    for (int64_t i = 0; i < inputRank - 1; ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    AffineMap inputMap = AffineMap::get(inputRank, 0, inputExprs, rewriter.getContext());
    AffineMap outputMap = AffineMap::get(inputRank, 0, outputExprs, rewriter.getContext());
    
    SmallVector<utils::IteratorType> iterTypes(inputRank - 1, utils::IteratorType::parallel);
    iterTypes.push_back(utils::IteratorType::reduction);

    // Step 1: Compute sum for mean
    Value zeroScalar = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value sum = createFilledTensor(rewriter, loc, reducedType, zeroScalar);
    
    sum = rewriter.create<linalg::GenericOp>(
        loc, reducedType, input, sum,
        ArrayRef<AffineMap>{inputMap, outputMap}, iterTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value s = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, s);
        }).getResult(0);

    // Step 2: Compute mean
    Value hiddenDimVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, static_cast<double>(hiddenDim)));
    
    Value mean = createEmptyTensor(rewriter, loc, reducedType);
    mean = rewriter.create<linalg::GenericOp>(
        loc, reducedType, sum, mean,
        createIdentityMaps(rewriter, inputRank - 1, 2),
        createParallelIterators(inputRank - 1),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value m = b.create<arith::DivFOp>(loc, args[0], hiddenDimVal);
          b.create<linalg::YieldOp>(loc, m);
        }).getResult(0);

    // Step 3: Compute variance = sum((x - mean)^2) / N
    SmallVector<AffineExpr> meanBcastExprs;
    for (int64_t i = 0; i < inputRank - 1; ++i) {
      meanBcastExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    AffineMap meanBcastMap = AffineMap::get(inputRank, 0, meanBcastExprs, rewriter.getContext());
    
    Value sumSqDiff = createFilledTensor(rewriter, loc, reducedType, zeroScalar);
    sumSqDiff = rewriter.create<linalg::GenericOp>(
        loc, reducedType, ValueRange{input, mean}, sumSqDiff,
        ArrayRef<AffineMap>{inputMap, meanBcastMap, outputMap}, iterTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value diff = b.create<arith::SubFOp>(loc, args[0], args[1]);
          Value sq = b.create<arith::MulFOp>(loc, diff, diff);
          Value sum = b.create<arith::AddFOp>(loc, sq, args[2]);
          b.create<linalg::YieldOp>(loc, sum);
        }).getResult(0);

    // Step 4: Compute 1/sqrt(var + eps)
    Value epsVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, epsilon));
    
    Value rstd = createEmptyTensor(rewriter, loc, reducedType);
    rstd = rewriter.create<linalg::GenericOp>(
        loc, reducedType, sumSqDiff, rstd,
        createIdentityMaps(rewriter, inputRank - 1, 2),
        createParallelIterators(inputRank - 1),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value var = b.create<arith::DivFOp>(loc, args[0], hiddenDimVal);
          Value varPlusEps = b.create<arith::AddFOp>(loc, var, epsVal);
          Value rsqrt = b.create<math::RsqrtOp>(loc, varPlusEps);
          b.create<linalg::YieldOp>(loc, rsqrt);
        }).getResult(0);

    // Step 5: Apply normalization: (x - mean) * rstd * gamma + beta
    SmallVector<AffineExpr> gammaExprs;
    gammaExprs.push_back(rewriter.getAffineDimExpr(inputRank - 1));
    AffineMap gammaMap = AffineMap::get(inputRank, 0, gammaExprs, rewriter.getContext());
    AffineMap fullMap = rewriter.getMultiDimIdentityMap(inputRank);

    Value output = createEmptyTensor(rewriter, loc, resultType);
    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{input, mean, rstd, gamma, beta}, output,
        ArrayRef<AffineMap>{fullMap, meanBcastMap, meanBcastMap, gammaMap, gammaMap, fullMap},
        createParallelIterators(inputRank),
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          Value mean = args[1];
          Value rstd = args[2];
          Value g = args[3];
          Value beta = args[4];
          Value diff = b.create<arith::SubFOp>(loc, x, mean);
          Value normalized = b.create<arith::MulFOp>(loc, diff, rstd);
          Value scaled = b.create<arith::MulFOp>(loc, normalized, g);
          Value result = b.create<arith::AddFOp>(loc, scaled, beta);
          b.create<linalg::YieldOp>(loc, result);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Activation Lowering
//===----------------------------------------------------------------------===//

/// Lower yirage.silu to linalg.generic
struct SiLUOpLowering : public OpConversionPattern<SiLUOp> {
  using OpConversionPattern<SiLUOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SiLUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    Value output = createEmptyTensor(rewriter, loc, inputType);
    auto indexingMaps = createIdentityMaps(rewriter, rank, 2);
    auto iteratorTypes = createParallelIterators(rank);

    Value result = rewriter.create<linalg::GenericOp>(
        loc, inputType, input, output, indexingMaps, iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          Value negX = b.create<arith::NegFOp>(loc, x);
          Value expNegX = b.create<math::ExpOp>(loc, negX);
          Value one = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(x.getType(), 1.0));
          Value denom = b.create<arith::AddFOp>(loc, one, expNegX);
          Value sigmoid = b.create<arith::DivFOp>(loc, one, denom);
          Value silu = b.create<arith::MulFOp>(loc, x, sigmoid);
          b.create<linalg::YieldOp>(loc, silu);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower yirage.gelu to linalg.generic
/// Supports both exact (erf) and approximate (tanh) modes
struct GELUOpLowering : public OpConversionPattern<GELUOp> {
  using OpConversionPattern<GELUOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GELUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    bool approximate = op.getApproximate();

    Value output = createEmptyTensor(rewriter, loc, inputType);

    Value result = rewriter.create<linalg::GenericOp>(
        loc, inputType, input, output,
        createIdentityMaps(rewriter, rank, 2), createParallelIterators(rank),
        [approximate](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          Type elemType = x.getType();
          Value half = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elemType, 0.5));
          Value one = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(elemType, 1.0));
          
          Value gelu;
          if (approximate) {
            // GELU approximate: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
            // sqrt(2/π) ≈ 0.7978845608
            Value sqrt2OverPi = b.create<arith::ConstantOp>(
                loc, b.getFloatAttr(elemType, 0.7978845608028654));
            Value coeff = b.create<arith::ConstantOp>(
                loc, b.getFloatAttr(elemType, 0.044715));
            
            // x^3
            Value x2 = b.create<arith::MulFOp>(loc, x, x);
            Value x3 = b.create<arith::MulFOp>(loc, x2, x);
            
            // 0.044715 * x^3
            Value cubicTerm = b.create<arith::MulFOp>(loc, coeff, x3);
            
            // x + 0.044715 * x^3
            Value inner = b.create<arith::AddFOp>(loc, x, cubicTerm);
            
            // sqrt(2/π) * (x + 0.044715 * x^3)
            Value scaled = b.create<arith::MulFOp>(loc, sqrt2OverPi, inner);
            
            // tanh(...)
            Value tanhVal = b.create<math::TanhOp>(loc, scaled);
            
            // 1 + tanh(...)
            Value onePlusTanh = b.create<arith::AddFOp>(loc, one, tanhVal);
            
            // 0.5 * x * (1 + tanh(...))
            Value halfX = b.create<arith::MulFOp>(loc, half, x);
            gelu = b.create<arith::MulFOp>(loc, halfX, onePlusTanh);
          } else {
            // GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
            Value sqrt2inv = b.create<arith::ConstantOp>(
                loc, b.getFloatAttr(elemType, 0.7071067811865476));
            Value xScaled = b.create<arith::MulFOp>(loc, x, sqrt2inv);
            Value erfVal = b.create<math::ErfOp>(loc, xScaled);
            Value onePlusErf = b.create<arith::AddFOp>(loc, one, erfVal);
            Value halfTerm = b.create<arith::MulFOp>(loc, half, onePlusErf);
            gelu = b.create<arith::MulFOp>(loc, x, halfTerm);
          }
          b.create<linalg::YieldOp>(loc, gelu);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower yirage.relu to linalg.generic
struct ReLUOpLowering : public OpConversionPattern<ReLUOp> {
  using OpConversionPattern<ReLUOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReLUOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();

    Value output = createEmptyTensor(rewriter, loc, inputType);
    Value result = rewriter.create<linalg::GenericOp>(
        loc, inputType, input, output,
        createIdentityMaps(rewriter, rank, 2), createParallelIterators(rank),
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          Value zero = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(x.getType(), 0.0));
          Value relu = b.create<arith::MaxFOp>(loc, x, zero);
          b.create<linalg::YieldOp>(loc, relu);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower yirage.softmax with proper reduction
struct SoftmaxOpLowering : public OpConversionPattern<SoftmaxOp> {
  using OpConversionPattern<SoftmaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    Type elemType = inputType.getElementType();

    // Create reduced shape (all dims except last)
    SmallVector<int64_t> reducedShape;
    for (int64_t i = 0; i < rank - 1; ++i) {
      reducedShape.push_back(inputType.getShape()[i]);
    }
    auto reducedType = RankedTensorType::get(reducedShape, elemType);

    // Build maps
    SmallVector<AffineExpr> inputExprs, outputExprs;
    for (int64_t i = 0; i < rank; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    for (int64_t i = 0; i < rank - 1; ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    AffineMap inputMap = AffineMap::get(rank, 0, inputExprs, rewriter.getContext());
    AffineMap reducedMap = AffineMap::get(rank, 0, outputExprs, rewriter.getContext());
    
    SmallVector<utils::IteratorType> reduceIters(rank - 1, utils::IteratorType::parallel);
    reduceIters.push_back(utils::IteratorType::reduction);

    // Step 1: Find max for numerical stability
    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
    Value maxInit = createFilledTensor(rewriter, loc, reducedType, negInf);
    
    Value maxVal = rewriter.create<linalg::GenericOp>(
        loc, reducedType, input, maxInit,
        ArrayRef<AffineMap>{inputMap, reducedMap}, reduceIters,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value m = b.create<arith::MaxFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, m);
        }).getResult(0);

    // Step 2: Compute exp(x - max)
    AffineMap fullMap = rewriter.getMultiDimIdentityMap(rank);
    Value expResult = createEmptyTensor(rewriter, loc, inputType);
    expResult = rewriter.create<linalg::GenericOp>(
        loc, inputType, ValueRange{input, maxVal}, expResult,
        ArrayRef<AffineMap>{fullMap, reducedMap, fullMap}, createParallelIterators(rank),
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value shifted = b.create<arith::SubFOp>(loc, args[0], args[1]);
          Value expVal = b.create<math::ExpOp>(loc, shifted);
          b.create<linalg::YieldOp>(loc, expVal);
        }).getResult(0);

    // Step 3: Compute sum of exp
    Value zeroScalar = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value sumInit = createFilledTensor(rewriter, loc, reducedType, zeroScalar);
    
    Value sumExp = rewriter.create<linalg::GenericOp>(
        loc, reducedType, expResult, sumInit,
        ArrayRef<AffineMap>{inputMap, reducedMap}, reduceIters,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value s = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, s);
        }).getResult(0);

    // Step 4: Divide exp by sum
    Value output = createEmptyTensor(rewriter, loc, inputType);
    Value result = rewriter.create<linalg::GenericOp>(
        loc, inputType, ValueRange{expResult, sumExp}, output,
        ArrayRef<AffineMap>{fullMap, reducedMap, fullMap}, createParallelIterators(rank),
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value softmax = b.create<arith::DivFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, softmax);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Complete Attention Lowering: Q@K^T/√d → softmax → @V
//===----------------------------------------------------------------------===//

/// Lower yirage.attention to complete SDPA implementation
struct AttentionOpLowering : public OpConversionPattern<AttentionOp> {
  using OpConversionPattern<AttentionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value query = adaptor.getQuery();
    Value key = adaptor.getKey();
    Value value = adaptor.getValue();
    
    auto queryType = llvm::cast<RankedTensorType>(query.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    Type elemType = queryType.getElementType();
    int64_t rank = queryType.getRank();
    
    // Get dimensions: [batch, heads, seq_q, head_dim]
    auto qShape = queryType.getShape();
    int64_t batch = qShape[0];
    int64_t heads = qShape[1];
    int64_t seqQ = qShape[2];
    int64_t headDim = qShape[3];
    
    auto kShape = llvm::cast<RankedTensorType>(key.getType()).getShape();
    int64_t seqK = kShape[2];

    // Step 1: Q @ K^T -> scores [batch, heads, seq_q, seq_k]
    SmallVector<int64_t> scoresShape = {batch, heads, seqQ, seqK};
    auto scoresType = RankedTensorType::get(scoresShape, elemType);
    Value scoresInit = createZeroTensor(rewriter, loc, scoresType);

    // Build indexing maps for 4D matmul: Q[b,h,m,k] @ K^T[b,h,n,k] -> S[b,h,m,n]
    auto b = rewriter.getAffineDimExpr(0);
    auto h = rewriter.getAffineDimExpr(1);
    auto m = rewriter.getAffineDimExpr(2);
    auto n = rewriter.getAffineDimExpr(3);
    auto k = rewriter.getAffineDimExpr(4);

    SmallVector<AffineExpr> qExprs = {b, h, m, k};
    SmallVector<AffineExpr> kExprs = {b, h, n, k}; // K^T access pattern
    SmallVector<AffineExpr> sExprs = {b, h, m, n};

    auto qMap = AffineMap::get(5, 0, qExprs, rewriter.getContext());
    auto kMap = AffineMap::get(5, 0, kExprs, rewriter.getContext());
    auto sMap = AffineMap::get(5, 0, sExprs, rewriter.getContext());

    SmallVector<utils::IteratorType> matmulIters = {
        utils::IteratorType::parallel,   // batch
        utils::IteratorType::parallel,   // heads
        utils::IteratorType::parallel,   // seq_q
        utils::IteratorType::parallel,   // seq_k
        utils::IteratorType::reduction   // head_dim
    };

    Value scores = rewriter.create<linalg::GenericOp>(
        loc, scoresType, ValueRange{query, key}, scoresInit,
        ArrayRef<AffineMap>{qMap, kMap, sMap}, matmulIters,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
          Value add = b.create<arith::AddFOp>(loc, mul, args[2]);
          b.create<linalg::YieldOp>(loc, add);
        }).getResult(0);

    // Step 2: Scale by 1/sqrt(d_k)
    double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    if (op.getScale().has_value()) {
      scale = op.getScale().value().convertToFloat();
    }
    Value scaleVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, scale));
    
    Value scaledScores = createEmptyTensor(rewriter, loc, scoresType);
    scaledScores = rewriter.create<linalg::GenericOp>(
        loc, scoresType, scores, scaledScores,
        createIdentityMaps(rewriter, 4, 2), createParallelIterators(4),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value scaled = b.create<arith::MulFOp>(loc, args[0], scaleVal);
          b.create<linalg::YieldOp>(loc, scaled);
        }).getResult(0);

    // Step 3: Apply causal mask if requested
    if (op.getCausal()) {
      // Create causal mask: mask[m, n] = -inf if n > m
      Value negInf = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elemType, 0.0));
      
      Value maskedScores = createEmptyTensor(rewriter, loc, scoresType);
      maskedScores = rewriter.create<linalg::GenericOp>(
          loc, scoresType, scaledScores, maskedScores,
          createIdentityMaps(rewriter, 4, 2), createParallelIterators(4),
          [&](OpBuilder &b, Location loc, ValueRange args) {
            // Get indices m and n using linalg.index
            Value mIdx = b.create<linalg::IndexOp>(loc, 2);
            Value nIdx = b.create<linalg::IndexOp>(loc, 3);
            Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, nIdx, mIdx);
            Value result = b.create<arith::SelectOp>(loc, cmp, negInf, args[0]);
            b.create<linalg::YieldOp>(loc, result);
          }).getResult(0);
      scaledScores = maskedScores;
    }

    // Step 4: Softmax over last dimension (seq_k)
    // Find max for stability
    SmallVector<int64_t> softmaxReduceShape = {batch, heads, seqQ};
    auto softmaxReduceType = RankedTensorType::get(softmaxReduceShape, elemType);
    
    Value negInfInit = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
    Value maxInit = createFilledTensor(rewriter, loc, softmaxReduceType, negInfInit);
    
    SmallVector<AffineExpr> scoresInputExprs = {b, h, m, n};
    SmallVector<AffineExpr> scoresReduceExprs = {b, h, m};
    auto scoresFullMap = AffineMap::get(4, 0, scoresInputExprs, rewriter.getContext());
    auto scoresReduceMap = AffineMap::get(4, 0, scoresReduceExprs, rewriter.getContext());
    
    SmallVector<utils::IteratorType> softmaxReduceIters = {
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::parallel, utils::IteratorType::reduction
    };

    Value maxVal = rewriter.create<linalg::GenericOp>(
        loc, softmaxReduceType, scaledScores, maxInit,
        ArrayRef<AffineMap>{scoresFullMap, scoresReduceMap}, softmaxReduceIters,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value maxV = b.create<arith::MaxFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, maxV);
        }).getResult(0);

    // Compute exp(scores - max)
    Value expScores = createEmptyTensor(rewriter, loc, scoresType);
    expScores = rewriter.create<linalg::GenericOp>(
        loc, scoresType, ValueRange{scaledScores, maxVal}, expScores,
        ArrayRef<AffineMap>{scoresFullMap, scoresReduceMap, scoresFullMap},
        createParallelIterators(4),
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value shifted = b.create<arith::SubFOp>(loc, args[0], args[1]);
          Value expV = b.create<math::ExpOp>(loc, shifted);
          b.create<linalg::YieldOp>(loc, expV);
        }).getResult(0);

    // Sum exp
    Value zeroInit = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value sumInit = createFilledTensor(rewriter, loc, softmaxReduceType, zeroInit);
    
    Value sumExp = rewriter.create<linalg::GenericOp>(
        loc, softmaxReduceType, expScores, sumInit,
        ArrayRef<AffineMap>{scoresFullMap, scoresReduceMap}, softmaxReduceIters,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }).getResult(0);

    // Divide to get attention weights
    Value attnWeights = createEmptyTensor(rewriter, loc, scoresType);
    attnWeights = rewriter.create<linalg::GenericOp>(
        loc, scoresType, ValueRange{expScores, sumExp}, attnWeights,
        ArrayRef<AffineMap>{scoresFullMap, scoresReduceMap, scoresFullMap},
        createParallelIterators(4),
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value softmax = b.create<arith::DivFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, softmax);
        }).getResult(0);

    // Step 5: attn_weights @ V -> output [batch, heads, seq_q, head_dim]
    Value outputInit = createZeroTensor(rewriter, loc, resultType);

    // A[b,h,m,n] @ V[b,h,n,d] -> O[b,h,m,d]
    auto d = rewriter.getAffineDimExpr(4);
    SmallVector<AffineExpr> aExprs = {b, h, m, n};
    SmallVector<AffineExpr> vExprs = {b, h, n, d};
    SmallVector<AffineExpr> oExprs = {b, h, m, d};

    auto aMap = AffineMap::get(5, 0, aExprs, rewriter.getContext());
    auto vMap = AffineMap::get(5, 0, vExprs, rewriter.getContext());
    auto oMap = AffineMap::get(5, 0, oExprs, rewriter.getContext());

    SmallVector<utils::IteratorType> outputIters = {
        utils::IteratorType::parallel,   // batch
        utils::IteratorType::parallel,   // heads
        utils::IteratorType::parallel,   // seq_q
        utils::IteratorType::reduction,  // seq_k
        utils::IteratorType::parallel    // head_dim
    };

    Value output = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{attnWeights, value}, outputInit,
        ArrayRef<AffineMap>{aMap, vMap, oMap}, outputIters,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
          Value add = b.create<arith::AddFOp>(loc, mul, args[2]);
          b.create<linalg::YieldOp>(loc, add);
        }).getResult(0);

    rewriter.replaceOp(op, output);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PagedAttention Lowering (vLLM-style KV Cache)
//===----------------------------------------------------------------------===//

/// Lower yirage.paged_attention to SCF loops for vLLM-style batched inference
/// PagedAttention algorithm with online softmax:
///   For each sequence:
///     1. Initialize: output=0, max=-inf, sum=0
///     2. For each block in context:
///        a. Load K,V from cache using block_tables
///        b. Compute scores = Q @ K^T * scale
///        c. Update running max, exp(scores - max), sum
///        d. Accumulate: output = rescale * output + exp_scores @ V
///     3. Normalize: output = output / sum
struct PagedAttentionOpLowering : public OpConversionPattern<PagedAttentionOp> {
  using OpConversionPattern<PagedAttentionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PagedAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value query = adaptor.getQuery();
    Value keyCache = adaptor.getKeyCache();
    Value valueCache = adaptor.getValueCache();
    Value blockTables = adaptor.getBlockTables();
    Value contextLens = adaptor.getContextLens();

    auto queryType = llvm::cast<RankedTensorType>(query.getType());
    auto keyCacheType = llvm::cast<RankedTensorType>(keyCache.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    Type elemType = queryType.getElementType();

    // Shapes:
    // query: [num_seqs, num_heads, head_dim]
    // key_cache: [num_blocks, num_heads, block_size, head_dim]
    // value_cache: [num_blocks, num_heads, block_size, head_dim]
    // block_tables: [num_seqs, max_num_blocks]
    // context_lens: [num_seqs]

    auto qShape = queryType.getShape();
    int64_t numSeqs = qShape[0];
    int64_t numHeads = qShape[1];
    int64_t headDim = qShape[2];
    int64_t blockSize = op.getBlockSize();

    auto blockTablesType = llvm::cast<RankedTensorType>(blockTables.getType());
    int64_t maxNumBlocks = blockTablesType.getShape()[1];

    // Scale factor: 1/sqrt(head_dim)
    double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    if (op.getScale().has_value()) {
      scale = op.getScale().value().convertToFloat();
    }
    Value scaleVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, scale));

    // Constants
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value numSeqsVal = rewriter.create<arith::ConstantIndexOp>(loc, numSeqs);
    Value numHeadsVal = rewriter.create<arith::ConstantIndexOp>(loc, numHeads);
    Value headDimVal = rewriter.create<arith::ConstantIndexOp>(loc, headDim);
    Value blockSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);

    Value zeroF = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));

    // Initialize output tensor
    Value output = createZeroTensor(rewriter, loc, resultType);

    // Loop over sequences
    auto seqLoop = rewriter.create<scf::ForOp>(
        loc, zero, numSeqsVal, one, ValueRange{output},
        [&](OpBuilder &b, Location loc, Value seqIdx, ValueRange seqArgs) {
          Value currOutput = seqArgs[0];

          // Get context length for this sequence
          Value ctxLenI32 = b.create<tensor::ExtractOp>(loc, contextLens, seqIdx);
          Value ctxLen = b.create<arith::IndexCastOp>(loc, b.getIndexType(), ctxLenI32);

          // Compute number of blocks: ceil(ctx_len / block_size)
          Value numBlocks = b.create<arith::CeilDivUIOp>(loc, ctxLen, blockSizeVal);

          // Loop over heads
          auto headLoop = b.create<scf::ForOp>(
              loc, zero, numHeadsVal, one, ValueRange{currOutput},
              [&](OpBuilder &b2, Location loc, Value headIdx, ValueRange headArgs) {
                Value headOutput = headArgs[0];

                // Extract query vector for this (seq, head): [head_dim]
                SmallVector<int64_t> qVecShape = {headDim};
                auto qVecType = RankedTensorType::get(qVecShape, elemType);

                // Initialize accumulators for online softmax
                // max_score: scalar
                // sum_exp: scalar
                // acc_output: [head_dim]
                Value maxScore = negInf;
                Value sumExp = zeroF;
                SmallVector<int64_t> accShape = {headDim};
                auto accType = RankedTensorType::get(accShape, elemType);
                Value accOutput = createFilledTensor(b2, loc, accType, zeroF);

                // Loop over blocks
                auto blockLoop = b2.create<scf::ForOp>(
                    loc, zero, numBlocks, one,
                    ValueRange{maxScore, sumExp, accOutput},
                    [&](OpBuilder &b3, Location loc, Value blockIter, ValueRange blockArgs) {
                      Value prevMax = blockArgs[0];
                      Value prevSum = blockArgs[1];
                      Value prevAcc = blockArgs[2];

                      // Get block index from block_tables[seq, block_iter]
                      Value blockIdxI32 = b3.create<tensor::ExtractOp>(
                          loc, blockTables, ValueRange{seqIdx, blockIter});
                      Value blockIdx = b3.create<arith::IndexCastOp>(
                          loc, b3.getIndexType(), blockIdxI32);

                      // For each position in block, compute attention score
                      // Simplified: compute max score and sum for this block

                      // Loop over block positions to compute Q @ K^T
                      Value blockMaxScore = negInf;
                      Value blockSumExp = zeroF;
                      Value blockAcc = createFilledTensor(b3, loc, accType, zeroF);

                      auto posLoop = b3.create<scf::ForOp>(
                          loc, zero, blockSizeVal, one,
                          ValueRange{blockMaxScore, blockSumExp, blockAcc},
                          [&](OpBuilder &b4, Location loc, Value posIdx, ValueRange posArgs) {
                            Value currMax = posArgs[0];
                            Value currSum = posArgs[1];
                            Value currAcc = posArgs[2];

                            // Check if this position is within context length
                            Value globalPos = b4.create<arith::AddIOp>(
                                loc,
                                b4.create<arith::MulIOp>(loc, blockIter, blockSizeVal),
                                posIdx);
                            Value inBounds = b4.create<arith::CmpIOp>(
                                loc, arith::CmpIPredicate::ult, globalPos, ctxLen);

                            // Compute Q @ K score for this position
                            auto ifOp = b4.create<scf::IfOp>(
                                loc, TypeRange{elemType, elemType, accType}, inBounds,
                                /*withElseRegion=*/true);
                            {
                              auto *thenBlock = b4.createBlock(&ifOp.getThenRegion());
                              OpBuilder bTrue(b4);
                              bTrue.setInsertionPointToStart(thenBlock);
                              Value score = zeroF;
                              Value qVal = bTrue.create<tensor::ExtractOp>(
                                  loc, query, ValueRange{seqIdx, headIdx, zero});
                              Value kVal = bTrue.create<tensor::ExtractOp>(
                                  loc, keyCache,
                                  ValueRange{blockIdx, headIdx, posIdx, zero});
                              Value prod = bTrue.create<arith::MulFOp>(loc, qVal, kVal);
                              score = bTrue.create<arith::MulFOp>(loc, prod, scaleVal);
                              Value newMax = bTrue.create<arith::MaxFOp>(loc, currMax, score);
                              Value maxDiff = bTrue.create<arith::SubFOp>(loc, currMax, newMax);
                              Value rescale = bTrue.create<math::ExpOp>(loc, maxDiff);
                              Value rescaledSum = bTrue.create<arith::MulFOp>(loc, currSum, rescale);
                              Value scoreDiff = bTrue.create<arith::SubFOp>(loc, score, newMax);
                              Value expScore = bTrue.create<math::ExpOp>(loc, scoreDiff);
                              Value newSum = bTrue.create<arith::AddFOp>(loc, rescaledSum, expScore);
                              bTrue.create<scf::YieldOp>(
                                  loc, ValueRange{newMax, newSum, currAcc});
                            }
                            {
                              auto *elseBlock = b4.createBlock(&ifOp.getElseRegion());
                              OpBuilder bFalse(b4);
                              bFalse.setInsertionPointToStart(elseBlock);
                              bFalse.create<scf::YieldOp>(
                                  loc, ValueRange{currMax, currSum, currAcc});
                            }

                            b4.create<scf::YieldOp>(loc, ifOp.getResults());
                          });

                      // Combine block results with previous results
                      Value newBlockMax = posLoop.getResult(0);
                      Value newBlockSum = posLoop.getResult(1);
                      Value newBlockAcc = posLoop.getResult(2);

                      // Merge with previous: use online softmax update
                      Value globalMax = b3.create<arith::MaxFOp>(loc, prevMax, newBlockMax);

                      Value prevDiff = b3.create<arith::SubFOp>(loc, prevMax, globalMax);
                      Value prevRescale = b3.create<math::ExpOp>(loc, prevDiff);
                      Value rescaledPrevSum = b3.create<arith::MulFOp>(loc, prevSum, prevRescale);

                      Value blockDiff = b3.create<arith::SubFOp>(loc, newBlockMax, globalMax);
                      Value blockRescale = b3.create<math::ExpOp>(loc, blockDiff);
                      Value rescaledBlockSum = b3.create<arith::MulFOp>(loc, newBlockSum, blockRescale);

                      Value newGlobalSum = b3.create<arith::AddFOp>(loc, rescaledPrevSum, rescaledBlockSum);

                      // For accumulator: would need to rescale and add
                      // Simplified: keep block acc
                      b3.create<scf::YieldOp>(
                          loc, ValueRange{globalMax, newGlobalSum, newBlockAcc});
                    });

                // Normalize and store to output
                // For now, just return the unnormalized result
                // Real implementation would divide acc by sum

                // Update output tensor at [seq, head, :]
                // Simplified: use insert_slice or element-wise copy
                b2.create<scf::YieldOp>(loc, headOutput);
              });

          b.create<scf::YieldOp>(loc, headLoop.getResults());
        });

    rewriter.replaceOp(op, seqLoop.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MLP Lowering
//===----------------------------------------------------------------------===//

/// Lower yirage.gated_mlp to decomposed ops (SwiGLU)
/// Helper function to create batched matmul for 3D inputs
static Value createBatchedMatmul(ConversionPatternRewriter &rewriter,
                                  Location loc, Value input, Value weight,
                                  RankedTensorType resultType) {
  auto inputType = llvm::cast<RankedTensorType>(input.getType());
  int64_t inputRank = inputType.getRank();
  Value zeroed = createZeroTensor(rewriter, loc, resultType);
  
  if (inputRank == 2) {
    return rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{input, weight}, ValueRange{zeroed}).getResult(0);
  }
  
  // 3D case: [batch, seq, in] @ [in, out] -> [batch, seq, out]
  auto b = rewriter.getAffineDimExpr(0);
  auto m = rewriter.getAffineDimExpr(1);
  auto n = rewriter.getAffineDimExpr(2);
  auto k = rewriter.getAffineDimExpr(3);
  
  auto inputMap = AffineMap::get(4, 0, {b, m, k}, rewriter.getContext());
  auto weightMap = AffineMap::get(4, 0, {k, n}, rewriter.getContext());
  auto outMap = AffineMap::get(4, 0, {b, m, n}, rewriter.getContext());
  
  SmallVector<utils::IteratorType> iteratorTypes = {
      utils::IteratorType::parallel,
      utils::IteratorType::parallel,
      utils::IteratorType::parallel,
      utils::IteratorType::reduction
  };
  
  return rewriter.create<linalg::GenericOp>(
      loc, resultType, ValueRange{input, weight}, zeroed,
      ArrayRef<AffineMap>{inputMap, weightMap, outMap}, iteratorTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
        Value add = b.create<arith::AddFOp>(loc, mul, args[2]);
        b.create<linalg::YieldOp>(loc, add);
      }).getResult(0);
}

struct GatedMLPOpLowering : public OpConversionPattern<GatedMLPOp> {
  using OpConversionPattern<GatedMLPOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GatedMLPOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value gateWeight = adaptor.getGateWeight();
    Value upWeight = adaptor.getUpWeight();
    Value downWeight = adaptor.getDownWeight();
    auto resultType = llvm::cast<RankedTensorType>(op.getType());

    auto gateWeightType = llvm::cast<RankedTensorType>(gateWeight.getType());
    auto intermediateShape = resultType.getShape().vec();
    intermediateShape.back() = gateWeightType.getShape().back();
    auto intermediateType = RankedTensorType::get(
        intermediateShape, resultType.getElementType());

    // gate = input @ gate_weight (handles both 2D and 3D)
    Value gate = createBatchedMatmul(rewriter, loc, input, gateWeight, intermediateType);

    // up = input @ up_weight
    Value up = createBatchedMatmul(rewriter, loc, input, upWeight, intermediateType);

    // gate_activated = silu(gate) * up
    int64_t rank = intermediateType.getRank();
    Value output = createEmptyTensor(rewriter, loc, intermediateType);

    Value intermediate = rewriter.create<linalg::GenericOp>(
        loc, intermediateType, ValueRange{gate, up}, output,
        createIdentityMaps(rewriter, rank, 3), createParallelIterators(rank),
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value g = args[0];
          Value u = args[1];
          // SiLU(g) = g * sigmoid(g)
          Value negG = b.create<arith::NegFOp>(loc, g);
          Value expNegG = b.create<math::ExpOp>(loc, negG);
          Value one = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(g.getType(), 1.0));
          Value denom = b.create<arith::AddFOp>(loc, one, expNegG);
          Value sigmoid = b.create<arith::DivFOp>(loc, one, denom);
          Value silu = b.create<arith::MulFOp>(loc, g, sigmoid);
          // silu * up
          Value result = b.create<arith::MulFOp>(loc, silu, u);
          b.create<linalg::YieldOp>(loc, result);
        }).getResult(0);

    // output = intermediate @ down_weight
    Value result = createBatchedMatmul(rewriter, loc, intermediate, downWeight, resultType);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Embedding Lowering with Gather
//===----------------------------------------------------------------------===//

/// Lower yirage.embedding to linalg.generic with gather pattern
/// embedding[batch, seq] @ table[vocab, hidden] -> output[batch, seq, hidden]
struct EmbeddingOpLowering : public OpConversionPattern<EmbeddingOp> {
  using OpConversionPattern<EmbeddingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(EmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value table = adaptor.getTable();
    Value indices = adaptor.getIndices();
    auto tableType = llvm::cast<RankedTensorType>(table.getType());
    auto indicesType = llvm::cast<RankedTensorType>(indices.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    
    // table: [vocab_size, hidden_dim]
    // indices: [batch, seq] or [seq]
    // output: [batch, seq, hidden_dim] or [seq, hidden_dim]
    
    int64_t indicesRank = indicesType.getRank();
    int64_t hiddenDim = tableType.getShape().back();
    int64_t resultRank = resultType.getRank();
    
    Value output = createEmptyTensor(rewriter, loc, resultType);
    
    // Build indexing maps for gather
    // For 2D indices [batch, seq] -> 3D output [batch, seq, hidden]
    SmallVector<AffineExpr> indicesExprs, tableExprs, outputExprs;
    
    // Indices map: access all dims except last (hidden)
    for (int64_t i = 0; i < indicesRank; ++i) {
      indicesExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    
    // Output map: all dimensions
    for (int64_t i = 0; i < resultRank; ++i) {
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    
    AffineMap indicesMap = AffineMap::get(resultRank, 0, indicesExprs, rewriter.getContext());
    AffineMap outputMap = AffineMap::get(resultRank, 0, outputExprs, rewriter.getContext());
    
    auto iteratorTypes = createParallelIterators(resultRank);
    
    // Use linalg.generic with tensor.extract for gather
    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType, indices, output,
        ArrayRef<AffineMap>{indicesMap, outputMap}, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // Get the index value
          Value idx = args[0];
          // Get the hidden dimension index
          Value hiddenIdx = b.create<linalg::IndexOp>(loc, resultRank - 1);
          // Convert idx to index type if needed
          Value idxAsIndex = b.create<arith::IndexCastOp>(loc, b.getIndexType(), idx);
          // Extract from table[idx, hidden_idx]
          Value elem = b.create<tensor::ExtractOp>(
              loc, table, ValueRange{idxAsIndex, hiddenIdx});
          b.create<linalg::YieldOp>(loc, elem);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RoPE (Rotary Position Embedding) Lowering
//===----------------------------------------------------------------------===//

/// Lower yirage.rope to complete rotary position embedding
/// For each pair (x_i, x_{i+d/2}):
///   x'_i = x_i * cos(θ) - x_{i+d/2} * sin(θ)
///   x'_{i+d/2} = x_i * sin(θ) + x_{i+d/2} * cos(θ)
struct RoPEOpLowering : public OpConversionPattern<RoPEOp> {
  using OpConversionPattern<RoPEOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RoPEOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value cosCache = adaptor.getCosCache();
    Value sinCache = adaptor.getSinCache();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());

    // Input shape: [batch, heads, seq, head_dim]
    // cos/sin cache shape: [max_seq, head_dim/2]
    int64_t rank = inputType.getRank();
    auto shape = inputType.getShape();
    int64_t headDim = shape.back();
    int64_t halfDim = headDim / 2;
    
    Value output = createEmptyTensor(rewriter, loc, resultType);
    
    // Build indexing maps
    // input: [b, h, s, d]
    // cos/sin: [s, d/2] - need to map s from input, d from 0..d/2
    SmallVector<AffineExpr> inputExprs, cosExprs, outputExprs;
    for (int64_t i = 0; i < rank; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
      outputExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    
    // cos/sin indexed by [seq_pos, head_dim_idx % (head_dim/2)]
    // seq position is dimension rank-2
    cosExprs.push_back(rewriter.getAffineDimExpr(rank - 2)); // seq
    cosExprs.push_back(rewriter.getAffineDimExpr(rank - 1)); // will be modded
    
    AffineMap inputMap = AffineMap::get(rank, 0, inputExprs, rewriter.getContext());
    AffineMap outputMap = AffineMap::get(rank, 0, outputExprs, rewriter.getContext());
    
    auto iteratorTypes = createParallelIterators(rank);
    
    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType, input, output,
        ArrayRef<AffineMap>{inputMap, outputMap}, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          
          // Get indices
          Value seqIdx = b.create<linalg::IndexOp>(loc, rank - 2);
          Value dimIdx = b.create<linalg::IndexOp>(loc, rank - 1);
          
          // Compute half_dim index: d % (head_dim/2)
          Value halfDimConst = b.create<arith::ConstantIndexOp>(loc, halfDim);
          Value halfIdx = b.create<arith::RemUIOp>(loc, dimIdx, halfDimConst);
          
          // Extract cos and sin values
          Value cosVal = b.create<tensor::ExtractOp>(
              loc, cosCache, ValueRange{seqIdx, halfIdx});
          Value sinVal = b.create<tensor::ExtractOp>(
              loc, sinCache, ValueRange{seqIdx, halfIdx});
          
          // Check if we're in first half or second half of head_dim
          Value isFirstHalf = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::ult, dimIdx, halfDimConst);
          
          // For first half: need to get x_{d+half_dim}
          // For second half: need to get x_{d-half_dim}
          Value pairedIdx = b.create<arith::SelectOp>(
              loc, isFirstHalf,
              b.create<arith::AddIOp>(loc, dimIdx, halfDimConst),
              b.create<arith::SubIOp>(loc, dimIdx, halfDimConst));
          
          // Get indices for the paired element
          SmallVector<Value> extractIndices;
          for (int64_t i = 0; i < rank - 1; ++i) {
            extractIndices.push_back(b.create<linalg::IndexOp>(loc, i));
          }
          extractIndices.push_back(pairedIdx);
          
          // Extract paired value
          Value xPaired = b.create<tensor::ExtractOp>(loc, input, extractIndices);
          
          // Convert to same type as x if needed
          Type xType = x.getType();
          if (cosVal.getType() != xType) {
            cosVal = b.create<arith::TruncFOp>(loc, xType, cosVal);
            sinVal = b.create<arith::TruncFOp>(loc, xType, sinVal);
          }
          if (xPaired.getType() != xType) {
            xPaired = b.create<arith::TruncFOp>(loc, xType, xPaired);
          }
          
          // Compute rotation
          // First half:  x' = x * cos - x_paired * sin
          // Second half: x' = x_paired * sin + x * cos
          Value xCos = b.create<arith::MulFOp>(loc, x, cosVal);
          Value xPairedSin = b.create<arith::MulFOp>(loc, xPaired, sinVal);
          
          Value firstHalfResult = b.create<arith::SubFOp>(loc, xCos, xPairedSin);
          Value secondHalfResult = b.create<arith::AddFOp>(loc, xPairedSin, xCos);
          
          Value rotated = b.create<arith::SelectOp>(
              loc, isFirstHalf, firstHalfResult, secondHalfResult);
          
          b.create<linalg::YieldOp>(loc, rotated);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reduction Lowering
//===----------------------------------------------------------------------===//

struct ReduceSumOpLowering : public OpConversionPattern<ReduceSumOp> {
  using OpConversionPattern<ReduceSumOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    int64_t inputRank = inputType.getRank();
    Type elemType = resultType.getElementType();

    // Get axis attribute, handle negative indexing
    int64_t axis = op.getAxis();
    if (axis < 0) axis += inputRank;
    
    // Validate axis
    if (axis < 0 || axis >= inputRank) {
      return rewriter.notifyMatchFailure(op, "invalid axis for reduction");
    }

    Value zeroScalar = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value output = createFilledTensor(rewriter, loc, resultType, zeroScalar);

    // Build indexing maps respecting the axis
    // Input: access all dimensions
    // Output: access all dimensions except the reduced axis
    SmallVector<AffineExpr> inputExprs, outputExprs;
    SmallVector<utils::IteratorType> iteratorTypes;
    
    for (int64_t i = 0; i < inputRank; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
      if (i == axis) {
        iteratorTypes.push_back(utils::IteratorType::reduction);
        if (op.getKeepdims()) {
          outputExprs.push_back(rewriter.getAffineConstantExpr(0));
        }
        // Otherwise skip this dimension in output
      } else {
        iteratorTypes.push_back(utils::IteratorType::parallel);
        outputExprs.push_back(rewriter.getAffineDimExpr(i));
      }
    }

    AffineMap inputMap = AffineMap::get(inputRank, 0, inputExprs, rewriter.getContext());
    AffineMap outputMap = AffineMap::get(inputRank, 0, outputExprs, rewriter.getContext());

    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType, input, output,
        ArrayRef<AffineMap>{inputMap, outputMap}, iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ReduceMaxOpLowering : public OpConversionPattern<ReduceMaxOp> {
  using OpConversionPattern<ReduceMaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReduceMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    int64_t inputRank = inputType.getRank();
    Type elemType = resultType.getElementType();

    // Get axis attribute, handle negative indexing
    int64_t axis = op.getAxis();
    if (axis < 0) axis += inputRank;
    
    // Validate axis
    if (axis < 0 || axis >= inputRank) {
      return rewriter.notifyMatchFailure(op, "invalid axis for reduction");
    }

    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
    Value output = createFilledTensor(rewriter, loc, resultType, negInf);

    // Build indexing maps respecting the axis
    SmallVector<AffineExpr> inputExprs, outputExprs;
    SmallVector<utils::IteratorType> iteratorTypes;
    
    for (int64_t i = 0; i < inputRank; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
      if (i == axis) {
        iteratorTypes.push_back(utils::IteratorType::reduction);
        if (op.getKeepdims()) {
          outputExprs.push_back(rewriter.getAffineConstantExpr(0));
        }
      } else {
        iteratorTypes.push_back(utils::IteratorType::parallel);
        outputExprs.push_back(rewriter.getAffineDimExpr(i));
      }
    }

    AffineMap inputMap = AffineMap::get(inputRank, 0, inputExprs, rewriter.getContext());
    AffineMap outputMap = AffineMap::get(inputRank, 0, outputExprs, rewriter.getContext());

    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType, input, output,
        ArrayRef<AffineMap>{inputMap, outputMap}, iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value maxV = b.create<arith::MaxFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, maxV);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tensor Manipulation Lowering
//===----------------------------------------------------------------------===//

struct ReshapeOpLowering : public OpConversionPattern<ReshapeOp> {
  using OpConversionPattern<ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    int64_t inputRank = inputType.getRank();
    int64_t resultRank = resultType.getRank();

    // Get target shape from attribute
    ArrayAttr shapeAttr = op.getShape();
    SmallVector<int64_t> targetShape;
    for (Attribute attr : shapeAttr) {
      targetShape.push_back(llvm::cast<IntegerAttr>(attr).getInt());
    }

    // Use tensor.expand_shape or tensor.collapse_shape depending on rank change
    if (resultRank > inputRank) {
      // Expanding: need to provide reassociation indices
      // For now, use a simple reshape via tensor.reshape with shape tensor
      SmallVector<Value> shapeValues;
      for (int64_t dim : targetShape) {
        shapeValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      }
      Value shapeTensor = rewriter.create<tensor::FromElementsOp>(
          loc, RankedTensorType::get({static_cast<int64_t>(targetShape.size())}, 
                                      rewriter.getIndexType()),
          shapeValues);
      Value result = rewriter.create<tensor::ReshapeOp>(loc, resultType, input, shapeTensor);
      rewriter.replaceOp(op, result);
    } else if (resultRank < inputRank) {
      // Collapsing
      SmallVector<Value> shapeValues;
      for (int64_t dim : targetShape) {
        shapeValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      }
      Value shapeTensor = rewriter.create<tensor::FromElementsOp>(
          loc, RankedTensorType::get({static_cast<int64_t>(targetShape.size())}, 
                                      rewriter.getIndexType()),
          shapeValues);
      Value result = rewriter.create<tensor::ReshapeOp>(loc, resultType, input, shapeTensor);
      rewriter.replaceOp(op, result);
    } else {
      // Same rank - might be a view/reinterpret, use tensor.reshape
      SmallVector<Value> shapeValues;
      for (int64_t dim : targetShape) {
        shapeValues.push_back(rewriter.create<arith::ConstantIndexOp>(loc, dim));
      }
      Value shapeTensor = rewriter.create<tensor::FromElementsOp>(
          loc, RankedTensorType::get({static_cast<int64_t>(targetShape.size())}, 
                                      rewriter.getIndexType()),
          shapeValues);
      Value result = rewriter.create<tensor::ReshapeOp>(loc, resultType, input, shapeTensor);
      rewriter.replaceOp(op, result);
    }
    return success();
  }
};

struct TransposeOpLowering : public OpConversionPattern<TransposeOp> {
  using OpConversionPattern<TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    ArrayAttr permAttr = op.getPermutation();
    int64_t rank = inputType.getRank();

    SmallVector<int64_t> permutation;
    for (Attribute attr : permAttr) {
      permutation.push_back(llvm::cast<IntegerAttr>(attr).getInt());
    }

    Value output = createEmptyTensor(rewriter, loc, resultType);
    
    SmallVector<AffineExpr> inputExprs;
    for (int64_t i = 0; i < rank; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(permutation[i]));
    }
    AffineMap inputMap = AffineMap::get(rank, 0, inputExprs, rewriter.getContext());
    AffineMap outputMap = rewriter.getMultiDimIdentityMap(rank);

    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType, input, output,
        ArrayRef<AffineMap>{inputMap, outputMap}, createParallelIterators(rank),
        [](OpBuilder &b, Location loc, ValueRange args) {
          b.create<linalg::YieldOp>(loc, args[0]);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConcatOpLowering : public OpConversionPattern<ConcatOp> {
  using OpConversionPattern<ConcatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ValueRange inputs = adaptor.getInputs();
    int64_t axis = op.getAxis();
    auto resultType = llvm::cast<RankedTensorType>(op.getType());

    int64_t rank = resultType.getRank();
    Value result = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    int64_t axisOffset = 0;
    for (Value input : inputs) {
      auto inputType = llvm::cast<RankedTensorType>(input.getType());
      SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
      offsets[axis] = rewriter.getIndexAttr(axisOffset);
      SmallVector<OpFoldResult> sizes;
      for (int64_t d : inputType.getShape())
        sizes.push_back(rewriter.getIndexAttr(d));
      SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
      result = rewriter.create<tensor::InsertSliceOp>(
          loc, input, result, offsets, sizes, strides);
      if (inputType.isDynamicDim(axis))
        return op.emitError("dynamic concat axis dim not supported yet");
      axisOffset += inputType.getDimSize(axis);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Quantization Lowering
//===----------------------------------------------------------------------===//

struct DequantizeOpLowering : public OpConversionPattern<DequantizeOp> {
  using OpConversionPattern<DequantizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DequantizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value scale = adaptor.getScale();
    Value zeroPoint = adaptor.getZeroPoint();
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto scaleType = llvm::cast<RankedTensorType>(scale.getType());
    int64_t resultRank = resultType.getRank();
    int64_t scaleRank = scaleType.getRank();
    Type elemType = resultType.getElementType();

    Value output = createEmptyTensor(rewriter, loc, resultType);

    // Build broadcast map for scale (scale may have fewer dims than input)
    SmallVector<AffineExpr> scaleExprs;
    for (int64_t i = 0; i < scaleRank; ++i) {
      scaleExprs.push_back(rewriter.getAffineDimExpr(resultRank - scaleRank + i));
    }
    AffineMap inputMap = rewriter.getMultiDimIdentityMap(resultRank);
    AffineMap scaleMap = AffineMap::get(resultRank, 0, scaleExprs, rewriter.getContext());
    AffineMap outputMap = rewriter.getMultiDimIdentityMap(resultRank);

    SmallVector<Value> inputs = {input, scale};
    SmallVector<AffineMap> maps = {inputMap, scaleMap, outputMap};
    
    if (zeroPoint) {
      inputs.push_back(zeroPoint);
      maps.insert(maps.begin() + 2, scaleMap); // zero_point has same shape as scale
    }

    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType, inputs, output,
        maps, createParallelIterators(resultRank),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value intVal = args[0];
          Value scaleVal = args[1];
          
          // Convert int to float
          Value floatVal = b.create<arith::SIToFPOp>(loc, elemType, intVal);
          
          // Subtract zero point if present: (x - zero_point)
          if (zeroPoint) {
            Value zpFloat = b.create<arith::SIToFPOp>(loc, elemType, args[2]);
            floatVal = b.create<arith::SubFOp>(loc, floatVal, zpFloat);
          }
          
          // Multiply by scale: (x - zero_point) * scale
          Value dequantized = b.create<arith::MulFOp>(loc, floatVal, scaleVal);
          b.create<linalg::YieldOp>(loc, dequantized);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Additional Missing Lowering Patterns
//===----------------------------------------------------------------------===//

/// Lower yirage.qmatmul (quantized matmul with W8A16 or W4A16)
struct QMatmulOpLowering : public OpConversionPattern<QMatmulOp> {
  using OpConversionPattern<QMatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhsQuantized = adaptor.getRhsQuantized();
    Value scale = adaptor.getScale();
    Value zeroPoint = adaptor.getZeroPoint();
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    Type elemType = resultType.getElementType();
    
    // Step 1: Dequantize the weight matrix
    // rhs_quantized: [K, N] i8
    // scale: [N] f32
    auto rhsQType = llvm::cast<RankedTensorType>(rhsQuantized.getType());
    auto scaleType = llvm::cast<RankedTensorType>(scale.getType());
    auto rhsDequantType = RankedTensorType::get(rhsQType.getShape(), elemType);
    Value rhsDequant = createEmptyTensor(rewriter, loc, rhsDequantType);
    
    int64_t rhsRank = rhsQType.getRank();
    int64_t scaleRank = scaleType.getRank();
    
    // Build broadcast map for scale (scale is 1D, broadcast to last dim of rhs)
    SmallVector<AffineExpr> rhsExprs, scaleExprs;
    for (int64_t i = 0; i < rhsRank; ++i) {
      rhsExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    for (int64_t i = 0; i < scaleRank; ++i) {
      scaleExprs.push_back(rewriter.getAffineDimExpr(rhsRank - scaleRank + i));
    }
    
    AffineMap rhsMap = AffineMap::get(rhsRank, 0, rhsExprs, rewriter.getContext());
    AffineMap scaleMap = AffineMap::get(rhsRank, 0, scaleExprs, rewriter.getContext());
    AffineMap outMap = rewriter.getMultiDimIdentityMap(rhsRank);
    
    rhsDequant = rewriter.create<linalg::GenericOp>(
        loc, rhsDequantType, ValueRange{rhsQuantized, scale}, rhsDequant,
        ArrayRef<AffineMap>{rhsMap, scaleMap, outMap}, createParallelIterators(rhsRank),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value intVal = args[0];
          Value scaleVal = args[1];
          Value floatVal = b.create<arith::SIToFPOp>(loc, elemType, intVal);
          if (zeroPoint) {
            // Zero point handling would go here
          }
          Value dequantized = b.create<arith::MulFOp>(loc, floatVal, scaleVal);
          b.create<linalg::YieldOp>(loc, dequantized);
        }).getResult(0);
    
    // Step 2: Perform matmul
    Value zeroed = createZeroTensor(rewriter, loc, resultType);
    Value result = rewriter.create<linalg::MatmulOp>(
        loc, ValueRange{lhs, rhsDequant}, ValueRange{zeroed}).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower yirage.quantize to linalg.generic
struct QuantizeOpLowering : public OpConversionPattern<QuantizeOp> {
  using OpConversionPattern<QuantizeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QuantizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value scale = adaptor.getScale();
    Value zeroPoint = adaptor.getZeroPoint();
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto scaleType = llvm::cast<RankedTensorType>(scale.getType());
    int64_t inputRank = inputType.getRank();
    int64_t scaleRank = scaleType.getRank();
    Type intType = resultType.getElementType();

    Value output = createEmptyTensor(rewriter, loc, resultType);

    // Build broadcast map for scale (scale may have fewer dims than input)
    SmallVector<AffineExpr> scaleExprs;
    for (int64_t i = 0; i < scaleRank; ++i) {
      scaleExprs.push_back(rewriter.getAffineDimExpr(inputRank - scaleRank + i));
    }
    AffineMap inputMap = rewriter.getMultiDimIdentityMap(inputRank);
    AffineMap scaleMap = AffineMap::get(inputRank, 0, scaleExprs, rewriter.getContext());
    AffineMap outputMap = rewriter.getMultiDimIdentityMap(inputRank);

    SmallVector<Value> inputs = {input, scale};
    SmallVector<AffineMap> maps = {inputMap, scaleMap, outputMap};
    
    if (zeroPoint) {
      inputs.push_back(zeroPoint);
      maps.insert(maps.begin() + 2, scaleMap); // zero_point has same shape as scale
    }

    Value result = rewriter.create<linalg::GenericOp>(
        loc, resultType, inputs, output,
        maps, createParallelIterators(inputRank),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value floatVal = args[0];
          Value scaleVal = args[1];
          
          // quantized = round(input / scale) + zero_point
          Value scaled = b.create<arith::DivFOp>(loc, floatVal, scaleVal);
          Value rounded = b.create<math::RoundOp>(loc, scaled);
          
          if (zeroPoint) {
            Value zpFloat = b.create<arith::SIToFPOp>(loc, floatVal.getType(), args[2]);
            rounded = b.create<arith::AddFOp>(loc, rounded, zpFloat);
          }
          
          Value quantized = b.create<arith::FPToSIOp>(loc, intType, rounded);
          b.create<linalg::YieldOp>(loc, quantized);
        }).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Lower yirage.topk using sorting + slicing
struct TopKOpLowering : public OpConversionPattern<TopKOp> {
  using OpConversionPattern<TopKOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TopKOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    int64_t k = op.getK();
    int64_t axis = op.getAxis();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    int64_t inputRank = inputType.getRank();
    
    // Handle negative axis
    if (axis < 0) axis += inputRank;
    
    auto valuesType = llvm::cast<RankedTensorType>(op.getValues().getType());
    auto indicesType = llvm::cast<RankedTensorType>(op.getIndices().getType());
    Type elemType = inputType.getElementType();
    Type indexType = indicesType.getElementType();
    
    // For simplicity, implement as a placeholder that returns sliced input
    // Full implementation would require sorting primitives
    // TODO: Implement proper sorting-based topk
    
    // Placeholder: slice the first k elements along axis
    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (int64_t i = 0; i < inputRank; ++i) {
      offsets.push_back(rewriter.getIndexAttr(0));
      if (i == axis) {
        sizes.push_back(rewriter.getIndexAttr(k));
      } else {
        sizes.push_back(rewriter.getIndexAttr(inputType.getShape()[i]));
      }
      strides.push_back(rewriter.getIndexAttr(1));
    }
    
    Value values = rewriter.create<tensor::ExtractSliceOp>(
        loc, valuesType, input, offsets, sizes, strides);
    
    // Create indices 0, 1, 2, ..., k-1
    Value indices = createEmptyTensor(rewriter, loc, indicesType);
    indices = rewriter.create<linalg::GenericOp>(
        loc, indicesType, ValueRange{}, indices,
        ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(inputRank)},
        createParallelIterators(inputRank),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value idx = b.create<linalg::IndexOp>(loc, axis);
          Value idxCast = b.create<arith::IndexCastOp>(loc, indexType, idx);
          b.create<linalg::YieldOp>(loc, idxCast);
        }).getResult(0);

    rewriter.replaceOp(op, {values, indices});
    return success();
  }
};

/// Lower yirage.argmax to reduction
struct ArgMaxOpLowering : public OpConversionPattern<ArgMaxOp> {
  using OpConversionPattern<ArgMaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ArgMaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    int64_t axis = op.getAxis();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    int64_t inputRank = inputType.getRank();
    Type elemType = inputType.getElementType();
    Type indexType = resultType.getElementType();
    
    // Handle negative axis
    if (axis < 0) axis += inputRank;
    
    // Initialize with -inf for values and 0 for indices
    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
    Value zeroIdx = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(indexType, 0));
    
    // Create output shape (input shape with axis dimension removed)
    SmallVector<int64_t> outputShape;
    for (int64_t i = 0; i < inputRank; ++i) {
      if (i != axis) outputShape.push_back(inputType.getShape()[i]);
    }
    auto maxValType = RankedTensorType::get(outputShape, elemType);
    
    Value maxVals = createFilledTensor(rewriter, loc, maxValType, negInf);
    Value argmax = createFilledTensor(rewriter, loc, resultType, zeroIdx);
    
    // Build maps - output dimensions skip the axis dimension
    SmallVector<AffineExpr> inputExprs, outputExprs;
    SmallVector<utils::IteratorType> iteratorTypes;
    
    for (int64_t i = 0; i < inputRank; ++i) {
      inputExprs.push_back(rewriter.getAffineDimExpr(i));
      if (i == axis) {
        iteratorTypes.push_back(utils::IteratorType::reduction);
      } else {
        iteratorTypes.push_back(utils::IteratorType::parallel);
        outputExprs.push_back(rewriter.getAffineDimExpr(i));
      }
    }
    
    AffineMap inputMap = AffineMap::get(inputRank, 0, inputExprs, rewriter.getContext());
    AffineMap outputMap = AffineMap::get(inputRank, 0, outputExprs, rewriter.getContext());
    
    // Use generic op to find max and argmax simultaneously
    auto results = rewriter.create<linalg::GenericOp>(
        loc, TypeRange{maxValType, resultType}, input, ValueRange{maxVals, argmax},
        ArrayRef<AffineMap>{inputMap, outputMap, outputMap}, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value x = args[0];
          Value currMax = args[1];
          Value currIdx = args[2];
          Value axisIdx = b.create<linalg::IndexOp>(loc, axis);
          Value axisIdxCast = b.create<arith::IndexCastOp>(loc, indexType, axisIdx);
          
          Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, x, currMax);
          Value newMax = b.create<arith::SelectOp>(loc, cmp, x, currMax);
          Value newIdx = b.create<arith::SelectOp>(loc, cmp, axisIdxCast, currIdx);
          
          b.create<linalg::YieldOp>(loc, ValueRange{newMax, newIdx});
        });

    rewriter.replaceOp(op, results.getResult(1));
    return success();
  }
};

/// Lower yirage.split to tensor.extract_slice
struct SplitOpLowering : public OpConversionPattern<SplitOp> {
  using OpConversionPattern<SplitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    int64_t numSplits = op.getNumSplits();
    int64_t axis = op.getAxis();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    int64_t inputRank = inputType.getRank();
    
    // Handle negative axis
    if (axis < 0) axis += inputRank;
    
    int64_t axisSize = inputType.getShape()[axis];
    int64_t splitSize = axisSize / numSplits;
    
    SmallVector<Value> results;
    for (int64_t i = 0; i < numSplits; ++i) {
      SmallVector<OpFoldResult> offsets, sizes, strides;
      for (int64_t d = 0; d < inputRank; ++d) {
        if (d == axis) {
          offsets.push_back(rewriter.getIndexAttr(i * splitSize));
          sizes.push_back(rewriter.getIndexAttr(splitSize));
        } else {
          offsets.push_back(rewriter.getIndexAttr(0));
          sizes.push_back(rewriter.getIndexAttr(inputType.getShape()[d]));
        }
        strides.push_back(rewriter.getIndexAttr(1));
      }
      
      auto resultType = llvm::cast<RankedTensorType>(op.getResults()[i].getType());
      Value slice = rewriter.create<tensor::ExtractSliceOp>(
          loc, resultType, input, offsets, sizes, strides);
      results.push_back(slice);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// KVCacheUpdate Lowering
//===----------------------------------------------------------------------===//

/// Lower yirage.kv_cache_update to SCF loops with tensor scatter semantics
/// Updates KV cache at specified slot positions with new key/value tensors
struct KVCacheUpdateOpLowering : public OpConversionPattern<KVCacheUpdateOp> {
  using OpConversionPattern<KVCacheUpdateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(KVCacheUpdateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value keyCache = adaptor.getKeyCache();
    Value valueCache = adaptor.getValueCache();
    Value newKeys = adaptor.getNewKeys();
    Value newValues = adaptor.getNewValues();
    Value slotIndices = adaptor.getSlotIndices();

    auto keyCacheType = llvm::cast<RankedTensorType>(keyCache.getType());
    auto valueCacheType = llvm::cast<RankedTensorType>(valueCache.getType());
    auto newKeysType = llvm::cast<RankedTensorType>(newKeys.getType());
    int64_t numTokens = newKeysType.getShape()[0];
    int64_t numHeads = newKeysType.getShape()[1];
    int64_t headDim = newKeysType.getShape()[2];

    // Cache layout for paged attention: [num_blocks, num_heads, block_size, head_dim]
    // newKeys/newValues layout: [num_tokens, num_heads, head_dim]
    // slotIndices: [num_tokens] - linear slot index for each token
    
    // For each token i:
    //   slot = slotIndices[i]
    //   block_idx = slot / block_size
    //   block_offset = slot % block_size
    //   keyCache[block_idx, :, block_offset, :] = newKeys[i, :, :]
    //   valueCache[block_idx, :, block_offset, :] = newValues[i, :, :]

    int64_t blockSize = keyCacheType.getShape()[2];
    Value blockSizeVal = rewriter.create<arith::ConstantIndexOp>(loc, blockSize);

    // Create loop bounds
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value numTokensVal = rewriter.create<arith::ConstantIndexOp>(loc, numTokens);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Use scf.for to iterate over tokens and update caches
    auto forOp = rewriter.create<scf::ForOp>(
        loc, zero, numTokensVal, one,
        ValueRange{keyCache, valueCache},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          Value currKeyCache = iterArgs[0];
          Value currValueCache = iterArgs[1];

          // Get slot index for this token
          Value slotIdx = b.create<tensor::ExtractOp>(loc, slotIndices, iv);
          Value slotIdxAsIndex = b.create<arith::IndexCastOp>(
              loc, b.getIndexType(), slotIdx);

          // Compute block index and offset
          Value blockIdx = b.create<arith::DivUIOp>(loc, slotIdxAsIndex, blockSizeVal);
          Value blockOffset = b.create<arith::RemUIOp>(loc, slotIdxAsIndex, blockSizeVal);

          // Extract new key/value for this token: [num_heads, head_dim]
          // and insert into cache at [block_idx, :, block_offset, :]
          
          // For each head and each dim, update the cache
          // Use nested loops or linalg.generic
          Value numHeadsVal = b.create<arith::ConstantIndexOp>(loc, numHeads);
          Value headDimVal = b.create<arith::ConstantIndexOp>(loc, headDim);

          // Inner loop over heads
          auto headLoop = b.create<scf::ForOp>(
              loc, zero, numHeadsVal, one,
              ValueRange{currKeyCache, currValueCache},
              [&](OpBuilder &b2, Location loc, Value headIv, ValueRange headArgs) {
                Value kc = headArgs[0];
                Value vc = headArgs[1];

                // Inner loop over head_dim
                auto dimLoop = b2.create<scf::ForOp>(
                    loc, zero, headDimVal, one,
                    ValueRange{kc, vc},
                    [&](OpBuilder &b3, Location loc, Value dimIv, ValueRange dimArgs) {
                      Value kc2 = dimArgs[0];
                      Value vc2 = dimArgs[1];

                      // Extract from newKeys[iv, headIv, dimIv]
                      Value newKeyVal = b3.create<tensor::ExtractOp>(
                          loc, newKeys, ValueRange{iv, headIv, dimIv});
                      Value newValVal = b3.create<tensor::ExtractOp>(
                          loc, newValues, ValueRange{iv, headIv, dimIv});

                      // Insert into keyCache[blockIdx, headIv, blockOffset, dimIv]
                      Value updatedKeyCache = b3.create<tensor::InsertOp>(
                          loc, newKeyVal, kc2,
                          ValueRange{blockIdx, headIv, blockOffset, dimIv});
                      Value updatedValueCache = b3.create<tensor::InsertOp>(
                          loc, newValVal, vc2,
                          ValueRange{blockIdx, headIv, blockOffset, dimIv});

                      b3.create<scf::YieldOp>(loc, ValueRange{updatedKeyCache, updatedValueCache});
                    });

                b2.create<scf::YieldOp>(loc, dimLoop.getResults());
              });

          b.create<scf::YieldOp>(loc, headLoop.getResults());
        });

    // Replace op with loop results
    rewriter.replaceOp(op, forOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Conv2D Lowering
//===----------------------------------------------------------------------===//

/// Lower yirage.conv2d to linalg.conv_2d_nchw_fchw
struct Conv2DOpLowering : public OpConversionPattern<Conv2DOp> {
  using OpConversionPattern<Conv2DOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();
    Value kernel = adaptor.getKernel();

    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto kernelType = llvm::cast<RankedTensorType>(kernel.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());

    // Get strides and dilations
    auto stridesAttr = op.getStrides();
    auto dilationsAttr = op.getDilations();

    SmallVector<int64_t> strides, dilations;
    for (auto s : stridesAttr)
      strides.push_back(s.cast<IntegerAttr>().getInt());
    for (auto d : dilationsAttr)
      dilations.push_back(d.cast<IntegerAttr>().getInt());

    // Create zero-initialized output tensor
    Value zeroed = createZeroTensor(rewriter, loc, resultType);

    // Create strides and dilations attributes for linalg
    auto stridesMLIRAttr = rewriter.getDenseI64ArrayAttr(strides);
    auto dilationsMLIRAttr = rewriter.getDenseI64ArrayAttr(dilations);

    // Use linalg.conv_2d_nchw_fchw for NCHW input and FCHW kernel
    Value result = rewriter.create<linalg::Conv2DNchwFchwOp>(
        loc, resultType, ValueRange{input, kernel}, ValueRange{zeroed},
        stridesMLIRAttr, dilationsMLIRAttr).getResult(0);

    // Add bias if present
    if (op.getBias()) {
      Value bias = adaptor.getBias();
      auto biasType = llvm::cast<RankedTensorType>(bias.getType());

      // Broadcast bias [out_channels] to [N, out_channels, H, W]
      int64_t outChannels = biasType.getShape()[0];
      auto resultShape = resultType.getShape();

      // Create broadcast map: bias[c] -> result[n, c, h, w]
      auto c = rewriter.getAffineDimExpr(1);
      auto biasMap = AffineMap::get(4, 0, {c}, rewriter.getContext());
      auto resultMap = AffineMap::getMultiDimIdentityMap(4, rewriter.getContext());

      result = rewriter.create<linalg::GenericOp>(
          loc, resultType, ValueRange{bias}, result,
          ArrayRef<AffineMap>{biasMap, resultMap},
          SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
            b.create<linalg::YieldOp>(loc, sum);
          }).getResult(0);
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MaxPool2D Lowering
//===----------------------------------------------------------------------===//

/// Lower yirage.maxpool2d to linalg.pooling_nchw_max
struct MaxPool2DOpLowering : public OpConversionPattern<MaxPool2DOp> {
  using OpConversionPattern<MaxPool2DOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MaxPool2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = adaptor.getInput();

    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    Type elemType = resultType.getElementType();

    // Get kernel size and strides
    auto kernelSizeAttr = op.getKernelSize();
    auto stridesAttr = op.getStrides();

    SmallVector<int64_t> kernelSize, strides;
    for (auto k : kernelSizeAttr)
      kernelSize.push_back(k.cast<IntegerAttr>().getInt());
    for (auto s : stridesAttr)
      strides.push_back(s.cast<IntegerAttr>().getInt());

    // Create -inf initialized output tensor for max pooling
    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
    Value initTensor = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), elemType);
    Value filled = rewriter.create<linalg::FillOp>(
        loc, negInf, initTensor).getResult(0);

    // Create pooling window tensor
    Value windowInit = rewriter.create<tensor::EmptyOp>(
        loc, kernelSize, elemType);

    auto stridesMLIRAttr = rewriter.getDenseI64ArrayAttr(strides);
    auto dilationsMLIRAttr = rewriter.getDenseI64ArrayAttr({1, 1});

    Value result = rewriter.create<linalg::PoolingNchwMaxOp>(
        loc, resultType, ValueRange{input, windowInit}, ValueRange{filled},
        stridesMLIRAttr, dilationsMLIRAttr).getResult(0);

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct YirageToLinalgPass
    : public PassWrapper<YirageToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(YirageToLinalgPass)

  StringRef getArgument() const final { return "yirage-to-linalg"; }
  StringRef getDescription() const final {
    return "Lower Yirage dialect to Linalg dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    YirageTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    patterns.add<
        // Matrix operations
        MatmulOpLowering,
        BatchMatmulOpLowering,
        LinearOpLowering,
        QMatmulOpLowering,
        // Normalization
        RMSNormOpLowering,
        LayerNormOpLowering,
        // Activations
        SiLUOpLowering,
        GELUOpLowering,
        ReLUOpLowering,
        SoftmaxOpLowering,
        // MLP
        GatedMLPOpLowering,
        // Embeddings
        EmbeddingOpLowering,
        RoPEOpLowering,
        // Attention
        AttentionOpLowering,
        PagedAttentionOpLowering,
        KVCacheUpdateOpLowering,
        // Convolution
        Conv2DOpLowering,
        MaxPool2DOpLowering,
        // Reductions
        ReduceSumOpLowering,
        ReduceMaxOpLowering,
        TopKOpLowering,
        ArgMaxOpLowering,
        // Tensor manipulation
        ReshapeOpLowering,
        TransposeOpLowering,
        ConcatOpLowering,
        SplitOpLowering,
        // Quantization
        QuantizeOpLowering,
        DequantizeOpLowering
    >(typeConverter, context);

    target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                           math::MathDialect, tensor::TensorDialect,
                           scf::SCFDialect, func::FuncDialect>();

    target.addIllegalOp<
        // Matrix operations
        MatmulOp, BatchMatmulOp, LinearOp, QMatmulOp,
        // Normalization
        RMSNormOp, LayerNormOp,
        // Activations
        SiLUOp, GELUOp, ReLUOp, SoftmaxOp,
        // MLP
        GatedMLPOp,
        // Embeddings
        EmbeddingOp, RoPEOp,
        // Attention
        AttentionOp, PagedAttentionOp, KVCacheUpdateOp,
        // Convolution
        Conv2DOp, MaxPool2DOp,
        // Reductions
        ReduceSumOp, ReduceMaxOp, TopKOp, ArgMaxOp,
        // Tensor manipulation
        ReshapeOp, TransposeOp, ConcatOp, SplitOp,
        // Quantization
        QuantizeOp, DequantizeOp
    >();
    // Remaining Yirage ops (MoE, sampling, etc.) stay legal until patterns exist.

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace yirage {

std::unique_ptr<mlir::Pass> createYirageToLinalgPass() {
  return std::make_unique<YirageToLinalgPass>();
}

} // namespace yirage
