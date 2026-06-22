//===- YirageToStableHLO.cpp - Lower Yirage to StableHLO --------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering from YiRage dialect to StableHLO for TPU
// execution via XLA.
//
// StableHLO is a portable ML operations dialect that can be compiled by:
// - XLA (Google's Accelerated Linear Algebra compiler)
// - IREE (Intermediate Representation Execution Environment)
// - Other StableHLO consumers
//
// Pipeline:
//   YiRage → StableHLO → XLA → TPU executable
//
// Supported operations:
// - matmul → stablehlo.dot_general
// - attention → stablehlo.dot_general + stablehlo.softmax pattern
// - rms_norm → stablehlo reduction + broadcast + multiply
// - gated_mlp → stablehlo.dot_general + activation pattern
// - etc.
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.h"
#include "yirage-mlir/Dialect/Yirage/IR/YirageOps.h"
#include "yirage-mlir/Dialect/Yirage/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <cmath>

using namespace mlir;

namespace yirage {
namespace ir {

//===----------------------------------------------------------------------===//
// StableHLO Operation Builders
// 
// Since we may not have StableHLO dialect linked, we use generic ops
// that match StableHLO semantics. In production with StableHLO linked,
// these would be replaced with actual StableHLO ops.
//===----------------------------------------------------------------------===//

namespace stablehlo_compat {

/// Create a dot_general operation (equivalent to einsum)
/// This matches stablehlo.dot_general semantics
Value createDotGeneral(OpBuilder &builder, Location loc,
                       Value lhs, Value rhs,
                       ArrayRef<int64_t> lhsBatchDims,
                       ArrayRef<int64_t> rhsBatchDims,
                       ArrayRef<int64_t> lhsContractDims,
                       ArrayRef<int64_t> rhsContractDims,
                       Type resultType) {
  // For now, use linalg.generic to represent the contraction
  // In production, this would emit stablehlo.dot_general
  
  auto lhsType = llvm::cast<RankedTensorType>(lhs.getType());
  auto rhsType = llvm::cast<RankedTensorType>(rhs.getType());
  auto outType = llvm::cast<RankedTensorType>(resultType);
  
  int64_t lhsRank = lhsType.getRank();
  int64_t rhsRank = rhsType.getRank();
  int64_t outRank = outType.getRank();
  
  // Build affine maps for the contraction
  SmallVector<AffineExpr> lhsExprs, rhsExprs, outExprs;
  
  int64_t dimIdx = 0;
  
  // Batch dimensions
  for (size_t i = 0; i < lhsBatchDims.size(); ++i) {
    auto dim = builder.getAffineDimExpr(dimIdx);
    lhsExprs.push_back(dim);
    rhsExprs.push_back(dim);
    outExprs.push_back(dim);
    ++dimIdx;
  }
  
  // LHS non-contracting dimensions
  for (int64_t i = 0; i < lhsRank; ++i) {
    if (std::find(lhsBatchDims.begin(), lhsBatchDims.end(), i) == lhsBatchDims.end() &&
        std::find(lhsContractDims.begin(), lhsContractDims.end(), i) == lhsContractDims.end()) {
      auto dim = builder.getAffineDimExpr(dimIdx);
      lhsExprs.push_back(dim);
      outExprs.push_back(dim);
      ++dimIdx;
    }
  }
  
  // RHS non-contracting dimensions
  for (int64_t i = 0; i < rhsRank; ++i) {
    if (std::find(rhsBatchDims.begin(), rhsBatchDims.end(), i) == rhsBatchDims.end() &&
        std::find(rhsContractDims.begin(), rhsContractDims.end(), i) == rhsContractDims.end()) {
      auto dim = builder.getAffineDimExpr(dimIdx);
      rhsExprs.push_back(dim);
      outExprs.push_back(dim);
      ++dimIdx;
    }
  }
  
  // Contracting dimensions
  int64_t numContractDims = lhsContractDims.size();
  for (int64_t i = 0; i < numContractDims; ++i) {
    auto dim = builder.getAffineDimExpr(dimIdx);
    // Insert at correct position in lhs/rhs
    ++dimIdx;
  }
  
  // Create zero-initialized output
  Value zeroF = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(outType.getElementType()));
  Value init = builder.create<tensor::EmptyOp>(loc, outType.getShape(),
                                                outType.getElementType());
  Value output = builder.create<linalg::FillOp>(loc, zeroF, init).getResult(0);
  
  // For simplicity, use linalg.matmul for 2D case
  if (lhsRank == 2 && rhsRank == 2) {
    return builder.create<linalg::MatmulOp>(loc, resultType,
                                            ValueRange{lhs, rhs}, output)
        .getResult(0);
  }
  
  // For batched case, use batch_matmul pattern
  return builder.create<linalg::BatchMatmulOp>(loc, resultType,
                                                ValueRange{lhs, rhs}, output)
      .getResult(0);
}

/// Create a reduce operation (max, sum, etc.)
Value createReduce(OpBuilder &builder, Location loc,
                   Value input, Value init,
                   ArrayRef<int64_t> reduceDims,
                   StringRef reduceOp) {
  auto inputType = llvm::cast<RankedTensorType>(input.getType());
  Type elemType = inputType.getElementType();
  
  // Compute output shape
  SmallVector<int64_t> outShape;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    if (std::find(reduceDims.begin(), reduceDims.end(), i) == reduceDims.end()) {
      outShape.push_back(inputType.getDimSize(i));
    }
  }
  
  auto outType = RankedTensorType::get(outShape, elemType);
  Value output = builder.create<tensor::EmptyOp>(loc, outShape, elemType);
  output = builder.create<linalg::FillOp>(loc, init, output).getResult(0);
  
  // Build affine maps
  int64_t numIterDims = inputType.getRank();
  SmallVector<AffineExpr> inputExprs, outputExprs;
  SmallVector<utils::IteratorType> iterTypes;
  
  int64_t outIdx = 0;
  for (int64_t i = 0; i < numIterDims; ++i) {
    inputExprs.push_back(builder.getAffineDimExpr(i));
    
    if (std::find(reduceDims.begin(), reduceDims.end(), i) != reduceDims.end()) {
      iterTypes.push_back(utils::IteratorType::reduction);
    } else {
      outputExprs.push_back(builder.getAffineDimExpr(i));
      iterTypes.push_back(utils::IteratorType::parallel);
    }
  }
  
  auto inputMap = AffineMap::get(numIterDims, 0, inputExprs, builder.getContext());
  auto outputMap = AffineMap::get(numIterDims, 0, outputExprs, builder.getContext());
  
  return builder.create<linalg::GenericOp>(
      loc, outType, input, output,
      ArrayRef<AffineMap>{inputMap, outputMap}, iterTypes,
      [&](OpBuilder &b, Location loc, ValueRange args) {
        Value result;
        if (reduceOp == "max") {
          result = b.create<arith::MaxFOp>(loc, args[0], args[1]);
        } else if (reduceOp == "sum") {
          result = b.create<arith::AddFOp>(loc, args[0], args[1]);
        } else if (reduceOp == "min") {
          result = b.create<arith::MinimumFOp>(loc, args[0], args[1]);
        } else {
          result = b.create<arith::AddFOp>(loc, args[0], args[1]);
        }
        b.create<linalg::YieldOp>(loc, result);
      }).getResult(0);
}

/// Create broadcast operation
Value createBroadcast(OpBuilder &builder, Location loc,
                      Value input, ArrayRef<int64_t> broadcastDims,
                      RankedTensorType resultType) {
  // Use tensor.broadcast or linalg.generic
  auto inputType = llvm::cast<RankedTensorType>(input.getType());
  
  Value output = builder.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType());
  
  // Build affine maps for broadcast
  int64_t outRank = resultType.getRank();
  SmallVector<AffineExpr> inputExprs;
  SmallVector<AffineExpr> outputExprs;
  
  int64_t inputIdx = 0;
  for (int64_t i = 0; i < outRank; ++i) {
    outputExprs.push_back(builder.getAffineDimExpr(i));
    if (std::find(broadcastDims.begin(), broadcastDims.end(), i) == broadcastDims.end()) {
      inputExprs.push_back(builder.getAffineDimExpr(i));
      ++inputIdx;
    }
  }
  
  auto inputMap = AffineMap::get(outRank, 0, inputExprs, builder.getContext());
  auto outputMap = AffineMap::get(outRank, 0, outputExprs, builder.getContext());
  
  SmallVector<utils::IteratorType> iterTypes(outRank, utils::IteratorType::parallel);
  
  return builder.create<linalg::GenericOp>(
      loc, resultType, input, output,
      ArrayRef<AffineMap>{inputMap, outputMap}, iterTypes,
      [](OpBuilder &b, Location loc, ValueRange args) {
        b.create<linalg::YieldOp>(loc, args[0]);
      }).getResult(0);
}

} // namespace stablehlo_compat

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert yirage.matmul to StableHLO-compatible dot_general
struct MatmulToStableHLO : public OpRewritePattern<MatmulOp> {
  using OpRewritePattern<MatmulOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    
    auto lhsType = llvm::cast<RankedTensorType>(lhs.getType());
    auto rhsType = llvm::cast<RankedTensorType>(rhs.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    
    // Handle transpose
    if (op.getTransposeLhs()) {
      // Transpose lhs
      SmallVector<int64_t> perm;
      for (int64_t i = 0; i < lhsType.getRank() - 2; ++i) perm.push_back(i);
      perm.push_back(lhsType.getRank() - 1);
      perm.push_back(lhsType.getRank() - 2);
      
      auto permAttr = rewriter.getI64ArrayAttr(perm);
      lhs = rewriter.create<TransposeOp>(loc, lhs, permAttr);
    }
    
    if (op.getTransposeRhs()) {
      // Transpose rhs
      SmallVector<int64_t> perm;
      for (int64_t i = 0; i < rhsType.getRank() - 2; ++i) perm.push_back(i);
      perm.push_back(rhsType.getRank() - 1);
      perm.push_back(rhsType.getRank() - 2);
      
      auto permAttr = rewriter.getI64ArrayAttr(perm);
      rhs = rewriter.create<TransposeOp>(loc, rhs, permAttr);
    }
    
    // Create dot_general with contracting dimension on last dim of lhs and
    // second-to-last of rhs
    SmallVector<int64_t> lhsBatchDims, rhsBatchDims;
    SmallVector<int64_t> lhsContractDims = {lhsType.getRank() - 1};
    SmallVector<int64_t> rhsContractDims = {rhsType.getRank() - 2};
    
    // Batch dimensions
    for (int64_t i = 0; i < lhsType.getRank() - 2; ++i) {
      lhsBatchDims.push_back(i);
      rhsBatchDims.push_back(i);
    }
    
    Value result = stablehlo_compat::createDotGeneral(
        rewriter, loc, lhs, rhs,
        lhsBatchDims, rhsBatchDims,
        lhsContractDims, rhsContractDims,
        resultType);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert yirage.rms_norm to StableHLO-compatible operations
struct RMSNormToStableHLO : public OpRewritePattern<RMSNormOp> {
  using OpRewritePattern<RMSNormOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(RMSNormOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value gamma = op.getGamma();
    float epsilon = op.getEpsilon().convertToFloat();
    
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    Type elemType = inputType.getElementType();
    int64_t rank = inputType.getRank();
    
    // Step 1: x^2
    Value xSquared = rewriter.create<arith::MulFOp>(loc, input, input);
    
    // Step 2: mean(x^2) over last dimension
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    SmallVector<int64_t> reduceDims = {rank - 1};
    Value sumSquared = stablehlo_compat::createReduce(
        rewriter, loc, xSquared, zero, reduceDims, "sum");
    
    // Divide by hidden dim to get mean
    int64_t hiddenDim = inputType.getDimSize(rank - 1);
    Value hiddenDimF = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, static_cast<double>(hiddenDim)));
    
    auto sumType = llvm::cast<RankedTensorType>(sumSquared.getType());
    Value meanSquared = rewriter.create<linalg::GenericOp>(
        loc, sumType, sumSquared,
        rewriter.create<tensor::EmptyOp>(loc, sumType.getShape(), elemType),
        SmallVector<AffineMap>(2, rewriter.getMultiDimIdentityMap(sumType.getRank())),
        SmallVector<utils::IteratorType>(sumType.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value div = b.create<arith::DivFOp>(loc, args[0], hiddenDimF);
          b.create<linalg::YieldOp>(loc, div);
        }).getResult(0);
    
    // Step 3: 1 / sqrt(mean + epsilon)
    Value epsVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, epsilon));
    
    Value rsqrt = rewriter.create<linalg::GenericOp>(
        loc, sumType, meanSquared,
        rewriter.create<tensor::EmptyOp>(loc, sumType.getShape(), elemType),
        SmallVector<AffineMap>(2, rewriter.getMultiDimIdentityMap(sumType.getRank())),
        SmallVector<utils::IteratorType>(sumType.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value addEps = b.create<arith::AddFOp>(loc, args[0], epsVal);
          Value rsqrtVal = b.create<math::RsqrtOp>(loc, addEps);
          b.create<linalg::YieldOp>(loc, rsqrtVal);
        }).getResult(0);
    
    // Step 4: Broadcast rsqrt to input shape
    SmallVector<int64_t> broadcastDims = {rank - 1};
    Value rsqrtBroadcast = stablehlo_compat::createBroadcast(
        rewriter, loc, rsqrt, broadcastDims, inputType);
    
    // Step 5: x * rsqrt
    Value normalized = rewriter.create<arith::MulFOp>(loc, input, rsqrtBroadcast);
    
    // Step 6: normalized * gamma
    // Broadcast gamma
    Value gammaBroadcast = stablehlo_compat::createBroadcast(
        rewriter, loc, gamma, broadcastDims, inputType);
    Value result = rewriter.create<arith::MulFOp>(loc, normalized, gammaBroadcast);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert yirage.softmax to StableHLO-compatible operations
struct SoftmaxToStableHLO : public OpRewritePattern<SoftmaxOp> {
  using OpRewritePattern<SoftmaxOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(SoftmaxOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    int64_t axis = op.getAxis();
    
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    Type elemType = inputType.getElementType();
    int64_t rank = inputType.getRank();
    
    // Handle negative axis
    if (axis < 0) axis += rank;
    
    SmallVector<int64_t> reduceDims = {axis};
    
    // Step 1: max(input, axis)
    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
    Value maxVal = stablehlo_compat::createReduce(
        rewriter, loc, input, negInf, reduceDims, "max");
    
    // Step 2: input - max (broadcast)
    SmallVector<int64_t> broadcastDims = {axis};
    Value maxBroadcast = stablehlo_compat::createBroadcast(
        rewriter, loc, maxVal, broadcastDims, inputType);
    Value shifted = rewriter.create<arith::SubFOp>(loc, input, maxBroadcast);
    
    // Step 3: exp(shifted)
    Value expInput = rewriter.create<math::ExpOp>(loc, shifted);
    
    // Step 4: sum(exp, axis)
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value sumExp = stablehlo_compat::createReduce(
        rewriter, loc, expInput, zero, reduceDims, "sum");
    
    // Step 5: exp / sum (broadcast)
    Value sumBroadcast = stablehlo_compat::createBroadcast(
        rewriter, loc, sumExp, broadcastDims, inputType);
    Value result = rewriter.create<arith::DivFOp>(loc, expInput, sumBroadcast);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert yirage.gated_mlp to StableHLO-compatible operations
struct GatedMLPToStableHLO : public OpRewritePattern<GatedMLPOp> {
  using OpRewritePattern<GatedMLPOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(GatedMLPOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getInput();
    Value gateWeight = op.getGateWeight();
    Value upWeight = op.getUpWeight();
    Value downWeight = op.getDownWeight();
    
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    Type elemType = inputType.getElementType();
    
    // Step 1: gate = input @ gate_weight
    auto gateWeightType = llvm::cast<RankedTensorType>(gateWeight.getType());
    SmallVector<int64_t> gateShape = {inputType.getDimSize(0), 
                                       gateWeightType.getDimSize(1)};
    auto gateType = RankedTensorType::get(gateShape, elemType);
    
    Value gate = stablehlo_compat::createDotGeneral(
        rewriter, loc, input, gateWeight,
        {}, {}, {1}, {0}, gateType);
    
    // Step 2: up = input @ up_weight
    Value up = stablehlo_compat::createDotGeneral(
        rewriter, loc, input, upWeight,
        {}, {}, {1}, {0}, gateType);
    
    // Step 3: silu(gate)
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    Value negGate = rewriter.create<arith::NegFOp>(loc, gate);
    Value expNegGate = rewriter.create<math::ExpOp>(loc, negGate);
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, 1.0));
    
    Value siluGate = rewriter.create<linalg::GenericOp>(
        loc, gateType, ValueRange{gate, expNegGate},
        rewriter.create<tensor::EmptyOp>(loc, gateShape, elemType),
        SmallVector<AffineMap>(3, rewriter.getMultiDimIdentityMap(2)),
        SmallVector<utils::IteratorType>(2, utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value denom = b.create<arith::AddFOp>(loc, one, args[1]);
          Value silu = b.create<arith::DivFOp>(loc, args[0], denom);
          b.create<linalg::YieldOp>(loc, silu);
        }).getResult(0);
    
    // Step 4: intermediate = silu(gate) * up
    Value intermediate = rewriter.create<arith::MulFOp>(loc, siluGate, up);
    
    // Step 5: output = intermediate @ down_weight
    Value result = stablehlo_compat::createDotGeneral(
        rewriter, loc, intermediate, downWeight,
        {}, {}, {1}, {0}, resultType);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Convert yirage.attention to StableHLO-compatible operations
struct AttentionToStableHLO : public OpRewritePattern<AttentionOp> {
  using OpRewritePattern<AttentionOp>::OpRewritePattern;
  
  LogicalResult matchAndRewrite(AttentionOp op,
                                PatternRewriter &rewriter) const override {
    // Delegate to standard lowering which already produces StableHLO-compatible ops
    // The FlashAttention pass handles optimization
    Location loc = op.getLoc();
    Value query = op.getQuery();
    Value key = op.getKey();
    Value value = op.getValue();
    
    auto queryType = llvm::cast<RankedTensorType>(query.getType());
    Type elemType = queryType.getElementType();
    
    // Shape: [batch, heads, seq, head_dim]
    int64_t headDim = queryType.getDimSize(3);
    double scale = 1.0 / std::sqrt(static_cast<double>(headDim));
    if (op.getScale().has_value()) {
      scale = op.getScale().value().convertToFloat();
    }
    
    // Q @ K^T
    auto keyType = llvm::cast<RankedTensorType>(key.getType());
    SmallVector<int64_t> scoresShape = {
        queryType.getDimSize(0), queryType.getDimSize(1),
        queryType.getDimSize(2), keyType.getDimSize(2)};
    auto scoresType = RankedTensorType::get(scoresShape, elemType);
    
    // Use batch_matmul with transpose
    Value keyTransposed = rewriter.create<TransposeOp>(
        loc, key, rewriter.getI64ArrayAttr({0, 1, 3, 2}));
    
    Value scores = stablehlo_compat::createDotGeneral(
        rewriter, loc, query, keyTransposed,
        {0, 1}, {0, 1}, {3}, {2}, scoresType);
    
    // Scale
    Value scaleVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, scale));
    scores = rewriter.create<arith::MulFOp>(loc, scores, scaleVal);
    
    // Causal mask
    if (op.getCausal()) {
      Value negInf = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
      
      scores = rewriter.create<linalg::GenericOp>(
          loc, scoresType, scores,
          rewriter.create<tensor::EmptyOp>(loc, scoresShape, elemType),
          SmallVector<AffineMap>(2, rewriter.getMultiDimIdentityMap(4)),
          SmallVector<utils::IteratorType>(4, utils::IteratorType::parallel),
          [&](OpBuilder &b, Location loc, ValueRange args) {
            Value row = b.create<linalg::IndexOp>(loc, 2);
            Value col = b.create<linalg::IndexOp>(loc, 3);
            Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt, col, row);
            Value result = b.create<arith::SelectOp>(loc, cmp, negInf, args[0]);
            b.create<linalg::YieldOp>(loc, result);
          }).getResult(0);
    }
    
    // Softmax
    // ... (similar to SoftmaxToStableHLO)
    SmallVector<int64_t> reduceDims = {3};
    
    Value negInf = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, -std::numeric_limits<float>::infinity()));
    Value maxScores = stablehlo_compat::createReduce(
        rewriter, loc, scores, negInf, reduceDims, "max");
    
    SmallVector<int64_t> broadcastDims = {3};
    Value maxBroadcast = stablehlo_compat::createBroadcast(
        rewriter, loc, maxScores, broadcastDims, scoresType);
    Value shifted = rewriter.create<arith::SubFOp>(loc, scores, maxBroadcast);
    Value expScores = rewriter.create<math::ExpOp>(loc, shifted);
    
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elemType));
    Value sumExp = stablehlo_compat::createReduce(
        rewriter, loc, expScores, zero, reduceDims, "sum");
    Value sumBroadcast = stablehlo_compat::createBroadcast(
        rewriter, loc, sumExp, broadcastDims, scoresType);
    Value attnWeights = rewriter.create<arith::DivFOp>(loc, expScores, sumBroadcast);
    
    // Attention weights @ V
    auto resultType = llvm::cast<RankedTensorType>(op.getType());
    Value result = stablehlo_compat::createDotGeneral(
        rewriter, loc, attnWeights, value,
        {0, 1}, {0, 1}, {3}, {2}, resultType);
    
    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct YirageToStableHLOPass
    : public PassWrapper<YirageToStableHLOPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(YirageToStableHLOPass)

  StringRef getArgument() const final { return "yirage-to-stablehlo"; }
  StringRef getDescription() const final {
    return "Lower YiRage dialect to StableHLO-compatible operations";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<MatmulToStableHLO>(context);
    patterns.add<RMSNormToStableHLO>(context);
    patterns.add<SoftmaxToStableHLO>(context);
    patterns.add<GatedMLPToStableHLO>(context);
    patterns.add<AttentionToStableHLO>(context);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace ir
} // namespace yirage

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

namespace yirage {

std::unique_ptr<mlir::Pass> createYirageToStableHLOPass() {
  return std::make_unique<ir::YirageToStableHLOPass>();
}

} // namespace yirage
