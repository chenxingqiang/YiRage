// RUN: yirage-opt %s -yirage-to-linalg | FileCheck %s
//
// Unit tests for bug fixes in YirageToLinalg.cpp
// Tests cover: Dequantize, Reshape, ReduceSum/Max axis, GELU approximate,
//              TopK, ArgMax, Split, Quantize, QMatmul

//===----------------------------------------------------------------------===//
// Test 1: Dequantize with scale and zero_point
// Fix: DequantizeOpLowering now applies scale correctly
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_dequantize_with_scale
// CHECK: arith.sitofp
// CHECK: arith.mulf
func.func @test_dequantize_with_scale(%input: tensor<4x8xi8>, %scale: tensor<8xf32>) -> tensor<4x8xf32> {
  %result = yirage.dequantize %input, %scale : tensor<4x8xi8>, tensor<8xf32> -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_dequantize_with_zero_point
// CHECK: arith.sitofp
// CHECK: arith.subf
// CHECK: arith.mulf
func.func @test_dequantize_with_zero_point(%input: tensor<4x8xi8>, %scale: tensor<8xf32>, %zp: tensor<8xi8>) -> tensor<4x8xf32> {
  %result = yirage.dequantize %input, %scale, %zp : tensor<8xi8> : tensor<4x8xi8>, tensor<8xf32> -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

//===----------------------------------------------------------------------===//
// Test 2: Reshape using tensor.reshape instead of tensor.cast
// Fix: ReshapeOpLowering now uses tensor.reshape
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_reshape_expand
// CHECK: tensor.reshape
func.func @test_reshape_expand(%input: tensor<16xf32>) -> tensor<4x4xf32> {
  %result = yirage.reshape %input {shape = [4, 4]} : tensor<16xf32> -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_reshape_collapse
// CHECK: tensor.reshape
func.func @test_reshape_collapse(%input: tensor<4x4xf32>) -> tensor<16xf32> {
  %result = yirage.reshape %input {shape = [16]} : tensor<4x4xf32> -> tensor<16xf32>
  return %result : tensor<16xf32>
}

// CHECK-LABEL: func.func @test_reshape_same_rank
// CHECK: tensor.reshape
func.func @test_reshape_same_rank(%input: tensor<2x8xf32>) -> tensor<4x4xf32> {
  %result = yirage.reshape %input {shape = [4, 4]} : tensor<2x8xf32> -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}

//===----------------------------------------------------------------------===//
// Test 3: ReduceSum/ReduceMax with axis attribute
// Fix: Now respects axis attribute instead of always reducing last dim
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_reduce_sum_axis0
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["reduction", "parallel"]
func.func @test_reduce_sum_axis0(%input: tensor<4x8xf32>) -> tensor<8xf32> {
  %result = yirage.reduce_sum %input {axis = 0 : i64} : tensor<4x8xf32> -> tensor<8xf32>
  return %result : tensor<8xf32>
}

// CHECK-LABEL: func.func @test_reduce_sum_axis_negative
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction"]
func.func @test_reduce_sum_axis_negative(%input: tensor<4x8xf32>) -> tensor<4xf32> {
  %result = yirage.reduce_sum %input {axis = -1 : i64} : tensor<4x8xf32> -> tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK-LABEL: func.func @test_reduce_max_axis1
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
func.func @test_reduce_max_axis1(%input: tensor<2x4x8xf32>) -> tensor<2x8xf32> {
  %result = yirage.reduce_max %input {axis = 1 : i64} : tensor<2x4x8xf32> -> tensor<2x8xf32>
  return %result : tensor<2x8xf32>
}

//===----------------------------------------------------------------------===//
// Test 4: GELU with approximate mode
// Fix: Now supports approximate=true using tanh
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_gelu_exact
// CHECK: math.erf
// CHECK-NOT: math.tanh
func.func @test_gelu_exact(%input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %result = yirage.gelu %input {approximate = false} : tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_gelu_approximate
// CHECK: math.tanh
// CHECK-NOT: math.erf
func.func @test_gelu_approximate(%input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %result = yirage.gelu %input {approximate = true} : tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

//===----------------------------------------------------------------------===//
// Test 5: TopK lowering
// Fix: New lowering pattern added
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_topk
// CHECK: tensor.extract_slice
func.func @test_topk(%input: tensor<8x16xf32>) -> (tensor<8x4xf32>, tensor<8x4xi64>) {
  %values, %indices = yirage.topk %input {k = 4 : i64, axis = -1 : i64} : tensor<8x16xf32> -> (tensor<8x4xf32>, tensor<8x4xi64>)
  return %values, %indices : tensor<8x4xf32>, tensor<8x4xi64>
}

//===----------------------------------------------------------------------===//
// Test 6: ArgMax lowering
// Fix: New lowering pattern added
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_argmax
// CHECK: linalg.generic
// CHECK: arith.cmpf ogt
// CHECK: arith.select
func.func @test_argmax(%input: tensor<8x16xf32>) -> tensor<8xi64> {
  %result = yirage.argmax %input {axis = -1 : i64} : tensor<8x16xf32> -> tensor<8xi64>
  return %result : tensor<8xi64>
}

// CHECK-LABEL: func.func @test_argmax_axis0
// CHECK: linalg.generic
func.func @test_argmax_axis0(%input: tensor<8x16xf32>) -> tensor<16xi64> {
  %result = yirage.argmax %input {axis = 0 : i64} : tensor<8x16xf32> -> tensor<16xi64>
  return %result : tensor<16xi64>
}

//===----------------------------------------------------------------------===//
// Test 7: Split lowering
// Fix: New lowering pattern added
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_split
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice
func.func @test_split(%input: tensor<8x16xf32>) -> (tensor<4x16xf32>, tensor<4x16xf32>) {
  %r0, %r1 = yirage.split %input {num_splits = 2 : i64, axis = 0 : i64} : tensor<8x16xf32> -> (tensor<4x16xf32>, tensor<4x16xf32>)
  return %r0, %r1 : tensor<4x16xf32>, tensor<4x16xf32>
}

// CHECK-LABEL: func.func @test_split_axis1
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice
// CHECK: tensor.extract_slice
func.func @test_split_axis1(%input: tensor<8x16xf32>) -> (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>) {
  %r0, %r1, %r2, %r3 = yirage.split %input {num_splits = 4 : i64, axis = 1 : i64} : tensor<8x16xf32> -> (tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>)
  return %r0, %r1, %r2, %r3 : tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>, tensor<8x4xf32>
}

//===----------------------------------------------------------------------===//
// Test 8: Quantize lowering
// Fix: New lowering pattern added
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_quantize
// CHECK: arith.divf
// CHECK: math.round
// CHECK: arith.fptosi
func.func @test_quantize(%input: tensor<4x8xf32>, %scale: tensor<8xf32>) -> tensor<4x8xi8> {
  %result = yirage.quantize %input, %scale {bits = 8 : i64} : tensor<4x8xf32>, tensor<8xf32> -> tensor<4x8xi8>
  return %result : tensor<4x8xi8>
}

//===----------------------------------------------------------------------===//
// Test 9: QMatmul lowering
// Fix: New lowering pattern added
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_qmatmul
// CHECK: arith.sitofp
// CHECK: arith.mulf
// CHECK: linalg.matmul
func.func @test_qmatmul(%lhs: tensor<4x8xf32>, %rhs_q: tensor<8x16xi8>, %scale: tensor<16xf32>) -> tensor<4x16xf32> {
  %result = yirage.qmatmul %lhs, %rhs_q, %scale {bits = 8 : i64} : tensor<4x8xf32>, tensor<8x16xi8>, tensor<16xf32> -> tensor<4x16xf32>
  return %result : tensor<4x16xf32>
}

//===----------------------------------------------------------------------===//
// Test 10: Existing ops still work correctly
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_matmul
// CHECK: linalg.matmul
func.func @test_matmul(%lhs: tensor<4x8xf32>, %rhs: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %result = yirage.matmul %lhs, %rhs : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
  return %result : tensor<4x16xf32>
}

// CHECK-LABEL: func.func @test_silu
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.divf
// CHECK: arith.mulf
func.func @test_silu(%input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %result = yirage.silu %input : tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_relu
// CHECK: arith.maximumf
func.func @test_relu(%input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %result = yirage.relu %input : tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_softmax
// CHECK: arith.maximumf
// CHECK: math.exp
// CHECK: arith.addf
// CHECK: arith.divf
func.func @test_softmax(%input: tensor<4x8xf32>) -> tensor<4x8xf32> {
  %result = yirage.softmax %input : tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_rms_norm
// CHECK: arith.mulf
// CHECK: math.rsqrt
func.func @test_rms_norm(%input: tensor<4x8xf32>, %gamma: tensor<8xf32>) -> tensor<4x8xf32> {
  %result = yirage.rms_norm %input, %gamma {epsilon = 1.0e-6 : f32} : tensor<4x8xf32>, tensor<8xf32> -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_layer_norm
// CHECK: math.rsqrt
// CHECK: arith.mulf
// CHECK: arith.addf
func.func @test_layer_norm(%input: tensor<4x8xf32>, %gamma: tensor<8xf32>, %beta: tensor<8xf32>) -> tensor<4x8xf32> {
  %result = yirage.layer_norm %input, %gamma, %beta {epsilon = 1.0e-6 : f32} : tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32> -> tensor<4x8xf32>
  return %result : tensor<4x8xf32>
}

// CHECK-LABEL: func.func @test_transpose
// CHECK: linalg.generic
func.func @test_transpose(%input: tensor<4x8xf32>) -> tensor<8x4xf32> {
  %result = yirage.transpose %input {permutation = [1, 0]} : tensor<4x8xf32> -> tensor<8x4xf32>
  return %result : tensor<8x4xf32>
}

// CHECK-LABEL: func.func @test_concat
// CHECK: tensor.concat
func.func @test_concat(%a: tensor<4x8xf32>, %b: tensor<4x8xf32>) -> tensor<8x8xf32> {
  %result = yirage.concat %a, %b {axis = 0 : i64} : tensor<4x8xf32>, tensor<4x8xf32> -> tensor<8x8xf32>
  return %result : tensor<8x8xf32>
}

// CHECK-LABEL: func.func @test_linear_with_bias
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: arith.addf
func.func @test_linear_with_bias(%input: tensor<4x8xf32>, %weight: tensor<8x16xf32>, %bias: tensor<16xf32>) -> tensor<4x16xf32> {
  %result = yirage.linear %input, %weight, %bias : tensor<16xf32> : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
  return %result : tensor<4x16xf32>
}
