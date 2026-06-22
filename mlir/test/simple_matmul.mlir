// RUN: yirage-opt %s -yirage-to-linalg | FileCheck %s

// Simple test for yirage.matmul lowering to linalg.matmul

func.func @test_matmul(%arg0: tensor<32x64xf32>, %arg1: tensor<64x128xf32>) -> tensor<32x128xf32> {
  // CHECK: linalg.matmul
  %0 = yirage.matmul %arg0, %arg1 : tensor<32x64xf32>, tensor<64x128xf32> -> tensor<32x128xf32>
  return %0 : tensor<32x128xf32>
}

func.func @test_silu(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.negf
  // CHECK: math.exp
  // CHECK: arith.divf
  // CHECK: arith.mulf
  %0 = yirage.silu %arg0 : tensor<32x128xf32>
  return %0 : tensor<32x128xf32>
}

func.func @test_gelu(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32> {
  // CHECK: linalg.generic
  // CHECK: math.erf
  %0 = yirage.gelu %arg0 : tensor<32x128xf32>
  return %0 : tensor<32x128xf32>
}

func.func @test_relu(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32> {
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  %0 = yirage.relu %arg0 : tensor<32x128xf32>
  return %0 : tensor<32x128xf32>
}
