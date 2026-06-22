// RUN: yirage-opt %s -yirage-to-linalg | FileCheck %s
//
// Basic LLM Operations Test
// Tests core LLM operators that have been verified to work

// ===----------------------------------------------------------------------===//
// Test 1: 2D Matrix Multiplication
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_matmul_2d
// CHECK: linalg.fill
// CHECK: linalg.matmul
func.func @test_matmul_2d(%input: tensor<2048x4096xf16>, %weight: tensor<4096x4096xf16>) -> tensor<2048x4096xf16> {
  %0 = yirage.matmul %input, %weight : tensor<2048x4096xf16>, tensor<4096x4096xf16> -> tensor<2048x4096xf16>
  return %0 : tensor<2048x4096xf16>
}

// ===----------------------------------------------------------------------===//
// Test 2: SiLU Activation
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_silu
// CHECK: linalg.generic
// CHECK: arith.negf
// CHECK: math.exp
// CHECK: arith.divf
// CHECK: arith.mulf
func.func @test_silu(%input: tensor<2048x11008xf16>) -> tensor<2048x11008xf16> {
  %0 = yirage.silu %input : tensor<2048x11008xf16>
  return %0 : tensor<2048x11008xf16>
}

// ===----------------------------------------------------------------------===//
// Test 3: GELU Activation
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_gelu
// CHECK: linalg.generic
// CHECK: math.erf
func.func @test_gelu(%input: tensor<197x3072xf16>) -> tensor<197x3072xf16> {
  %0 = yirage.gelu %input : tensor<197x3072xf16>
  return %0 : tensor<197x3072xf16>
}

// ===----------------------------------------------------------------------===//
// Test 4: ReLU Activation
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_relu
// CHECK: linalg.generic
// CHECK: arith.maximumf
func.func @test_relu(%input: tensor<32x128xf16>) -> tensor<32x128xf16> {
  %0 = yirage.relu %input : tensor<32x128xf16>
  return %0 : tensor<32x128xf16>
}

// ===----------------------------------------------------------------------===//
// Test 5: Softmax
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_softmax
// CHECK: linalg.generic
// CHECK: math.exp
func.func @test_softmax(%input: tensor<32x2048x2048xf16>) -> tensor<32x2048x2048xf16> {
  %0 = yirage.softmax %input {axis = -1 : i64} : tensor<32x2048x2048xf16>
  return %0 : tensor<32x2048x2048xf16>
}

// ===----------------------------------------------------------------------===//
// Test 6: Gated MLP (SwiGLU)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_gated_mlp
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: linalg.fill
// CHECK: linalg.matmul
func.func @test_gated_mlp(
    %input: tensor<2048x4096xf16>,
    %gate_weight: tensor<4096x11008xf16>,
    %up_weight: tensor<4096x11008xf16>,
    %down_weight: tensor<11008x4096xf16>) -> tensor<2048x4096xf16> {
  %0 = yirage.gated_mlp %input, %gate_weight, %up_weight, %down_weight 
      : tensor<2048x4096xf16>, tensor<4096x11008xf16>, tensor<4096x11008xf16>, tensor<11008x4096xf16> 
      -> tensor<2048x4096xf16>
  return %0 : tensor<2048x4096xf16>
}

// ===----------------------------------------------------------------------===//
// Test 7: Linear Layer
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_linear
// CHECK: linalg.fill
// CHECK: linalg.matmul
func.func @test_linear(%input: tensor<2048x4096xf16>, %weight: tensor<4096x12288xf16>) -> tensor<2048x12288xf16> {
  %0 = yirage.linear %input, %weight : tensor<2048x4096xf16>, tensor<4096x12288xf16> -> tensor<2048x12288xf16>
  return %0 : tensor<2048x12288xf16>
}

// ===----------------------------------------------------------------------===//
// Test 8: Transpose
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_transpose
// CHECK: linalg.generic
func.func @test_transpose(%input: tensor<2048x32x128xf16>) -> tensor<32x2048x128xf16> {
  %0 = yirage.transpose %input {permutation = [1, 0, 2]} : tensor<2048x32x128xf16> -> tensor<32x2048x128xf16>
  return %0 : tensor<32x2048x128xf16>
}

// ===----------------------------------------------------------------------===//
// Test 9: Concat
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_concat
// CHECK: tensor.concat
func.func @test_concat(%a: tensor<1024x4096xf16>, %b: tensor<1024x4096xf16>) -> tensor<2048x4096xf16> {
  %0 = yirage.concat %a, %b {axis = 0 : i64} : tensor<1024x4096xf16>, tensor<1024x4096xf16> -> tensor<2048x4096xf16>
  return %0 : tensor<2048x4096xf16>
}

// ===----------------------------------------------------------------------===//
// Test 10: Complete MLP Block (Practical LLM Pattern)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @llm_mlp_block
func.func @llm_mlp_block(
    %hidden_states: tensor<2048x4096xf16>,
    %gate_proj_weight: tensor<4096x11008xf16>,
    %up_proj_weight: tensor<4096x11008xf16>,
    %down_proj_weight: tensor<11008x4096xf16>
) -> tensor<2048x4096xf16> {
    
    // gate = hidden @ gate_proj
    // CHECK: linalg.matmul
    %gate = yirage.linear %hidden_states, %gate_proj_weight
        : tensor<2048x4096xf16>, tensor<4096x11008xf16> -> tensor<2048x11008xf16>
    
    // up = hidden @ up_proj
    // CHECK: linalg.matmul
    %up = yirage.linear %hidden_states, %up_proj_weight
        : tensor<2048x4096xf16>, tensor<4096x11008xf16> -> tensor<2048x11008xf16>
    
    // gate_activated = silu(gate)
    // CHECK: linalg.generic
    %gate_activated = yirage.silu %gate : tensor<2048x11008xf16>
    
    // intermediate = gate_activated * up
    %intermediate = arith.mulf %gate_activated, %up : tensor<2048x11008xf16>
    
    // output = intermediate @ down_proj
    // CHECK: linalg.matmul
    %output = yirage.linear %intermediate, %down_proj_weight
        : tensor<2048x11008xf16>, tensor<11008x4096xf16> -> tensor<2048x4096xf16>
    
    return %output : tensor<2048x4096xf16>
}

// ===----------------------------------------------------------------------===//
// Test 11: QKV Projection Pattern
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @qkv_projection
func.func @qkv_projection(
    %hidden_states: tensor<2048x4096xf16>,
    %q_proj: tensor<4096x4096xf16>,
    %k_proj: tensor<4096x4096xf16>,
    %v_proj: tensor<4096x4096xf16>
) -> (tensor<2048x4096xf16>, tensor<2048x4096xf16>, tensor<2048x4096xf16>) {
    
    // Q = hidden @ q_proj
    // CHECK: linalg.matmul
    %q = yirage.linear %hidden_states, %q_proj
        : tensor<2048x4096xf16>, tensor<4096x4096xf16> -> tensor<2048x4096xf16>
    
    // K = hidden @ k_proj
    // CHECK: linalg.matmul
    %k = yirage.linear %hidden_states, %k_proj
        : tensor<2048x4096xf16>, tensor<4096x4096xf16> -> tensor<2048x4096xf16>
    
    // V = hidden @ v_proj
    // CHECK: linalg.matmul
    %v = yirage.linear %hidden_states, %v_proj
        : tensor<2048x4096xf16>, tensor<4096x4096xf16> -> tensor<2048x4096xf16>
    
    return %q, %k, %v : tensor<2048x4096xf16>, tensor<2048x4096xf16>, tensor<2048x4096xf16>
}
