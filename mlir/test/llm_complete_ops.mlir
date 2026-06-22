// RUN: yirage-opt %s -yirage-to-linalg | FileCheck %s
//
// Complete LLM Operations Test
// Tests all major LLM operators with proper lowering
//
// Configuration (LLaMA-7B style):
//   - Hidden dim: 4096
//   - Intermediate dim: 11008 (SwiGLU)
//   - Num heads: 32
//   - Head dim: 128

// ===----------------------------------------------------------------------===//
// Test 1: RMSNorm with Proper Broadcasting
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_rms_norm
func.func @test_rms_norm(%input: tensor<2x2048x4096xf32>, %gamma: tensor<4096xf32>) -> tensor<2x2048x4096xf32> {
  // Should compute: output = x / sqrt(mean(x^2) + eps) * gamma
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: arith.divf
  // CHECK: math.rsqrt
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.mulf
  %0 = yirage.rms_norm %input, %gamma {epsilon = 1.0e-6 : f32}
      : tensor<2x2048x4096xf32>, tensor<4096xf32> -> tensor<2x2048x4096xf32>
  return %0 : tensor<2x2048x4096xf32>
}

// ===----------------------------------------------------------------------===//
// Test 2: LayerNorm with Proper Broadcasting
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_layer_norm
func.func @test_layer_norm(%input: tensor<2x197x768xf32>, %gamma: tensor<768xf32>, %beta: tensor<768xf32>) -> tensor<2x197x768xf32> {
  // Should compute: output = (x - mean) / sqrt(var + eps) * gamma + beta
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: arith.divf
  // CHECK: linalg.generic
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: math.rsqrt
  // CHECK: linalg.generic
  // CHECK: arith.subf
  // CHECK: arith.mulf
  // CHECK: arith.mulf
  // CHECK: arith.addf
  %0 = yirage.layer_norm %input, %gamma, %beta {epsilon = 1.0e-6 : f32}
      : tensor<2x197x768xf32>, tensor<768xf32>, tensor<768xf32> -> tensor<2x197x768xf32>
  return %0 : tensor<2x197x768xf32>
}

// ===----------------------------------------------------------------------===//
// Test 3: Complete Softmax with Numerical Stability
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_softmax
func.func @test_softmax(%input: tensor<32x2048x2048xf32>) -> tensor<32x2048x2048xf32> {
  // Should compute: exp(x - max) / sum(exp(x - max))
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  // CHECK: linalg.generic
  // CHECK: arith.subf
  // CHECK: math.exp
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: arith.divf
  %0 = yirage.softmax %input {axis = -1 : i64} : tensor<32x2048x2048xf32>
  return %0 : tensor<32x2048x2048xf32>
}

// ===----------------------------------------------------------------------===//
// Test 4: Complete Attention (Q@K^T/sqrt(d) -> softmax -> @V)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_attention
func.func @test_attention(
    %query: tensor<1x32x2048x128xf32>,
    %key: tensor<1x32x2048x128xf32>,
    %value: tensor<1x32x2048x128xf32>) -> tensor<1x32x2048x128xf32> {
  // Should compute:
  // 1. scores = Q @ K^T (batch matmul with transpose)
  // 2. scaled_scores = scores / sqrt(head_dim)
  // 3. causal_scores = apply_causal_mask(scaled_scores)
  // 4. attn_weights = softmax(causal_scores)
  // 5. output = attn_weights @ V
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: linalg.generic
  // CHECK: arith.select
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  // CHECK: linalg.generic
  // CHECK: arith.subf
  // CHECK: math.exp
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: arith.divf
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.addf
  %0 = yirage.attention %query, %key, %value {causal = true}
      : tensor<1x32x2048x128xf32>, tensor<1x32x2048x128xf32>, tensor<1x32x2048x128xf32>
      -> tensor<1x32x2048x128xf32>
  return %0 : tensor<1x32x2048x128xf32>
}

// ===----------------------------------------------------------------------===//
// Test 5: Gated MLP (SwiGLU)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_gated_mlp
func.func @test_gated_mlp(
    %input: tensor<2048x4096xf32>,
    %gate_weight: tensor<4096x11008xf32>,
    %up_weight: tensor<4096x11008xf32>,
    %down_weight: tensor<11008x4096xf32>) -> tensor<2048x4096xf32> {
  // Should compute:
  // gate = input @ gate_weight
  // up = input @ up_weight
  // intermediate = silu(gate) * up
  // output = intermediate @ down_weight
  // CHECK: linalg.fill
  // CHECK: linalg.matmul
  // CHECK: linalg.fill
  // CHECK: linalg.matmul
  // CHECK: linalg.generic
  // CHECK: arith.negf
  // CHECK: math.exp
  // CHECK: arith.divf
  // CHECK: arith.mulf
  // CHECK: arith.mulf
  // CHECK: linalg.fill
  // CHECK: linalg.matmul
  %0 = yirage.gated_mlp %input, %gate_weight, %up_weight, %down_weight
      : tensor<2048x4096xf32>, tensor<4096x11008xf32>, tensor<4096x11008xf32>, tensor<11008x4096xf32>
      -> tensor<2048x4096xf32>
  return %0 : tensor<2048x4096xf32>
}

// ===----------------------------------------------------------------------===//
// Test 6: 4D Batch Matmul (Attention Patterns)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_batch_matmul_4d
func.func @test_batch_matmul_4d(
    %q: tensor<1x32x2048x128xf32>,
    %k: tensor<1x32x2048x128xf32>) -> tensor<1x32x2048x2048xf32> {
  // Q @ K^T pattern for attention
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.addf
  %0 = yirage.batch_matmul %q, %k {transpose_rhs = true}
      : tensor<1x32x2048x128xf32>, tensor<1x32x2048x128xf32>
      -> tensor<1x32x2048x2048xf32>
  return %0 : tensor<1x32x2048x2048xf32>
}

// ===----------------------------------------------------------------------===//
// Test 7: Reduce Sum with Reduction
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_reduce_sum
func.func @test_reduce_sum(%input: tensor<32x2048x128xf32>) -> tensor<32x2048xf32> {
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.addf
  %0 = yirage.reduce_sum %input {axis = -1 : i64}
      : tensor<32x2048x128xf32> -> tensor<32x2048xf32>
  return %0 : tensor<32x2048xf32>
}

// ===----------------------------------------------------------------------===//
// Test 8: Reduce Max with Reduction
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_reduce_max
func.func @test_reduce_max(%input: tensor<32x2048x128xf32>) -> tensor<32x2048xf32> {
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  %0 = yirage.reduce_max %input {axis = -1 : i64}
      : tensor<32x2048x128xf32> -> tensor<32x2048xf32>
  return %0 : tensor<32x2048xf32>
}

// ===----------------------------------------------------------------------===//
// Test 9: Complete LLaMA Decoder Layer
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @llama_decoder_layer
func.func @llama_decoder_layer(
    %hidden_states: tensor<2048x4096xf32>,
    %input_layernorm_weight: tensor<4096xf32>,
    %post_attention_layernorm_weight: tensor<4096xf32>,
    %q_proj_weight: tensor<4096x4096xf32>,
    %k_proj_weight: tensor<4096x4096xf32>,
    %v_proj_weight: tensor<4096x4096xf32>,
    %o_proj_weight: tensor<4096x4096xf32>,
    %gate_proj_weight: tensor<4096x11008xf32>,
    %up_proj_weight: tensor<4096x11008xf32>,
    %down_proj_weight: tensor<11008x4096xf32>
) -> tensor<2048x4096xf32> {
    
    // === Pre-Attention RMSNorm ===
    // CHECK: linalg.generic
    %normed = yirage.rms_norm %hidden_states, %input_layernorm_weight {epsilon = 1.0e-6 : f32}
        : tensor<2048x4096xf32>, tensor<4096xf32> -> tensor<2048x4096xf32>
    
    // === QKV Projection ===
    // CHECK: linalg.matmul
    %q = yirage.linear %normed, %q_proj_weight
        : tensor<2048x4096xf32>, tensor<4096x4096xf32> -> tensor<2048x4096xf32>
    // CHECK: linalg.matmul
    %k = yirage.linear %normed, %k_proj_weight
        : tensor<2048x4096xf32>, tensor<4096x4096xf32> -> tensor<2048x4096xf32>
    // CHECK: linalg.matmul
    %v = yirage.linear %normed, %v_proj_weight
        : tensor<2048x4096xf32>, tensor<4096x4096xf32> -> tensor<2048x4096xf32>
    
    // === Output Projection (simplified, skip attention) ===
    // CHECK: linalg.matmul
    %attn_out = yirage.linear %q, %o_proj_weight
        : tensor<2048x4096xf32>, tensor<4096x4096xf32> -> tensor<2048x4096xf32>
    
    // === Residual Connection ===
    %residual1 = arith.addf %hidden_states, %attn_out : tensor<2048x4096xf32>
    
    // === Post-Attention RMSNorm ===
    // CHECK: linalg.generic
    %normed2 = yirage.rms_norm %residual1, %post_attention_layernorm_weight {epsilon = 1.0e-6 : f32}
        : tensor<2048x4096xf32>, tensor<4096xf32> -> tensor<2048x4096xf32>
    
    // === Gated MLP (SwiGLU) ===
    // CHECK: linalg.matmul
    // CHECK: linalg.matmul
    // CHECK: linalg.generic
    // CHECK: linalg.matmul
    %mlp_out = yirage.gated_mlp %normed2, %gate_proj_weight, %up_proj_weight, %down_proj_weight
        : tensor<2048x4096xf32>, tensor<4096x11008xf32>, tensor<4096x11008xf32>, tensor<11008x4096xf32>
        -> tensor<2048x4096xf32>
    
    // === Final Residual Connection ===
    %output = arith.addf %residual1, %mlp_out : tensor<2048x4096xf32>
    
    return %output : tensor<2048x4096xf32>
}
