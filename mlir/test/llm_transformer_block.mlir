// RUN: yirage-opt %s -yirage-to-linalg | FileCheck %s
//
// Comprehensive LLM Transformer Block Test
// Tests all major LLM operators in a realistic transformer configuration
//
// Configuration (LLaMA-7B style):
//   - Hidden dim: 4096
//   - Intermediate dim: 11008 (for SwiGLU MLP)
//   - Num heads: 32
//   - Head dim: 128
//   - Sequence length: 2048
//   - Batch size: 1

// ===----------------------------------------------------------------------===//
// Test 1: Matrix Multiplication (Matmul)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_matmul
func.func @test_matmul(%input: tensor<2048x4096xf16>, %weight: tensor<4096x4096xf16>) -> tensor<2048x4096xf16> {
  // CHECK: linalg.matmul
  %0 = yirage.matmul %input, %weight : tensor<2048x4096xf16>, tensor<4096x4096xf16> -> tensor<2048x4096xf16>
  return %0 : tensor<2048x4096xf16>
}

// ===----------------------------------------------------------------------===//
// Test 2: Batch Matrix Multiplication
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_batch_matmul
func.func @test_batch_matmul(%q: tensor<1x32x2048x128xf16>, %k: tensor<1x32x2048x128xf16>) -> tensor<1x32x2048x2048xf16> {
  // CHECK: linalg.batch_matmul
  %0 = yirage.batch_matmul %q, %k {transpose_rhs = true} : tensor<1x32x2048x128xf16>, tensor<1x32x2048x128xf16> -> tensor<1x32x2048x2048xf16>
  return %0 : tensor<1x32x2048x2048xf16>
}

// ===----------------------------------------------------------------------===//
// Test 3: RMS Normalization (Pre-attention norm)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_rms_norm
func.func @test_rms_norm(%input: tensor<1x2048x4096xf16>, %gamma: tensor<4096xf16>) -> tensor<1x2048x4096xf16> {
  // CHECK: linalg.generic
  %0 = yirage.rms_norm %input, %gamma {epsilon = 1.0e-6 : f32} : tensor<1x2048x4096xf16>, tensor<4096xf16> -> tensor<1x2048x4096xf16>
  return %0 : tensor<1x2048x4096xf16>
}

// ===----------------------------------------------------------------------===//
// Test 4: Layer Normalization
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_layer_norm
func.func @test_layer_norm(%input: tensor<1x197x768xf16>, %gamma: tensor<768xf16>, %beta: tensor<768xf16>) -> tensor<1x197x768xf16> {
  // CHECK: linalg.generic
  %0 = yirage.layer_norm %input, %gamma, %beta {epsilon = 1.0e-6 : f32} : tensor<1x197x768xf16>, tensor<768xf16>, tensor<768xf16> -> tensor<1x197x768xf16>
  return %0 : tensor<1x197x768xf16>
}

// ===----------------------------------------------------------------------===//
// Test 5: SiLU Activation (for SwiGLU MLP)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_silu
func.func @test_silu(%input: tensor<1x2048x11008xf16>) -> tensor<1x2048x11008xf16> {
  // CHECK: linalg.generic
  // CHECK: arith.negf
  // CHECK: math.exp
  // CHECK: arith.mulf
  %0 = yirage.silu %input : tensor<1x2048x11008xf16>
  return %0 : tensor<1x2048x11008xf16>
}

// ===----------------------------------------------------------------------===//
// Test 6: GELU Activation
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_gelu
func.func @test_gelu(%input: tensor<1x197x3072xf16>) -> tensor<1x197x3072xf16> {
  // CHECK: linalg.generic
  // CHECK: math.erf
  %0 = yirage.gelu %input : tensor<1x197x3072xf16>
  return %0 : tensor<1x197x3072xf16>
}

// ===----------------------------------------------------------------------===//
// Test 7: ReLU Activation
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_relu
func.func @test_relu(%input: tensor<32x128xf16>) -> tensor<32x128xf16> {
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  %0 = yirage.relu %input : tensor<32x128xf16>
  return %0 : tensor<32x128xf16>
}

// ===----------------------------------------------------------------------===//
// Test 8: Softmax
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_softmax
func.func @test_softmax(%input: tensor<1x32x2048x2048xf16>) -> tensor<1x32x2048x2048xf16> {
  // CHECK: linalg.generic
  // CHECK: math.exp
  %0 = yirage.softmax %input {axis = -1 : i64} : tensor<1x32x2048x2048xf16>
  return %0 : tensor<1x32x2048x2048xf16>
}

// ===----------------------------------------------------------------------===//
// Test 9: Gated MLP (SwiGLU)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_gated_mlp
func.func @test_gated_mlp(
    %input: tensor<2048x4096xf16>,
    %gate_weight: tensor<4096x11008xf16>,
    %up_weight: tensor<4096x11008xf16>,
    %down_weight: tensor<11008x4096xf16>) -> tensor<2048x4096xf16> {
  // CHECK: linalg.matmul
  // CHECK: linalg.matmul
  // CHECK: linalg.generic
  // CHECK: linalg.matmul
  %0 = yirage.gated_mlp %input, %gate_weight, %up_weight, %down_weight : tensor<2048x4096xf16>, tensor<4096x11008xf16>, tensor<4096x11008xf16>, tensor<11008x4096xf16> -> tensor<2048x4096xf16>
  return %0 : tensor<2048x4096xf16>
}

// ===----------------------------------------------------------------------===//
// Test 10: Linear Layer
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_linear
func.func @test_linear(%input: tensor<2048x4096xf16>, %weight: tensor<4096x12288xf16>) -> tensor<2048x12288xf16> {
  // CHECK: linalg.matmul
  %0 = yirage.linear %input, %weight : tensor<2048x4096xf16>, tensor<4096x12288xf16> -> tensor<2048x12288xf16>
  return %0 : tensor<2048x12288xf16>
}

// ===----------------------------------------------------------------------===//
// Test 11: Attention (Scaled Dot-Product)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_attention
func.func @test_attention(
    %query: tensor<1x32x2048x128xf16>,
    %key: tensor<1x32x2048x128xf16>,
    %value: tensor<1x32x2048x128xf16>) -> tensor<1x32x2048x128xf16> {
  // CHECK: linalg.generic
  %0 = yirage.attention %query, %key, %value {causal = true} : tensor<1x32x2048x128xf16>, tensor<1x32x2048x128xf16>, tensor<1x32x2048x128xf16> -> tensor<1x32x2048x128xf16>
  return %0 : tensor<1x32x2048x128xf16>
}

// ===----------------------------------------------------------------------===//
// Test 12: Rotary Position Embedding (RoPE)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_rope
func.func @test_rope(
    %input: tensor<1x32x2048x128xf16>,
    %cos_cache: tensor<2048x64xf32>,
    %sin_cache: tensor<2048x64xf32>) -> tensor<1x32x2048x128xf16> {
  // CHECK: linalg.generic
  %0 = yirage.rope %input, %cos_cache, %sin_cache : tensor<1x32x2048x128xf16>, tensor<2048x64xf32>, tensor<2048x64xf32> -> tensor<1x32x2048x128xf16>
  return %0 : tensor<1x32x2048x128xf16>
}

// ===----------------------------------------------------------------------===//
// Test 13: Embedding Lookup
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_embedding
func.func @test_embedding(%table: tensor<128256x4096xf16>, %indices: tensor<1x2048xi32>) -> tensor<1x2048x4096xf16> {
  // CHECK: tensor.empty
  %0 = yirage.embedding %table, %indices : tensor<128256x4096xf16>, tensor<1x2048xi32> -> tensor<1x2048x4096xf16>
  return %0 : tensor<1x2048x4096xf16>
}

// ===----------------------------------------------------------------------===//
// Test 14: Transpose
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_transpose
func.func @test_transpose(%input: tensor<1x2048x32x128xf16>) -> tensor<1x32x2048x128xf16> {
  // CHECK: linalg.generic
  %0 = yirage.transpose %input {permutation = [0, 2, 1, 3]} : tensor<1x2048x32x128xf16> -> tensor<1x32x2048x128xf16>
  return %0 : tensor<1x32x2048x128xf16>
}

// ===----------------------------------------------------------------------===//
// Test 15: Concat
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_concat
func.func @test_concat(%a: tensor<1x1024x4096xf16>, %b: tensor<1x1024x4096xf16>) -> tensor<1x2048x4096xf16> {
  // CHECK: tensor.concat
  %0 = yirage.concat %a, %b {axis = 1 : i64} : tensor<1x1024x4096xf16>, tensor<1x1024x4096xf16> -> tensor<1x2048x4096xf16>
  return %0 : tensor<1x2048x4096xf16>
}

// ===----------------------------------------------------------------------===//
// Test 16: Reduce Sum
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_reduce_sum
func.func @test_reduce_sum(%input: tensor<32x2048x128xf16>) -> tensor<32x2048xf16> {
  // CHECK: linalg.generic
  // CHECK: arith.addf
  %0 = yirage.reduce_sum %input {axis = -1 : i64} : tensor<32x2048x128xf16> -> tensor<32x2048xf16>
  return %0 : tensor<32x2048xf16>
}

// ===----------------------------------------------------------------------===//
// Test 17: Reduce Max
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_reduce_max
func.func @test_reduce_max(%input: tensor<32x2048x128xf16>) -> tensor<32x2048xf16> {
  // CHECK: linalg.fill
  %0 = yirage.reduce_max %input {axis = -1 : i64} : tensor<32x2048x128xf16> -> tensor<32x2048xf16>
  return %0 : tensor<32x2048xf16>
}

// ===----------------------------------------------------------------------===//
// Test 18: Dequantize
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_dequantize
func.func @test_dequantize(%input: tensor<4096x4096xi8>, %scale: tensor<4096xf16>) -> tensor<4096x4096xf16> {
  // CHECK: linalg.generic
  %0 = yirage.dequantize %input, %scale : tensor<4096x4096xi8>, tensor<4096xf16> -> tensor<4096x4096xf16>
  return %0 : tensor<4096x4096xf16>
}

// ===----------------------------------------------------------------------===//
// Test 19: Complete LLaMA-style Transformer Decoder Layer
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @llama_decoder_layer
func.func @llama_decoder_layer(
    // Input hidden states
    %hidden_states: tensor<2048x4096xf16>,
    // RMSNorm weights
    %input_layernorm_weight: tensor<4096xf16>,
    %post_attention_layernorm_weight: tensor<4096xf16>,
    // Attention weights (QKV projection)
    %qkv_weight: tensor<4096x12288xf16>,
    %o_proj_weight: tensor<4096x4096xf16>,
    // MLP weights
    %gate_proj_weight: tensor<4096x11008xf16>,
    %up_proj_weight: tensor<4096x11008xf16>,
    %down_proj_weight: tensor<11008x4096xf16>
) -> tensor<2048x4096xf16> {
    
    // === Pre-Attention RMSNorm ===
    // CHECK: linalg.generic
    %normed = yirage.rms_norm %hidden_states, %input_layernorm_weight {epsilon = 1.0e-6 : f32}
        : tensor<2048x4096xf16>, tensor<4096xf16> -> tensor<2048x4096xf16>
    
    // === QKV Projection ===
    // CHECK: linalg.matmul
    %qkv = yirage.linear %normed, %qkv_weight
        : tensor<2048x4096xf16>, tensor<4096x12288xf16> -> tensor<2048x12288xf16>
    
    // === Output Projection ===
    // (Skipping attention computation for simplicity)
    // CHECK: linalg.matmul
    %attn_out = yirage.linear %normed, %o_proj_weight
        : tensor<2048x4096xf16>, tensor<4096x4096xf16> -> tensor<2048x4096xf16>
    
    // === Residual Connection ===
    %residual1 = arith.addf %hidden_states, %attn_out : tensor<2048x4096xf16>
    
    // === Post-Attention RMSNorm ===
    // CHECK: linalg.generic
    %normed2 = yirage.rms_norm %residual1, %post_attention_layernorm_weight {epsilon = 1.0e-6 : f32}
        : tensor<2048x4096xf16>, tensor<4096xf16> -> tensor<2048x4096xf16>
    
    // === Gated MLP (SwiGLU) ===
    // CHECK: linalg.matmul
    %mlp_out = yirage.gated_mlp %normed2, %gate_proj_weight, %up_proj_weight, %down_proj_weight
        : tensor<2048x4096xf16>, tensor<4096x11008xf16>, tensor<4096x11008xf16>, tensor<11008x4096xf16> -> tensor<2048x4096xf16>
    
    // === Final Residual Connection ===
    %output = arith.addf %residual1, %mlp_out : tensor<2048x4096xf16>
    
    return %output : tensor<2048x4096xf16>
}
