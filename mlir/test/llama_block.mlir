// RUN: yirage-opt %s --yirage-to-linalg | FileCheck %s

// Test: LLaMA-style transformer block operations

// CHECK-LABEL: func.func @test_rms_norm
func.func @test_rms_norm(
    %input: tensor<1x2048x4096xf16>,
    %gamma: tensor<4096xf16>
) -> tensor<1x2048x4096xf16> {
  // CHECK: linalg.generic
  // CHECK: math.sqrt
  %out = yirage.rms_norm %input, %gamma {epsilon = 1.0e-6 : f32}
      : tensor<1x2048x4096xf16>, tensor<4096xf16> -> tensor<1x2048x4096xf16>
  return %out : tensor<1x2048x4096xf16>
}

// CHECK-LABEL: func.func @test_matmul
func.func @test_matmul(
    %A: tensor<1024x4096xf16>,
    %B: tensor<4096x4096xf16>
) -> tensor<1024x4096xf16> {
  // CHECK: linalg.matmul
  %C = yirage.matmul %A, %B
      : tensor<1024x4096xf16>, tensor<4096x4096xf16> -> tensor<1024x4096xf16>
  return %C : tensor<1024x4096xf16>
}

// CHECK-LABEL: func.func @test_silu
func.func @test_silu(%input: tensor<1x2048x11008xf16>) -> tensor<1x2048x11008xf16> {
  // CHECK: linalg.generic
  // CHECK: arith.negf
  // CHECK: math.exp
  // CHECK: arith.divf
  // CHECK: arith.mulf
  %out = yirage.silu %input : tensor<1x2048x11008xf16>
  return %out : tensor<1x2048x11008xf16>
}

// CHECK-LABEL: func.func @test_gelu
func.func @test_gelu(%input: tensor<1x197x3072xf16>) -> tensor<1x197x3072xf16> {
  // CHECK: linalg.generic
  // CHECK: math.erf
  %out = yirage.gelu %input : tensor<1x197x3072xf16>
  return %out : tensor<1x197x3072xf16>
}

// CHECK-LABEL: func.func @test_softmax
func.func @test_softmax(%input: tensor<1x32x2048x2048xf32>) -> tensor<1x32x2048x2048xf32> {
  // CHECK: linalg.softmax
  %out = yirage.softmax %input {axis = -1 : i64} : tensor<1x32x2048x2048xf32>
  return %out : tensor<1x32x2048x2048xf32>
}

// Test: Full LLaMA transformer block (not lowered, just parsed)
func.func @llama_transformer_block(
    %hidden: tensor<1x2048x4096xf16>,
    %rms_attn: tensor<4096xf16>,
    %wq: tensor<4096x4096xf16>,
    %wk: tensor<4096x1024xf16>,
    %wv: tensor<4096x1024xf16>,
    %wo: tensor<4096x4096xf16>,
    %rms_mlp: tensor<4096xf16>,
    %gate: tensor<4096x11008xf16>,
    %up: tensor<4096x11008xf16>,
    %down: tensor<11008x4096xf16>,
    %cos: tensor<2048x64xf32>,
    %sin: tensor<2048x64xf32>
) -> tensor<1x2048x4096xf16> {
  
  // Self-attention block
  %normed1 = yirage.rms_norm %hidden, %rms_attn {epsilon = 1.0e-6 : f32}
      : tensor<1x2048x4096xf16>, tensor<4096xf16> -> tensor<1x2048x4096xf16>
  
  %q = yirage.matmul %normed1, %wq
      : tensor<1x2048x4096xf16>, tensor<4096x4096xf16> -> tensor<1x2048x4096xf16>
  %k = yirage.matmul %normed1, %wk
      : tensor<1x2048x4096xf16>, tensor<4096x1024xf16> -> tensor<1x2048x1024xf16>
  %v = yirage.matmul %normed1, %wv
      : tensor<1x2048x4096xf16>, tensor<4096x1024xf16> -> tensor<1x2048x1024xf16>
  
  // Reshape to heads: [B, S, H*D] -> [B, H, S, D]
  %q_heads = yirage.reshape %q {shape = [1, 2048, 32, 128]}
      : tensor<1x2048x4096xf16> -> tensor<1x2048x32x128xf16>
  %q_t = yirage.transpose %q_heads {permutation = [0, 2, 1, 3]}
      : tensor<1x2048x32x128xf16> -> tensor<1x32x2048x128xf16>
  
  %k_heads = yirage.reshape %k {shape = [1, 2048, 8, 128]}
      : tensor<1x2048x1024xf16> -> tensor<1x2048x8x128xf16>
  %k_t = yirage.transpose %k_heads {permutation = [0, 2, 1, 3]}
      : tensor<1x2048x8x128xf16> -> tensor<1x8x2048x128xf16>
  
  %v_heads = yirage.reshape %v {shape = [1, 2048, 8, 128]}
      : tensor<1x2048x1024xf16> -> tensor<1x2048x8x128xf16>
  %v_t = yirage.transpose %v_heads {permutation = [0, 2, 1, 3]}
      : tensor<1x2048x8x128xf16> -> tensor<1x8x2048x128xf16>
  
  // Apply RoPE
  %q_rope = yirage.rope %q_t, %cos, %sin
      : tensor<1x32x2048x128xf16>, tensor<2048x64xf32>, tensor<2048x64xf32>
      -> tensor<1x32x2048x128xf16>
  %k_rope = yirage.rope %k_t, %cos, %sin
      : tensor<1x8x2048x128xf16>, tensor<2048x64xf32>, tensor<2048x64xf32>
      -> tensor<1x8x2048x128xf16>
  
  // Attention (GQA with 8 KV heads)
  %attn = yirage.attention %q_rope, %k_rope, %v_t {
    causal = true,
    num_kv_heads = 8 : i64
  } : tensor<1x32x2048x128xf16>, tensor<1x8x2048x128xf16>, tensor<1x8x2048x128xf16>
    -> tensor<1x32x2048x128xf16>
  
  // Reshape back: [B, H, S, D] -> [B, S, H*D]
  %attn_t = yirage.transpose %attn {permutation = [0, 2, 1, 3]}
      : tensor<1x32x2048x128xf16> -> tensor<1x2048x32x128xf16>
  %attn_flat = yirage.reshape %attn_t {shape = [1, 2048, 4096]}
      : tensor<1x2048x32x128xf16> -> tensor<1x2048x4096xf16>
  
  // Output projection
  %attn_proj = yirage.matmul %attn_flat, %wo
      : tensor<1x2048x4096xf16>, tensor<4096x4096xf16> -> tensor<1x2048x4096xf16>
  
  // Residual
  %attn_res = arith.addf %hidden, %attn_proj : tensor<1x2048x4096xf16>
  
  // MLP block
  %normed2 = yirage.rms_norm %attn_res, %rms_mlp {epsilon = 1.0e-6 : f32}
      : tensor<1x2048x4096xf16>, tensor<4096xf16> -> tensor<1x2048x4096xf16>
  
  %mlp_out = yirage.gated_mlp %normed2, %gate, %up, %down
      : tensor<1x2048x4096xf16>, tensor<4096x11008xf16>,
        tensor<4096x11008xf16>, tensor<11008x4096xf16>
      -> tensor<1x2048x4096xf16>
  
  // Final residual
  %output = arith.addf %attn_res, %mlp_out : tensor<1x2048x4096xf16>
  
  return %output : tensor<1x2048x4096xf16>
}
