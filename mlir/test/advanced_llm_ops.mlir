// RUN: yirage-opt %s -yirage-to-linalg -cse | FileCheck %s
// Test file for advanced LLM operations in YiRage MLIR dialect

//===----------------------------------------------------------------------===//
// Mixture of Experts (MoE) Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @moe_router
func.func @moe_router(
    %input: tensor<2048x4096xf16>,
    %gate_weight: tensor<4096x8xf16>
) -> (tensor<2048x2xf16>, tensor<2048x2xi32>) {
  // CHECK: yirage.moe_router
  %weights, %indices = yirage.moe_router %input, %gate_weight {
    num_experts = 8 : i64,
    top_k = 2 : i64,
    capacity_factor = 1.25 : f32,
    normalize_weights = true
  } : tensor<2048x4096xf16>, tensor<4096x8xf16>
    -> tensor<2048x2xf16>, tensor<2048x2xi32>
  
  return %weights, %indices : tensor<2048x2xf16>, tensor<2048x2xi32>
}

// CHECK-LABEL: func.func @moe_dispatch
func.func @moe_dispatch(
    %input: tensor<2048x4096xf16>,
    %expert_indices: tensor<2048x2xi32>,
    %routing_weights: tensor<2048x2xf16>
) -> tensor<8x512x4096xf16> {
  // CHECK: yirage.moe_dispatch
  %dispatched = yirage.moe_dispatch %input, %expert_indices, %routing_weights
    : tensor<2048x4096xf16>, tensor<2048x2xi32>, tensor<2048x2xf16>
    -> tensor<8x512x4096xf16>
  
  return %dispatched : tensor<8x512x4096xf16>
}

// CHECK-LABEL: func.func @moe_combine
func.func @moe_combine(
    %expert_outputs: tensor<8x512x4096xf16>,
    %expert_indices: tensor<2048x2xi32>,
    %routing_weights: tensor<2048x2xf16>
) -> tensor<2048x4096xf16> {
  // CHECK: yirage.moe_combine
  %result = yirage.moe_combine %expert_outputs, %expert_indices, %routing_weights
    : tensor<8x512x4096xf16>, tensor<2048x2xi32>, tensor<2048x2xf16>
    -> tensor<2048x4096xf16>
  
  return %result : tensor<2048x4096xf16>
}

// CHECK-LABEL: func.func @moe_layer_complete
func.func @moe_layer_complete(
    %input: tensor<2048x4096xf16>,
    %gate_weight: tensor<4096x8xf16>,
    %expert_gate_weights: tensor<8x4096x11008xf16>,
    %expert_up_weights: tensor<8x4096x11008xf16>,
    %expert_down_weights: tensor<8x11008x4096xf16>
) -> tensor<2048x4096xf16> {
  // CHECK: yirage.moe_layer
  %result = yirage.moe_layer %input, %gate_weight, %expert_gate_weights,
      %expert_up_weights, %expert_down_weights {
    num_experts = 8 : i64,
    top_k = 2 : i64,
    capacity_factor = 1.25 : f32
  } : tensor<2048x4096xf16>, tensor<4096x8xf16>, tensor<8x4096x11008xf16>,
      tensor<8x4096x11008xf16>, tensor<8x11008x4096xf16>
    -> tensor<2048x4096xf16>
  
  return %result : tensor<2048x4096xf16>
}

//===----------------------------------------------------------------------===//
// Multi-Latent Attention (MLA) Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mla_compress
func.func @mla_compress(
    %key: tensor<1x32x2048x128xf16>,
    %value: tensor<1x32x2048x128xf16>,
    %down_proj: tensor<256x512xf16>
) -> tensor<1x2048x512xf16> {
  // CHECK: yirage.mla_compress
  %compressed = yirage.mla_compress %key, %value, %down_proj {
    compressed_dim = 512 : i64
  } : tensor<1x32x2048x128xf16>, tensor<1x32x2048x128xf16>,
      tensor<256x512xf16> -> tensor<1x2048x512xf16>
  
  return %compressed : tensor<1x2048x512xf16>
}

// CHECK-LABEL: func.func @mla_decompress
func.func @mla_decompress(
    %compressed_kv: tensor<1x2048x512xf16>,
    %up_proj: tensor<512x256xf16>
) -> (tensor<1x32x2048x128xf16>, tensor<1x32x2048x128xf16>) {
  // CHECK: yirage.mla_decompress
  %key, %value = yirage.mla_decompress %compressed_kv, %up_proj {
    num_heads = 32 : i64,
    head_dim = 128 : i64
  } : tensor<1x2048x512xf16>, tensor<512x256xf16>
    -> tensor<1x32x2048x128xf16>, tensor<1x32x2048x128xf16>
  
  return %key, %value : tensor<1x32x2048x128xf16>, tensor<1x32x2048x128xf16>
}

// CHECK-LABEL: func.func @ml_attention_complete
func.func @ml_attention_complete(
    %query: tensor<1x32x2048x128xf16>,
    %compressed_kv: tensor<1x2048x512xf16>,
    %kv_down_proj: tensor<2048x512xf16>,
    %kv_up_proj: tensor<512x2048xf16>
) -> tensor<1x32x2048x128xf16> {
  // CHECK: yirage.ml_attention
  %result = yirage.ml_attention %query, %compressed_kv, %kv_down_proj, %kv_up_proj {
    num_heads = 32 : i64,
    num_kv_heads = 8 : i64,
    head_dim = 128 : i64,
    compressed_dim = 512 : i64,
    causal = true
  } : tensor<1x32x2048x128xf16>, tensor<1x2048x512xf16>,
      tensor<2048x512xf16>, tensor<512x2048xf16>
    -> tensor<1x32x2048x128xf16>
  
  return %result : tensor<1x32x2048x128xf16>
}

//===----------------------------------------------------------------------===//
// Speculative Decoding Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @spec_draft
func.func @spec_draft(
    %input_hidden: tensor<1x2048xf16>,
    %draft_lm_head: tensor<2048x32000xf16>
) -> (tensor<1x5xi32>, tensor<1x5x32000xf16>) {
  // CHECK: yirage.spec_draft
  %draft_tokens, %draft_probs = yirage.spec_draft %input_hidden, %draft_lm_head {
    num_draft_tokens = 5 : i64,
    temperature = 1.0 : f32,
    top_k = 1 : i64
  } : tensor<1x2048xf16>, tensor<2048x32000xf16>
    -> tensor<1x5xi32>, tensor<1x5x32000xf16>
  
  return %draft_tokens, %draft_probs : tensor<1x5xi32>, tensor<1x5x32000xf16>
}

// CHECK-LABEL: func.func @spec_verify
func.func @spec_verify(
    %draft_tokens: tensor<1x5xi32>,
    %draft_probs: tensor<1x5x32000xf16>,
    %target_probs: tensor<1x5x32000xf16>
) -> (tensor<1x5xi32>, tensor<1xi32>) {
  // CHECK: yirage.spec_verify
  %accepted_tokens, %num_accepted = yirage.spec_verify %draft_tokens,
      %draft_probs, %target_probs
    : tensor<1x5xi32>, tensor<1x5x32000xf16>, tensor<1x5x32000xf16>
    -> tensor<1x5xi32>, tensor<1xi32>
  
  return %accepted_tokens, %num_accepted : tensor<1x5xi32>, tensor<1xi32>
}

// CHECK-LABEL: func.func @lookahead_decode
func.func @lookahead_decode(
    %input: tensor<1x2048xf16>,
    %ngram_pool: tensor<32000x32000x32000xi32>
) -> tensor<1x5xi32> {
  // CHECK: yirage.lookahead_decode
  %output_tokens = yirage.lookahead_decode %input, %ngram_pool {
    window_size = 5 : i64,
    ngram_size = 3 : i64,
    guess_set_size = 10 : i64
  } : tensor<1x2048xf16>, tensor<32000x32000x32000xi32>
    -> tensor<1x5xi32>
  
  return %output_tokens : tensor<1x5xi32>
}

//===----------------------------------------------------------------------===//
// Sliding Window Attention
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @sliding_window_attention
func.func @sliding_window_attention(
    %query: tensor<1x32x16384x128xf16>,
    %key: tensor<1x32x16384x128xf16>,
    %value: tensor<1x32x16384x128xf16>
) -> tensor<1x32x16384x128xf16> {
  // CHECK: yirage.sliding_window_attention
  %result = yirage.sliding_window_attention %query, %key, %value {
    window_size = 4096 : i64,
    causal = true
  } : tensor<1x32x16384x128xf16>, tensor<1x32x16384x128xf16>,
      tensor<1x32x16384x128xf16> -> tensor<1x32x16384x128xf16>
  
  return %result : tensor<1x32x16384x128xf16>
}

//===----------------------------------------------------------------------===//
// Cross-Attention
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @cross_attention
func.func @cross_attention(
    %decoder_q: tensor<1x32x512x128xf16>,
    %encoder_k: tensor<1x32x1024x128xf16>,
    %encoder_v: tensor<1x32x1024x128xf16>
) -> tensor<1x32x512x128xf16> {
  // CHECK: yirage.cross_attention
  %result = yirage.cross_attention %decoder_q, %encoder_k, %encoder_v
    : tensor<1x32x512x128xf16>, tensor<1x32x1024x128xf16>,
      tensor<1x32x1024x128xf16> -> tensor<1x32x512x128xf16>
  
  return %result : tensor<1x32x512x128xf16>
}

//===----------------------------------------------------------------------===//
// Sampling Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @sample_top_p
func.func @sample_top_p(%logits: tensor<1x32000xf16>) -> tensor<1xi32> {
  // CHECK: yirage.sample_top_p
  %token = yirage.sample_top_p %logits {
    top_p = 0.9 : f32,
    temperature = 1.0 : f32
  } : tensor<1x32000xf16> -> tensor<1xi32>
  
  return %token : tensor<1xi32>
}

// CHECK-LABEL: func.func @sample_top_k
func.func @sample_top_k(%logits: tensor<1x32000xf16>) -> tensor<1xi32> {
  // CHECK: yirage.sample_top_k
  %token = yirage.sample_top_k %logits {
    top_k = 50 : i64,
    temperature = 1.0 : f32
  } : tensor<1x32000xf16> -> tensor<1xi32>
  
  return %token : tensor<1xi32>
}

//===----------------------------------------------------------------------===//
// Complete Transformer Blocks
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @mixtral_moe_block
// Mixtral-style MoE block with sliding window attention
func.func @mixtral_moe_block(
    %input: tensor<1x2048x4096xf16>,
    %attn_norm_weight: tensor<4096xf16>,
    %ffn_norm_weight: tensor<4096xf16>,
    %wq: tensor<4096x4096xf16>,
    %wk: tensor<4096x1024xf16>,
    %wv: tensor<4096x1024xf16>,
    %wo: tensor<4096x4096xf16>,
    %gate_weight: tensor<4096x8xf16>,
    %expert_gate_weights: tensor<8x4096x14336xf16>,
    %expert_up_weights: tensor<8x4096x14336xf16>,
    %expert_down_weights: tensor<8x14336x4096xf16>
) -> tensor<1x2048x4096xf16> {
  
  // Attention norm
  %attn_normed = yirage.rms_norm %input, %attn_norm_weight {
    epsilon = 1e-5 : f32
  } : tensor<1x2048x4096xf16>, tensor<4096xf16> -> tensor<1x2048x4096xf16>
  
  // QKV projection (simplified - would reshape for GQA)
  %q = yirage.matmul %attn_normed, %wq
    : tensor<1x2048x4096xf16>, tensor<4096x4096xf16> -> tensor<1x2048x4096xf16>
  %k = yirage.matmul %attn_normed, %wk
    : tensor<1x2048x4096xf16>, tensor<4096x1024xf16> -> tensor<1x2048x1024xf16>
  %v = yirage.matmul %attn_normed, %wv
    : tensor<1x2048x4096xf16>, tensor<4096x1024xf16> -> tensor<1x2048x1024xf16>
  
  // Sliding window attention (would need reshaping in real impl)
  // Simplified: using regular attention
  %attn_out = yirage.attention %q, %k, %v {
    causal = true
  } : tensor<1x2048x4096xf16>, tensor<1x2048x1024xf16>,
      tensor<1x2048x1024xf16> -> tensor<1x2048x4096xf16>
  
  // Output projection
  %attn_proj = yirage.matmul %attn_out, %wo
    : tensor<1x2048x4096xf16>, tensor<4096x4096xf16> -> tensor<1x2048x4096xf16>
  
  // Residual
  %attn_res = arith.addf %input, %attn_proj : tensor<1x2048x4096xf16>
  
  // FFN norm
  %ffn_normed = yirage.rms_norm %attn_res, %ffn_norm_weight {
    epsilon = 1e-5 : f32
  } : tensor<1x2048x4096xf16>, tensor<4096xf16> -> tensor<1x2048x4096xf16>
  
  // Reshape for MoE (batch * seq -> tokens)
  %ffn_flat = tensor.reshape %ffn_normed : tensor<1x2048x4096xf16> -> tensor<2048x4096xf16>
  
  // MoE layer
  %moe_out = yirage.moe_layer %ffn_flat, %gate_weight, %expert_gate_weights,
      %expert_up_weights, %expert_down_weights {
    num_experts = 8 : i64,
    top_k = 2 : i64
  } : tensor<2048x4096xf16>, tensor<4096x8xf16>, tensor<8x4096x14336xf16>,
      tensor<8x4096x14336xf16>, tensor<8x14336x4096xf16>
    -> tensor<2048x4096xf16>
  
  // Reshape back
  %moe_reshaped = tensor.reshape %moe_out : tensor<2048x4096xf16> -> tensor<1x2048x4096xf16>
  
  // Final residual
  %output = arith.addf %attn_res, %moe_reshaped : tensor<1x2048x4096xf16>
  
  return %output : tensor<1x2048x4096xf16>
}

// CHECK-LABEL: func.func @deepseek_mla_block
// DeepSeek V2 style block with Multi-Latent Attention
func.func @deepseek_mla_block(
    %input: tensor<1x2048x4096xf16>,
    %norm_weight: tensor<4096xf16>,
    %wq: tensor<4096x4096xf16>,
    %compressed_kv: tensor<1x2048x512xf16>,
    %kv_down_proj: tensor<2048x512xf16>,
    %kv_up_proj: tensor<512x2048xf16>,
    %wo: tensor<4096x4096xf16>
) -> tensor<1x2048x4096xf16> {
  
  // Pre-norm
  %normed = yirage.rms_norm %input, %norm_weight {
    epsilon = 1e-5 : f32
  } : tensor<1x2048x4096xf16>, tensor<4096xf16> -> tensor<1x2048x4096xf16>
  
  // Query projection
  %q = yirage.matmul %normed, %wq
    : tensor<1x2048x4096xf16>, tensor<4096x4096xf16> -> tensor<1x2048x4096xf16>
  
  // Reshape Q for multi-head
  %q_reshaped = tensor.reshape %q : tensor<1x2048x4096xf16> -> tensor<1x32x2048x128xf16>
  
  // Multi-Latent Attention
  %attn_out = yirage.ml_attention %q_reshaped, %compressed_kv, %kv_down_proj, %kv_up_proj {
    num_heads = 32 : i64,
    num_kv_heads = 8 : i64,
    head_dim = 128 : i64,
    compressed_dim = 512 : i64,
    causal = true
  } : tensor<1x32x2048x128xf16>, tensor<1x2048x512xf16>,
      tensor<2048x512xf16>, tensor<512x2048xf16>
    -> tensor<1x32x2048x128xf16>
  
  // Reshape back
  %attn_flat = tensor.reshape %attn_out : tensor<1x32x2048x128xf16> -> tensor<1x2048x4096xf16>
  
  // Output projection
  %projected = yirage.matmul %attn_flat, %wo
    : tensor<1x2048x4096xf16>, tensor<4096x4096xf16> -> tensor<1x2048x4096xf16>
  
  // Residual
  %output = arith.addf %input, %projected : tensor<1x2048x4096xf16>
  
  return %output : tensor<1x2048x4096xf16>
}
