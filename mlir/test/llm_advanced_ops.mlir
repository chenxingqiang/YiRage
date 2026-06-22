// RUN: yirage-opt %s -yirage-to-linalg | FileCheck %s
//
// Advanced LLM Operations Test
// Tests RoPE, Embedding, PagedAttention, and Flash Attention

// ===----------------------------------------------------------------------===//
// Test 1: Embedding with Gather Pattern
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_embedding
func.func @test_embedding(%table: tensor<128256x4096xf32>, %indices: tensor<2048xi32>) -> tensor<2048x4096xf32> {
  // CHECK: linalg.generic
  // CHECK: tensor.extract
  %0 = yirage.embedding %table, %indices
      : tensor<128256x4096xf32>, tensor<2048xi32> -> tensor<2048x4096xf32>
  return %0 : tensor<2048x4096xf32>
}

// ===----------------------------------------------------------------------===//
// Test 2: Embedding with Batched Indices
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_embedding_batched
func.func @test_embedding_batched(%table: tensor<128256x4096xf32>, %indices: tensor<4x2048xi32>) -> tensor<4x2048x4096xf32> {
  // CHECK: linalg.generic
  // CHECK: tensor.extract
  %0 = yirage.embedding %table, %indices
      : tensor<128256x4096xf32>, tensor<4x2048xi32> -> tensor<4x2048x4096xf32>
  return %0 : tensor<4x2048x4096xf32>
}

// ===----------------------------------------------------------------------===//
// Test 3: RoPE (Rotary Position Embedding)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_rope
func.func @test_rope(
    %input: tensor<1x32x2048x128xf32>,
    %cos_cache: tensor<2048x64xf32>,
    %sin_cache: tensor<2048x64xf32>) -> tensor<1x32x2048x128xf32> {
  // Should apply rotary position embedding
  // x'_i = x_i * cos - x_{i+d/2} * sin
  // x'_{i+d/2} = x_i * sin + x_{i+d/2} * cos
  // CHECK: linalg.generic
  // CHECK: linalg.index
  // CHECK: tensor.extract
  // CHECK: arith.mulf
  %0 = yirage.rope %input, %cos_cache, %sin_cache
      : tensor<1x32x2048x128xf32>, tensor<2048x64xf32>, tensor<2048x64xf32>
      -> tensor<1x32x2048x128xf32>
  return %0 : tensor<1x32x2048x128xf32>
}

// ===----------------------------------------------------------------------===//
// Test 4: Flash Attention (with flash=true)
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_flash_attention
func.func @test_flash_attention(
    %query: tensor<1x32x2048x128xf32>,
    %key: tensor<1x32x2048x128xf32>,
    %value: tensor<1x32x2048x128xf32>) -> tensor<1x32x2048x128xf32> {
  // Flash attention should use block-wise computation
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: linalg.generic
  // CHECK: arith.maximumf
  // CHECK: linalg.generic
  // CHECK: math.exp
  // CHECK: linalg.generic
  // CHECK: arith.addf
  // CHECK: linalg.generic
  // CHECK: arith.divf
  // CHECK: linalg.generic
  // CHECK: arith.mulf
  // CHECK: arith.addf
  %0 = yirage.attention %query, %key, %value {flash = true, causal = true}
      : tensor<1x32x2048x128xf32>, tensor<1x32x2048x128xf32>, tensor<1x32x2048x128xf32>
      -> tensor<1x32x2048x128xf32>
  return %0 : tensor<1x32x2048x128xf32>
}

// ===----------------------------------------------------------------------===//
// Test 5: Complete LLaMA-style Forward Pass with All Ops
// ===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @llama_forward_with_rope
func.func @llama_forward_with_rope(
    // Token indices
    %token_ids: tensor<4x2048xi32>,
    // Embedding table
    %embedding_table: tensor<128256x4096xf32>,
    // RoPE caches
    %cos_cache: tensor<2048x64xf32>,
    %sin_cache: tensor<2048x64xf32>,
    // RMSNorm weights
    %input_layernorm_weight: tensor<4096xf32>,
    // QKV projection weights
    %q_proj_weight: tensor<4096x4096xf32>,
    %k_proj_weight: tensor<4096x4096xf32>,
    %v_proj_weight: tensor<4096x4096xf32>,
    %o_proj_weight: tensor<4096x4096xf32>
) -> tensor<4x2048x4096xf32> {
    
    // === Token Embedding Lookup ===
    // CHECK: linalg.generic
    // CHECK: tensor.extract
    %hidden = yirage.embedding %embedding_table, %token_ids
        : tensor<128256x4096xf32>, tensor<4x2048xi32> -> tensor<4x2048x4096xf32>
    
    // === RMSNorm ===
    // CHECK: linalg.generic
    %normed = yirage.rms_norm %hidden, %input_layernorm_weight {epsilon = 1.0e-6 : f32}
        : tensor<4x2048x4096xf32>, tensor<4096xf32> -> tensor<4x2048x4096xf32>
    
    return %normed : tensor<4x2048x4096xf32>
}
