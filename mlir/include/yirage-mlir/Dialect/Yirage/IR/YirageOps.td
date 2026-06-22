//===- YirageOps.td - Yirage Operations Definition ----------*- tablegen -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the Yirage dialect.
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_OPS
#define YIRAGE_OPS

include "yirage-mlir/Dialect/Yirage/IR/YirageDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// Base Yirage Operation Class
//===----------------------------------------------------------------------===//

class Yirage_Op<string mnemonic, list<Trait> traits = []> :
    Op<Yirage_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Matrix Operations
//===----------------------------------------------------------------------===//

def Yirage_MatmulOp : Yirage_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    Performs matrix multiplication: C = A @ B
    
    Supports optional transpose of inputs and accumulation.
    
    Example:
    ```mlir
    %C = yirage.matmul %A, %B : tensor<M x K x f16>, tensor<K x N x f16> -> tensor<M x N x f16>
    %C = yirage.matmul %A, %B {transpose_rhs = true} : ...
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
  
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Yirage_BatchMatmulOp : Yirage_Op<"batch_matmul", [Pure]> {
  let summary = "Batched matrix multiplication";
  let description = [{
    Performs batched matrix multiplication.
    
    Example:
    ```mlir
    %C = yirage.batch_matmul %A, %B
        : tensor<B x M x K x f16>, tensor<B x K x N x f16> -> tensor<B x M x N x f16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_lhs,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_rhs
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

def Yirage_QMatmulOp : Yirage_Op<"qmatmul", [Pure]> {
  let summary = "Quantized matrix multiplication (W8A16 or W4A16)";
  let description = [{
    Performs quantized matrix multiplication with on-the-fly dequantization.
    
    The weight matrix is stored in INT8 or INT4, dequantized during computation.
    
    Example:
    ```mlir
    %C = yirage.qmatmul %A, %B_quant, %scale {bits = 8}
        : tensor<M x K x f16>, tensor<K x N x i8>, tensor<N x f16> -> tensor<M x N x f16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs_quantized,
    AnyRankedTensor:$scale,
    Optional<AnyRankedTensor>:$zero_point,
    DefaultValuedAttr<I64Attr, "8">:$bits,
    DefaultValuedAttr<I64Attr, "128">:$group_size
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs_quantized `,` $scale (`,` $zero_point^ `:` type($zero_point))? attr-dict
    `:` type($lhs) `,` type($rhs_quantized) `,` type($scale) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Attention Operations
//===----------------------------------------------------------------------===//

def Yirage_AttentionOp : Yirage_Op<"attention", [Pure]> {
  let summary = "Scaled dot-product attention";
  let description = [{
    Computes scaled dot-product attention:
      output = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
    
    Supports:
    - Causal masking
    - Multi-Query Attention (MQA)
    - Grouped-Query Attention (GQA)
    - Flash Attention optimization
    
    Example:
    ```mlir
    %out = yirage.attention %Q, %K, %V {causal = true, num_kv_heads = 8}
        : tensor<1x32x2048x128xf16> -> tensor<1x32x2048x128xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$key,
    AnyRankedTensor:$value,
    Optional<AnyRankedTensor>:$attention_mask,
    DefaultValuedAttr<BoolAttr, "false">:$causal,
    OptionalAttr<F32Attr>:$scale,
    DefaultValuedAttr<I64Attr, "0">:$num_kv_heads,
    DefaultValuedAttr<BoolAttr, "true">:$flash
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $query `,` $key `,` $value (`,` `mask` `=` $attention_mask^ `:` type($attention_mask))? attr-dict
    `:` type($query) `,` type($key) `,` type($value) `->` type($result)
  }];
  
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Yirage_PagedAttentionOp : Yirage_Op<"paged_attention", [Pure]> {
  let summary = "Paged attention for efficient KV cache management";
  let description = [{
    Paged attention operation for vLLM-style batched inference with
    block-structured KV cache.
    
    Example:
    ```mlir
    %out = yirage.paged_attention %query, %key_cache, %value_cache, %block_tables
        : ... -> tensor<batch x heads x head_dim x f16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$key_cache,
    AnyRankedTensor:$value_cache,
    AnyRankedTensor:$block_tables,
    AnyRankedTensor:$context_lens,
    DefaultValuedAttr<I64Attr, "16">:$block_size,
    OptionalAttr<F32Attr>:$scale
  );
  let results = (outs AnyRankedTensor:$result);
  
  let hasVerifier = 1;
}

def Yirage_KVCacheUpdateOp : Yirage_Op<"kv_cache_update", [Pure]> {
  let summary = "Update KV cache with new keys and values";
  let description = [{
    Updates the KV cache with new key and value tensors at specified slot indices.
    Returns the updated key and value caches.
    
    For paged attention, the cache layout is typically:
      [num_blocks, num_heads, block_size, head_dim]
    
    Example:
    ```mlir
    %new_key_cache, %new_value_cache = yirage.kv_cache_update
        %key_cache, %value_cache, %new_keys, %new_values, %slot_indices
        : tensor<1024x8x16x128xf16>, tensor<1024x8x16x128xf16>,
          tensor<32x8x128xf16>, tensor<32x8x128xf16>, tensor<32xi32>
        -> tensor<1024x8x16x128xf16>, tensor<1024x8x16x128xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$key_cache,
    AnyRankedTensor:$value_cache,
    AnyRankedTensor:$new_keys,
    AnyRankedTensor:$new_values,
    AnyRankedTensor:$slot_indices
  );
  let results = (outs
    AnyRankedTensor:$updated_key_cache,
    AnyRankedTensor:$updated_value_cache
  );
  
  let assemblyFormat = [{
    $key_cache `,` $value_cache `,` $new_keys `,` $new_values `,` $slot_indices attr-dict
    `:` type($key_cache) `,` type($value_cache) `,` type($new_keys) `,` type($new_values) `,` type($slot_indices)
    `->` type($updated_key_cache) `,` type($updated_value_cache)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Normalization Operations
//===----------------------------------------------------------------------===//

def Yirage_RMSNormOp : Yirage_Op<"rms_norm", [Pure]> {
  let summary = "Root Mean Square Layer Normalization";
  let description = [{
    Computes RMSNorm: output = x / sqrt(mean(x^2) + eps) * gamma
    
    Example:
    ```mlir
    %out = yirage.rms_norm %input, %gamma {epsilon = 1.0e-6}
        : tensor<1x2048x4096xf16>, tensor<4096xf16> -> tensor<1x2048x4096xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$gamma,
    DefaultValuedAttr<F32Attr, "1.0e-6">:$epsilon
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $gamma attr-dict `:` type($input) `,` type($gamma) `->` type($result)
  }];
  
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Yirage_LayerNormOp : Yirage_Op<"layer_norm", [Pure]> {
  let summary = "Layer Normalization";
  let description = [{
    Computes LayerNorm: output = (x - mean) / sqrt(var + eps) * gamma + beta
    
    Example:
    ```mlir
    %out = yirage.layer_norm %input, %gamma, %beta {epsilon = 1.0e-6}
        : tensor<1x197x768xf16>, tensor<768xf16>, tensor<768xf16> -> tensor<1x197x768xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$gamma,
    AnyRankedTensor:$beta,
    DefaultValuedAttr<F32Attr, "1.0e-6">:$epsilon
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $gamma `,` $beta attr-dict
    `:` type($input) `,` type($gamma) `,` type($beta) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Activation Operations
//===----------------------------------------------------------------------===//

def Yirage_SiLUOp : Yirage_Op<"silu", [Pure, SameOperandsAndResultType]> {
  let summary = "SiLU (Swish) activation function";
  let description = [{
    Computes SiLU: output = x * sigmoid(x)
  }];
  
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let hasFolder = 1;
}

def Yirage_GELUOp : Yirage_Op<"gelu", [Pure, SameOperandsAndResultType]> {
  let summary = "GELU activation function";
  let description = [{
    Computes GELU: output = x * 0.5 * (1 + erf(x / sqrt(2)))
    
    Optional approximate mode uses tanh approximation.
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<BoolAttr, "false">:$approximate
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let hasFolder = 1;
}

def Yirage_ReLUOp : Yirage_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation function";
  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let hasFolder = 1;
}

def Yirage_SoftmaxOp : Yirage_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Softmax operation along specified axis";
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MLP Operations
//===----------------------------------------------------------------------===//

def Yirage_GatedMLPOp : Yirage_Op<"gated_mlp", [Pure]> {
  let summary = "Gated MLP block (SwiGLU style)";
  let description = [{
    Computes gated MLP: output = down(silu(gate(x)) * up(x))
    
    Example:
    ```mlir
    %out = yirage.gated_mlp %input, %gate_weight, %up_weight, %down_weight
        : tensor<1x2048x4096xf16>, ... -> tensor<1x2048x4096xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$gate_weight,
    AnyRankedTensor:$up_weight,
    AnyRankedTensor:$down_weight
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $gate_weight `,` $up_weight `,` $down_weight attr-dict
    `:` type($input) `,` type($gate_weight) `,` type($up_weight) `,` type($down_weight) `->` type($result)
  }];
  
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Yirage_LinearOp : Yirage_Op<"linear", [Pure]> {
  let summary = "Linear layer with optional bias";
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $weight (`,` $bias^ `:` type($bias))? attr-dict
    `:` type($input) `,` type($weight) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Embedding Operations
//===----------------------------------------------------------------------===//

def Yirage_EmbeddingOp : Yirage_Op<"embedding", [Pure]> {
  let summary = "Token embedding lookup";
  let description = [{
    Looks up embeddings from a table using token indices.
    
    Example:
    ```mlir
    %out = yirage.embedding %table, %indices
        : tensor<128256x4096xf16>, tensor<1x2048xi32> -> tensor<1x2048x4096xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$table,
    AnyRankedTensor:$indices
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $table `,` $indices attr-dict `:` type($table) `,` type($indices) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

def Yirage_RoPEOp : Yirage_Op<"rope", [Pure]> {
  let summary = "Rotary Position Embedding";
  let description = [{
    Applies Rotary Position Embedding to Q/K tensors.
    
    Example:
    ```mlir
    %out = yirage.rope %input, %cos_cache, %sin_cache
        : tensor<1x32x2048x128xf16>, tensor<2048x64xf32> -> tensor<1x32x2048x128xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$cos_cache,
    AnyRankedTensor:$sin_cache,
    DefaultValuedAttr<I64Attr, "0">:$position_offset
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $cos_cache `,` $sin_cache attr-dict
    `:` type($input) `,` type($cos_cache) `,` type($sin_cache) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reduction Operations
//===----------------------------------------------------------------------===//

def Yirage_ReduceSumOp : Yirage_Op<"reduce_sum", [Pure]> {
  let summary = "Sum reduction along axis";
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

def Yirage_ReduceMaxOp : Yirage_Op<"reduce_max", [Pure]> {
  let summary = "Max reduction along axis";
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

def Yirage_TopKOp : Yirage_Op<"topk", [Pure]> {
  let summary = "TopK operation";
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$k,
    DefaultValuedAttr<I64Attr, "-1">:$axis,
    DefaultValuedAttr<BoolAttr, "true">:$largest
  );
  let results = (outs
    AnyRankedTensor:$values,
    AnyRankedTensor:$indices
  );
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` `(` type($values) `,` type($indices) `)`
  }];
  
  let hasVerifier = 1;
}

def Yirage_ArgMaxOp : Yirage_Op<"argmax", [Pure]> {
  let summary = "Argmax operation";
  let arguments = (ins
    AnyRankedTensor:$input,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Tensor Manipulation Operations
//===----------------------------------------------------------------------===//

def Yirage_ReshapeOp : Yirage_Op<"reshape", [Pure]> {
  let summary = "Reshape tensor";
  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$shape
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Yirage_TransposeOp : Yirage_Op<"transpose", [Pure]> {
  let summary = "Transpose tensor dimensions";
  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$permutation
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Yirage_ConcatOp : Yirage_Op<"concat", [Pure]> {
  let summary = "Concatenate tensors along axis";
  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    I64Attr:$axis
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $inputs attr-dict `:` type($inputs) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

def Yirage_SplitOp : Yirage_Op<"split", [Pure]> {
  let summary = "Split tensor along axis";
  let arguments = (ins
    AnyRankedTensor:$input,
    I64Attr:$num_splits,
    I64Attr:$axis
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` `(` type($results) `)`
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Convolution Operations
//===----------------------------------------------------------------------===//

def Yirage_Conv2DOp : Yirage_Op<"conv2d", [Pure]> {
  let summary = "2D convolution";
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$kernel,
    Optional<AnyRankedTensor>:$bias,
    I64ArrayAttr:$strides,
    I64ArrayAttr:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1}">:$dilations
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $kernel (`,` $bias^ `:` type($bias))? attr-dict
    `:` type($input) `,` type($kernel) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

def Yirage_MaxPool2DOp : Yirage_Op<"max_pool2d", [Pure]> {
  let summary = "2D max pooling";
  let arguments = (ins
    AnyRankedTensor:$input,
    I64ArrayAttr:$kernel_size,
    I64ArrayAttr:$strides,
    DefaultValuedAttr<I64ArrayAttr, "{0, 0, 0, 0}">:$padding
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Quantization Operations
//===----------------------------------------------------------------------===//

def Yirage_QuantizeOp : Yirage_Op<"quantize", [Pure]> {
  let summary = "Quantize tensor to INT8 or INT4";
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$scale,
    Optional<AnyRankedTensor>:$zero_point,
    DefaultValuedAttr<I64Attr, "8">:$bits
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $scale (`,` $zero_point^ `:` type($zero_point))? attr-dict
    `:` type($input) `,` type($scale) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

def Yirage_DequantizeOp : Yirage_Op<"dequantize", [Pure]> {
  let summary = "Dequantize INT8/INT4 tensor to float";
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$scale,
    Optional<AnyRankedTensor>:$zero_point
  );
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = [{
    $input `,` $scale (`,` $zero_point^ `:` type($zero_point))? attr-dict
    `:` type($input) `,` type($scale) `->` type($result)
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Mixture of Experts (MoE) Operations
//===----------------------------------------------------------------------===//

def Yirage_MoERouterOp : Yirage_Op<"moe_router", [Pure]> {
  let summary = "MoE Router (gating/dispatch)";
  let description = [{
    Computes expert routing weights and dispatch indices for MoE layers.
    
    Implements top-k gating: select top-k experts per token.
    
    Example:
    ```mlir
    %weights, %indices = yirage.moe_router %input, %gate_weight {
        num_experts = 8, top_k = 2, capacity_factor = 1.25
    } : tensor<2048x4096xf16>, tensor<4096x8xf16>
        -> tensor<2048x2xf16>, tensor<2048x2xi32>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$gate_weight,
    I64Attr:$num_experts,
    DefaultValuedAttr<I64Attr, "2">:$top_k,
    DefaultValuedAttr<F32Attr, "1.25">:$capacity_factor,
    DefaultValuedAttr<BoolAttr, "true">:$normalize_weights,
    DefaultValuedAttr<BoolAttr, "false">:$use_aux_loss
  );
  let results = (outs
    AnyRankedTensor:$routing_weights,
    AnyRankedTensor:$expert_indices
  );
  
  let hasVerifier = 1;
}

def Yirage_MoEDispatchOp : Yirage_Op<"moe_dispatch", [Pure]> {
  let summary = "MoE Dispatch (scatter tokens to experts)";
  let description = [{
    Dispatches tokens to their assigned experts based on routing indices.
    Returns permuted tokens ready for expert computation.
    
    Example:
    ```mlir
    %dispatched = yirage.moe_dispatch %input, %expert_indices, %routing_weights
        : tensor<2048x4096xf16>, tensor<2048x2xi32>, tensor<2048x2xf16>
        -> tensor<8x512x4096xf16>  // [num_experts, capacity, hidden]
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$expert_indices,
    AnyRankedTensor:$routing_weights
  );
  let results = (outs AnyRankedTensor:$dispatched);
  
  let hasVerifier = 1;
}

def Yirage_MoECombineOp : Yirage_Op<"moe_combine", [Pure]> {
  let summary = "MoE Combine (gather expert outputs)";
  let description = [{
    Combines expert outputs back to original token order using routing weights.
    
    Example:
    ```mlir
    %output = yirage.moe_combine %expert_outputs, %expert_indices, %routing_weights
        : tensor<8x512x4096xf16>, tensor<2048x2xi32>, tensor<2048x2xf16>
        -> tensor<2048x4096xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$expert_outputs,
    AnyRankedTensor:$expert_indices,
    AnyRankedTensor:$routing_weights
  );
  let results = (outs AnyRankedTensor:$result);
  
  let hasVerifier = 1;
}

def Yirage_MoEExpertOp : Yirage_Op<"moe_expert", [Pure]> {
  let summary = "Single MoE Expert (parallel FFN)";
  let description = [{
    Applies a single expert FFN to dispatched tokens.
    Typically used within a loop over experts.
    
    Example:
    ```mlir
    %out = yirage.moe_expert %input, %w1, %w2, %w3 {expert_id = 0}
        : tensor<512x4096xf16>, tensor<4096x11008xf16>,
          tensor<4096x11008xf16>, tensor<11008x4096xf16>
        -> tensor<512x4096xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$gate_weight,
    AnyRankedTensor:$up_weight,
    AnyRankedTensor:$down_weight,
    I64Attr:$expert_id
  );
  let results = (outs AnyRankedTensor:$result);
  
  let hasVerifier = 1;
}

def Yirage_MoELayerOp : Yirage_Op<"moe_layer", [Pure]> {
  let summary = "Complete MoE Layer (router + dispatch + experts + combine)";
  let description = [{
    Complete Mixture of Experts layer combining all MoE operations.
    
    This is a high-level op that encompasses:
    1. Router: Compute expert assignment
    2. Dispatch: Scatter tokens to experts
    3. Experts: Apply expert FFNs
    4. Combine: Gather and weight outputs
    
    Example:
    ```mlir
    %out = yirage.moe_layer %input, %gate, %experts {
        num_experts = 8, top_k = 2
    } : tensor<2048x4096xf16>, tensor<4096x8xf16>,
        tensor<8x4096x11008xf16>  // [num_experts, hidden, intermediate]
        -> tensor<2048x4096xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$gate_weight,
    AnyRankedTensor:$expert_gate_weights,
    AnyRankedTensor:$expert_up_weights,
    AnyRankedTensor:$expert_down_weights,
    I64Attr:$num_experts,
    DefaultValuedAttr<I64Attr, "2">:$top_k,
    DefaultValuedAttr<F32Attr, "1.25">:$capacity_factor
  );
  let results = (outs AnyRankedTensor:$result);
  
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Speculative Decoding Operations
//===----------------------------------------------------------------------===//

def Yirage_SpecDraftOp : Yirage_Op<"spec_draft", [Pure]> {
  let summary = "Speculative Decoding Draft (generate candidate tokens)";
  let description = [{
    Generate draft tokens using a smaller/faster draft model.
    
    Example:
    ```mlir
    %draft_tokens, %draft_probs = yirage.spec_draft %input, %draft_model {
        num_draft_tokens = 5
    } : tensor<1x2048xf16>, !yirage.model
        -> tensor<1x5xi32>, tensor<1x5xvocab_sizexf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input_hidden,
    AnyRankedTensor:$draft_lm_head,
    I64Attr:$num_draft_tokens,
    DefaultValuedAttr<F32Attr, "1.0">:$temperature,
    DefaultValuedAttr<I64Attr, "1">:$top_k
  );
  let results = (outs
    AnyRankedTensor:$draft_tokens,
    AnyRankedTensor:$draft_probs
  );
  
  let hasVerifier = 1;
}

def Yirage_SpecVerifyOp : Yirage_Op<"spec_verify", [Pure]> {
  let summary = "Speculative Decoding Verify (validate draft tokens)";
  let description = [{
    Verify draft tokens against target model probabilities.
    Uses rejection sampling to accept/reject tokens.
    
    Example:
    ```mlir
    %accepted_tokens, %num_accepted = yirage.spec_verify
        %draft_tokens, %draft_probs, %target_probs
        : tensor<1x5xi32>, tensor<1x5xvocab_sizexf16>, tensor<1x5xvocab_sizexf16>
        -> tensor<1x5xi32>, tensor<1xi32>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$draft_tokens,
    AnyRankedTensor:$draft_probs,
    AnyRankedTensor:$target_probs
  );
  let results = (outs
    AnyRankedTensor:$accepted_tokens,
    AnyRankedTensor:$num_accepted
  );
  
  let hasVerifier = 1;
}

def Yirage_LookaheadDecodeOp : Yirage_Op<"lookahead_decode", [Pure]> {
  let summary = "Lookahead Decoding (parallel verification)";
  let description = [{
    Lookahead decoding with n-gram based speculation.
    
    Example:
    ```mlir
    %output_tokens = yirage.lookahead_decode %input, %ngram_pool {
        window_size = 5, ngram_size = 3, guess_set_size = 10
    } : tensor<1x2048xf16>, tensor<vocab_sizexvocab_sizexvocab_sizexi32>
        -> tensor<1x5xi32>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$ngram_pool,
    I64Attr:$window_size,
    DefaultValuedAttr<I64Attr, "3">:$ngram_size,
    DefaultValuedAttr<I64Attr, "10">:$guess_set_size
  );
  let results = (outs AnyRankedTensor:$output_tokens);
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Multi-Latent Attention (MLA) Operations - DeepSeek style
//===----------------------------------------------------------------------===//

def Yirage_MLACompressOp : Yirage_Op<"mla_compress", [Pure]> {
  let summary = "MLA Key-Value Compression";
  let description = [{
    Compress key-value pairs using low-rank projection (DeepSeek MLA).
    
    Computes: c_kv = down_proj(concat(k, v))
    
    This reduces KV cache size by projecting to lower dimension.
    
    Example:
    ```mlir
    %c_kv = yirage.mla_compress %key, %value, %down_proj {
        compressed_dim = 512
    } : tensor<1x32x2048x128xf16>, tensor<1x32x2048x128xf16>,
        tensor<256x512xf16> -> tensor<1x2048x512xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$key,
    AnyRankedTensor:$value,
    AnyRankedTensor:$down_proj,
    I64Attr:$compressed_dim
  );
  let results = (outs AnyRankedTensor:$compressed_kv);
  
  let hasVerifier = 1;
}

def Yirage_MLADecompressOp : Yirage_Op<"mla_decompress", [Pure]> {
  let summary = "MLA Key-Value Decompression";
  let description = [{
    Decompress key-value pairs from low-rank representation.
    
    Computes: k, v = split(up_proj(c_kv))
    
    Example:
    ```mlir
    %key, %value = yirage.mla_decompress %c_kv, %up_proj {
        num_heads = 32, head_dim = 128
    } : tensor<1x2048x512xf16>, tensor<512x256xf16>
        -> tensor<1x32x2048x128xf16>, tensor<1x32x2048x128xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$compressed_kv,
    AnyRankedTensor:$up_proj,
    I64Attr:$num_heads,
    I64Attr:$head_dim
  );
  let results = (outs
    AnyRankedTensor:$key,
    AnyRankedTensor:$value
  );
  
  let hasVerifier = 1;
}

def Yirage_MLAttentionOp : Yirage_Op<"ml_attention", [Pure]> {
  let summary = "Multi-Latent Attention (complete MLA operation)";
  let description = [{
    Complete Multi-Latent Attention as used in DeepSeek-V2.
    
    MLA reduces KV cache memory by:
    1. Compressing K,V to low-rank latent space
    2. Storing compressed KV in cache
    3. Decompressing on-the-fly during attention
    
    Equivalent to:
    ```
    c_kv = down_proj(concat(k, v))  # Compress
    k', v' = split(up_proj(c_kv))    # Decompress
    out = attention(q, k', v')       # Standard attention
    ```
    
    Example:
    ```mlir
    %out = yirage.ml_attention %query, %compressed_kv,
        %kv_down_proj, %kv_up_proj, %q_rope_proj {
        num_heads = 32, num_kv_heads = 8, head_dim = 128,
        compressed_dim = 512, causal = true
    } : ... -> tensor<1x32x2048x128xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$compressed_kv,
    AnyRankedTensor:$kv_down_proj,
    AnyRankedTensor:$kv_up_proj,
    Optional<AnyRankedTensor>:$q_rope_embed,
    I64Attr:$num_heads,
    I64Attr:$num_kv_heads,
    I64Attr:$head_dim,
    I64Attr:$compressed_dim,
    DefaultValuedAttr<BoolAttr, "true">:$causal,
    OptionalAttr<F32Attr>:$scale
  );
  let results = (outs AnyRankedTensor:$result);
  
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// Sliding Window Attention Operations
//===----------------------------------------------------------------------===//

def Yirage_SlidingWindowAttentionOp : Yirage_Op<"sliding_window_attention", [Pure]> {
  let summary = "Sliding Window Attention (Mistral/Mixtral style)";
  let description = [{
    Attention with sliding window for efficient long context.
    
    Each token attends to at most window_size previous tokens.
    
    Example:
    ```mlir
    %out = yirage.sliding_window_attention %Q, %K, %V {
        window_size = 4096, causal = true
    } : tensor<1x32x16384x128xf16> -> tensor<1x32x16384x128xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$key,
    AnyRankedTensor:$value,
    I64Attr:$window_size,
    DefaultValuedAttr<BoolAttr, "true">:$causal,
    OptionalAttr<F32Attr>:$scale
  );
  let results = (outs AnyRankedTensor:$result);
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Cross-Attention Operations
//===----------------------------------------------------------------------===//

def Yirage_CrossAttentionOp : Yirage_Op<"cross_attention", [Pure]> {
  let summary = "Cross-Attention (encoder-decoder attention)";
  let description = [{
    Cross-attention between decoder queries and encoder key-values.
    Used in encoder-decoder models (T5, BART) and vision-language models.
    
    Example:
    ```mlir
    %out = yirage.cross_attention %decoder_q, %encoder_k, %encoder_v
        : tensor<1x32x512x128xf16>, tensor<1x32x1024x128xf16>,
          tensor<1x32x1024x128xf16> -> tensor<1x32x512x128xf16>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$query,
    AnyRankedTensor:$key,
    AnyRankedTensor:$value,
    Optional<AnyRankedTensor>:$attention_mask,
    OptionalAttr<F32Attr>:$scale
  );
  let results = (outs AnyRankedTensor:$result);
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Prefix/Prompt Caching Operations
//===----------------------------------------------------------------------===//

def Yirage_PrefixCacheLookupOp : Yirage_Op<"prefix_cache_lookup", [Pure]> {
  let summary = "Lookup cached KV for prefix/prompt";
  let description = [{
    Check if prefix KV is cached and return cache hit info.
    Used for prompt caching in serving systems.
    
    Example:
    ```mlir
    %cached_kv, %cache_hit = yirage.prefix_cache_lookup %prefix_hash, %cache_table
        : tensor<1xi64>, !yirage.kv_cache -> tensor<1x32x1024x128xf16>, tensor<1xi1>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$prefix_hash,
    AnyRankedTensor:$cache_table
  );
  let results = (outs
    AnyRankedTensor:$cached_kv,
    AnyRankedTensor:$cache_hit
  );
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Sampling Operations
//===----------------------------------------------------------------------===//

def Yirage_SampleTopPOp : Yirage_Op<"sample_top_p", [Pure]> {
  let summary = "Top-p (nucleus) sampling";
  let description = [{
    Sample from logits using top-p (nucleus) sampling.
    
    Example:
    ```mlir
    %token = yirage.sample_top_p %logits {top_p = 0.9, temperature = 1.0}
        : tensor<1xvocab_sizexf16> -> tensor<1xi32>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$logits,
    DefaultValuedAttr<F32Attr, "0.9">:$top_p,
    DefaultValuedAttr<F32Attr, "1.0">:$temperature
  );
  let results = (outs AnyRankedTensor:$token);
  
  let hasVerifier = 1;
}

def Yirage_SampleTopKOp : Yirage_Op<"sample_top_k", [Pure]> {
  let summary = "Top-k sampling";
  let description = [{
    Sample from logits using top-k sampling.
    
    Example:
    ```mlir
    %token = yirage.sample_top_k %logits {top_k = 50, temperature = 1.0}
        : tensor<1xvocab_sizexf16> -> tensor<1xi32>
    ```
  }];
  
  let arguments = (ins
    AnyRankedTensor:$logits,
    I64Attr:$top_k,
    DefaultValuedAttr<F32Attr, "1.0">:$temperature
  );
  let results = (outs AnyRankedTensor:$token);
  
  let hasVerifier = 1;
}

#endif // YIRAGE_OPS
