//===- YirageDialect.td - Yirage Dialect Definition --------*- tablegen -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file defines the Yirage dialect for MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_DIALECT
#define YIRAGE_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// Yirage Dialect Definition
//===----------------------------------------------------------------------===//

def Yirage_Dialect : Dialect {
  let name = "yirage";
  let cppNamespace = "::yirage::ir";
  let summary = "YiRage dialect for high-performance AI/LLM operators";
  let description = [{
    The Yirage dialect provides a set of high-level operations for AI and LLM
    workloads. It serves as an intermediate representation between framework
    graphs (PyTorch, JAX) and low-level target dialects (Linalg, GPU, LLVM).
    
    Key features:
    - LLM-specific operations (attention, RoPE, RMSNorm, etc.)
    - Quantization-aware operations (INT8/INT4 matmul)
    - Multi-backend targeting (CUDA, ROCm, TPU, CPU, etc.)
    - Automatic kernel fusion and optimization
    
    Example:
    ```mlir
    %output = yirage.attention %Q, %K, %V {causal = true}
        : tensor<1x32x2048x128xf16> -> tensor<1x32x2048x128xf16>
    ```
  }];
  
  // No dependent dialects for now - keep it minimal for initial testing
}

// Note: Custom types and attributes are defined in separate files for clarity.
// For initial testing, we use built-in MLIR types (tensor, memref) directly.

#endif // YIRAGE_DIALECT
