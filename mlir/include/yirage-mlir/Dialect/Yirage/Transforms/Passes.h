//===- Passes.h - Yirage Transformation Passes ------------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_MLIR_DIALECT_YIRAGE_TRANSFORMS_PASSES_H
#define YIRAGE_MLIR_DIALECT_YIRAGE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
namespace func {
class FuncOp;
} // namespace func
namespace gpu {
class GPUModuleOp;
} // namespace gpu
} // namespace mlir

namespace yirage {

//===----------------------------------------------------------------------===//
// Optimization Passes
//===----------------------------------------------------------------------===//

/// Creates a pass to optimize Yirage operations.
std::unique_ptr<mlir::Pass> createYirageOptimizePass();

/// Creates a pass to fuse Yirage operations.
std::unique_ptr<mlir::Pass> createYirageFuseOpsPass();

//===----------------------------------------------------------------------===//
// Lowering Passes
//===----------------------------------------------------------------------===//

/// Creates a pass to lower Yirage to Linalg dialect.
std::unique_ptr<mlir::Pass> createYirageToLinalgPass();

/// Creates a pass to lower Yirage to StableHLO dialect.
std::unique_ptr<mlir::Pass> createYirageToStableHLOPass();

/// Creates a pass to lower Yirage to TOSA dialect.
std::unique_ptr<mlir::Pass> createYirageToTOSAPass();

//===----------------------------------------------------------------------===//
// Tiling and Vectorization Passes
//===----------------------------------------------------------------------===//

/// Creates a pass to tile and fuse operations.
std::unique_ptr<mlir::Pass> createYirageTileAndFusePass();

/// Creates a pass to vectorize operations.
std::unique_ptr<mlir::Pass> createYirageVectorizePass();

//===----------------------------------------------------------------------===//
// Attention Optimization Passes
//===----------------------------------------------------------------------===//

/// Creates a pass to optimize attention using Flash Attention algorithm.
/// Flash Attention reduces memory usage from O(N^2) to O(N) by using
/// online softmax and block-wise computation.
std::unique_ptr<mlir::Pass> createFlashAttentionPass();

/// Creates a Flash Attention pass with custom block sizes.
/// @param blockQ Block size for query tiling (default: 64)
/// @param blockKV Block size for key/value tiling (default: 64)
std::unique_ptr<mlir::Pass> createFlashAttentionPass(int64_t blockQ, int64_t blockKV);

//===----------------------------------------------------------------------===//
// GPU Lowering Passes
//===----------------------------------------------------------------------===//

/// Creates a pass to tile Linalg operations for GPU execution.
std::unique_ptr<mlir::Pass> createLinalgTilingPass();

/// Creates a pass to convert tensor to memref (bufferization).
std::unique_ptr<mlir::Pass> createTensorToMemRefPass();

/// Creates a pass to map SCF loops to GPU launch.
std::unique_ptr<mlir::Pass> createSCFToGPUPass();

/// Creates a full GPU lowering pipeline pass.
std::unique_ptr<mlir::Pass> createLinalgToGPUPipelinePass(llvm::StringRef target = "cuda");

/// Creates a pass to lower Linalg to GPU dialect.
std::unique_ptr<mlir::Pass> createYirageLinalgToGPUPass();

/// Creates a pass to lower GPU to NVVM (CUDA).
std::unique_ptr<mlir::Pass> createYirageGPUToNVVMPass();

/// Creates a pass to lower GPU to ROCDL (AMD ROCm).
std::unique_ptr<mlir::Pass> createYirageGPUToROCDLPass();

/// Creates a pass to lower GPU to SPIR-V (Intel XPU, Vulkan).
std::unique_ptr<mlir::Pass> createYirageGPUToSPIRVPass();

/// Creates a pass to lower GPU to Metal (Apple MPS).
std::unique_ptr<mlir::Pass> createYirageGPUToMetalPass();

/// Creates a pass to lower GPU to MACA (MetaX).
std::unique_ptr<mlir::Pass> createYirageGPUToMACAPass();

//===----------------------------------------------------------------------===//
// CPU Lowering Passes
//===----------------------------------------------------------------------===//

/// Creates a pass to vectorize Linalg operations for SIMD.
std::unique_ptr<mlir::Pass> createLinalgVectorizationPass();

/// Creates a pass to optimize loops for CPU cache hierarchy.
std::unique_ptr<mlir::Pass> createLoopOptimizationPass();

/// Rewrites ``(x, w) -> memref`` into ``(x, w, out)`` for CPU JIT invoke ABI.
std::unique_ptr<mlir::Pass> createCpuJitOutParamPass();

/// Creates a full CPU lowering pipeline pass.
std::unique_ptr<mlir::Pass> createLinalgToCPUPipelinePass(llvm::StringRef arch = "x86-64");

/// Creates a pass to lower Linalg to LLVM.
std::unique_ptr<mlir::Pass> createYirageLinalgToLLVMPass();

//===----------------------------------------------------------------------===//
// Accelerator-Specific Passes
//===----------------------------------------------------------------------===//

/// Creates a pass to lower Linalg to TBE (Huawei Ascend).
std::unique_ptr<mlir::Pass> createYirageLinalgToTBEPass();

/// Creates a pass to lower Affine to HLS (FPGA).
std::unique_ptr<mlir::Pass> createYirageAffineToHLSPass();

//===----------------------------------------------------------------------===//
// Pipeline Passes
//===----------------------------------------------------------------------===//

/// Creates a complete GPU lowering pipeline.
std::unique_ptr<mlir::Pass> createYirageGPUPipelinePass();

/// Creates a complete CPU lowering pipeline.
std::unique_ptr<mlir::Pass> createYirageCPUPipelinePass();

/// Creates a complete TPU lowering pipeline.
std::unique_ptr<mlir::Pass> createYirageTPUPipelinePass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Generate pass declarations and definitions.
#define GEN_PASS_DECL
#include "yirage-mlir/Dialect/Yirage/Transforms/Passes.h.inc"

/// Register all Yirage passes.
void registerYiragePasses();

/// Register all Yirage pass pipelines.
void registerYiragePassPipelines();

} // namespace yirage

#endif // YIRAGE_MLIR_DIALECT_YIRAGE_TRANSFORMS_PASSES_H
