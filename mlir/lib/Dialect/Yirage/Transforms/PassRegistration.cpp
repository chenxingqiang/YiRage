//===- PassRegistration.cpp - Register Yirage passes -------------*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file registers all YiRage MLIR passes and pipelines.
//
// Complete Compilation Pipeline:
//
//   muGraph (Python/C++)
//       ↓ mugraph_to_mlir.py
//   YiRage MLIR dialect (high-level operators)
//       ↓ yirage-to-linalg
//   Linalg dialect (tensor semantics)
//       ↓ yirage-linalg-tiling
//   Tiled Linalg (blocked for hardware)
//       ↓ yirage-tensor-to-memref (bufferization)
//   Linalg on memrefs (buffer semantics)
//       ↓ convert-linalg-to-loops
//   SCF/Affine loops
//       ↓ [branch by target]
//   ┌─────────────────────────────────────────────────────────────┐
//   │ GPU Path            │ CPU Path           │ Accelerator Path │
//   │ yirage-scf-to-gpu   │ yirage-vectorize   │ target-specific  │
//   │ gpu-to-nvvm/rocdl   │ convert-to-llvm    │ lowering         │
//   └─────────────────────────────────────────────────────────────┘
//       ↓
//   Target IR (NVVM, ROCDL, SPIRV, LLVM, etc.)
//       ↓ LLVM backend
//   Binary (PTX/cubin, GCN/hsaco, SPIR-V, native)
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Dialect/Yirage/Transforms/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace yirage {

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void registerYiragePasses() {
  // Core lowering passes
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createYirageToLinalgPass();
  });
  
  // GPU-related passes
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLinalgTilingPass();
  });
  
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createTensorToMemRefPass();
  });
  
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSCFToGPUPass();
  });
  
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLinalgToGPUPipelinePass("cuda");
  });
  
  // CPU-related passes
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLinalgVectorizationPass();
  });
  
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLoopOptimizationPass();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createCpuJitOutParamPass();
  });
  
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLinalgToCPUPipelinePass("x86-64");
  });
  
  // Attention optimization passes
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createFlashAttentionPass();
  });
}

void registerYiragePassPipelines() {
  //===--------------------------------------------------------------------===//
  // GPU Pipeline: Full path to GPU kernel generation
  //===--------------------------------------------------------------------===//
  mlir::PassPipelineRegistration<>(
      "yirage-gpu-pipeline",
      "Complete GPU lowering: Yirage → Linalg → Tiling → GPU → NVVM/ROCDL",
      [](mlir::OpPassManager &pm) {
        // Step 1: YiRage → Linalg (high-level to mid-level)
        pm.addPass(createYirageToLinalgPass());
        
        // Step 2: Canonicalization and CSE
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        
        // Step 3: Linalg tiling for GPU blocks
        pm.addPass(createLinalgTilingPass());
        
        // Step 4: Bufferization (tensor → memref)
        pm.addPass(createTensorToMemRefPass());
        
        // Step 5: Lower Linalg to loops (SCF)
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
        
        // Step 6: Map loops to GPU
        pm.addNestedPass<mlir::func::FuncOp>(createSCFToGPUPass());
        
        // Step 7: Further GPU lowering would happen here
        // (GPU to NVVM/ROCDL/SPIRV based on target)
      });

  //===--------------------------------------------------------------------===//
  // CUDA Pipeline: Optimized for NVIDIA GPUs
  //===--------------------------------------------------------------------===//
  mlir::PassPipelineRegistration<>(
      "yirage-cuda-pipeline",
      "NVIDIA CUDA lowering: Yirage → Linalg → GPU → NVVM → PTX",
      [](mlir::OpPassManager &pm) {
        pm.addPass(createYirageToLinalgPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(createLinalgTilingPass());
        pm.addPass(createTensorToMemRefPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
        pm.addNestedPass<mlir::func::FuncOp>(createSCFToGPUPass());
        // GPU to NVVM lowering would be added here
      });

  //===--------------------------------------------------------------------===//
  // ROCm Pipeline: Optimized for AMD GPUs
  //===--------------------------------------------------------------------===//
  mlir::PassPipelineRegistration<>(
      "yirage-rocm-pipeline",
      "AMD ROCm lowering: Yirage → Linalg → GPU → ROCDL → GCN",
      [](mlir::OpPassManager &pm) {
        pm.addPass(createYirageToLinalgPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(createLinalgTilingPass());
        pm.addPass(createTensorToMemRefPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
        pm.addNestedPass<mlir::func::FuncOp>(createSCFToGPUPass());
        // GPU to ROCDL lowering would be added here
      });

  //===--------------------------------------------------------------------===//
  // CPU Pipeline: Full path to native code
  //===--------------------------------------------------------------------===//
  mlir::PassPipelineRegistration<>(
      "yirage-cpu-pipeline",
      "Complete CPU lowering: Yirage → Linalg → Loops (tensor semantics)",
      [](mlir::OpPassManager &pm) {
        // Step 1: YiRage → Linalg
        pm.addPass(createYirageToLinalgPass());
        
        // Step 2: Canonicalization and CSE
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        
        // Step 3: Vectorization for SIMD (placeholder)
        pm.addNestedPass<mlir::func::FuncOp>(createLinalgVectorizationPass());
        
        // Step 4: Loop optimizations (placeholder)
        pm.addNestedPass<mlir::func::FuncOp>(createLoopOptimizationPass());
        
        // Note: Full LLVM lowering requires bufferization first
        // Use yirage-to-llvm pipeline for complete native code generation
      });

  //===--------------------------------------------------------------------===//
  // CPU JIT Pipeline: YiRage → Linalg → buffers → loops (then JITRunner)
  //===--------------------------------------------------------------------===//
  mlir::PassPipelineRegistration<>(
      "yirage-cpu-jit-pipeline",
      "CPU JIT prep: Yirage → Linalg → one-shot bufferize → loops",
      [](mlir::OpPassManager &pm) {
        pm.addPass(createYirageToLinalgPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(
            mlir::bufferization::createEmptyTensorToAllocTensorPass());
        {
          mlir::bufferization::OneShotBufferizationOptions opts;
          opts.bufferizeFunctionBoundaries = true;
          opts.allowReturnAllocs = true;
          opts.setFunctionBoundaryTypeConversion(
              mlir::bufferization::LayoutMapOption::IdentityLayoutMap);
          pm.addPass(
              mlir::bufferization::createOneShotBufferizePass(opts));
        }
        pm.addNestedPass<mlir::func::FuncOp>(
            mlir::createConvertLinalgToLoopsPass());
        pm.addNestedPass<mlir::func::FuncOp>(createCpuJitOutParamPass());
        pm.addPass(mlir::memref::createExpandStridedMetadataPass());
        pm.addPass(mlir::createCanonicalizerPass());
      });

  //===--------------------------------------------------------------------===//
  // LLVM Pipeline: Lower to LLVM IR (for JIT or AOT compilation)
  //===--------------------------------------------------------------------===//
  mlir::PassPipelineRegistration<>(
      "yirage-to-llvm",
      "Full LLVM lowering: Yirage → Linalg → Buffers → Loops → LLVM IR",
      [](mlir::OpPassManager &pm) {
        pm.addPass(createYirageToLinalgPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(createTensorToMemRefPass());
        
        // Step 2: Lower Linalg to loops
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
        
        // Step 3: Lower SCF to control flow
        pm.addPass(mlir::createConvertSCFToCFPass());
        
        // Step 4: Lower to LLVM dialect
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        
        // Step 5: Cleanup - reconcile unrealized casts
        pm.addPass(mlir::createCanonicalizerPass());
      });

  //===--------------------------------------------------------------------===//
  // MPS Pipeline: Apple Silicon GPU
  //===--------------------------------------------------------------------===//
  mlir::PassPipelineRegistration<>(
      "yirage-mps-pipeline",
      "Apple MPS lowering: Yirage → Linalg → GPU → Metal",
      [](mlir::OpPassManager &pm) {
        pm.addPass(createYirageToLinalgPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(createLinalgTilingPass());
        pm.addPass(createTensorToMemRefPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
        // Metal-specific lowering would be added here
      });

  //===--------------------------------------------------------------------===//
  // Ascend Pipeline: Huawei NPU
  //===--------------------------------------------------------------------===//
  mlir::PassPipelineRegistration<>(
      "yirage-ascend-pipeline",
      "Huawei Ascend lowering: Yirage → Linalg → TBE",
      [](mlir::OpPassManager &pm) {
        pm.addPass(createYirageToLinalgPass());
        pm.addPass(mlir::createCanonicalizerPass());
        // Ascend TBE-specific lowering would be added here
      });

  //===--------------------------------------------------------------------===//
  // TPU Pipeline: Google TPU
  //===--------------------------------------------------------------------===//
  mlir::PassPipelineRegistration<>(
      "yirage-tpu-pipeline",
      "Google TPU lowering: Yirage → StableHLO → XLA",
      [](mlir::OpPassManager &pm) {
        // For TPU, we would lower to StableHLO instead of Linalg
        // StableHLO is the standard dialect for XLA-based compilation
        pm.addPass(createYirageToLinalgPass());  // Placeholder
        // createYirageToStableHLOPass() would be used here
      });

  //===--------------------------------------------------------------------===//
  // FPGA Pipeline: High-Level Synthesis
  //===--------------------------------------------------------------------===//
  mlir::PassPipelineRegistration<>(
      "yirage-fpga-pipeline",
      "FPGA HLS lowering: Yirage → Affine → HLS C++",
      [](mlir::OpPassManager &pm) {
        pm.addPass(createYirageToLinalgPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToAffineLoopsPass());
        // Affine loop optimizations for HLS would be added here
      });
}

} // namespace yirage
