//===- Passes.td - Yirage Transformation Passes ------------*- tablegen -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file defines transformation passes for the Yirage dialect.
//
//===----------------------------------------------------------------------===//

#ifndef YIRAGE_TRANSFORMS_PASSES
#define YIRAGE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Optimization Passes
//===----------------------------------------------------------------------===//

def YirageOptimize : Pass<"yirage-optimize", "mlir::func::FuncOp"> {
  let summary = "Apply Yirage-specific optimizations";
  let description = [{
    This pass applies various optimizations specific to the Yirage dialect:
    - Constant folding
    - Dead code elimination
    - Strength reduction
  }];
  let constructor = "yirage::createYirageOptimizePass()";
}

def YirageFuseOps : Pass<"yirage-fuse-ops", "mlir::func::FuncOp"> {
  let summary = "Fuse Yirage operations for performance";
  let description = [{
    This pass identifies opportunities to fuse multiple Yirage operations
    into single fused operations:
    - RMSNorm + Attention
    - GatedMLP (gate + up + silu + mul + down)
    - Embedding + RoPE
  }];
  let constructor = "yirage::createYirageFuseOpsPass()";
  let options = [
    Option<"fusionLevel", "fusion-level", "int", /*default=*/"2",
           "Level of fusion aggressiveness (0-3)">
  ];
}

//===----------------------------------------------------------------------===//
// Lowering Passes - To Linalg
//===----------------------------------------------------------------------===//

def YirageToLinalg : Pass<"yirage-to-linalg", "mlir::ModuleOp"> {
  let summary = "Lower Yirage dialect to Linalg dialect";
  let description = [{
    This pass lowers Yirage operations to equivalent Linalg operations.
    This is the primary lowering path for CPU and generic GPU targets.
    
    Lowering mappings:
    - yirage.matmul -> linalg.matmul
    - yirage.attention -> linalg.batch_matmul + softmax decomposition
    - yirage.rms_norm -> linalg.generic
    - yirage.silu/gelu -> linalg.generic with math ops
  }];
  let constructor = "yirage::createYirageToLinalgPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Lowering Passes - To StableHLO (TPU/XLA)
//===----------------------------------------------------------------------===//

def YirageToStableHLO : Pass<"yirage-to-stablehlo", "mlir::ModuleOp"> {
  let summary = "Lower Yirage dialect to StableHLO dialect";
  let description = [{
    This pass lowers Yirage operations to StableHLO for TPU and XLA targets.
    
    Lowering mappings:
    - yirage.matmul -> stablehlo.dot_general
    - yirage.attention -> stablehlo ops composition
    - yirage.embedding -> stablehlo.gather
    - yirage.softmax -> stablehlo reduce + exp + div
  }];
  let constructor = "yirage::createYirageToStableHLOPass()";
  let dependentDialects = [
    "mlir::stablehlo::StablehloDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Lowering Passes - To TOSA
//===----------------------------------------------------------------------===//

def YirageToTOSA : Pass<"yirage-to-tosa", "mlir::ModuleOp"> {
  let summary = "Lower Yirage dialect to TOSA dialect";
  let description = [{
    This pass lowers Yirage operations to TOSA for edge/mobile targets.
    TOSA provides good coverage for neural network operations.
  }];
  let constructor = "yirage::createYirageToTOSAPass()";
  let dependentDialects = [
    "mlir::tosa::TosaDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Tiling and Fusion Passes
//===----------------------------------------------------------------------===//

def YirageTileAndFuse : Pass<"yirage-tile-and-fuse", "mlir::func::FuncOp"> {
  let summary = "Tile and fuse Yirage operations";
  let description = [{
    This pass tiles operations for cache efficiency and fuses
    producer-consumer pairs to minimize memory traffic.
  }];
  let constructor = "yirage::createYirageTileAndFusePass()";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes for each dimension">,
    Option<"enableFusion", "enable-fusion", "bool", /*default=*/"true",
           "Enable fusion after tiling">
  ];
}

def YirageVectorize : Pass<"yirage-vectorize", "mlir::func::FuncOp"> {
  let summary = "Vectorize Yirage operations";
  let description = [{
    This pass vectorizes Yirage operations using MLIR's vector dialect.
    Supports different vector widths for various targets.
  }];
  let constructor = "yirage::createYirageVectorizePass()";
  let options = [
    Option<"vectorWidth", "vector-width", "int", /*default=*/"8",
           "Vector width for SIMD operations">
  ];
  let dependentDialects = [
    "mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// GPU Lowering Passes
//===----------------------------------------------------------------------===//

def YirageLinalgToGPU : Pass<"yirage-linalg-to-gpu", "mlir::func::FuncOp"> {
  let summary = "Lower Linalg to GPU dialect";
  let description = [{
    This pass converts Linalg operations to GPU kernels with
    appropriate thread/block mappings.
  }];
  let constructor = "yirage::createYirageLinalgToGPUPass()";
  let options = [
    Option<"blockSizeX", "block-size-x", "int", /*default=*/"256",
           "Thread block size in X dimension">,
    Option<"blockSizeY", "block-size-y", "int", /*default=*/"1",
           "Thread block size in Y dimension">
  ];
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def YirageGPUToNVVM : Pass<"yirage-gpu-to-nvvm", "mlir::gpu::GPUModuleOp"> {
  let summary = "Lower GPU dialect to NVVM (CUDA PTX)";
  let description = [{
    This pass lowers GPU operations to NVVM intrinsics for NVIDIA GPUs.
    Handles shared memory, tensor cores (WMMA), and other CUDA-specific features.
  }];
  let constructor = "yirage::createYirageGPUToNVVMPass()";
  let options = [
    Option<"useTensorCores", "use-tensor-cores", "bool", /*default=*/"true",
           "Use tensor core operations when possible">,
    Option<"smVersion", "sm-version", "int", /*default=*/"80",
           "Target SM version">
  ];
  let dependentDialects = [
    "mlir::NVVM::NVVMDialect"
  ];
}

def YirageGPUToROCDL : Pass<"yirage-gpu-to-rocdl", "mlir::gpu::GPUModuleOp"> {
  let summary = "Lower GPU dialect to ROCDL (AMD HIP)";
  let description = [{
    This pass lowers GPU operations to ROCDL intrinsics for AMD GPUs.
    Handles LDS, MFMA, and other AMD-specific features.
  }];
  let constructor = "yirage::createYirageGPUToROCDLPass()";
  let options = [
    Option<"useMFMA", "use-mfma", "bool", /*default=*/"true",
           "Use MFMA matrix operations when possible">,
    Option<"gfxVersion", "gfx-version", "std::string", /*default=*/"\"gfx90a\"",
           "Target GFX version">
  ];
  let dependentDialects = [
    "mlir::ROCDL::ROCDLDialect"
  ];
}

def YirageGPUToSPIRV : Pass<"yirage-gpu-to-spirv", "mlir::ModuleOp"> {
  let summary = "Lower GPU dialect to SPIR-V (Intel XPU, Vulkan)";
  let description = [{
    This pass lowers GPU operations to SPIR-V for Intel XPU, Vulkan, and OpenCL.
    Handles subgroup operations and Intel-specific extensions.
  }];
  let constructor = "yirage::createYirageGPUToSPIRVPass()";
  let options = [
    Option<"useCooperativeMatrices", "use-cooperative-matrices", "bool",
           /*default=*/"true", "Use SPIR-V cooperative matrix operations">
  ];
  let dependentDialects = [
    "mlir::spirv::SPIRVDialect"
  ];
}

//===----------------------------------------------------------------------===//
// CPU Lowering Passes
//===----------------------------------------------------------------------===//

def YirageLinalgToLLVM : Pass<"yirage-linalg-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower Linalg to LLVM dialect for CPU";
  let description = [{
    This pass lowers Linalg operations through SCF and Affine to LLVM dialect.
    Optimizes for CPU SIMD (AVX2, AVX-512, NEON, SVE).
  }];
  let constructor = "yirage::createYirageLinalgToLLVMPass()";
  let options = [
    Option<"targetArch", "target-arch", "std::string", /*default=*/"\"x86-64-v3\"",
           "Target CPU architecture">,
    Option<"enableOpenMP", "enable-openmp", "bool", /*default=*/"true",
           "Enable OpenMP parallelization">
  ];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::scf::SCFDialect",
    "mlir::affine::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Backend-Specific Passes
//===----------------------------------------------------------------------===//

def YirageGPUToMetal : Pass<"yirage-gpu-to-metal", "mlir::gpu::GPUModuleOp"> {
  let summary = "Lower GPU dialect to Metal (Apple MPS)";
  let constructor = "yirage::createYirageGPUToMetalPass()";
  let options = [
    Option<"useSimdGroup", "use-simd-group", "bool", /*default=*/"true",
           "Use SIMD group operations">
  ];
}

def YirageGPUToMACA : Pass<"yirage-gpu-to-maca", "mlir::gpu::GPUModuleOp"> {
  let summary = "Lower GPU dialect to MACA (MetaX GPU)";
  let constructor = "yirage::createYirageGPUToMACAPass()";
}

def YirageLinalgToTBE : Pass<"yirage-linalg-to-tbe", "mlir::func::FuncOp"> {
  let summary = "Lower Linalg to TBE for Huawei Ascend";
  let constructor = "yirage::createYirageLinalgToTBEPass()";
  let options = [
    Option<"ascendVersion", "ascend-version", "std::string",
           /*default=*/"\"910B\"", "Target Ascend version">
  ];
}

def YirageAffineToHLS : Pass<"yirage-affine-to-hls", "mlir::func::FuncOp"> {
  let summary = "Lower Affine dialect to HLS C++ for FPGA";
  let constructor = "yirage::createYirageAffineToHLSPass()";
  let options = [
    Option<"targetFPGA", "target-fpga", "std::string",
           /*default=*/"\"xilinx\"", "Target FPGA vendor">
  ];
}

//===----------------------------------------------------------------------===//
// Pass Pipelines
//===----------------------------------------------------------------------===//

def YirageGPUPipeline : Pass<"yirage-gpu-pipeline", "mlir::ModuleOp"> {
  let summary = "Run full GPU optimization and lowering pipeline";
  let description = [{
    This meta-pass runs the complete pipeline for GPU targets:
    1. yirage-optimize
    2. yirage-fuse-ops
    3. yirage-to-linalg
    4. yirage-tile-and-fuse
    5. yirage-linalg-to-gpu
    6. yirage-gpu-to-{nvvm|rocdl|spirv}
  }];
  let constructor = "yirage::createYirageGPUPipelinePass()";
  let options = [
    Option<"targetGPU", "target-gpu", "std::string", /*default=*/"\"cuda\"",
           "Target GPU platform: cuda, rocm, spirv, metal, maca">
  ];
}

def YirageCPUPipeline : Pass<"yirage-cpu-pipeline", "mlir::ModuleOp"> {
  let summary = "Run full CPU optimization and lowering pipeline";
  let constructor = "yirage::createYirageCPUPipelinePass()";
  let options = [
    Option<"targetArch", "target-arch", "std::string", /*default=*/"\"native\"",
           "Target CPU architecture">
  ];
}

def YirageTPUPipeline : Pass<"yirage-tpu-pipeline", "mlir::ModuleOp"> {
  let summary = "Run full TPU optimization and lowering pipeline";
  let constructor = "yirage::createYirageTPUPipelinePass()";
}

#endif // YIRAGE_TRANSFORMS_PASSES
