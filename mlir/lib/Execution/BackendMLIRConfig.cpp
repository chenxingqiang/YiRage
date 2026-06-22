//===- BackendMLIRConfig.cpp - Backend-Specific MLIR Config -----*- C++ -*-===//
//
// Part of the YiRage Project, under the Apache License v2.0.
// See LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file provides backend-specific MLIR compilation configurations for
// all supported hardware targets.
//
//===----------------------------------------------------------------------===//

#include "yirage-mlir/Execution/GPUCodeGen.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include <cstdlib>
#include <string>
#include <unordered_map>

namespace yirage {

//===----------------------------------------------------------------------===//
// Backend Configuration Data
//===----------------------------------------------------------------------===//

/// Hardware profile for a specific backend
struct BackendHardwareProfile {
  std::string name;
  std::string arch;
  std::string triple;
  
  // Compute capabilities
  double peakTFLOPS_FP16;
  double peakTFLOPS_FP32;
  double peakTFLOPS_INT8;
  
  // Memory
  double dramBandwidth_GBps;
  double onChipBandwidth_GBps;
  size_t sharedMemorySize_KB;
  size_t l2CacheSize_MB;
  
  // Parallelism
  int maxWarps;
  int maxRegisters;
  int maxThreadsPerBlock;
  
  // Tiling
  int preferredTileM;
  int preferredTileN;
  int preferredTileK;
  
  // Optimizations
  bool supportsTensorCore;
  bool supportsAsyncCopy;
  bool supportsDynamicParallelism;
  int vectorWidth;
};

/// Get hardware profiles for all backends
std::unordered_map<std::string, BackendHardwareProfile> getBackendProfiles() {
  return {
    //========================================================================
    // NVIDIA CUDA Profiles
    //========================================================================
    {"cuda-sm_70", {
      .name = "NVIDIA V100",
      .arch = "sm_70",
      .triple = "nvptx64-nvidia-cuda",
      .peakTFLOPS_FP16 = 125.0,
      .peakTFLOPS_FP32 = 15.7,
      .peakTFLOPS_INT8 = 125.0,
      .dramBandwidth_GBps = 900.0,
      .onChipBandwidth_GBps = 13800.0,
      .sharedMemorySize_KB = 96,
      .l2CacheSize_MB = 6,
      .maxWarps = 64,
      .maxRegisters = 65536,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 128,
      .preferredTileN = 128,
      .preferredTileK = 32,
      .supportsTensorCore = true,
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = true,
      .vectorWidth = 8
    }},
    
    {"cuda-sm_80", {
      .name = "NVIDIA A100",
      .arch = "sm_80",
      .triple = "nvptx64-nvidia-cuda",
      .peakTFLOPS_FP16 = 312.0,
      .peakTFLOPS_FP32 = 19.5,
      .peakTFLOPS_INT8 = 624.0,
      .dramBandwidth_GBps = 2039.0,
      .onChipBandwidth_GBps = 19500.0,
      .sharedMemorySize_KB = 164,
      .l2CacheSize_MB = 40,
      .maxWarps = 64,
      .maxRegisters = 65536,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 128,
      .preferredTileN = 256,
      .preferredTileK = 64,
      .supportsTensorCore = true,
      .supportsAsyncCopy = true,
      .supportsDynamicParallelism = true,
      .vectorWidth = 8
    }},
    
    {"cuda-sm_90", {
      .name = "NVIDIA H100",
      .arch = "sm_90",
      .triple = "nvptx64-nvidia-cuda",
      .peakTFLOPS_FP16 = 989.0,
      .peakTFLOPS_FP32 = 67.0,
      .peakTFLOPS_INT8 = 1979.0,
      .dramBandwidth_GBps = 3350.0,
      .onChipBandwidth_GBps = 33000.0,
      .sharedMemorySize_KB = 228,
      .l2CacheSize_MB = 50,
      .maxWarps = 64,
      .maxRegisters = 65536,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 256,
      .preferredTileN = 256,
      .preferredTileK = 64,
      .supportsTensorCore = true,
      .supportsAsyncCopy = true,
      .supportsDynamicParallelism = true,
      .vectorWidth = 8
    }},
    
    //========================================================================
    // AMD ROCm Profiles
    //========================================================================
    {"rocm-gfx908", {
      .name = "AMD MI100",
      .arch = "gfx908",
      .triple = "amdgcn-amd-amdhsa",
      .peakTFLOPS_FP16 = 184.6,
      .peakTFLOPS_FP32 = 23.1,
      .peakTFLOPS_INT8 = 184.6,
      .dramBandwidth_GBps = 1228.8,
      .onChipBandwidth_GBps = 10000.0,
      .sharedMemorySize_KB = 64,
      .l2CacheSize_MB = 8,
      .maxWarps = 32,
      .maxRegisters = 65536,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 128,
      .preferredTileN = 128,
      .preferredTileK = 32,
      .supportsTensorCore = true,  // Matrix cores
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = false,
      .vectorWidth = 8
    }},
    
    {"rocm-gfx90a", {
      .name = "AMD MI250",
      .arch = "gfx90a",
      .triple = "amdgcn-amd-amdhsa",
      .peakTFLOPS_FP16 = 383.0,
      .peakTFLOPS_FP32 = 47.9,
      .peakTFLOPS_INT8 = 383.0,
      .dramBandwidth_GBps = 3276.8,
      .onChipBandwidth_GBps = 15000.0,
      .sharedMemorySize_KB = 64,
      .l2CacheSize_MB = 8,
      .maxWarps = 32,
      .maxRegisters = 65536,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 128,
      .preferredTileN = 256,
      .preferredTileK = 64,
      .supportsTensorCore = true,
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = false,
      .vectorWidth = 8
    }},
    
    {"rocm-gfx942", {
      .name = "AMD MI300X",
      .arch = "gfx942",
      .triple = "amdgcn-amd-amdhsa",
      .peakTFLOPS_FP16 = 1307.4,
      .peakTFLOPS_FP32 = 163.4,
      .peakTFLOPS_INT8 = 2614.9,
      .dramBandwidth_GBps = 5300.0,
      .onChipBandwidth_GBps = 25000.0,
      .sharedMemorySize_KB = 64,
      .l2CacheSize_MB = 256,
      .maxWarps = 32,
      .maxRegisters = 65536,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 256,
      .preferredTileN = 256,
      .preferredTileK = 64,
      .supportsTensorCore = true,
      .supportsAsyncCopy = true,
      .supportsDynamicParallelism = false,
      .vectorWidth = 8
    }},
    
    //========================================================================
    // Intel XPU Profiles
    //========================================================================
    {"xpu-pvc", {
      .name = "Intel Max 1550 (Ponte Vecchio)",
      .arch = "pvc",
      .triple = "spir64-intel-gpu",
      .peakTFLOPS_FP16 = 839.0,
      .peakTFLOPS_FP32 = 52.0,
      .peakTFLOPS_INT8 = 1678.0,
      .dramBandwidth_GBps = 3276.0,
      .onChipBandwidth_GBps = 20000.0,
      .sharedMemorySize_KB = 64,
      .l2CacheSize_MB = 408,
      .maxWarps = 64,
      .maxRegisters = 65536,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 256,
      .preferredTileN = 256,
      .preferredTileK = 32,
      .supportsTensorCore = true,  // XMX
      .supportsAsyncCopy = true,
      .supportsDynamicParallelism = false,
      .vectorWidth = 16
    }},
    
    //========================================================================
    // Google TPU Profiles
    //========================================================================
    {"tpu-v4", {
      .name = "Google TPU v4",
      .arch = "tpu-v4",
      .triple = "tpu",
      .peakTFLOPS_FP16 = 275.0,
      .peakTFLOPS_FP32 = 275.0,  // BF16 mainly
      .peakTFLOPS_INT8 = 550.0,
      .dramBandwidth_GBps = 1228.0,
      .onChipBandwidth_GBps = 50000.0,  // HBM3
      .sharedMemorySize_KB = 128,  // VMEM
      .l2CacheSize_MB = 32,
      .maxWarps = 128,
      .maxRegisters = 65536,
      .maxThreadsPerBlock = 2048,
      .preferredTileM = 128,
      .preferredTileN = 128,
      .preferredTileK = 128,
      .supportsTensorCore = true,  // MXU
      .supportsAsyncCopy = true,
      .supportsDynamicParallelism = false,
      .vectorWidth = 128
    }},
    
    {"tpu-v5e", {
      .name = "Google TPU v5e",
      .arch = "tpu-v5e",
      .triple = "tpu",
      .peakTFLOPS_FP16 = 197.0,
      .peakTFLOPS_FP32 = 197.0,
      .peakTFLOPS_INT8 = 394.0,
      .dramBandwidth_GBps = 1600.0,
      .onChipBandwidth_GBps = 40000.0,
      .sharedMemorySize_KB = 128,
      .l2CacheSize_MB = 32,
      .maxWarps = 128,
      .maxRegisters = 65536,
      .maxThreadsPerBlock = 2048,
      .preferredTileM = 128,
      .preferredTileN = 128,
      .preferredTileK = 128,
      .supportsTensorCore = true,
      .supportsAsyncCopy = true,
      .supportsDynamicParallelism = false,
      .vectorWidth = 128
    }},
    
    //========================================================================
    // Huawei Ascend Profiles
    //========================================================================
    {"ascend-910b", {
      .name = "Huawei Ascend 910B",
      .arch = "ascend-910b",
      .triple = "ascend",
      .peakTFLOPS_FP16 = 320.0,
      .peakTFLOPS_FP32 = 160.0,
      .peakTFLOPS_INT8 = 640.0,
      .dramBandwidth_GBps = 1500.0,
      .onChipBandwidth_GBps = 20000.0,
      .sharedMemorySize_KB = 512,
      .l2CacheSize_MB = 32,
      .maxWarps = 64,
      .maxRegisters = 65536,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 128,
      .preferredTileN = 256,
      .preferredTileK = 64,
      .supportsTensorCore = true,  // AI Core
      .supportsAsyncCopy = true,
      .supportsDynamicParallelism = false,
      .vectorWidth = 16
    }},
    
    //========================================================================
    // MetaX MACA Profiles
    //========================================================================
    {"maca-mxc500", {
      .name = "MetaX MXC500",
      .arch = "mxc500",
      .triple = "maca",
      .peakTFLOPS_FP16 = 200.0,
      .peakTFLOPS_FP32 = 100.0,
      .peakTFLOPS_INT8 = 400.0,
      .dramBandwidth_GBps = 1200.0,
      .onChipBandwidth_GBps = 15000.0,
      .sharedMemorySize_KB = 128,
      .l2CacheSize_MB = 32,
      .maxWarps = 32,
      .maxRegisters = 32768,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 128,
      .preferredTileN = 128,
      .preferredTileK = 32,
      .supportsTensorCore = true,
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = false,
      .vectorWidth = 8
    }},
    
    //========================================================================
    // Apple Metal Profiles
    //========================================================================
    {"metal-m1", {
      .name = "Apple M1 Ultra",
      .arch = "apple-m1",
      .triple = "arm64-apple-macos",
      .peakTFLOPS_FP16 = 21.0,
      .peakTFLOPS_FP32 = 21.0,
      .peakTFLOPS_INT8 = 42.0,
      .dramBandwidth_GBps = 800.0,
      .onChipBandwidth_GBps = 4000.0,
      .sharedMemorySize_KB = 32,
      .l2CacheSize_MB = 48,
      .maxWarps = 32,
      .maxRegisters = 32768,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 64,
      .preferredTileN = 64,
      .preferredTileK = 32,
      .supportsTensorCore = true,  // AMX
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = false,
      .vectorWidth = 4
    }},
    {"metal-m2", {
      .name = "Apple M2 Ultra",
      .arch = "apple-m2",
      .triple = "arm64-apple-macos",
      .peakTFLOPS_FP16 = 27.2,
      .peakTFLOPS_FP32 = 27.2,
      .peakTFLOPS_INT8 = 54.4,
      .dramBandwidth_GBps = 800.0,
      .onChipBandwidth_GBps = 4500.0,
      .sharedMemorySize_KB = 32,
      .l2CacheSize_MB = 48,
      .maxWarps = 32,
      .maxRegisters = 32768,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 64,
      .preferredTileN = 64,
      .preferredTileK = 32,
      .supportsTensorCore = true,  // AMX
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = false,
      .vectorWidth = 4
    }},
    {"metal-m3", {
      .name = "Apple M3 Max",
      .arch = "apple-m3",
      .triple = "arm64-apple-macos",
      .peakTFLOPS_FP16 = 14.2,
      .peakTFLOPS_FP32 = 14.2,
      .peakTFLOPS_INT8 = 28.4,
      .dramBandwidth_GBps = 400.0,
      .onChipBandwidth_GBps = 5000.0,
      .sharedMemorySize_KB = 32,
      .l2CacheSize_MB = 48,
      .maxWarps = 32,
      .maxRegisters = 32768,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 64,
      .preferredTileN = 64,
      .preferredTileK = 32,
      .supportsTensorCore = true,  // AMX
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = false,
      .vectorWidth = 4
    }},
    {"metal-m4", {
      .name = "Apple M4 Max",
      .arch = "apple-m4",
      .triple = "arm64-apple-macos",
      .peakTFLOPS_FP16 = 18.0,
      .peakTFLOPS_FP32 = 18.0,
      .peakTFLOPS_INT8 = 36.0,
      .dramBandwidth_GBps = 546.0,
      .onChipBandwidth_GBps = 5500.0,
      .sharedMemorySize_KB = 32,
      .l2CacheSize_MB = 48,
      .maxWarps = 32,
      .maxRegisters = 32768,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 64,
      .preferredTileN = 64,
      .preferredTileK = 32,
      .supportsTensorCore = true,  // AMX
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = false,
      .vectorWidth = 4
    }},
    {"metal-m5", {
      .name = "Apple M5 Ultra",
      .arch = "apple-m5",
      .triple = "arm64-apple-macos",
      .peakTFLOPS_FP16 = 36.0,
      .peakTFLOPS_FP32 = 36.0,
      .peakTFLOPS_INT8 = 72.0,
      .dramBandwidth_GBps = 1000.0,
      .onChipBandwidth_GBps = 6000.0,
      .sharedMemorySize_KB = 32,
      .l2CacheSize_MB = 64,
      .maxWarps = 32,
      .maxRegisters = 32768,
      .maxThreadsPerBlock = 1024,
      .preferredTileM = 64,
      .preferredTileN = 64,
      .preferredTileK = 32,
      .supportsTensorCore = true,  // AMX
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = false,
      .vectorWidth = 4
    }},

    //========================================================================
    // CPU Profiles
    //========================================================================
    {"cpu-avx512", {
      .name = "x86-64 AVX-512",
      .arch = "x86-64-v4",
      .triple = "x86_64-unknown-linux-gnu",
      .peakTFLOPS_FP16 = 2.0,  // With AMX
      .peakTFLOPS_FP32 = 4.0,
      .peakTFLOPS_INT8 = 8.0,
      .dramBandwidth_GBps = 200.0,
      .onChipBandwidth_GBps = 500.0,
      .sharedMemorySize_KB = 0,
      .l2CacheSize_MB = 2,
      .maxWarps = 1,
      .maxRegisters = 32,
      .maxThreadsPerBlock = 1,
      .preferredTileM = 32,
      .preferredTileN = 32,
      .preferredTileK = 32,
      .supportsTensorCore = true,  // AMX
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = false,
      .vectorWidth = 16
    }},
    
    {"cpu-neon", {
      .name = "ARM NEON",
      .arch = "armv8.2-a+fp16",
      .triple = "aarch64-unknown-linux-gnu",
      .peakTFLOPS_FP16 = 0.5,
      .peakTFLOPS_FP32 = 0.25,
      .peakTFLOPS_INT8 = 1.0,
      .dramBandwidth_GBps = 100.0,
      .onChipBandwidth_GBps = 200.0,
      .sharedMemorySize_KB = 0,
      .l2CacheSize_MB = 1,
      .maxWarps = 1,
      .maxRegisters = 32,
      .maxThreadsPerBlock = 1,
      .preferredTileM = 16,
      .preferredTileN = 16,
      .preferredTileK = 16,
      .supportsTensorCore = false,
      .supportsAsyncCopy = false,
      .supportsDynamicParallelism = false,
      .vectorWidth = 4
    }},
    
    //========================================================================
    // FPGA Profiles
    //========================================================================
    {"fpga-xilinx", {
      .name = "Xilinx Alveo U250",
      .arch = "xilinx-u250",
      .triple = "fpga",
      .peakTFLOPS_FP16 = 10.0,
      .peakTFLOPS_FP32 = 5.0,
      .peakTFLOPS_INT8 = 20.0,
      .dramBandwidth_GBps = 77.0,
      .onChipBandwidth_GBps = 1000.0,
      .sharedMemorySize_KB = 54000,  // BRAM
      .l2CacheSize_MB = 0,
      .maxWarps = 1,
      .maxRegisters = 0,
      .maxThreadsPerBlock = 1,
      .preferredTileM = 64,
      .preferredTileN = 64,
      .preferredTileK = 64,
      .supportsTensorCore = false,
      .supportsAsyncCopy = true,
      .supportsDynamicParallelism = false,
      .vectorWidth = 16
    }}
  };
}

//===----------------------------------------------------------------------===//
// Target Configuration Factory
//===----------------------------------------------------------------------===//

GPUTargetConfig GPUTargetConfig::forCUDA(int computeCapability) {
  GPUTargetConfig config;
  config.backend = GPUBackend::CUDA;
  config.arch = "sm_" + std::to_string(computeCapability);
  config.triple = "nvptx64-nvidia-cuda";
  
  auto profiles = getBackendProfiles();
  std::string key = "cuda-" + config.arch;
  if (profiles.count(key)) {
    auto& p = profiles[key];
    config.features = p.supportsTensorCore ? "+ptx80" : "";
  }
  
  return config;
}

GPUTargetConfig GPUTargetConfig::forROCm(llvm::StringRef gpuArch) {
  GPUTargetConfig config;
  config.backend = GPUBackend::ROCm;
  config.arch = gpuArch.str();
  config.triple = "amdgcn-amd-amdhsa";
  return config;
}

GPUTargetConfig GPUTargetConfig::forSPIRV() {
  GPUTargetConfig config;
  config.backend = GPUBackend::SPIRV;
  config.arch = "spirv64";
  config.triple = "spir64-unknown-unknown";
  return config;
}

GPUTargetConfig GPUTargetConfig::forMetal(llvm::StringRef arch) {
  GPUTargetConfig config;
  config.backend = GPUBackend::Metal;

  // Resolve the canonical profile key: "m3" → "metal-m3", etc.
  std::string key = "metal-" + arch.str();
  auto profiles = getBackendProfiles();
  if (profiles.count(key)) {
    auto &p = profiles[key];
    config.arch = p.arch;
    config.triple = p.triple;
  } else {
    // Fallback: treat arch as a raw GPU arch string
    config.arch = arch.str();
    config.triple = "air64-apple-macos";
  }
  return config;
}

GPUTargetConfig GPUTargetConfig::forTPU(llvm::StringRef tpuVersion) {
  GPUTargetConfig config;
  config.backend = GPUBackend::TPU;
  config.arch = "tpu-" + tpuVersion.str();
  config.triple = "tpu";
  return config;
}

GPUTargetConfig GPUTargetConfig::forAscend(llvm::StringRef chipType) {
  GPUTargetConfig config;
  config.backend = GPUBackend::Ascend;
  config.arch = "ascend-" + chipType.str();
  config.triple = "ascend";
  return config;
}

GPUTargetConfig GPUTargetConfig::forMACA(llvm::StringRef gpuArch) {
  GPUTargetConfig config;
  config.backend = GPUBackend::MACA;
  config.arch = gpuArch.str();
  config.triple = "maca";
  return config;
}

GPUTargetConfig GPUTargetConfig::forFPGA(llvm::StringRef vendor) {
  GPUTargetConfig config;
  config.backend = GPUBackend::FPGA_OpenCL;
  config.arch = vendor.str();
  config.triple = "fpga";
  return config;
}

GPUTargetConfig GPUTargetConfig::forCPU(llvm::StringRef cpuArch) {
  GPUTargetConfig config;
  config.backend = GPUBackend::CPU_LLVM;
  config.arch = cpuArch.str();
  
  // Detect host triple
#if defined(__x86_64__) || defined(_M_X64)
  config.triple = "x86_64-unknown-linux-gnu";
#elif defined(__aarch64__)
  config.triple = "aarch64-unknown-linux-gnu";
#else
  config.triple = "unknown-unknown-unknown";
#endif
  
  return config;
}

//===----------------------------------------------------------------------===//
// Factory Functions
//===----------------------------------------------------------------------===//

std::unique_ptr<GPUCodeGen> createCUDACodeGen(mlir::MLIRContext *context,
                                               int computeCapability) {
  return std::make_unique<GPUCodeGen>(context, 
                                       GPUTargetConfig::forCUDA(computeCapability));
}

std::unique_ptr<GPUCodeGen> createROCmCodeGen(mlir::MLIRContext *context,
                                               llvm::StringRef arch) {
  return std::make_unique<GPUCodeGen>(context, 
                                       GPUTargetConfig::forROCm(arch));
}

std::unique_ptr<GPUCodeGen> createSPIRVCodeGen(mlir::MLIRContext *context) {
  return std::make_unique<GPUCodeGen>(context, 
                                       GPUTargetConfig::forSPIRV());
}

std::unique_ptr<GPUCodeGen> createMetalCodeGen(mlir::MLIRContext *context,
                                               llvm::StringRef arch) {
  return std::make_unique<GPUCodeGen>(context,
                                       GPUTargetConfig::forMetal(arch));
}

std::unique_ptr<GPUCodeGen> createTPUCodeGen(mlir::MLIRContext *context,
                                              llvm::StringRef version) {
  return std::make_unique<GPUCodeGen>(context, 
                                       GPUTargetConfig::forTPU(version));
}

std::unique_ptr<GPUCodeGen> createAscendCodeGen(mlir::MLIRContext *context,
                                                 llvm::StringRef chip) {
  return std::make_unique<GPUCodeGen>(context, 
                                       GPUTargetConfig::forAscend(chip));
}

std::unique_ptr<GPUCodeGen> createMACACodeGen(mlir::MLIRContext *context,
                                               llvm::StringRef arch) {
  return std::make_unique<GPUCodeGen>(context, 
                                       GPUTargetConfig::forMACA(arch));
}

std::unique_ptr<GPUCodeGen> createFPGACodeGen(mlir::MLIRContext *context,
                                               llvm::StringRef vendor) {
  return std::make_unique<GPUCodeGen>(context, 
                                       GPUTargetConfig::forFPGA(vendor));
}

std::unique_ptr<GPUCodeGen> createCPUCodeGen(mlir::MLIRContext *context,
                                              llvm::StringRef arch) {
  return std::make_unique<GPUCodeGen>(context, 
                                       GPUTargetConfig::forCPU(arch));
}

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

std::vector<GPUBackend> getAvailableBackends() {
  std::vector<GPUBackend> available;
  
  // Check CUDA
  if (std::getenv("CUDA_HOME") || std::getenv("CUDA_PATH")) {
    available.push_back(GPUBackend::CUDA);
  }
  
  // Check ROCm
  if (std::getenv("ROCM_PATH") || std::getenv("HIP_PATH")) {
    available.push_back(GPUBackend::ROCm);
  }
  
  // Check Ascend
  if (std::getenv("ASCEND_HOME")) {
    available.push_back(GPUBackend::Ascend);
  }
  
  // Check MACA
  if (std::getenv("MACA_HOME")) {
    available.push_back(GPUBackend::MACA);
  }
  
  // CPU always available
  available.push_back(GPUBackend::CPU_LLVM);
  
  return available;
}

llvm::StringRef backendToString(GPUBackend backend) {
  switch (backend) {
  case GPUBackend::CUDA: return "cuda";
  case GPUBackend::ROCm: return "rocm";
  case GPUBackend::SPIRV: return "spirv";
  case GPUBackend::Metal: return "metal";
  case GPUBackend::MACA: return "maca";
  case GPUBackend::TPU: return "tpu";
  case GPUBackend::Ascend: return "ascend";
  case GPUBackend::FPGA_OpenCL: return "fpga-opencl";
  case GPUBackend::FPGA_HLS: return "fpga-hls";
  case GPUBackend::CPU_LLVM: return "cpu-llvm";
  case GPUBackend::CPU_OpenMP: return "cpu-openmp";
  }
  return "unknown";
}

GPUBackend stringToBackend(llvm::StringRef name) {
  if (name == "cuda") return GPUBackend::CUDA;
  if (name == "rocm") return GPUBackend::ROCm;
  if (name == "spirv") return GPUBackend::SPIRV;
  if (name == "metal") return GPUBackend::Metal;
  if (name == "maca") return GPUBackend::MACA;
  if (name == "tpu") return GPUBackend::TPU;
  if (name == "ascend") return GPUBackend::Ascend;
  if (name == "fpga-opencl") return GPUBackend::FPGA_OpenCL;
  if (name == "fpga-hls") return GPUBackend::FPGA_HLS;
  if (name == "cpu" || name == "cpu-llvm") return GPUBackend::CPU_LLVM;
  if (name == "cpu-openmp") return GPUBackend::CPU_OpenMP;
  return GPUBackend::CPU_LLVM;  // Default
}

bool GPUCodeGen::isBackendAvailable(GPUBackend backend) {
  auto available = getAvailableBackends();
  return std::find(available.begin(), available.end(), backend) != available.end();
}

} // namespace yirage
