/* Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file backend_test_common.h
 * @brief Common definitions for YiRage backend tests
 * 
 * Defines the 12 backends × 5 layers test matrix
 */

#ifndef YIRAGE_BACKEND_TEST_COMMON_H
#define YIRAGE_BACKEND_TEST_COMMON_H

#include <string>
#include <vector>
#include <functional>

namespace yirage {
namespace test {

// =============================================================================
// Backend Enumeration (12 Backends)
// =============================================================================

enum class Backend {
    CUDA,      // NVIDIA GPU
    ROCM,      // AMD GPU (HIP)
    CPU,       // x86/ARM
    MPS,       // Apple Silicon
    ASCEND,    // Huawei NPU
    MACA,      // MetaX GPU
    TPU,       // Google Cloud
    XPU,       // Intel GPU
    FPGA,      // Intel/Xilinx
    TRITON,    // OpenAI Triton
    NKI,       // AWS Neuron
    MLIR,      // Multi-target IR
    COUNT      // Total count = 12
};

inline const char* backend_name(Backend b) {
    switch (b) {
        case Backend::CUDA:   return "cuda";
        case Backend::ROCM:   return "rocm";
        case Backend::CPU:    return "cpu";
        case Backend::MPS:    return "mps";
        case Backend::ASCEND: return "ascend";
        case Backend::MACA:   return "maca";
        case Backend::TPU:    return "tpu";
        case Backend::XPU:    return "xpu";
        case Backend::FPGA:   return "fpga";
        case Backend::TRITON: return "triton";
        case Backend::NKI:    return "nki";
        case Backend::MLIR:   return "mlir";
        default:              return "unknown";
    }
}

inline const char* backend_display_name(Backend b) {
    switch (b) {
        case Backend::CUDA:   return "NVIDIA CUDA";
        case Backend::ROCM:   return "AMD ROCm/HIP";
        case Backend::CPU:    return "CPU (x86/ARM)";
        case Backend::MPS:    return "Apple Metal/MPS";
        case Backend::ASCEND: return "Huawei Ascend";
        case Backend::MACA:   return "MetaX MACA";
        case Backend::TPU:    return "Google TPU";
        case Backend::XPU:    return "Intel XPU";
        case Backend::FPGA:   return "FPGA (OpenCL)";
        case Backend::TRITON: return "OpenAI Triton";
        case Backend::NKI:    return "AWS Neuron NKI";
        case Backend::MLIR:   return "LLVM MLIR";
        default:              return "Unknown";
    }
}

inline std::vector<Backend> all_backends() {
    return {
        Backend::CUDA, Backend::ROCM, Backend::CPU, Backend::MPS,
        Backend::ASCEND, Backend::MACA, Backend::TPU, Backend::XPU,
        Backend::FPGA, Backend::TRITON, Backend::NKI, Backend::MLIR
    };
}

// =============================================================================
// Test Layer Enumeration (5 Layers)
// =============================================================================

enum class Layer {
    BACKEND_API,    // Layer 1: Backend interface and capabilities
    STRATEGY,       // Layer 2: Search strategy
    KERNEL,         // Layer 3: Kernel generation
    THREADBLOCK,    // Layer 4: Threadblock operations
    PK_RUNTIME,     // Layer 5: Persistent Kernel runtime
    COUNT           // Total count = 5
};

inline const char* layer_name(Layer l) {
    switch (l) {
        case Layer::BACKEND_API:  return "backend_api";
        case Layer::STRATEGY:     return "strategy";
        case Layer::KERNEL:       return "kernel";
        case Layer::THREADBLOCK:  return "threadblock";
        case Layer::PK_RUNTIME:   return "pk_runtime";
        default:                  return "unknown";
    }
}

inline const char* layer_display_name(Layer l) {
    switch (l) {
        case Layer::BACKEND_API:  return "Backend API";
        case Layer::STRATEGY:     return "Search Strategy";
        case Layer::KERNEL:       return "Kernel Generation";
        case Layer::THREADBLOCK:  return "Threadblock Ops";
        case Layer::PK_RUNTIME:   return "PK Runtime";
        default:                  return "Unknown";
    }
}

inline std::vector<Layer> all_layers() {
    return {
        Layer::BACKEND_API, Layer::STRATEGY, Layer::KERNEL,
        Layer::THREADBLOCK, Layer::PK_RUNTIME
    };
}

// =============================================================================
// Hardware Architecture Info
// =============================================================================

struct HardwareInfo {
    Backend backend;
    
    // Thread model
    int warp_size;           // Warp/Wavefront size
    std::string thread_model;
    
    // Matrix unit
    std::string matrix_unit;
    int matrix_size;         // e.g., 16 for Tensor Core, 128 for MXU
    
    // Memory hierarchy
    std::vector<std::string> memory_levels;
    
    // Native data types
    std::vector<std::string> native_dtypes;
    
    // Compilation
    std::string compiler;
    std::string target_ir;
};

inline HardwareInfo get_hardware_info(Backend b) {
    HardwareInfo info;
    info.backend = b;
    
    switch (b) {
        case Backend::CUDA:
            info.warp_size = 32;
            info.thread_model = "Warp (32 threads)";
            info.matrix_unit = "Tensor Core";
            info.matrix_size = 16;
            info.memory_levels = {"Registers", "Shared Memory", "L2 Cache", "HBM"};
            info.native_dtypes = {"FP32", "FP16", "TF32", "BF16", "INT8"};
            info.compiler = "NVCC";
            info.target_ir = "PTX";
            break;
            
        case Backend::ROCM:
            info.warp_size = 64;
            info.thread_model = "Wavefront (64 threads)";
            info.matrix_unit = "Matrix Core";
            info.matrix_size = 16;
            info.memory_levels = {"VGPR", "LDS", "L2 Cache", "HBM"};
            info.native_dtypes = {"FP32", "FP16", "BF16", "INT8"};
            info.compiler = "HIPCC";
            info.target_ir = "HSACO";
            break;
            
        case Backend::MPS:
            info.warp_size = 32;
            info.thread_model = "SIMD Group";
            info.matrix_unit = "Apple GPU";
            info.matrix_size = 8;
            info.memory_levels = {"Threadgroup Memory", "Device Memory", "Unified Memory"};
            info.native_dtypes = {"FP32", "FP16", "BF16"};
            info.compiler = "Metal";
            info.target_ir = "MetalIR";
            break;
            
        case Backend::ASCEND:
            info.warp_size = 0;  // AI Core based
            info.thread_model = "AI Core";
            info.matrix_unit = "Cube Unit";
            info.matrix_size = 16;
            info.memory_levels = {"L0 Buffer", "L1 Buffer", "L2 Cache", "HBM"};
            info.native_dtypes = {"FP32", "FP16", "INT8"};
            info.compiler = "BiSheng/Triton-Ascend";
            info.target_ir = "CANN IR";
            break;
            
        case Backend::MACA:
            info.warp_size = 64;  // MetaX uses 64-thread warps
            info.thread_model = "Warp (64 threads)";
            info.matrix_unit = "Tensor Core";
            info.matrix_size = 16;
            info.memory_levels = {"Registers", "Shared Memory", "L2 Cache", "HBM"};
            info.native_dtypes = {"FP32", "FP16", "BF16"};
            info.compiler = "MACA Compiler";
            info.target_ir = "MACA IR";
            break;
            
        case Backend::TPU:
            info.warp_size = 0;  // Systolic array
            info.thread_model = "Systolic Array";
            info.matrix_unit = "MXU";
            info.matrix_size = 128;
            info.memory_levels = {"VMEM", "HBM"};
            info.native_dtypes = {"BF16", "FP32", "INT8"};
            info.compiler = "XLA";
            info.target_ir = "HLO";
            break;
            
        case Backend::XPU:
            info.warp_size = 16;  // Subgroup
            info.thread_model = "Xe Subgroup";
            info.matrix_unit = "XMX";
            info.matrix_size = 8;
            info.memory_levels = {"SLM", "L3 Cache", "HBM"};
            info.native_dtypes = {"FP32", "FP16", "BF16", "INT8"};
            info.compiler = "DPC++/SYCL";
            info.target_ir = "SPIRV";
            break;
            
        case Backend::FPGA:
            info.warp_size = 0;  // Pipeline based
            info.thread_model = "Pipeline";
            info.matrix_unit = "DSP Block";
            info.matrix_size = 0;  // Configurable
            info.memory_levels = {"BRAM", "URAM", "DDR/HBM"};
            info.native_dtypes = {"FP32", "FP16", "INT8", "Custom"};
            info.compiler = "OpenCL/HLS";
            info.target_ir = "Bitstream";
            break;
            
        case Backend::TRITON:
            info.warp_size = 32;  // Underlying GPU
            info.thread_model = "Tile-based";
            info.matrix_unit = "Target GPU";
            info.matrix_size = 0;
            info.memory_levels = {"Shared", "Global"};
            info.native_dtypes = {"FP32", "FP16", "BF16"};
            info.compiler = "Triton JIT";
            info.target_ir = "PTX/HSACO";
            break;
            
        case Backend::NKI:
            info.warp_size = 0;  // Engine based
            info.thread_model = "Tensor/Vector Engine";
            info.matrix_unit = "Tensor Engine";
            info.matrix_size = 128;
            info.memory_levels = {"SBUF", "HBM"};
            info.native_dtypes = {"BF16", "FP32", "INT8"};
            info.compiler = "neuronx-cc";
            info.target_ir = "NEFF";
            break;
            
        case Backend::MLIR:
            info.warp_size = 0;  // Target dependent
            info.thread_model = "Target dependent";
            info.matrix_unit = "Target dependent";
            info.matrix_size = 0;
            info.memory_levels = {"Target dependent"};
            info.native_dtypes = {"FP32", "FP16", "BF16", "INT8"};
            info.compiler = "MLIR";
            info.target_ir = "LLVM/NVVM/SPIRV";
            break;
            
        default:
            info.warp_size = 1;
            info.thread_model = "Sequential";
            info.matrix_unit = "None";
            info.matrix_size = 1;
            info.memory_levels = {"Memory"};
            info.native_dtypes = {"FP32"};
            info.compiler = "Unknown";
            info.target_ir = "Unknown";
            break;
    }
    
    return info;
}

// =============================================================================
// Test Configuration
// =============================================================================

struct TestConfig {
    std::vector<Backend> enabled_backends;
    std::vector<Layer> enabled_layers;
    bool require_hardware;  // Skip if hardware not available
    bool verbose;
    int timeout_ms;
    
    TestConfig() 
        : enabled_backends(all_backends())
        , enabled_layers(all_layers())
        , require_hardware(false)
        , verbose(false)
        , timeout_ms(30000) {}
};

// =============================================================================
// Backend Availability Check
// =============================================================================

bool is_backend_available(Backend b);

// Forward declarations for backend checks
bool check_cuda_available();
bool check_rocm_available();
bool check_mps_available();
bool check_ascend_available();
bool check_maca_available();
bool check_tpu_available();
bool check_xpu_available();
bool check_fpga_available();
bool check_triton_available();
bool check_nki_available();
bool check_mlir_available();

}  // namespace test
}  // namespace yirage

#endif  // YIRAGE_BACKEND_TEST_COMMON_H
