# =============================================================================
# YiRage MLIR Backend Configuration
# =============================================================================
# This configuration enables MLIR-based compilation for all backends.
# 
# Features:
#   - Unified compilation pipeline via MLIR
#   - CPU codegen via LLVM
#   - GPU codegen via NVVM/ROCDL/SPIRV
#   - End-to-end optimization passes
#
# Requirements:
#   - LLVM/MLIR 17+ (built from submodule or system-installed)
#
# Usage:
#   cp cmake/backends/mlir.cmake config.cmake
#   mkdir build && cd build
#   cmake ..
#   make -j$(nproc)
#
# =============================================================================

# =============================================================================
# MLIR Ecosystem (Primary)
# =============================================================================
set(USE_MLIR ON)         # Enable MLIR integration
set(USE_STABLEHLO OFF)   # StableHLO (optional, for XLA compatibility)
set(USE_TVM OFF)         # Apache TVM (optional)
set(USE_IREE OFF)        # IREE runtime (optional)

# LLVM build configuration
set(YIRAGE_LLVM_SOURCE "submodule" CACHE STRING "LLVM source: submodule, fetch, prebuilt, system")
set(YIRAGE_LLVM_VERSION "17" CACHE STRING "LLVM version")
set(YIRAGE_LLVM_BUILD_TYPE "Release" CACHE STRING "LLVM build type")

# MLIR target backends
set(YIRAGE_MLIR_ENABLE_LLVM ON)    # CPU via LLVM
set(YIRAGE_MLIR_ENABLE_NVVM ON)    # NVIDIA via NVVM
set(YIRAGE_MLIR_ENABLE_ROCDL ON)   # AMD via ROCDL
set(YIRAGE_MLIR_ENABLE_SPIRV ON)   # Intel/Vulkan via SPIR-V

# =============================================================================
# Hardware Backends (via MLIR lowering)
# =============================================================================

# NVIDIA GPU - compiled via MLIR → NVVM → PTX
set(USE_CUDA ON)         # Enable CUDA runtime
set(USE_CUDNN OFF)       # Disable cuDNN (using MLIR codegen instead)
set(USE_CUSPARSELT OFF)  # Disable cuSPARSELt
set(USE_CUTLASS OFF)     # Disable CUTLASS (using MLIR codegen instead)

# AMD GPU - compiled via MLIR → ROCDL → GCN
set(USE_ROCM OFF)        # Enable for ROCm target (set ON when building for AMD)

# Apple Silicon - compiled via MLIR → Metal/MPS
set(USE_MPS OFF)         # Enable for Apple Silicon (set ON on macOS)

# Intel GPU - compiled via MLIR → SPIR-V → Level Zero
set(USE_XPU OFF)         # Enable for Intel XPU (set ON when building for Intel)

# Huawei NPU - uses CANN, not MLIR (yet)
set(USE_ASCEND OFF)      # Huawei Ascend NPU

# MetaX GPU - uses MACA SDK
set(USE_MACA OFF)        # MetaX GPU

# Google TPU - compiled via MLIR → StableHLO → XLA
set(USE_TPU OFF)         # Google TPU (requires StableHLO)

# FPGA - compiled via MLIR → HLS
set(USE_FPGA OFF)        # FPGA acceleration

# =============================================================================
# CPU Backends
# =============================================================================
set(USE_CPU ON)          # Always available as fallback
set(USE_MKL OFF)         # Intel MKL (optional acceleration)
set(USE_MKLDNN OFF)      # oneDNN (optional)
set(USE_OPENMP ON)       # OpenMP for parallel execution
set(USE_XEON OFF)        # Intel Xeon specific (AVX-512)

# =============================================================================
# DSL/Compiler Backends (Alternative to MLIR)
# =============================================================================
set(USE_NKI OFF)         # AWS Neuron Kernel Interface
set(USE_TRITON OFF)      # OpenAI Triton

# =============================================================================
# Specialized Backends
# =============================================================================
set(USE_MHA OFF)         # Multi-Head Attention library
set(USE_NNPACK OFF)      # Facebook NNPACK
set(USE_OPT_EINSUM OFF)  # Einsum optimization

# =============================================================================
# Build Options
# =============================================================================
set(BUILD_CPP_EXAMPLES ON)        # Build C++ examples
set(USE_FORMAL_VERIFIER OFF)      # Formal verification
set(YIRAGE_BUILD_UNIT_TEST ON)    # Build unit tests
set(YIRAGE_BUILD_MLIR_TOOLS ON)   # Build yirage-opt tool

# =============================================================================
# MLIR-specific Build Options
# =============================================================================

# Pass pipeline defaults
set(YIRAGE_MLIR_DEFAULT_OPT_LEVEL 3)
set(YIRAGE_MLIR_ENABLE_VECTORIZATION ON)
set(YIRAGE_MLIR_ENABLE_LOOP_TILING ON)
set(YIRAGE_MLIR_ENABLE_FUSION ON)

# GPU-specific options
set(YIRAGE_MLIR_CUDA_COMPUTE_CAPABILITY "80" CACHE STRING "CUDA compute capability (e.g., 70, 80, 90)")
set(YIRAGE_MLIR_ROCM_GPU_TARGET "gfx908" CACHE STRING "ROCm GPU target (e.g., gfx908, gfx90a)")

# =============================================================================
# Documentation
# =============================================================================
#
# MLIR Compilation Pipeline:
# 
#   PyTorch/JAX Model
#         ↓
#   YiRage Dialect (yirage.matmul, yirage.attention, etc.)
#         ↓
#   Linalg + Tensor + SCF (tiling, fusion, vectorization)
#         ↓
#   ┌─────────────────────────────────────────────────────┐
#   │  CPU          │  CUDA       │  ROCm      │  Metal   │
#   │  LLVM IR      │  NVVM/PTX   │  ROCDL     │  SPIR-V  │
#   │     ↓         │      ↓      │     ↓      │     ↓    │
#   │  Native .so   │  .cubin     │  .hsaco    │  .msl    │
#   └─────────────────────────────────────────────────────┘
#
# To customize the pipeline, modify:
#   - mlir/lib/Dialect/Yirage/Transforms/YirageToLinalg.cpp
#   - mlir/lib/Execution/GPUCodeGen.cpp
#   - mlir/lib/Execution/JITRunner.cpp
#
# =============================================================================
