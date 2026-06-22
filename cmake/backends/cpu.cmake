# =============================================================================
# Backend Configuration for CPU-Only Systems
# =============================================================================
# Target: Any x86_64 or ARM64 CPU
# Supports: Intel, AMD, Apple M-series, ARM Neoverse
# =============================================================================

# All GPU Backends - DISABLED
set(USE_CUDA OFF)
set(USE_CUDNN OFF)
set(USE_CUSPARSELT OFF)
set(USE_CUTLASS OFF)
set(USE_ROCM OFF)
set(USE_MPS OFF)
set(USE_XPU OFF)

# NPU/Accelerator Backends - DISABLED
set(USE_ASCEND OFF)
set(USE_MACA OFF)
set(USE_TPU OFF)
set(USE_FPGA OFF)

# CPU Backends - ENABLED
set(USE_CPU ON)
set(USE_OPENMP ON)       # Parallel execution

# Optional Intel optimizations (enable if on Intel platform)
set(USE_MKL OFF)         # Intel Math Kernel Library
set(USE_MKLDNN OFF)      # oneDNN (Intel Deep Neural Network Library)
set(USE_XEON OFF)        # Intel Xeon specific (AVX-512, AMX)

# DSL Backends
set(USE_NKI OFF)
set(USE_TRITON OFF)

# MLIR Ecosystem
set(USE_MLIR OFF)
set(USE_STABLEHLO OFF)
set(USE_TVM OFF)
set(USE_IREE OFF)

# Library Backends
set(USE_MHA OFF)
set(USE_NNPACK OFF)
set(USE_OPT_EINSUM OFF)

# Build Options
set(BUILD_CPP_EXAMPLES OFF)
set(USE_FORMAL_VERIFIER OFF)
set(YIRAGE_BUILD_UNIT_TEST OFF)

# =============================================================================
# CPU-Specific Configuration
# =============================================================================
# SIMD Support (auto-detected):
#   - x86_64: SSE4.2, AVX, AVX2, AVX-512, AMX
#   - ARM64: NEON, SVE, SVE2
#
# Thread configuration:
#   - Uses OpenMP for parallel search
#   - Default: number of physical cores
#
# Usage:
#   cp cmake/backends/cpu.cmake config.cmake
#   mkdir build && cd build
#   cmake ..
#   make -j$(nproc)
#
# Python:
#   import yirage as yr
#   graph.superoptimize(backend='cpu')
# =============================================================================
