# =============================================================================
# Backend Configuration for Apple Silicon (MPS)
# =============================================================================
# Target: Apple M1/M2/M3/M4 series chips with Metal GPU
# Requires: macOS 12.0+, Xcode Command Line Tools
# =============================================================================

# GPU Backends
set(USE_CUDA OFF)        # Not available on Mac
set(USE_CUDNN OFF)       # Requires CUDA
set(USE_CUSPARSELT OFF)  # Requires CUDA
set(USE_CUTLASS OFF)     # Requires CUDA
set(USE_ROCM OFF)        # Requires AMD GPU
set(USE_XPU OFF)         # Requires Intel GPU

# Apple Silicon GPU - ENABLED
set(USE_MPS ON)

# NPU/Accelerator Backends
set(USE_ASCEND OFF)      # Huawei NPU
set(USE_MACA OFF)        # MetaX GPU
set(USE_TPU OFF)         # Google TPU
set(USE_FPGA OFF)        # FPGA

# CPU Backends
set(USE_CPU ON)          # Fallback
set(USE_MKL OFF)         # Intel library
set(USE_MKLDNN OFF)      # Intel library
set(USE_OPENMP ON)       # brew install libomp
set(USE_XEON OFF)        # Intel-specific

# DSL Backends
set(USE_NKI OFF)         # AWS Neuron
set(USE_TRITON OFF)      # Requires CUDA

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
# MPS-Specific Configuration
# =============================================================================
# Apple Metal SIMD width: 32 threads
# Threadgroup memory: 32KB
# Max threads per threadgroup: 1024
#
# Usage:
#   cp cmake/backends/mps.cmake config.cmake
#   mkdir build && cd build
#   cmake ..
#   make -j
#
# Python:
#   import yirage as yr
#   graph.superoptimize(backend='mps')
# =============================================================================
