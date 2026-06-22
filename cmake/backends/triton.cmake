# =============================================================================
# Backend Configuration for Triton DSL
# =============================================================================
# Target: NVIDIA GPU with Triton compiler
# Requires: CUDA 11.4+, triton-lang
# =============================================================================

# Triton - ENABLED
set(USE_TRITON ON)

# CUDA backend required for Triton
set(USE_CUDA ON)
set(USE_CUDNN OFF)       # Optional
set(USE_CUSPARSELT OFF)  # Optional
set(USE_CUTLASS OFF)     # Use Triton instead

# Other GPU Backends
set(USE_ROCM OFF)        # Use triton-rocm fork if needed
set(USE_MPS OFF)
set(USE_XPU OFF)

# NPU/Accelerator Backends
set(USE_ASCEND OFF)      # Use triton-ascend if needed
set(USE_MACA OFF)
set(USE_TPU OFF)
set(USE_FPGA OFF)

# AWS Neuron
set(USE_NKI OFF)

# CPU Backends
set(USE_CPU ON)          # Fallback
set(USE_MKL OFF)
set(USE_MKLDNN OFF)
set(USE_OPENMP ON)
set(USE_XEON OFF)

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
# Triton-Specific Configuration
# =============================================================================
# Triton features:
#   - JIT compilation to PTX/CUBIN
#   - Automatic optimization (tiling, vectorization)
#   - Flash Attention, GEMM kernels
#
# Supported GPU architectures:
#   - Ampere (SM80): A100, RTX 30xx
#   - Hopper (SM90): H100
#   - Ada (SM89): RTX 40xx
#
# Environment:
#   pip install triton
#   export TRITON_CACHE_DIR=~/.triton/cache
#
# Usage:
#   cp cmake/backends/triton.cmake config.cmake
#   mkdir build && cd build
#   cmake ..
#   make -j
#
# Python:
#   import yirage as yr
#   import triton
#   graph.superoptimize(backend='triton')
# =============================================================================
