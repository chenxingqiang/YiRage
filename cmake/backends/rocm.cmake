# =============================================================================
# Backend Configuration for AMD GPU (ROCm/HIP)
# =============================================================================
# Target: AMD Instinct MI100/MI200/MI300 series, Radeon RX 7000 series
# Requires: ROCm 5.0+, hipcc compiler
# =============================================================================

# AMD GPU - ENABLED
set(USE_ROCM ON)
set(USE_HIP ON)

# Other GPU Backends
set(USE_CUDA OFF)        # NVIDIA GPU
set(USE_CUDNN OFF)       # Requires CUDA
set(USE_CUSPARSELT OFF)  # Requires CUDA
set(USE_CUTLASS OFF)     # Requires CUDA
set(USE_MPS OFF)         # Apple Silicon
set(USE_XPU OFF)         # Intel GPU

# NPU/Accelerator Backends
set(USE_ASCEND OFF)      # Huawei NPU
set(USE_MACA OFF)        # MetaX GPU
set(USE_TPU OFF)         # Google TPU
set(USE_FPGA OFF)        # FPGA

# CPU Backends
set(USE_CPU ON)          # Fallback
set(USE_MKL OFF)         # Intel library (use rocBLAS instead)
set(USE_MKLDNN OFF)      # Intel library
set(USE_OPENMP ON)       # Parallel search
set(USE_XEON OFF)        # Intel-specific

# DSL Backends
set(USE_NKI OFF)         # AWS Neuron
set(USE_TRITON OFF)      # Use triton-rocm fork if needed

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
# ROCm-Specific Configuration
# =============================================================================
# AMD wavefront size: 64 threads (default), 32 for RDNA
# LDS (Shared memory): 64KB per workgroup
# Max threads per workgroup: 1024
#
# Environment:
#   export ROCM_PATH=/opt/rocm
#   export HIP_PLATFORM=amd
#
# Usage:
#   cp cmake/backends/rocm.cmake config.cmake
#   mkdir build && cd build
#   cmake .. -DCMAKE_CXX_COMPILER=hipcc
#   make -j
#
# Python:
#   import yirage as yr
#   graph.superoptimize(backend='rocm')
# =============================================================================
