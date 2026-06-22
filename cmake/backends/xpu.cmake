# =============================================================================
# Backend Configuration for Intel GPU (XPU/oneAPI)
# =============================================================================
# Target: Intel Arc, Data Center GPU Max Series (Ponte Vecchio)
# Requires: Intel oneAPI Base Toolkit 2023.0+, Level Zero
# =============================================================================

# Intel GPU - ENABLED
set(USE_XPU ON)
set(USE_SYCL ON)

# Other GPU Backends
set(USE_CUDA OFF)        # NVIDIA GPU
set(USE_CUDNN OFF)       # Requires CUDA
set(USE_CUSPARSELT OFF)  # Requires CUDA
set(USE_CUTLASS OFF)     # Requires CUDA
set(USE_ROCM OFF)        # AMD GPU
set(USE_MPS OFF)         # Apple Silicon

# NPU/Accelerator Backends
set(USE_ASCEND OFF)      # Huawei NPU
set(USE_MACA OFF)        # MetaX GPU
set(USE_TPU OFF)         # Google TPU
set(USE_FPGA OFF)        # FPGA (can enable for Intel FPGA)

# CPU Backends - Enable Intel optimizations
set(USE_CPU ON)          # Fallback
set(USE_MKL ON)          # Intel Math Kernel Library
set(USE_MKLDNN ON)       # oneDNN
set(USE_OPENMP ON)       # Parallel execution
set(USE_XEON ON)         # Intel Xeon optimizations

# DSL Backends
set(USE_NKI OFF)         # AWS Neuron
set(USE_TRITON OFF)      # Use intel-extension-for-pytorch

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
# XPU-Specific Configuration
# =============================================================================
# Intel GPU architecture:
#   - Xe-HPG (Arc): 256 execution units, 16GB GDDR6
#   - Xe-HPC (Max): 128 Xe cores, 128GB HBM2e
#
# SYCL configuration:
#   - Subgroup size: 16 or 32
#   - Work group size: up to 1024
#   - Shared local memory: 64KB
#
# Environment:
#   source /opt/intel/oneapi/setvars.sh
#   export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
#
# Usage:
#   cp cmake/backends/xpu.cmake config.cmake
#   mkdir build && cd build
#   cmake .. -DCMAKE_CXX_COMPILER=icpx
#   make -j
#
# Python:
#   import yirage as yr
#   import intel_extension_for_pytorch as ipex
#   graph.superoptimize(backend='xpu')
# =============================================================================
