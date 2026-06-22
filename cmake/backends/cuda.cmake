# =============================================================================
# Backend Configuration for NVIDIA GPU (CUDA)
# =============================================================================
# Target: NVIDIA RTX/Quadro/Tesla/A100/H100/B100 series
# Requires: CUDA Toolkit 11.4+, cuDNN (optional)
# =============================================================================

# NVIDIA GPU - ENABLED
set(USE_CUDA ON)
set(USE_CUDNN OFF)        # Enable if cuDNN installed
set(USE_CUSPARSELT OFF)   # Enable for sparse operations
set(USE_CUTLASS ON)       # CUTLASS templates

# Other GPU Backends
set(USE_ROCM OFF)         # AMD GPU
set(USE_MPS OFF)          # Apple Silicon
set(USE_XPU OFF)          # Intel GPU

# NPU/Accelerator Backends
set(USE_ASCEND OFF)       # Huawei NPU
set(USE_MACA OFF)         # MetaX GPU
set(USE_TPU OFF)          # Google TPU
set(USE_FPGA OFF)         # FPGA

# CPU Backends
set(USE_CPU ON)           # Fallback
set(USE_MKL OFF)          # Intel library
set(USE_MKLDNN OFF)       # Intel library
set(USE_OPENMP ON)        # Parallel search
set(USE_XEON OFF)         # Intel-specific

# DSL Backends
set(USE_NKI OFF)          # AWS Neuron
set(USE_TRITON ON)        # OpenAI Triton (CUDA)

# MLIR Ecosystem
set(USE_MLIR OFF)
set(USE_STABLEHLO OFF)
set(USE_TVM OFF)
set(USE_IREE OFF)

# Library Backends
set(USE_MHA ON)           # Multi-Head Attention
set(USE_NNPACK OFF)
set(USE_OPT_EINSUM OFF)

# Build Options
set(BUILD_CPP_EXAMPLES OFF)
set(USE_FORMAL_VERIFIER OFF)
set(YIRAGE_BUILD_UNIT_TEST OFF)

# =============================================================================
# CUDA-Specific Configuration
# =============================================================================
# NVIDIA GPU architecture:
#   - Ampere (A100): 312 TFLOPS FP16, 80GB HBM2e
#   - Hopper (H100): 989 TFLOPS FP16, 80GB HBM3
#   - Blackwell (B100): 1800 TFLOPS FP8
#
# Memory:
#   - Shared memory: 48-164 KB per SM
#   - L2 cache: 40-50 MB
#   - Warp size: 32 threads
#
# Environment:
#   export CUDA_HOME=/usr/local/cuda
#   export PATH=$CUDA_HOME/bin:$PATH
#
# Usage:
#   cp cmake/backends/cuda.cmake config.cmake
#   mkdir build && cd build
#   cmake ..
#   make -j
#
# Python:
#   import yirage as yr
#   graph.superoptimize(backend='cuda')
# =============================================================================
