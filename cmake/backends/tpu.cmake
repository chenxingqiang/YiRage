# =============================================================================
# Backend Configuration for Google TPU
# =============================================================================
# Target: TPU v2, v3, v4, v5e (Cloud TPU or TPU VM)
# Requires: JAX, libtpu, Google Cloud access
# =============================================================================

# Google TPU - ENABLED
set(USE_TPU ON)

# Other GPU Backends
set(USE_CUDA OFF)
set(USE_CUDNN OFF)
set(USE_CUSPARSELT OFF)
set(USE_CUTLASS OFF)
set(USE_ROCM OFF)
set(USE_MPS OFF)
set(USE_XPU OFF)

# NPU/Accelerator Backends
set(USE_ASCEND OFF)
set(USE_MACA OFF)
set(USE_FPGA OFF)

# CPU Backends
set(USE_CPU ON)          # Fallback for host operations
set(USE_MKL OFF)
set(USE_MKLDNN OFF)
set(USE_OPENMP ON)
set(USE_XEON OFF)

# DSL Backends
set(USE_NKI OFF)
set(USE_TRITON OFF)

# MLIR Ecosystem - TPU uses XLA/StableHLO
set(USE_MLIR ON)
set(USE_STABLEHLO ON)    # XLA compatibility
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
# TPU-Specific Configuration
# =============================================================================
# TPU architecture:
#   - TPU v4: 275 TFLOPS BF16, 32GB HBM
#   - TPU v5e: 197 TFLOPS BF16, 16GB HBM (cost-optimized)
#   - MXU: 128x128 systolic array
#
# Memory:
#   - VMEM (vector memory): 16MB per core
#   - CMEM (scalar memory): 2MB per core
#   - HBM: shared across cores
#
# Environment (TPU VM):
#   export TPU_NAME=your-tpu-name
#   export JAX_PLATFORMS=tpu
#
# Usage:
#   cp cmake/backends/tpu.cmake config.cmake
#   mkdir build && cd build
#   cmake ..
#   make -j
#
# Python:
#   import yirage as yr
#   import jax
#   jax.devices('tpu')  # Verify TPU access
#   graph.superoptimize(backend='tpu')
# =============================================================================
