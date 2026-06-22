# =============================================================================
# Backend Configuration for FPGA Acceleration
# =============================================================================
# Target: Xilinx Alveo, Intel Agilex/Stratix, AWS F1
# Requires: Vitis HLS (Xilinx) or Intel HLS Compiler
# =============================================================================

# FPGA - ENABLED
set(USE_FPGA ON)

# GPU Backends
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
set(USE_TPU OFF)

# CPU Backends
set(USE_CPU ON)          # Host CPU for control
set(USE_MKL OFF)
set(USE_MKLDNN OFF)
set(USE_OPENMP ON)
set(USE_XEON OFF)

# DSL Backends
set(USE_NKI OFF)
set(USE_TRITON OFF)

# MLIR Ecosystem - FPGA may use CIRCT
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
# FPGA-Specific Configuration
# =============================================================================
# Supported FPGA platforms:
#   - Xilinx Alveo U50/U200/U250/U280
#   - Intel Agilex/Stratix 10
#   - AWS F1 (Xilinx VU9P)
#
# Resources:
#   - DSP blocks: 1000-10000+
#   - Block RAM: 10-100 MB
#   - HBM (Alveo U280): 8GB
#
# Toolchain:
#   Xilinx: Vitis 2022.2+
#   Intel: Intel FPGA SDK for OpenCL / oneAPI
#
# Environment (Xilinx):
#   source /opt/xilinx/Vitis/2022.2/settings64.sh
#   source /opt/xilinx/xrt/setup.sh
#
# Usage:
#   cp cmake/backends/fpga.cmake config.cmake
#   mkdir build && cd build
#   cmake .. -DFPGA_PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1
#   make -j
#
# Python:
#   import yirage as yr
#   graph.superoptimize(backend='fpga')
# =============================================================================
