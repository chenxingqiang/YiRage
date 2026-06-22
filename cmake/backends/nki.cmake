# =============================================================================
# Backend Configuration for AWS Neuron (NKI)
# =============================================================================
# Target: AWS Trainium, Inferentia2
# Requires: AWS Neuron SDK, neuronx-cc
# =============================================================================

# AWS Neuron - ENABLED
set(USE_NKI ON)

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
set(USE_FPGA OFF)

# CPU Backends
set(USE_CPU ON)          # Host CPU
set(USE_MKL OFF)
set(USE_MKLDNN OFF)
set(USE_OPENMP ON)
set(USE_XEON OFF)

# DSL Backends
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
# NKI-Specific Configuration
# =============================================================================
# AWS Neuron architecture:
#   - Trainium: 2 NeuronCores, 32GB HBM per chip
#   - Trainium2: 4 NeuronCores, 96GB HBM per chip
#   - Inferentia2: 2 NeuronCores, 32GB HBM per chip
#
# NeuronCore resources:
#   - Tensor Engine: 128x128 systolic array
#   - Vector Engine: 128 parallel lanes
#   - SBUF (scratchpad): 24MB per core
#
# Environment (EC2 trn1/inf2):
#   source /opt/aws_neuron_venv/bin/activate
#   pip install neuronx-cc torch-neuronx
#
# Usage:
#   cp cmake/backends/nki.cmake config.cmake
#   mkdir build && cd build
#   cmake ..
#   make -j
#
# Python:
#   import yirage as yr
#   import torch_neuronx
#   graph.superoptimize(backend='nki')
# =============================================================================
