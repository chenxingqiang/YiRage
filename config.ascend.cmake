# Backend Configuration for Huawei Ascend System
# This config is for systems with Ascend 910/910B/310P NPUs
# Requires: CANN toolkit, torch_npu, triton-ascend

# GPU Backends
set(USE_CUDA OFF)        # Not available on Ascend systems
set(USE_CUDNN OFF)       # Requires CUDA
set(USE_MPS OFF)         # Apple Silicon only
set(USE_CUSPARSELT OFF)  # Requires CUDA

# CPU Backends
set(USE_CPU ON)          # Always available as fallback
set(USE_MKL OFF)         # Intel library (optional)
set(USE_MKLDNN OFF)      # Intel library (optional)
set(USE_OPENMP ON)       # Enable for CPU parallelism
set(USE_XEON OFF)        # Intel-specific

# NPU Backends
set(USE_ASCEND ON)       # ★ Huawei Ascend NPU - ENABLED

# Specialized Backends  
set(USE_NKI OFF)         # AWS only
set(USE_TRITON OFF)      # Uses triton-ascend instead
set(USE_MHA OFF)
set(USE_NNPACK OFF)
set(USE_OPT_EINSUM OFF)

# Other Options
set(BUILD_CPP_EXAMPLES OFF)
set(USE_FORMAL_VERIFIER OFF)

# ============================================================
# Usage on Ascend System:
# ============================================================
#
# 1. Install CANN toolkit:
#    Download from https://www.hiascend.com/cann
#
# 2. Install Python dependencies:
#    pip install torch-npu
#    pip install triton-ascend
#
# 3. Build YiRage with Ascend support:
#    cp config.ascend.cmake config.cmake
#    mkdir build && cd build
#    cmake ..
#    make -j
#
# 4. Run tests:
#    python tests/ascend/test_triton_integration.py
#
# 5. Use in Python:
#    import yirage as yr
#    graph.superoptimize(backend='ascend')
# ============================================================

