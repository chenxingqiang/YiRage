# Backend Configuration for M3 Mac
# GPU Backends
set(USE_CUDA OFF)        # Mac has no NVIDIA GPU
set(USE_CUDNN OFF)       # Requires CUDA
set(USE_MPS ON)          # Apple Silicon GPU - M3
set(USE_CUSPARSELT OFF)  # Requires CUDA

# CPU Backends
set(USE_CPU ON)          # Always available as fallback
set(USE_MKL OFF)         # Intel library (optional on Mac)
set(USE_MKLDNN OFF)      # Intel library (optional on Mac)
set(USE_OPENMP OFF)      # macOS clang doesn't support -fopenmp
set(USE_XEON OFF)        # Intel-specific

# NPU Backends
set(USE_ASCEND OFF)      # Huawei Ascend NPU (requires CANN + torch_npu)

# Specialized Backends
set(USE_NKI OFF)         # AWS only
set(USE_TRITON OFF)      # Requires CUDA
set(USE_MHA OFF)
set(USE_NNPACK OFF)
set(USE_OPT_EINSUM OFF)

# Other Options
set(BUILD_CPP_EXAMPLES OFF)
set(USE_FORMAL_VERIFIER OFF)
