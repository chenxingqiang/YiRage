# MetaX MACA Backend Quick Start Guide

## Overview

YiRage supports **MetaX MACA** (MetaX Architecture for Compute Acceleration) GPU backend. MACA provides a CUDA-compatible programming model, allowing users to leverage MetaX GPU hardware with familiar CUDA APIs and code.

**MetaX Developer Community**: https://developer.metax-tech.com/

## Hardware Support

MetaX MACA backend supports:
- MetaX C500 series GPUs
- MetaX C500 Pro series GPUs
- Future MetaX GPU products

### Hardware Specifications (MetaX C500)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **warpSize** | **64** | ⚠️ Different from NVIDIA's 32! |
| Compute Capability | 10.0 | MetaX-specific |
| Streaming Multiprocessors | 104 | |
| Max Threads/Block | 1024 | |
| Max Threads/SM | 2048 | |
| Registers/Block | 131072 | 2x more than NVIDIA |
| Shared Memory/Block | 64 KB | |
| L2 Cache | 8 MB | |
| Memory Bus Width | 4096 bits | Wide memory bus |
| HBM Memory | 64 GB | |

### Key Differences from NVIDIA GPUs

> **⚠️ Important**: MetaX GPUs use **64-thread warps** instead of NVIDIA's 32-thread warps!

| Feature | MetaX MACA | NVIDIA CUDA |
|---------|------------|-------------|
| **warpSize** | **64** | 32 |
| Warp shuffle iterations | 6 (log₂64) | 5 (log₂32) |
| Warp mask type | `uint64_t` | `uint32_t` |
| Full warp mask | `0xFFFFFFFFFFFFFFFF` | `0xFFFFFFFF` |

This affects:
- All `__shfl_sync` operations
- Warp reduction algorithms
- Ballot operations
- Thread synchronization within warps

## Prerequisites

### 1. MACA SDK Installation

Download and install the MACA SDK from the MetaX Developer Community:

```bash
# Set environment variables
export MACA_HOME=/opt/maca
export PATH=$MACA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$MACA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2. mcPytorch (Optional)

For PyTorch integration, install mcPytorch:

```bash
# mcPytorch provides PyTorch with MACA backend support
pip install torch-maca  # or build from source
```

## Building YiRage with MACA Support

### CMake Configuration

```bash
mkdir build && cd build
cmake .. -DUSE_MACA=ON -DUSE_CUDA=OFF
make -j$(nproc)
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MACA_HOME` | Primary MACA SDK path | `/opt/maca` |
| `MACA_PATH` | Alternative SDK path | - |

## Usage

### Python API

```python
import yirage as yr

# Check MACA availability
if yr.is_backend_available('maca'):
    print("MetaX MACA GPU available!")

# Create kernel graph
graph = yr.new_kernel_graph()
X = graph.new_input(dims=(8, 4096), dtype=yr.float16)
W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
O = graph.matmul(X, W)
graph.mark_output(O)

# Optimize for MACA backend
optimized = graph.superoptimize(backend='maca')
```

### C++ API

```cpp
#include "yirage/backend/maca_backend.h"

// Check availability
auto* backend = yirage::backend::BackendRegistry::get_instance()
    .get_backend(yirage::type::BT_MACA);

if (backend && backend->is_available()) {
    std::cout << "MACA backend ready!" << std::endl;
    std::cout << "Device: " << backend->get_display_name() << std::endl;
    std::cout << "Memory: " << backend->get_max_memory() / (1024*1024*1024) 
              << " GB" << std::endl;
}
```

### Native MACA Kernel Example

```cpp
#include <mc_runtime.h>
#include <mc_common.h>

// MACA kernel using __global__ (same as CUDA)
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int n = 50000;
    size_t size = n * sizeof(float);
    
    // Allocate device memory using MACA API (mc* prefix)
    float *d_A, *d_B, *d_C;
    mcMalloc(&d_A, size);
    mcMalloc(&d_B, size);
    mcMalloc(&d_C, size);
    
    // Copy data to device
    mcMemcpy(d_A, h_A, size, mcMemcpyHostToDevice);
    mcMemcpy(d_B, h_B, size, mcMemcpyHostToDevice);
    
    // Launch kernel (same syntax as CUDA)
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Copy result back
    mcMemcpy(h_C, d_C, size, mcMemcpyDeviceToHost);
    
    // Free device memory
    mcFree(d_A);
    mcFree(d_B);
    mcFree(d_C);
    
    return 0;
}
```

### Kernel Configuration

MACA uses CUDA-compatible kernel configurations with MACA-specific optimizations:

```cpp
#include "yirage/kernel/maca/maca_kernel_config.h"

using namespace yirage::kernel::maca;

// Get architecture configuration for MetaX C500
auto arch = get_maca_arch_config(100);  // Compute Cap 10.0

// Note: warpSize is 64 on MetaX!
std::cout << "Warp size: " << arch.warp_size << std::endl;  // 64

// Configure matmul kernel
auto matmul_config = get_optimal_matmul_config(1024, 1024, 1024, arch);

std::cout << "Tile M: " << matmul_config.tile_m << std::endl;
std::cout << "Tile N: " << matmul_config.tile_n << std::endl;
std::cout << "Tile K: " << matmul_config.tile_k << std::endl;
std::cout << "Stages: " << matmul_config.num_stages << std::endl;
```

### MACA Warp Utilities

For kernels using warp-level primitives, use the MACA warp utilities:

```cpp
#include "yirage/kernel/maca/maca_warp_utils.h"

using namespace yirage::kernel::maca;

// MACA warp constants
constexpr int WARP_SIZE = MACA_WARP_SIZE;  // 64
constexpr auto FULL_MASK = MACA_FULL_WARP_MASK;  // 64-bit mask

__global__ void reduction_kernel(float* data, int n) {
    float val = data[threadIdx.x];
    
    // Use MACA warp reduce (6 iterations for 64 threads)
    float sum = maca_warp_reduce_sum(val);
    
    // Lane/warp ID for 64-thread warps
    int lane = maca_lane_id();  // 0-63
    int warp = maca_warp_id();
    
    if (lane == 0) {
        // First thread of each warp writes result
        output[warp] = sum;
    }
}
```

## Key Features

### MACA Native API

MACA provides its own native runtime API with `mc*` prefix:

| CUDA API | MACA Native API |
|----------|-----------------|
| `cudaMalloc` | `mcMalloc` |
| `cudaMemcpy` | `mcMemcpy` |
| `cudaFree` | `mcFree` |
| `cudaGetDeviceCount` | `mcGetDeviceCount` |
| `cudaGetDeviceProperties` | `mcGetDeviceProperties` |
| `cudaDeviceSynchronize` | `mcDeviceSynchronize` |
| `cudaError_t` | `mcError_t` |
| `cudaSuccess` | `mcSuccess` |
| `__global__` kernels | ✅ Supported |
| Tensor Cores | ✅ Supported (hardware-dependent) |

### Library Mapping

| NVIDIA | MetaX MACA | Description |
|--------|------------|-------------|
| cudart | mc_runtime | Runtime API |
| cublas | mcblas, mcblasLt | BLAS operations |
| CUTLASS | mctlass | Tile-based GEMM |
| nccl | mccl | Collective comm |
| curand | mcrand | Random numbers |
| cusolver | mcsolver | Linear algebra |
| cudnn | mcdnn | Deep learning primitives |
| Flash Attention | mcflashattn | Flash Attention 2.0 |

### MACA High-Performance Libraries

MetaX provides optimized libraries:

**mctlass** (CUTLASS equivalent):
- Tile-based GEMM with 64-thread warp support
- Supports SM75/SM80-like configurations
- Software pipelining and swizzling

**mcflashattn** (Flash Attention):
```cpp
#include <flash_attn/flash_attn.h>

// Create attention tensors
Tensor_t Q = make_contiguous_tensor4d(q_ptr, MCFLASHATTN_DATATYPE_FP16, 
                                       batch, seqlen, heads, head_dim);
Tensor_t K = make_contiguous_tensor4d(k_ptr, MCFLASHATTN_DATATYPE_FP16,
                                       batch, seqlen, heads, head_dim);
Tensor_t V = make_contiguous_tensor4d(v_ptr, MCFLASHATTN_DATATYPE_FP16,
                                       batch, seqlen, heads, head_dim);

// Run flash attention
mcflashattn_forward(Q, K, V, output, softmax_scale, is_causal);
```

**mcblas / mcblasLt**:
- High-performance matrix operations
- Tensor core support
- Auto-tuning for optimal performance

### Compilation

Use `mxcc` compiler with `-x maca` flag:

```bash
mxcc -x maca your_kernel.cpp -o output --maca-path=/opt/maca
```

## Performance Optimization

### Recommended Practices

1. **Use FP16/BF16**: MACA GPUs have excellent half-precision performance
2. **Align dimensions**: Use multiples of 16 for tensor dimensions
3. **Leverage MCCL**: For multi-GPU workloads, use MCCL collective operations
4. **Profile with MACA tools**: Use MetaX profiling tools for optimization

### Example: Optimized Attention

```python
import yirage as yr

# Configure for MACA attention
graph = yr.new_kernel_graph()

# Use FP16 for better performance
Q = graph.new_input(dims=(batch, heads, seq, head_dim), dtype=yr.float16)
K = graph.new_input(dims=(batch, heads, seq, head_dim), dtype=yr.float16)
V = graph.new_input(dims=(batch, heads, seq, head_dim), dtype=yr.float16)

# Flash attention optimized for MACA
attention = graph.flash_attention(Q, K, V, causal=True)
graph.mark_output(attention)

# Superoptimize for MACA
optimized = graph.superoptimize(
    backend='maca',
    options={
        'use_tensor_cores': True,
        'tile_size': 128,
    }
)
```

## Troubleshooting

### Common Issues

**Issue**: MACA SDK not found
```
Solution: Set MACA_HOME or MACA_PATH environment variable
export MACA_HOME=/path/to/maca/sdk
```

**Issue**: Library not found
```
Solution: Add MACA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$MACA_HOME/lib64:$LD_LIBRARY_PATH
```

**Issue**: Compiler not found
```
Solution: Ensure mxcc is in PATH
export PATH=$MACA_HOME/bin:$PATH
```

## Resources

- **MetaX Developer Community**: https://developer.metax-tech.com/
- **vLLM-metax**: https://github.com/MetaX-MACA/vLLM-metax
- **mcPytorch**: https://github.com/MetaX-MACA/mcPytorch
- **mcTVM**: https://github.com/MetaX-MACA/mcTVM

## Related Projects

- [vLLM-metax](https://github.com/MetaX-MACA/vLLM-metax) - High-performance LLM inference on MetaX GPUs
- [FlashMLA](https://github.com/MetaX-MACA/FlashMLA) - Flash attention for MACA
- [mcPytorch](https://github.com/MetaX-MACA/mcPytorch) - PyTorch with MACA support

