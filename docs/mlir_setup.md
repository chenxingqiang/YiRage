# YiRage MLIR Setup Guide

This guide explains how to set up and use MLIR with YiRage for end-to-end compiler-based optimization.

## Overview

YiRage uses MLIR (Multi-Level Intermediate Representation) as its core compiler infrastructure to:

- **Unified IR**: Represent AI/LLM operations in a high-level dialect (`yirage.*`)
- **Optimization**: Apply cross-operator fusion, tiling, and vectorization
- **Codegen**: Lower to multiple targets (CPU/LLVM, CUDA/NVVM, ROCm/ROCDL, SPIR-V)

## Quick Start

### Option 1: Prebuilt Binaries (Recommended)

Download prebuilt LLVM/MLIR from GitHub Releases:

```bash
# Linux x86_64
wget https://github.com/chenxingqiang/YiRage/releases/download/llvm-prebuilt-v17/llvm-17-linux-x86_64.tar.gz
sudo mkdir -p /opt/llvm
sudo tar -xzf llvm-17-linux-x86_64.tar.gz -C /opt/llvm
export MLIR_DIR=/opt/llvm/lib/cmake/mlir

# Build YiRage
cp cmake/backends/mlir.cmake config.cmake
mkdir build && cd build
cmake .. -DMLIR_DIR=$MLIR_DIR
make -j$(nproc)
```

### Option 2: Build from Submodule

```bash
# Clone with submodules
git clone --recursive https://github.com/chenxingqiang/YiRage.git
cd YiRage

# Initialize LLVM submodule (shallow clone)
git submodule update --init --depth 1 deps/llvm-project

# Configure and build (LLVM will be built automatically)
cp cmake/backends/mlir.cmake config.cmake
mkdir build && cd build
cmake .. -DYIRAGE_LLVM_SOURCE=submodule
make -j$(nproc)
```

**Note**: Building LLVM from source takes 30-60 minutes and requires ~20GB disk space.

### Option 3: System LLVM

If you have LLVM/MLIR installed system-wide:

```bash
# Ubuntu/Debian
sudo apt install llvm-17-dev libmlir-17-dev

# macOS
brew install llvm@17

# Configure YiRage
export MLIR_DIR=/usr/lib/llvm-17/lib/cmake/mlir  # Ubuntu
# or
export MLIR_DIR=$(brew --prefix llvm@17)/lib/cmake/mlir  # macOS

cp cmake/backends/mlir.cmake config.cmake
mkdir build && cd build
cmake .. -DYIRAGE_LLVM_SOURCE=system
make -j$(nproc)
```

## Configuration Options

Edit `config.cmake` to customize MLIR settings:

```cmake
# MLIR source options
set(YIRAGE_LLVM_SOURCE "submodule")  # submodule, prebuilt, system
set(YIRAGE_LLVM_VERSION "17")

# Enable MLIR targets
set(YIRAGE_MLIR_ENABLE_LLVM ON)    # CPU codegen
set(YIRAGE_MLIR_ENABLE_NVVM ON)    # NVIDIA GPU
set(YIRAGE_MLIR_ENABLE_ROCDL ON)   # AMD GPU
set(YIRAGE_MLIR_ENABLE_SPIRV ON)   # Intel/Vulkan

# GPU target architecture
set(YIRAGE_MLIR_CUDA_COMPUTE_CAPABILITY "80")  # Ampere
set(YIRAGE_MLIR_ROCM_GPU_TARGET "gfx90a")      # MI200
```

## YiRage MLIR Dialect

The YiRage MLIR dialect provides high-level operations for LLM inference:

### Operations

```mlir
// Matrix operations
yirage.matmul %A, %B : tensor<M×K×f16>, tensor<K×N×f16> -> tensor<M×N×f16>
yirage.batch_matmul %A, %B : tensor<B×M×K×f16>, tensor<B×K×N×f16> -> tensor<B×M×N×f16>

// Attention
yirage.attention %Q, %K, %V {causal=true} : ... -> tensor<B×H×S×D×f16>
yirage.paged_attention %Q, %K_cache, %V_cache {block_size=16} : ...

// Normalization
yirage.rms_norm %X, %gamma {eps=1e-6} : tensor<B×S×D×f16> -> tensor<B×S×D×f16>
yirage.layer_norm %X, %gamma, %beta : ...

// Activations
yirage.silu %X : tensor<...×f16> -> tensor<...×f16>
yirage.gelu %X {approximate=true} : ...

// MLP
yirage.gated_mlp %X, %W_gate, %W_up, %W_down : ...
```

### Example

```mlir
func.func @llama_attention(
    %Q: tensor<1x32x2048x128xf16>,
    %K: tensor<1x8x2048x128xf16>,
    %V: tensor<1x8x2048x128xf16>
) -> tensor<1x32x2048x128xf16> {
  %out = yirage.attention %Q, %K, %V {
    causal = true,
    num_kv_heads = 8,
    flash = true
  } : ... -> tensor<1x32x2048x128xf16>
  return %out : tensor<1x32x2048x128xf16>
}
```

## Compilation Pipeline

```
PyTorch/JAX Model
       ↓
   torch.export / jax.jit
       ↓
┌──────────────────────────────────────┐
│  YiRage Dialect                      │
│  (yirage.attention, yirage.matmul)   │
└──────────────────────────────────────┘
       ↓  yirage-to-linalg
┌──────────────────────────────────────┐
│  Linalg + Tensor + SCF               │
│  (tiling, fusion, vectorization)     │
└──────────────────────────────────────┘
       ↓  target lowering
┌─────────┬─────────┬─────────┬────────┐
│  CPU    │  CUDA   │  ROCm   │  SPIR-V│
│ LLVM IR │ NVVM    │ ROCDL   │ spirv  │
│   ↓     │   ↓     │   ↓     │   ↓    │
│ .so/.dll│ .cubin  │ .hsaco  │ .spv   │
└─────────┴─────────┴─────────┴────────┘
```

## CLI Tools

### yirage-opt

The `yirage-opt` tool runs MLIR passes:

```bash
# Lower YiRage dialect to Linalg
yirage-opt input.mlir --yirage-to-linalg -o output.mlir

# Full GPU pipeline
yirage-opt input.mlir \
  --yirage-to-linalg \
  --linalg-tile="tile-sizes=64,64,32" \
  --linalg-fuse \
  --convert-linalg-to-gpu \
  --gpu-to-nvvm \
  -o output.mlir

# Generate PTX
yirage-opt input.mlir --yirage-gpu-pipeline="target=cuda" -o kernel.ptx
```

### Python API

```python
import yirage
from yirage.mlir import compile_model, JITRunner

# Compile PyTorch model to MLIR
model = MyLLaMAModel()
mlir_module = yirage.compile(model, target="cuda")

# JIT execute
runner = JITRunner(mlir_module)
output = runner.invoke("forward", inputs)

# Or generate kernel code
ptx_code = yirage.codegen(mlir_module, target="cuda", arch="sm_80")
```

## Troubleshooting

### MLIR not found

```
CMake Error: MLIR not found
```

**Solution**: Set `MLIR_DIR` explicitly:
```bash
cmake .. -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir
```

### LLVM version mismatch

```
MLIR requires LLVM 17, but found LLVM 16
```

**Solution**: Install the correct LLVM version or build from submodule:
```bash
cmake .. -DYIRAGE_LLVM_SOURCE=submodule -DYIRAGE_LLVM_VERSION=17
```

### Build fails with OOM

Building LLVM requires significant RAM (8GB+). Use fewer parallel jobs:
```bash
cmake --build . -j2  # Use only 2 cores
```

### Missing TableGen

```
Cannot find TableGen
```

**Solution**: Ensure LLVM is fully installed:
```bash
sudo apt install llvm-17-tools  # Ubuntu
```

## Development

### Adding New Operations

1. Define the operation in `mlir/include/yirage-mlir/Dialect/Yirage/IR/YirageOps.td`
2. Implement lowering in `mlir/lib/Dialect/Yirage/Transforms/YirageToLinalg.cpp`
3. Add tests in `mlir/test/`
4. Regenerate with `make yirage-opt`

### Running Tests

```bash
cd build
ctest -R mlir  # Run MLIR-related tests
```

## References

- [MLIR Documentation](https://mlir.llvm.org/)
- [LLVM Getting Started](https://llvm.org/docs/GettingStarted.html)
- [Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [GPU Dialect](https://mlir.llvm.org/docs/Dialects/GPU/)
