# YiRage MLIR Dialect

This directory contains the YiRage MLIR dialect implementation, providing a high-level intermediate representation for AI/LLM workloads that can be lowered to various hardware backends.

## Directory Structure

```
mlir/
├── CMakeLists.txt                  # Build configuration
├── README.md                       # This file
├── include/
│   └── yirage-mlir/
│       └── Dialect/
│           └── Yirage/
│               ├── IR/
│               │   ├── YirageDialect.td    # Dialect definition (TableGen)
│               │   ├── YirageDialect.h     # Dialect C++ header
│               │   ├── YirageOps.td        # Operations definition (TableGen)
│               │   └── YirageOps.h         # Operations C++ header
│               └── Transforms/
│                   ├── Passes.td           # Pass definitions (TableGen)
│                   └── Passes.h            # Pass C++ header
├── lib/
│   └── Dialect/
│       └── Yirage/
│           ├── IR/
│           │   ├── YirageDialect.cpp       # Dialect implementation
│           │   └── YirageOps.cpp           # Operations implementation
│           └── Transforms/
│               ├── YirageToLinalg.cpp      # Linalg lowering pass
│               └── PassRegistration.cpp    # Pass registration
├── tools/
│   └── yirage-opt.cpp              # Optimizer tool
└── test/                           # Test cases (to be added)
```

## Requirements

- LLVM/MLIR 17+ (with development headers)
- CMake 3.20+
- C++17 compatible compiler

## Building

```bash
# Create build directory
mkdir build && cd build

# Configure with MLIR path
cmake .. -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir

# Build
make -j$(nproc)

# Install (optional)
make install
```

## Dialect Overview

### Operations

The Yirage dialect provides the following operations:

#### Matrix Operations
- `yirage.matmul` - Matrix multiplication
- `yirage.batch_matmul` - Batched matrix multiplication
- `yirage.qmatmul` - Quantized matrix multiplication (INT8/INT4)

#### Attention Operations
- `yirage.attention` - Scaled dot-product attention (supports MQA/GQA/Flash)
- `yirage.paged_attention` - Paged attention for KV cache
- `yirage.kv_cache_update` - KV cache update operation

#### Normalization
- `yirage.rms_norm` - RMSNorm (used in LLaMA, etc.)
- `yirage.layer_norm` - LayerNorm (used in BERT, ViT, etc.)

#### Activation Functions
- `yirage.silu` - SiLU/Swish activation
- `yirage.gelu` - GELU activation (exact and approximate)
- `yirage.relu` - ReLU activation
- `yirage.softmax` - Softmax operation

#### MLP Operations
- `yirage.gated_mlp` - Gated MLP block (SwiGLU style)
- `yirage.linear` - Linear layer with optional bias

#### Embedding Operations
- `yirage.embedding` - Token embedding lookup
- `yirage.rope` - Rotary Position Embedding

#### Reduction Operations
- `yirage.reduce_sum` - Sum reduction
- `yirage.reduce_max` - Max reduction
- `yirage.topk` - TopK operation
- `yirage.argmax` - Argmax operation

#### Tensor Operations
- `yirage.reshape` - Reshape tensor
- `yirage.transpose` - Transpose tensor
- `yirage.concat` - Concatenate tensors
- `yirage.split` - Split tensor

#### Convolution (for Vision)
- `yirage.conv2d` - 2D convolution
- `yirage.max_pool2d` - Max pooling

#### Quantization
- `yirage.quantize` - Quantize to INT8/INT4
- `yirage.dequantize` - Dequantize to float

### Types

- `!yirage.qtensor` - Quantized tensor type with scale and zero point
- `!yirage.kvcache` - KV cache type for paged attention

## Lowering Pipelines

### GPU Pipeline
```
yirage-opt input.mlir --yirage-gpu-pipeline --target-gpu=cuda
```

Pipeline: `Yirage → Linalg → Tile/Fuse → GPU → NVVM/ROCDL/SPIRV`

Supported targets:
- `cuda` - NVIDIA GPUs (NVVM/PTX)
- `rocm` - AMD GPUs (ROCDL/GCN)
- `spirv` - Intel XPU, Vulkan
- `metal` - Apple MPS
- `maca` - MetaX GPUs

### CPU Pipeline
```
yirage-opt input.mlir --yirage-cpu-pipeline --target-arch=x86-64-v3
```

Pipeline: `Yirage → Linalg → Tile → Vectorize → LLVM`

Supported architectures:
- `x86-64-v3` - AVX2
- `x86-64-v4` - AVX-512
- `aarch64` - ARM NEON
- `aarch64+sve` - ARM SVE

### TPU Pipeline
```
yirage-opt input.mlir --yirage-tpu-pipeline
```

Pipeline: `Yirage → StableHLO → XLA`

## Example Usage

### Input MLIR
```mlir
func.func @llama_attention(
    %Q: tensor<1x32x2048x128xf16>,
    %K: tensor<1x8x2048x128xf16>,
    %V: tensor<1x8x2048x128xf16>
) -> tensor<1x32x2048x128xf16> {
  %out = yirage.attention %Q, %K, %V {
    causal = true,
    num_kv_heads = 8
  } : tensor<1x32x2048x128xf16>, tensor<1x8x2048x128xf16>,
      tensor<1x8x2048x128xf16> -> tensor<1x32x2048x128xf16>
  return %out : tensor<1x32x2048x128xf16>
}
```

### Lowered to Linalg
```bash
yirage-opt input.mlir --yirage-to-linalg -o output.mlir
```

## Integration with YiRage

The MLIR dialect is designed to integrate with the YiRage superoptimizer:

1. **Import**: Convert PyTorch/JAX graphs to Yirage MLIR
2. **Optimize**: Apply Yirage-specific optimizations
3. **Lower**: Lower to target-specific MLIR dialects
4. **Codegen**: Generate optimized kernels

## License

Apache License 2.0 - See LICENSE file in the project root.
