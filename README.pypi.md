# YiRage - Yield Revolutionary AGile Engine

Multi-Backend LLM Inference Optimization Framework

## Quick Start

```bash
pip install yirage
```

```python
import yirage as yr

# Query available backends
backends = yr.get_available_backends()
print(f"Available backends: {backends}")

# Create and optimize a kernel graph
graph = yr.new_kernel_graph()
X = graph.new_input(dims=(8, 4096), dtype=yr.float16)
W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
Y = graph.matmul(X, W)
graph.mark_output(Y)

# Superoptimize for your hardware
optimized = graph.superoptimize(backend="cuda")  # or "maca", "ascend", "cpu"
```

## Multi-Backend Support

YiRage supports multiple hardware backends for maximum portability:

| Backend | Hardware | Status | Notes |
|---------|----------|--------|-------|
| **CUDA** | NVIDIA GPU | Production | Hopper, Blackwell optimized |
| **MACA** | MetaX GPU | Production | 64-thread warp support |
| **Ascend** | Huawei NPU | Production | Triton via BiSheng |
| **MPS** | Apple Silicon | Beta | M1/M2/M3/M4 support |
| **CPU** | x86/ARM | Beta | AVX2/AVX512 optimized |
| **Triton** | NVIDIA/AMD | Production | JIT compilation |
| **NKI** | AWS Trainium | Beta | Neuron SDK integration |

### Backend Selection

```python
import yirage as yr

# Automatic backend selection
default_backend = yr.get_default_backend()
print(f"Default: {default_backend}")

# Check specific backend availability
if yr.is_backend_available("maca"):
    print("MetaX MACA GPU available!")

# Manual backend selection
yr.set_default_backend("ascend")
```

### Multi-Backend Kernel Compilation

```python
from yirage import MultiBackendKernel, KernelConfig, KernelBackend

# Create kernel with automatic fallback
kernel = MultiBackendKernel(
    source_code=kernel_source,
    config=KernelConfig(
        backend=KernelBackend.CUDA,
        fallback_backends=[KernelBackend.MACA, KernelBackend.CPU],
    )
)

# Compile (automatically selects available backend)
kernel.compile()
print(f"Compiled for: {kernel.active_backend}")

# Execute
kernel.execute(input_tensors, output_tensors)
```

## Features

- **9 Backend Implementations**: CUDA, CPU, MPS, MACA, Ascend, Triton, NKI, cuDNN, MKL
- **Hardware-Aware Optimizers**: 42+ optimization methods
- **Search Strategies**: Backend-specific strategies with auto-tuning
- **Unified API**: Write once, run on any supported hardware
- **Production Ready**: 22,000+ lines of tested code

## Ascend NPU Support

YiRage supports Huawei Ascend NPUs via multiple code generation paths:

```python
from yirage import transpile_to_ascend, AscendTranspileConfig, CodeGenPath

# Configure for Ascend 910B
config = AscendTranspileConfig(
    codegen_path=CodeGenPath.TRITON,  # Uses BiSheng compiler
    enable_bf16=True,
)

# Transpile kernel
result = transpile_to_ascend(kernel_spec, config)
print(result.code)
```

## MetaX MACA GPU Support

For MetaX C500/C600 GPUs with MACA SDK:

```python
import yirage as yr

# MACA is detected automatically when mcPytorch is used
graph = yr.new_kernel_graph()
# ... define graph ...

# Optimize for MACA (64-thread warps)
optimized = graph.superoptimize(backend="maca")
```

## Installation

### For NVIDIA CUDA
```bash
pip install yirage
```

### For Huawei Ascend
```bash
# Requires CANN toolkit and torch_npu
pip install yirage torch-npu
```

### For MetaX MACA
```bash
# Requires MACA SDK and mcPytorch
pip install yirage
```

## Documentation

- Full documentation: https://github.com/chenxingqiang/YiRage
- API Reference: https://github.com/chenxingqiang/YiRage/docs
- Examples: https://github.com/chenxingqiang/YiRage/demo

## License

Apache License 2.0

Based on Mirage by CMU. Copyright 2025 Chen Xingqiang.
