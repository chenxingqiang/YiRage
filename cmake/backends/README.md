# YiRage Backend Configurations

This directory contains CMake configuration files for all supported hardware backends plus the MLIR unified compilation pipeline.

## Quick Start

```bash
# Copy your target backend config to the project root
cp cmake/backends/<backend>.cmake config.cmake

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Or use pip with auto-detection
YIRAGE_BACKEND=cuda pip install -e .
```

## Available Backends

| File | Backend | Hardware | Status | Requirements |
|------|---------|----------|--------|--------------|
| `cuda.cmake` | NVIDIA CUDA | RTX/A100/H100/B100 | ✅ Production | CUDA Toolkit 11.4+ |
| `mps.cmake` | Apple Metal | M1/M2/M3/M4 | ✅ Production | macOS 12.0+, Xcode |
| `rocm.cmake` | AMD ROCm | MI100/MI200/MI300 | ✅ Beta | ROCm 5.0+ |
| `ascend.cmake` | Huawei Ascend | 910/910B/310P | ✅ Beta | CANN, torch_npu |
| `tpu.cmake` | Google TPU | v4/v5e | 🔧 Alpha | JAX, Cloud TPU |
| `xpu.cmake` | Intel XPU | Arc/Data Center GPU | 🔧 Alpha | oneAPI 2023+ |
| `maca.cmake` | MetaX MACA | MetaX GPU | 🔧 Alpha | MACA SDK |
| `fpga.cmake` | FPGA | Alveo/Agilex | 🔬 Experimental | Vitis/Intel HLS |
| `nki.cmake` | AWS Neuron | Trainium/Inferentia2 | 🔧 Alpha | Neuron SDK |
| `triton.cmake` | Triton DSL | NVIDIA GPU | ✅ Beta | triton-lang |
| `cpu.cmake` | CPU Only | x86_64/ARM64 | ✅ Production | OpenMP |
| **`mlir.cmake`** | **MLIR/LLVM** | **All (Unified)** | 🆕 **Alpha** | **LLVM 17+** |

## MLIR Backend (NEW)

The MLIR backend provides a **unified compilation pipeline** for all hardware targets:

```bash
# Use MLIR for unified codegen
cp cmake/backends/mlir.cmake config.cmake

# Build with MLIR
mkdir build && cd build
cmake .. -DYIRAGE_LLVM_SOURCE=submodule  # Or: system, prebuilt
make -j$(nproc)
```

### MLIR Compilation Pipeline

```
PyTorch/JAX Model
       ↓
YiRage Dialect (yirage.matmul, yirage.attention)
       ↓
Linalg + Tensor + SCF (tiling, fusion)
       ↓
┌─────────┬─────────┬─────────┬─────────┐
│  CPU    │  CUDA   │  ROCm   │ SPIR-V  │
│ LLVM IR │  NVVM   │  ROCDL  │  spirv  │
│   ↓     │    ↓    │    ↓    │    ↓    │
│ .so/.a  │ .cubin  │ .hsaco  │  .spv   │
└─────────┴─────────┴─────────┴─────────┘
```

### MLIR vs Direct Backends

| Feature | Direct Backends | MLIR Backend |
|---------|-----------------|--------------|
| Setup complexity | Backend-specific | Unified |
| Cross-backend optimization | ❌ | ✅ Operator fusion |
| Custom operations | Manual | TableGen |
| Debugging | Varies | MLIR IR at each stage |
| New hardware | New implementation | Add target lowering |

See `docs/mlir_setup.md` for detailed setup instructions.

## Backend Selection Guide

### By Cloud Provider

| Cloud | Instance Type | Backend |
|-------|---------------|---------|
| AWS | p4d/p5 (A100/H100) | `cuda.cmake` |
| AWS | trn1/inf2 | `nki.cmake` |
| AWS | F1 FPGA | `fpga.cmake` |
| GCP | TPU v4/v5 | `tpu.cmake` |
| GCP | A100/H100 | `cuda.cmake` |
| Azure | ND A100 | `cuda.cmake` |
| Huawei Cloud | Ascend | `ascend.cmake` |
| Alibaba Cloud | GPU | `cuda.cmake` |
| 沐曦云 | MetaX GPU | `maca.cmake` |

### By Use Case

| Use Case | Recommended Backend |
|----------|---------------------|
| Local Mac development | `mps.cmake` |
| NVIDIA data center | `cuda.cmake` |
| Cost-effective cloud | `tpu.cmake` (GCP) |
| ML training at scale | `nki.cmake` (AWS Trainium) |
| AMD-based systems | `rocm.cmake` |
| Intel platforms | `xpu.cmake` + `cpu.cmake` |
| Edge deployment | `cpu.cmake` / `ascend.cmake` |
| **Unified pipeline** | **`mlir.cmake`** |

## Configuration Options

All backend configs follow a common structure:

```cmake
# GPU Backends
set(USE_CUDA OFF/ON)
set(USE_ROCM OFF/ON)
set(USE_MPS OFF/ON)
set(USE_XPU OFF/ON)

# NPU/Accelerator Backends
set(USE_ASCEND OFF/ON)
set(USE_MACA OFF/ON)
set(USE_TPU OFF/ON)
set(USE_FPGA OFF/ON)

# CPU Backends
set(USE_CPU ON)           # Always available
set(USE_MKL OFF/ON)       # Intel Math Kernel Library
set(USE_OPENMP ON)        # Parallel execution

# DSL Backends
set(USE_NKI OFF/ON)       # AWS Neuron Kernel Interface
set(USE_TRITON OFF/ON)    # OpenAI Triton

# MLIR Ecosystem (NEW)
set(USE_MLIR OFF/ON)      # MLIR compilation
set(USE_STABLEHLO OFF/ON) # StableHLO/XLA compatibility
set(USE_TVM OFF/ON)       # Apache TVM
set(USE_IREE OFF/ON)      # IREE runtime

# Build Options
set(BUILD_CPP_EXAMPLES OFF)
set(USE_FORMAL_VERIFIER OFF)
```

## Multi-Backend Support

YiRage supports multiple backends simultaneously:

```python
import yirage as yr

# Auto-detect best available backend
graph.superoptimize()

# Specify backend explicitly
graph.superoptimize(backend='cuda')
graph.superoptimize(backend='mps')
graph.superoptimize(backend='cpu')
graph.superoptimize(backend='mlir')  # NEW: Use MLIR pipeline
```

## Custom Configuration

Create your own config by combining multiple backends:

```cmake
# config.cmake - Custom multi-backend config
set(USE_CUDA ON)         # Primary: NVIDIA GPU
set(USE_CPU ON)          # Fallback: CPU
set(USE_OPENMP ON)       # Parallel search
set(USE_MKL ON)          # Intel optimizations
set(USE_MLIR ON)         # MLIR for advanced optimization
```

## Troubleshooting

### Backend not detected

```bash
# Check available backends
python -c "import yirage; print(yirage.get_available_backends())"
```

### Missing dependencies

```bash
# CUDA
nvidia-smi

# ROCm
rocminfo

# MPS (macOS)
system_profiler SPDisplaysDataType

# Ascend
npu-smi info

# MLIR/LLVM
llvm-config --version
mlir-opt --version
```
