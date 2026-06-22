# YiRage - Yield Revolutionary AGile Engine

<div align="center">

**Multi-Backend LLM Inference Optimization**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/GitHub-YiRage-blue)](https://github.com/chenxingqiang/YiRage)

</div>

---

## 🎯 About YiRage

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/chenxingqiang/YiRage)

**YiRage** (Yield Revolutionary AGile Engine) provides comprehensive **multi-backend support** for LLM inference optimization across diverse hardware platforms.

### Multi-Backend Optimization Focus

- Unified optimization workflow across CUDA, ROCm, CPU, MPS, Ascend, MACA, TPU, XPU, FPGA, Triton, NKI, and MLIR backends
- **Same-backend hardware optimization**: search, profile, cache, and execute on the **installed target** (CPU on CPU, CUDA on GPU)—not cross-backend mixes such as CPU search with CUDA execution
- Hardware-aware search, profiling, and kernel generation for deployment-focused LLM inference
- Extensible backend architecture for adding new hardware targets and compiler integrations

📖 Design rationale: **[docs/HARDWARE_OPTIMIZATION.md](docs/HARDWARE_OPTIMIZATION.md)**

---

## 🚀 Quick Start

### Installation

> 📖 For the full reference, see **[docs/INSTALLATION.md](docs/INSTALLATION.md)** and the per-backend
> **[Ascend Guide](docs/ascend_installation_guide.md)**. The sections below cover everything you need
> for typical setups.

**Native runtime (required).** `import yirage` requires the Cython extension `yirage.core` linked
against the C++ library `libyirage_runtime`. There is **no Python-only mode**. You can either
install a prebuilt wheel that already includes the native code, or build from source. Optional
PyPI extras (`yirage[cuda]`, `yirage[all]`, …) only add **Python** dependencies — they do **not**
remove the need for the native runtime.

#### 1. Prerequisites

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python      | 3.8 – 3.12 | CPython recommended |
| CMake       | ≥ 3.24 | Used to build the C++ runtime (`pip install cmake` works) |
| C++ compiler | C++17 capable | GCC ≥ 9, Clang ≥ 10, Apple Clang ≥ 13, or MSVC 2019+ |
| Cython      | ≥ 0.29 | Installed automatically as a build requirement |
| Git         | any | Required to clone and to fetch submodules |
| Backend SDK | varies | See [Backend-specific installation](#3-backend-specific-installation) below |

After cloning, initialize git submodules (vendored third-party code lives under `deps/`):

```bash
git clone --recursive https://github.com/chenxingqiang/YiRage.git
cd YiRage

# Or, if you already cloned without --recursive:
git submodule update --init --recursive
```

#### 2. Install from PyPI

When a prebuilt wheel is available for your platform (Linux x86_64 / Linux aarch64 / macOS arm64):

```bash
# Core install (native runtime ships with the wheel when published)
pip install yirage

# With optional Python extras
pip install "yirage[cuda]"          # NVIDIA CUDA stack (torch, triton)
pip install "yirage[rocm]"          # AMD ROCm stack
pip install "yirage[mps]"           # Apple Silicon
pip install "yirage[ascend]"        # Huawei NPU (also needs torch-npu, see below)
pip install "yirage[llm]"           # transformers, accelerate, safetensors
pip install "yirage[distributed]"   # Ray distributed runtime
pip install "yirage[profiling]"     # TensorBoard
pip install "yirage[dev]"           # pytest, black, ruff, mypy, …
pip install "yirage[all]"           # Everything that can be installed via pip
```

If no wheel is published for your platform/Python/ABI combination, pip will fall back to building
from source, in which case the next section applies.

#### 3. Install from Source

The recommended path for development or when you need to pick a specific backend.

```bash
git clone --recursive https://github.com/chenxingqiang/YiRage.git
cd YiRage

# (Optional) full dev-environment requirements file
pip install -r requirements.txt

# Auto-detect hardware (CUDA / MPS / CPU) and build everything
pip install -e .

# Editable + dev tooling
pip install -e ".[dev]"
```

##### Specify a backend explicitly

```bash
# Single backend via YIRAGE_BACKEND
YIRAGE_BACKEND=cuda   pip install -e .   # NVIDIA GPU
YIRAGE_BACKEND=rocm   pip install -e .   # AMD GPU
YIRAGE_BACKEND=mps    pip install -e .   # Apple Silicon
YIRAGE_BACKEND=ascend pip install -e .   # Huawei NPU
YIRAGE_BACKEND=maca   pip install -e .   # MetaX GPU
YIRAGE_BACKEND=xpu    pip install -e .   # Intel GPU
YIRAGE_BACKEND=tpu    pip install -e .   # Google TPU
YIRAGE_BACKEND=cpu    pip install -e .   # CPU (still a full native build)

# Multiple backends in one build
YIRAGE_BACKEND=cuda,cpu pip install -e .

# Equivalent low-level toggles
USE_CUDA=ON USE_CPU=ON pip install -e .
```

#### 4. Backend-Specific Installation

##### NVIDIA CUDA

```bash
# Prerequisites: CUDA Toolkit 11.8+ or 12.x   (https://developer.nvidia.com/cuda-toolkit)
nvidia-smi && nvcc --version   # sanity check

# Optional Python extras (PyTorch + Triton)
pip install "yirage[cuda]"

# Build from source against a specific CUDA install
CUDA_HOME=/usr/local/cuda-12.1 YIRAGE_BACKEND=cuda pip install -e .
```

##### AMD ROCm

```bash
# Prerequisites: ROCm 5.x or 6.x   (https://rocm.docs.amd.com/)
rocm-smi && hipcc --version

# Install ROCm-built PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Then build YiRage
ROCM_PATH=/opt/rocm-6.0 YIRAGE_BACKEND=rocm pip install -e .
```

##### Apple Silicon (MPS)

```bash
# Prerequisites: macOS 12.3+ on M1 / M2 / M3 / M4
brew install cmake libomp           # libomp enables parallel search

pip install "yirage[mps]"           # or:
YIRAGE_BACKEND=mps pip install -e . # explicit
```

##### Huawei Ascend NPU

📖 Full walk-through: **[docs/ascend_installation_guide.md](docs/ascend_installation_guide.md)**

```bash
# Prerequisites: CANN 7.0+   (https://www.hiascend.com/)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Install torch-npu (use Huawei mirror if the default index fails)
pip install torch-npu
# or:
pip install torch-npu -i https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/ascend-repo/simple/

# Build YiRage
YIRAGE_BACKEND=ascend pip install -e .
```

##### MetaX MACA

```bash
# Prerequisites: MACA SDK from MetaX
export MACA_PATH=/opt/maca
YIRAGE_BACKEND=maca pip install -e .
```

##### Intel XPU

```bash
# Prerequisites: Intel oneAPI   (https://www.intel.com/oneapi)
pip install intel-extension-for-pytorch
YIRAGE_BACKEND=xpu pip install -e .
```

##### Google TPU

```bash
# JAX for TPU (separate index)
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
YIRAGE_BACKEND=tpu pip install -e .
```

##### CPU-only

```bash
pip install yirage                       # uses prebuilt wheel if available
# or build from source:
YIRAGE_BACKEND=cpu pip install -e .
```

#### 5. Build-Time Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `YIRAGE_BACKEND` | Comma-separated list of backends to enable | `cuda`, `mps`, `cuda,cpu` |
| `USE_CUDA` / `USE_ROCM` / `USE_MPS` / `USE_ASCEND` / `USE_MACA` / `USE_CPU` | Low-level CMake toggles | `ON` / `OFF` |
| `CUDA_HOME` | Path to CUDA toolkit | `/usr/local/cuda-12.1` |
| `ROCM_PATH` | Path to ROCm install | `/opt/rocm-6.0` |
| `ASCEND_HOME` | Path to CANN toolkit | `/usr/local/Ascend/ascend-toolkit/latest` |
| `MACA_PATH` | Path to MACA SDK | `/opt/maca` |
| `SKIP_BUILD` | Skip the C++ rebuild (re-uses existing artifacts) | `1` |
| `CMAKE_BUILD_PARALLEL_LEVEL` | Parallel compile jobs | `8` |

#### 6. Docker Installation

```bash
# CUDA image
docker build -t yirage:cuda -f docker/Dockerfile.cuda .
docker run --gpus all -it yirage:cuda

# CPU image
docker build -t yirage:cpu -f docker/Dockerfile .
docker run -it yirage:cpu
```

#### 7. Verify the Installation

```python
import yirage as yr

print(f"YiRage version: {yr.__version__}")

backends = yr.get_available_backends()
print(f"Available backends: {backends}")

for name in ("cuda", "mps", "ascend", "rocm", "cpu"):
    if yr.is_backend_available(name):
        print(f"✅ {name} backend ready")
```

Run the test suite to confirm the native runtime works end-to-end:

```bash
pytest tests/python -v
```

#### 8. Troubleshooting

<details>
<summary><strong>CMake not found</strong></summary>

```bash
brew install cmake        # macOS
sudo apt install cmake    # Debian/Ubuntu
pip install cmake         # cross-platform fallback
```
</details>

<details>
<summary><strong><code>libz3.dylib</code> not found on macOS</strong></summary>

```bash
export DYLD_LIBRARY_PATH="$(python -c 'import z3, os; print(os.path.dirname(z3.__file__) + "/lib")'):$DYLD_LIBRARY_PATH"
```
</details>

<details>
<summary><strong>CUDA not detected</strong></summary>

```bash
nvidia-smi
nvcc --version
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```
</details>

<details>
<summary><strong>ROCm not detected</strong></summary>

```bash
rocm-smi
hipcc --version
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
```
</details>

<details>
<summary><strong><code>ImportError: yirage.core</code> after install</strong></summary>

The Cython extension failed to build or load. Reinstall with a clean tree:

```bash
pip uninstall -y yirage
rm -rf build dist *.egg-info
YIRAGE_BACKEND=<your-backend> pip install -e . -v
```
</details>

#### 9. Platform Support

| Platform | Backends | Status |
|----------|----------|--------|
| Linux x86_64 | CUDA, ROCm, CPU | ✅ Full support |
| Linux aarch64 | Ascend, CPU | ✅ Full support |
| macOS arm64 | MPS, CPU | ✅ Full support |
| macOS x86_64 | CPU | ✅ CPU only |
| Windows x86_64 | CUDA, CPU | 🔧 Experimental |

📖 **[Full Installation Guide](docs/INSTALLATION.md)** — all backends, options, and advanced configuration.

### Basic Usage

```python
import yirage as yr

# Query available backends
backends = yr.get_available_backends()
print(f"Available backends: {backends}")
# Output: ['cuda', 'cpu', 'mps']  # depends on your hardware

# Check specific backend
if yr.is_backend_available('mps'):
    print("Apple Silicon GPU ready!")

# Create kernel with backend selection
mpk = yr.PersistentKernel(
    mode="decode",
    backend="mps",              # Specify backend
    fallback_backends=["cpu"],  # Auto fallback
    world_size=1,
    mpi_rank=0,
    # ... other parameters
)
```

### Using Hardware-Specific Optimizers

```python
# CUDA optimization
from yirage.backends.cuda.config import CUDAArch, get_cuda_search_config

cuda_config = get_cuda_search_config(CUDAArch.AMPERE)
print(f"CUDA tile candidates: {cuda_config['block_dims_to_explore'][:3]}")

# CPU optimization
from yirage.backends.cpu.config import get_cpu_search_config

cpu_config = get_cpu_search_config()
print(f"CPU SIMD: {cpu_config['simd_type']} across {cpu_config['num_cores']} cores")
# Auto-detects: SIMD type, CPU cores, cache-aware search space

# MPS optimization (Apple Silicon)
from yirage.backends.mps.config import AppleChipFamily, get_mps_search_config

mps_config = get_mps_search_config(AppleChipFamily.M3_MAX)
print(f"MPS chip: {mps_config['chip_family']} with {mps_config['gpu_cores']} GPU cores")
# Auto-configures: GPU family, cores, memory/search limits
```

---

## 🏗️ Architecture

### Five-Layer Backend Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              YiRage Backend Architecture                                │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                         Layer 1: Python API                                     │    │
│  │  yirage.new_kernel_graph() → UnifiedCompiler → CoreBridge → superoptimize()     │    │
│  │  HardwareRegistry.instance() → ChipArchitecture → detect_current_chip()         │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                                │
│  ┌─────────────────────────────────────▼───────────────────────────────────────────┐    │
│  │                         Layer 2: Backend Manager (C++)                          │    │
│  │  BackendRegistry (thread-safe) ← BackendFactory ← StrategyFactory               │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                                │
│  ┌─────────────────────────────────────▼───────────────────────────────────────────┐    │
│  │                         Layer 3: Search & Strategy                              │    │
│  │  Hardware-aware Search │ Fingerprint Verification │ Performance Profiling       │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                                │
│  ┌─────────────────────────────────────▼───────────────────────────────────────────┐    │
│  │                         Layer 4: Threadblock Operations                         │    │
│  │  MatMul │ Attention │ RMSNorm │ SwiGLU │ Softmax │ Reduce │ Elementwise         │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                                │
│  ┌─────────────────────────────────────▼───────────────────────────────────────────┐    │
│  │                         Layer 5: Persistent Kernel Runtime                      │    │
│  │  Memory Management │ Kernel Launch │ Synchronization │ JIT Compilation          │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                        │                                                │
│  ┌─────────────────────────────────────▼───────────────────────────────────────────┐    │
│  │                              Hardware Layer                                     │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌───────┐ ┌──────┐ ┌──────┐ ┌─────┐ ┌─────┐ ┌──────┐   │    │
│  │  │CUDA │ │ROCm │ │ MPS │ │Ascend │ │ MACA │ │ TPU  │ │ XPU │ │FPGA │ │ CPU  │   │    │
│  │  │NVIDA│ │ AMD │ │Apple│ │Huawei │ │MetaX │ │Google│ │Intel│ │Xilinx││x86/ARM│  │    │
│  │  └─────┘ └─────┘ └─────┘ └───────┘ └──────┘ └──────┘ └─────┘ └─────┘ └──────┘   │    │
│  │  ┌───────┐ ┌─────┐ ┌──────┐                                                     │    │
│  │  │Triton │ │ NKI │ │ MLIR │  ← Compiler Backends                                │    │
│  │  │OpenAI │ │ AWS │ │ LLVM │                                                     │    │
│  │  └───────┘ └─────┘ └──────┘                                                     │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### Backend Support Matrix (12 Backends × 5 Layers)

| Backend | Hardware | Backend API | Strategy | Kernel | Threadblock | PK Runtime |
|---------|----------|-------------|----------|--------|-------------|------------|
| **CUDA** | NVIDIA GPU | ✅ | ✅ | ✅ | ✅ | ✅ |
| **ROCm** | AMD GPU | ✅ | ✅ | ✅ | ✅ | ✅ |
| **CPU** | x86/ARM | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MPS** | Apple Silicon | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Ascend** | Huawei NPU | ✅ | ✅ | ✅ | ✅ | ✅ |
| **MACA** | MetaX GPU | ✅ | ✅ | ✅ | ✅ | ✅ |
| **TPU** | Google Cloud | ✅ | ✅ | ✅ | ✅ | ✅ |
| **XPU** | Intel GPU | ✅ | ✅ | ✅ | ✅ | ✅ |
| **FPGA** | Intel/Xilinx | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Triton** | Compiler | ✅ | ✅ | ✅ | ✅ | ✅ |
| **NKI** | AWS Neuron | ✅ | ✅ | 🚧 | 🚧 | 🚧 |
| **MLIR** | Multi-target | ✅ | ✅ | ✅ | ✅ | ✅ |

> Status note: ✅ means the YiRage interface and modeling path exist; 🚧 means
> the backend is limited to modeling/code-generation paths and is not yet
> available through the runtime execution API. Some
> vendor-specific threadblock and persistent-kernel implementations remain
> experimental and are excluded from the default CMake build until their vendor
> toolchains and interfaces are complete.

### Five-Layer Design

**Layer 1: Python API**
- Backend query and selection (`get_available_backends()`)
- **Hardware Device Registry** (`HardwareRegistry` — register/query chip architectures at runtime)
- Unified compiler interface (`UnifiedCompiler`)
- Core bridge to C++ (`CoreBridge`)
- Hardware-specific optimizers

**Layer 2: Backend Manager (C++)**
- BackendRegistry (singleton, thread-safe)
- Factory patterns for backends and strategies
- Automatic initialization on import

**Layer 3: Search & Strategy**
- Hardware-aware kernel search
- Fingerprint-based verification
- Performance profiling and modeling

**Layer 4: Threadblock Operations**
- Optimized LLM operators (MatMul, Attention, RMSNorm, SwiGLU)
- Hardware-specific implementations
- Code generation for Triton/NKI/MLIR

**Layer 5: Persistent Kernel Runtime**
- Device memory management
- Kernel launch and synchronization
- JIT compilation support

---

## ✨ Key Features

### 🚀 12 Backend Targets (Core + Experimental)

| Backend | Hardware | Key Features | Architecture |
|---------|----------|--------------|--------------|
| **CUDA** | NVIDIA GPU | Tensor Core, 32-thread Warp, cuBLAS | SM, Shared Memory |
| **ROCm** | AMD GPU | Matrix Core, 64-thread Wavefront, rocBLAS | GCN/CDNA, LDS |
| **CPU** | x86/ARM | AVX512/NEON SIMD, Cache Blocking, OpenMP | Multi-core, L1/L2/L3 |
| **MPS** | Apple Silicon | Metal, Threadgroup, Unified Memory | M1/M2/M3/M4 |
| **Ascend** | Huawei NPU | Cube Unit 16×16, AI Core, L1 Buffer | Ascend 910/310 |
| **MACA** | MetaX GPU | 64-thread Warp, CUDA-compat, Tensor Core | C500 Series |
| **TPU** | Google Cloud | MXU 128×128, BF16 Native, PJRT | TPU v2/v3/v4/v5 |
| **XPU** | Intel GPU | XMX 8×8, SYCL/oneAPI, SLM | Arc/Max/Gaudi |
| **FPGA** | Intel/Xilinx | DSP Blocks, Pipeline, BRAM/HBM | OpenCL Kernel |
| **Triton** | Compiler | Auto-tuning, Tile Fusion, MMA | PTX/HSACO |
| **NKI** | AWS Neuron | Tensor Engine 128×128, SBUF 24MB | Trainium/Inferentia |
| **MLIR** | Multi-target | JIT, Linalg, Pass Pipeline | LLVM/NVVM/SPIRV |

### 🔧 Hardware Architecture Differences

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│                        Hardware Architecture Comparison                            │
├────────────┬─────────────────┬─────────────────┬───────────────────────────────────┤
│ Backend    │ Thread Model    │ Matrix Unit     │ Memory Hierarchy                  │
├────────────┼─────────────────┼─────────────────┼───────────────────────────────────┤
│ CUDA       │ 32-thread Warp  │ Tensor Core     │ Registers → Shared → L2 → HBM     │
│ ROCm       │ 64-thread Wave  │ Matrix Core     │ VGPR → LDS → L2 → HBM             │
│ MPS        │ SIMD Group      │ Apple GPU       │ Threadgroup → Device → Unified    │
│ Ascend     │ AI Core         │ Cube 16×16      │ L0 → L1 → L2 → HBM                │
│ MACA       │ 64-thread Warp  │ Tensor Core     │ Shared → L2 → HBM                 │
│ TPU        │ MXU Systolic    │ MXU 128×128     │ VMEM → HBM                        │
│ XPU        │ Xe Subgroup     │ XMX 8×8         │ SLM → L3 → HBM                    │
│ FPGA       │ Pipeline        │ DSP Block       │ BRAM/URAM → DDR/HBM               │
└────────────┴─────────────────┴─────────────────┴───────────────────────────────────┘
```

### 🎯 Hardware-Aware Kernel Optimizers

- **60+ Optimization Methods** across all 12 backends
- **Automatic Configuration** based on hardware capabilities
- **Performance Modeling** for each backend
- **Code Generation** for Triton/NKI/MLIR

#### Example: CUDA Optimizer
```python
from yirage.backends.cuda.config import CUDAArch, get_cuda_search_config

config = get_cuda_search_config(CUDAArch.AMPERE)
print(config["arch"], config["warp_size"], config["has_tensor_cores"])
# Auto-configured: Tensor Core, warps, shared memory, search space
```

#### Example: MPS Optimizer (Apple Silicon)
```python
from yirage.backends.mps.config import AppleChipFamily, get_mps_search_config

config = get_mps_search_config(AppleChipFamily.M3_MAX)
print(config["chip_family"], config["gpu_cores"], config["max_threads_per_threadgroup"])
# Auto-configures: M-series family, GPU cores, threadgroup size
```

#### Example: Ascend Optimizer (Huawei NPU)
```python
import yirage as yr

# Create and optimize for Ascend NPU
graph = yr.new_kernel_graph()
X = graph.new_input(dims=(8, 4096), dtype=yr.float16)
W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
O = graph.matmul(X, W)
graph.mark_output(O)

# Optimize using Ascend backend (via BiSheng + Triton)
optimized = graph.superoptimize(backend='ascend')
# Auto-configures: AI Core blocks, Cube unit tiles, L1 buffer
```

#### Example: MACA Optimizer (MetaX GPU)
```python
import yirage as yr

# Create and optimize for MetaX MACA GPU
graph = yr.new_kernel_graph()
X = graph.new_input(dims=(8, 4096), dtype=yr.float16)
W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
O = graph.matmul(X, W)
graph.mark_output(O)

# Optimize using MACA backend (64-thread warps!)
optimized = graph.superoptimize(backend='maca')
# Auto-configures: 64-thread warps, tile sizes, shared memory

# Environment: export MACA_HOME=/opt/maca
```

#### Example: ROCm Optimizer (AMD GPU) 🆕
```python
import yirage as yr

# Create and optimize for AMD GPU
graph = yr.new_kernel_graph()
X = graph.new_input(dims=(8, 4096), dtype=yr.float16)
W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
O = graph.matmul(X, W)
graph.mark_output(O)

# Optimize using ROCm backend
optimized = graph.superoptimize(backend='rocm')
# Auto-configures: 64-thread wavefronts, LDS, Matrix Cores (MI200/MI300)

# Environment: export ROCM_PATH=/opt/rocm
```

#### Example: TPU Optimizer (Google Cloud) 🆕
```python
import yirage as yr

# Create and optimize for Google TPU
graph = yr.new_kernel_graph()
X = graph.new_input(dims=(8, 4096), dtype=yr.bfloat16)
W = graph.new_input(dims=(4096, 4096), dtype=yr.bfloat16)
O = graph.matmul(X, W)
graph.mark_output(O)

# Optimize using TPU backend
optimized = graph.superoptimize(backend='tpu')
# Auto-configures: 128x128 MXU, BF16 native, VMEM tiling
```

#### Example: MLIR JIT Compiler 🆕
```python
from yirage.pk import MLIRPKBackend
from yirage.threadblock.mlir_ops import MLIRCodeGenerator, MLIRTileConfig

# Generate MLIR for MatMul
config = MLIRTileConfig(tile_sizes=[32, 32, 32], vectorize=True)
mlir_code = MLIRCodeGenerator.generate_matmul(1024, 1024, 1024, 
                                               dtype=yr.float16, config=config)

# JIT compile and execute
backend = MLIRPKBackend(target=MLIRPKBackend.JIT_TARGET_CPU)
backend.initialize()
backend.jit_compile(mlir_code)
backend.execute("matmul", [A_ptr, B_ptr, C_ptr], 3)
```

### 🔍 Backend-Specific Search Strategies

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              Search & Optimization Flow                              │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────┐     ┌──────────────────────────────────────────────────────────┐   │
│  │ Kernel Graph │────▶│                  Search Engine                           │   │
│  └──────────────┘     │  ┌────────────────┐  ┌───────────────┐  ┌─────────────┐  │   │
│                       │  │ Candidate Gen  │──│ Fingerprint   │──│ Performance │  │   │
│                       │  │ (µGraph Space) │  │ Verification  │  │  Profiler   │  │   │
│                       │  └────────────────┘  └───────────────┘  └─────────────┘  │   │
│                       └─────────────────────────────┬────────────────────────────┘   │
│                                                     │                                │
│  ┌──────────────────────────────────────────────────▼───────────────────────────┐    │
│  │                       Backend-Specific Strategies                            │    │
│  ├────────────┬────────────┬────────────┬────────────┬────────────┬─────────────┤    │
│  │   CUDA     │   ROCm     │   MPS      │  Ascend    │   MACA     │    TPU      │    │
│  │ TensorCore │ MatrixCore │ ThreadGrp  │  CubeUnit  │  64-Warp   │    MXU      │    │
│  │  32-Warp   │  64-Wave   │   SIMD     │  AI Core   │ TensorCore │  128×128    │    │
│  ├────────────┼────────────┼────────────┼────────────┼────────────┼─────────────┤    │
│  │    XPU     │   FPGA     │  Triton    │    NKI     │   MLIR     │    CPU      │    │
│  │    XMX     │  Pipeline  │ AutoTune   │ TensorEng  │ LinalgOpt  │  SIMD/OMP   │    │
│  │   SYCL     │    DSP     │ TileFuse   │   SBUF     │ JIT/AOT    │ CacheBlock  │    │
│  └────────────┴────────────┴────────────┴────────────┴────────────┴─────────────┘    │
│                                       │                                              │
│                         ┌─────────────▼─────────────┐                                │
│                         │    Optimized Kernel       │                                │
│                         │  (Best Configuration)     │                                │
│                         └───────────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

- **12 Independent Search Strategies** with hardware-specific optimization
- **20+ Candidate Generation Dimensions**
- **15 Performance Evaluation Metrics**
- Auto-tuning and performance modeling
- Code generation for compiler backends (Triton, NKI, MLIR)

---

## 🔌 Hardware Device Management (New!)

YiRage provides a **unified hardware registry** that allows new chip architectures to be registered at runtime — no code changes required. This is the highest level of hardware adaptation: any new chip can be plugged into the system by describing its architecture once.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     Hardware Device Management                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────┐     ┌──────────────────────────────────────┐    │
│  │  HardwareRegistry   │     │  ChipArchitecture Dataclass          │    │
│  │  (Thread-safe       │────▶│  ┌──────────┐ ┌───────────┐          │    │
│  │   Singleton)        │     │  │MemorySpec│ │ComputeSpec│          │    │
│  │                     │     │  └──────────┘ └───────────┘          │    │
│  │  • register()       │     │  ┌────────────┐ ┌────────┐           │    │
│  │  • get()            │     │  │FeatureFlags│ │Metadata│           │    │
│  │  • list_by_vendor() │     │  └────────────┘ └────────┘           │    │
│  │  • list_by_backend()│     └──────────────────────────────────────┘    │
│  │  • import_json()    │                                                 │
│  │  • export_json()    │     ┌──────────────────────────────────────┐    │
│  │  • on_register()    │     │  Built-in: 20+ Chips Pre-registered  │    │
│  └─────────────────────┘     │  NVIDIA V100→B200 │ AMD MI250X/MI300X│    │
│                              │  Ascend 910/910B  │ MetaX C500       │    │
│  ┌─────────────────────┐     │  Apple M2–M4      │ Google TPU v4/v5e│    │
│  │  Auto-Detection     │     │  Intel PVC │ Xilinx Alveo │ AWS Trn2 │    │ 
│  │  nvidia-smi/rocm-smi│     └──────────────────────────────────────┘    │
│  │  npu-smi/mx-smi/MPS │                                                 │
│  └─────────────────────┘                                                 │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 20+ Built-in Chip Architectures

| Vendor | Chips | Backend | Category |
|--------|-------|---------|----------|
| **NVIDIA** | V100, T4, A100, RTX 3090, RTX 4090, H100, B200 | `cuda` | GPU |
| **AMD** | MI250X, MI300X | `rocm` | GPU |
| **Intel** | Data Center GPU Max 1550 (PVC) | `xpu` | GPU |
| **Huawei** | Ascend 910, 910B, 310P | `ascend` | NPU |
| **MetaX** | C500, C500 Pro | `maca` | GPU |
| **Apple** | M2 Ultra, M3 Max, M4 Max | `mps` | GPU |
| **Google** | TPU v4, TPU v5e | `tpu` | TPU |
| **Xilinx** | Alveo U250 | `fpga` | FPGA |
| **AWS** | Trainium2 | `nki` | DSA |

### Quick Start — Query Built-in Chips

```python
from yirage.hardware import HardwareRegistry

reg = HardwareRegistry.instance()

# Look up a chip
h100 = reg.get("nvidia_h100")
print(h100.summary())
# NVIDIA H100 SXM5 | 132 CUs | 80GB HBM3 | 989 TFLOPS FP16

# List chips by vendor / backend / category
nvidia_chips = reg.list_by_vendor("nvidia")   # 7 chips
cuda_chips   = reg.list_by_backend("cuda")    # all CUDA-mapped chips
gpu_chips    = reg.list_by_category("gpu")    # all GPUs across vendors
```

### Register a New Chip at Runtime

```python
from yirage.hardware import (
    HardwareRegistry, ChipArchitecture,
    ChipVendor, ChipCategory,
    ComputeSpec, MemorySpec, MemoryType, FeatureFlags,
)

reg = HardwareRegistry.instance()

# Define the new chip
new_chip = ChipArchitecture(
    chip_id="myvendor_x1",
    chip_name="MyVendor X1 Accelerator",
    vendor=ChipVendor.OTHER,
    category=ChipCategory.DSA,
    arch_name="X1",
    arch_code="x1_v1",
    backend="cuda",                          # maps to YiRage backend
    memory=MemorySpec(
        capacity_gb=128,
        bandwidth_gbps=6000,
        memory_type=MemoryType.HBM3E,
    ),
    compute=ComputeSpec(
        warp_size=32,
        num_compute_units=256,
        peak_tflops_fp16=2000,
    ),
    features=FeatureFlags(
        tensor_cores=True,
        fp8=True,
        bf16=True,
    ),
)

reg.register(new_chip)
print(f"Registry now has {reg.size} chips")
```

### Bulk Import / Export (JSON)

```python
# Export entire registry to a file
reg.export_json("/path/to/chips.json")

# Import chips from a JSON file (e.g. from a partner's chip catalog)
count = reg.import_json("/path/to/new_chips.json")
print(f"Imported {count} new chips")
```

### Auto-detect Current Hardware

```python
from yirage.hardware import detect_current_chip

chip = detect_current_chip()
if chip:
    print(f"Detected: {chip.summary()}")
    print(f"Backend:  {chip.backend}")
    print(f"Memory:   {chip.memory.capacity_gb} GB {chip.memory.memory_type.value}")
    print(f"FP16:     {chip.compute.peak_tflops_fp16} TFLOPS")
```

### React to New Registrations (Callback)

```python
from yirage.hardware import ChipArchitecture, HardwareRegistry

reg = HardwareRegistry.instance()

def on_new_chip(chip):
    print(f"🆕 New chip registered: {chip.chip_name} ({chip.chip_id})")

another_chip = ChipArchitecture(
    chip_id="callback_demo",
    chip_name="Callback Demo",
    backend="cpu",
)

reg.on_register(on_new_chip)
reg.register(another_chip, overwrite=True)   # triggers callback
```

### Module Structure

```
python/yirage/hardware/
├── __init__.py          # Public API — auto-populates built-in chips on import
├── chip_arch.py         # ChipArchitecture, MemorySpec, ComputeSpec, FeatureFlags
├── registry.py          # HardwareRegistry (thread-safe singleton)
├── builtin_chips.py     # 20+ pre-registered chip definitions
└── detector.py          # Runtime auto-detection (nvidia-smi, npu-smi, etc.)
```

---

## 📊 Performance

### M3 Mac Benchmarks

| Benchmark | MPS (ms) | CPU (ms) |
|-----------|----------|----------|
| gated_mlp | 0.677 | 1.268 |
| rms_norm | 0.463 | 0.115 |
| lora | 0.637 | 0.590 |
| gqa | 0.554 | - |
| norm_transformer | 1.195 | - |

*All benchmarks support CUDA, MPS, and CPU backends*

---

## 🤖 RL-Guided Kernel Search

YiRage now supports **RL-guided kernel search** using Ray/RLlib, enabling intelligent exploration of the kernel configuration space.

### Hierarchical Closed-Loop Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  RL-YiRage Hierarchical Closed Loop                 │
│                                                                     │
│  Level 1: Config Policy                                             │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ HardwareConfig (grid_dim, block_dim, forloop) ────────────┐ │    │
│  └─────────────────────────────────────────────────────────┐ │ │    │
│                                                            │ │ │    │
│  Level 2: Graph Policy (constrained by Level 1)            │ │ │    │
│  ┌─────────────────────────────────────────────────────┐   │ │ │    │
│  │ µGraph actions ─▶ C++ Search ─▶ GPU Verify ─▶ reward│◀──┘ │ │    │
│  └─────────────────────────────────────────────────────┘     │ │    │
│         ▲                                                    │ │    │
│         └──── µGraph features (from C++) ◀───────────────────┘ │    │
│                                                                │    │
│                      policy update (RLlib)  ◀──────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Quick Start

```bash
# Run integration tests (no GPU required)
python scripts/test_rl_integration.py

# Test locally (no GPU required)
python scripts/train_rl_kernel_search.py --mode local --test-episodes 10

# Train with Ray/RLlib (requires GPU for verification)
python scripts/train_rl_kernel_search.py --mode train \
    --algorithm PPO \
    --num-workers 8 \
    --max-iterations 1000

# Search with trained policy
python scripts/train_rl_kernel_search.py --mode search \
    --checkpoint /path/to/checkpoint \
    --target-graph examples/matmul.json
```

### Python API

```python
from yirage.rl import YiRageSearchEnv, EnvConfig, train_rl_search

# Create environment
env_config = EnvConfig(
    target_graph_json=target_graph,
    backend="cuda",
    num_gpus=4,
)

# Option 1: Use as Gymnasium environment
env = YiRageSearchEnv(vars(env_config))
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)

# Option 2: Train with RLlib
from yirage.rl import TrainingConfig

config = TrainingConfig(
    algorithm="PPO",
    num_workers=8,
    max_iterations=500,
)
results = train_rl_search(config)
```

### Hierarchical Search

```python
from yirage.rl.search import (
    HardwareConfig, SearchSpaceConstraints,
    ConstrainedGraphActionSpace, HierarchicalSearchEnv
)

# Level 1: Configure hardware parameters
config = HardwareConfig(
    grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
    block_dim_x=128, block_dim_y=1, block_dim_z=1,
    forloop_range=16,
    shared_memory_size=49152
)

# Level 2: Get constraints for graph search
constraints = SearchSpaceConstraints(config)
print(f"Valid imaps: {len(constraints.valid_imaps)}")
print(f"Max operators: {constraints.max_operators}")

# Create constrained graph action space
graph_space = ConstrainedGraphActionSpace(constraints)
```

### µGraph Feature Extraction

```python
from yirage.rl.features import MuGraphFeature, FeatureProcessor

# Features extracted from C++ layer (or simulated JSON)
features = MuGraphFeature.from_json(features_json)
print(f"Operators: {len(features.operators)}")
print(f"Graph depth: {features.graph_depth}")

# Process for neural network input
processor = FeatureProcessor()
processed = processor.process(features)
# node_features: (num_nodes, 16)
# edge_index: (2, num_edges)
# global_features: (48,)
```

### Key Features

- **Hierarchical Search**: Level 1 (config) constrains Level 2 (µGraph)
- **Complete Closed Loop**: RL decisions → C++ search → GPU verification → reward
- **AccelForge Pre-screening**: virtual-hardware latency/energy/area/power modeling before physical profiling
- **µGraph Feature Extraction**: Rich features from C++ layer for RL model input
- **Multi-objective Reward**: Balances validity, performance, efficiency, exploration
- **Ray Integration**: Distributed CPU workers + GPU verification
- **Action Masking**: Prevents invalid actions based on search state
- **Model Persistence**: Save/load trained policies, export to ONNX

### Hardware-Aware Training

```python
from yirage.rl.hardware import detect_hardware, get_optimal_config
from yirage.rl.training import GRPOConfig, GRPOTrainer

# Auto-detect hardware
hardware = detect_hardware()
print(f"Detected: {hardware.backend} - {hardware.device_name}")
print(f"Peak FP16: {hardware.peak_tflops_fp16} TFLOPS")

# Get optimal config for workload
config = get_optimal_config(hardware, workload)

# Train with GRPO (supports LoRA fine-tuning)
grpo_config = GRPOConfig(
    group_size=8,
    learning_rate=1e-4,
    use_lora=True,
    lora_rank=16,
)
```

### AccelForge Hardware Co-design

YiRage can use [AccelForge](https://github.com/Accelergy-Project/accelforge)
as a virtual hardware oracle for accelerator design-space exploration and
kernel candidate pre-screening:

```bash
pip install "yirage[accelforge]"
```

See [YiRage × AccelForge Quick Start](docs/accelforge_quick_start.md) for
availability diagnostics, µGraph workload conversion, multi-objective metrics,
pre-screening, and Pareto-front examples.

### LLM Fine-tuning with TRL

```python
from yirage.rl.training import FineTuningConfig, MuGraphPolicyTrainer

# Configure fine-tuning with TRL
config = FineTuningConfig(
    strategy="dpo",  # sft, dpo, grpo, ppo
    model_name_or_path="meta-llama/Llama-2-7b-hf",
    use_lora=True,
    use_4bit=True,  # QLoRA
    lora_r=16,
)

# Train policy model
trainer = MuGraphPolicyTrainer(config)
trainer.train(train_data)

# Generate optimal configs
configs = trainer.generate_config(target_graph, hardware)
```

### Universal Compute Optimization 

Optimize **any compute task** on **any hardware** at **any cluster scale** with a single function call:

```python
from yirage.rl.cluster import optimize_any_task

# Optimize with one line
result = optimize_any_task(
    {"type": "attention", "batch": 32, "seq_len": 2048, "num_heads": 32},
    cluster_spec={"type": "multi_node", "num_nodes": 4, "gpus_per_node": 8}
)

print(f"Strategy: {result.result.parallelism_strategy}")  # e.g., "tensor_parallel_8"
print(f"Latency: {result.result.estimated_latency_ms:.2f} ms")
print(f"Throughput: {result.result.estimated_throughput_tps:.1f} samples/sec")

# Get kernel configs for YiRage search
for op_id, config in result.kernel_configs.items():
    print(f"{op_id}: {config}")
```

**Device Registry (25+ Pre-defined Devices)**:

```python
from yirage.rl.cluster import (
    ClusterTopology, DeviceRegistry, get_device_spec, register_custom_device
)

# Create heterogeneous cluster from registry
cluster = ClusterTopology.create_from_registry([
    "H100_SXM:4",      # 4x NVIDIA H100
    "MI300X:2",        # 2x AMD MI300X
    "TPUv4:2",         # 2x Google TPU v4
    "Ascend910B:2",    # 2x Huawei Ascend
])

# Register custom hardware
register_custom_device("MyAccelerator", {
    "device_type": "custom",
    "compute_units": 128,
    "peak_tflops_fp16": 500.0,
    "memory_gb": 64.0,
    "memory_bandwidth_gbps": 2000.0,
})
```

**Supported Device Types**:

| Category | Devices |
|----------|---------|
| **NVIDIA GPU** | H100, A100, V100, RTX 4090, RTX 3090 |
| **AMD GPU** | MI300X, MI250X |
| **Intel** | Max 1550 (XPU) |
| **Google** | TPU v4, TPU v5e |
| **Huawei** | Ascend 910B, Ascend 910, Ascend 310 |
| **AWS** | Trainium2, Inferentia2 |
| **Apple** | M2 Ultra, M3 Max (MPS) |
| **MetaX** | C500 (MACA) |
| **CPU** | EPYC 9654, Xeon 8480 |
| **FPGA** | Alveo U280 |
| **Custom** | User-defined devices |

Key features:
- **Any Task**: MatMul, Attention, MLP, Transformer, or custom graphs
- **Any Hardware**: CPU, GPU, NPU, TPU, FPGA, or custom accelerators
- **Any Scale**: Single device to multi-node clusters
- **Simulation-based**: Accurate communication modeling without real cluster
- **µGraph Integration**: Generates search space for YiRage kernel optimization
- **Device Registry**: 25+ pre-defined devices with full specs

### Design Documents

- [RL Closed-Loop Design](docs/design/rl_yirage_closed_loop_design.md)
- [Hierarchical Search Design](docs/design/hierarchical_search_design.md)
- [Feature Extraction Design](docs/design/rl_model_feature_design.md)
- [Hardware-Aware Training Design](docs/design/hardware_aware_training_design.md)
- [Universal Optimization Design](docs/design/universal_optimization_design.md)

---

## 🔥 COMET: Compound Operations with Explicit Collectives

YiRage integrates the **COMET** framework for modeling and optimizing compound operation dataflows with explicit collective communication, based on the research paper:

> *"COMET: A Framework for Modeling Compound Operation Dataflows with Explicit Collectives"* (Negi et al.)

### Key Features

- **Compound Operations**: Fused execution of GEMM-Softmax, GEMM-LayerNorm, Self-Attention, Gated MLP
- **Explicit Collectives**: AllReduce, AllGather, ReduceScatter, Broadcast with accurate cost modeling
- **Data Staging Model**: Ramp-up/steady-state/ramp-down phases for memory hierarchy
- **Scheduling Strategies**: Sequential, Pipelined, Parallel execution modes
- **Energy & Latency Estimation**: Detailed breakdown for optimization decisions

### Compound Operations API

```python
import yirage as yr

# Create kernel graph
graph = yr.new_kernel_graph()

# GEMM-Softmax fusion (reduces DRAM traffic by keeping intermediate on-chip)
A = graph.new_input(dims=(1024, 512), dtype=yr.float16)
B = graph.new_input(dims=(512, 1024), dtype=yr.float16)
result = graph.gemm_softmax(A, B, dim=-1)

# GEMM-LayerNorm fusion
result_ln = graph.gemm_layernorm(A, B, normalized_shape=(1024,))

# Self-Attention (FlashAttention-style fusion)
Q = graph.new_input(dims=(8, 1024, 64), dtype=yr.float16)  # [H, S, D]
K = graph.new_input(dims=(8, 64, 1024), dtype=yr.float16)  # [H, D, S] (transposed)
V = graph.new_input(dims=(8, 1024, 64), dtype=yr.float16)  # [H, S, D]
attn_out = graph.self_attention(Q, K, V)

# Gated MLP (LLM-style with SiLU activation)
X = graph.new_input(dims=(8, 1024, 4096), dtype=yr.float16)
W_gate = graph.new_input(dims=(4096, 11008), dtype=yr.float16)
W_up = graph.new_input(dims=(4096, 11008), dtype=yr.float16)
W_down = graph.new_input(dims=(11008, 4096), dtype=yr.float16)
mlp_out = graph.gated_mlp(X, W_gate, W_up, W_down, activation="silu")

# RMSNorm + Linear (common in attention QKV projection)
norm_out = graph.rms_norm_linear(X, W_gate, normalized_shape=(4096,))

graph.mark_output(result)
optimized = graph.superoptimize(backend="cuda")
```

### COMET Cost Model

```python
from yirage.rl.cluster.simulator import (
    COMETCostModel, COMETHardwareConfig,
    SchedulingStrategy, MemoryLevel, CommunicationType
)

# Create cost model with hardware config
hw_config = COMETHardwareConfig(
    dram_bandwidth_gbps=900.0,      # HBM2e
    global_buffer_bandwidth_gbps=3000.0,  # On-chip L2
    num_compute_units=108,          # SMs on A100
    peak_tflops_fp16=312.0,
)
cost_model = COMETCostModel(hw_config=hw_config)

# Estimate compound operation latency and energy
latency, energy = cost_model.estimate_compound_operation(
    op_name="gemm_softmax",
    input_shapes=[(2048, 1024), (1024, 2048)],
    dtype_bytes=2,  # FP16
    num_devices=4,
    strategy=SchedulingStrategy.PIPELINED,
)

print(f"Total latency: {latency.total_latency_ms:.3f} ms")
print(f"  - Compute: {latency.compute_latency_ms:.3f} ms")
print(f"  - Memory: {latency.total_memory_latency_ms:.3f} ms")
print(f"  - Collective: {latency.collective_latency_ms:.3f} ms")
print(f"Total energy: {energy.total_energy_mj:.3f} mJ")

# Compare distributed variants (local vs distributed execution)
results = cost_model.compare_distributed_variants(
    op_name="gemm_softmax",
    input_shapes=[(4096, 2048), (2048, 4096)],
    num_devices=8,
)
print(f"Speedup with distribution: {results['speedup']:.2f}x")
```

### Collective Communication Cost Model

```python
from yirage.rl.cluster.simulator import CommunicationModel, CommunicationType

comm_model = CommunicationModel()

# Ring AllReduce latency (Eq. 3-4 from COMET paper)
latency_ms = comm_model.all_reduce_time_ms(
    size_bytes=100 * 1024 * 1024,  # 100 MB
    num_devices=8,
    bandwidth_gbps=200.0,  # NVLink
    latency_us=1.0,
    algorithm="ring",
)
print(f"AllReduce latency: {latency_ms:.3f} ms")

# AllGather and ReduceScatter
gather_time = comm_model.all_gather_time_ms(
    size_bytes=50 * 1024 * 1024,
    num_devices=8,
    bandwidth_gbps=200.0,
    latency_us=1.0,
)
print(f"AllGather latency: {gather_time:.3f} ms")
```

### Latency Breakdown (COMET Equations)

The cost model implements the COMET paper equations:

| Equation | Description | Formula |
|----------|-------------|---------|
| **Eq. 1** | Memory Transaction | `MemLat(T) = DV / BW` |
| **Eq. 2** | Data Staging | `TotalMem = RampUp + Steady + RampDown` |
| **Eq. 3-4** | Ring Collective | `CollLat = 2(n-1)/n × size / bw` |
| **Eq. 5-7** | Scheduling | `Stall = CS + OS + CF` |

Where:
- **DV**: Data Volume, **BW**: Bandwidth
- **CS**: Compulsory Stall (data dependency)
- **OS**: Optional Stall (resource blocking)
- **CF**: Conflict Stall (resource contention)

### COMET Search Strategy

YiRage provides a complete search strategy for COMET compound operations:

```python
from yirage.search import (
    COMETSearchStrategy,
    get_backend_config,
    detect_compound_patterns,
    optimize_compound_graph,
)

# Auto-detect compound patterns in a graph
op_types = ["matmul", "exp", "reduction", "div", "matmul"]  # Self-attention
patterns = detect_compound_patterns(op_types)
print(f"Found {len(patterns)} compound patterns: {[p.op_type.name for p in patterns]}")

# Get backend-specific configuration (15 hardware profiles)
config = get_backend_config("cuda", "a100")  # NVIDIA A100
# Or: get_backend_config("rocm", "mi300x")   # AMD MI300X
# Or: get_backend_config("tpu", "v5e")       # Google TPU v5e
# Or: get_backend_config("ascend", "910b")   # Huawei Ascend

# Run COMET search to find optimal configuration
strategy = COMETSearchStrategy(config)
result = strategy.search(
    op_types=op_types,
    problem_dims={"M": 4096, "K": 4096, "N": 4096}
)

print(f"Best tile config: M={result.tile_config.tile_m}, N={result.tile_config.tile_n}")
print(f"Scheduling: {result.scheduling.name}")
print(f"Estimated latency: {result.latency_ns:.2f} ns")
```

### Backend Hardware Profiles

| Backend | Variant | DRAM BW (GB/s) | Peak TFLOPS | Tile Sizes |
|---------|---------|----------------|-------------|------------|
| **CUDA** | H100 | 3350 | 989 | 64, 128, 256 |
| **CUDA** | A100 | 2039 | 312 | 64, 128, 256 |
| **CUDA** | V100 | 900 | 125 | 32, 64, 128 |
| **ROCm** | MI300X | 5300 | 1307 | 64, 128, 256 |
| **ROCm** | MI250X | 3200 | 383 | 64, 128, 256 |
| **XPU** | Ponte Vecchio | 3200 | 420 | 32, 64, 128, 256 |
| **Ascend** | 910B | 1600 | 320 | 64, 128, 256 |
| **TPU** | v5e | 1600 | 197 | 128, 256, 512 |
| **TPU** | v4 | 1200 | 275 | 128, 256, 512 |
| **MACA** | MXC500 | 2000 | 256 | 64, 128, 256 |
| **MPS** | M3 Max | 400 | 14.2 | 32, 64, 128 |
| **MPS** | M2 Ultra | 800 | 27.2 | 32, 64, 128 |
| **CPU** | Xeon | 200 | 4.0 | 32, 64, 128, 256 |
| **CPU** | EPYC | 460 | 5.0 | 32, 64, 128, 256 |
| **FPGA** | Alveo | 77 | 4.0 | 16, 32, 64, 128 |

---

## 🚀 Deep Ray Integration

YiRage provides production-grade distributed optimization with deep Ray integration:

### Features

| Feature | Description |
|---------|-------------|
| **C++ Binding** | Direct `search_partition()` API via Cython for native performance |
| **Object Store** | `ray.put()` for efficient large graph data sharing |
| **Placement Groups** | GPU affinity with PACK/SPREAD strategies for NVLink |
| **Fault Tolerance** | Exponential backoff retry + checkpoint/restore |
| **Collective Ops** | Efficient all-reduce for gradient synchronization |

### Quick Start

```python
from yirage.distributed import (
    RayDeepIntegration,
    DeepIntegrationConfig,
    GPUPlacementConfig,
    RetryConfig,
    RetryStrategy,
)

# Configure distributed optimization
config = DeepIntegrationConfig(
    num_workers=8,
    gpu_placement=GPUPlacementConfig(
        gpus_per_worker=1,
        strategy="PACK",  # NVLink locality
    ),
    retry=RetryConfig(
        strategy=RetryStrategy.EXPONENTIAL,
        max_retries=5,
    ),
    use_object_store=True,
)

# Create engine and optimize
engine = RayDeepIntegration(config)
result = engine.optimize(
    graph={"type": "matmul", "input_shapes": [[1024, 2048], [2048, 4096]]},
    search_space={"grid_dims": [(1,1,1), (2,1,1), (4,1,1)], "block_dims": [(128,1,1), (256,1,1)]},
)

print(f"Best latency: {result['best_latency_ms']:.3f} ms")
print(f"Workers used: {result['num_workers']}")
```

### All-Reduce for Gradients

```python
# Distributed gradient synchronization
gradients = [{"layer1": 0.1}, {"layer1": 0.3}, {"layer1": 0.2}, {"layer1": 0.4}]
reduced = engine.all_reduce_gradients(gradients, reduce_op="mean")
# reduced["layer1"] = 0.25
```

### Run Demo

```bash
python examples/cluster/deep_ray_integration_demo.py
```

---

## 📚 Documentation

- **[Quick Start](docs/quickstart.md)** - Get started in 5 minutes
- **[API Reference](docs/api.md)** - Complete API documentation
- **[Backend Guide](docs/mpk/backend_usage.md)** - Backend usage and configuration
- **[Architecture Design](docs/mpk/multi_backend_design.md)** - System design

### Hardware Device Management

| Module | Description |
|--------|-------------|
| `yirage.hardware` | **Hardware Registry** — register, query, and auto-detect chip architectures at runtime |
| `yirage.hardware.ChipArchitecture` | Unified dataclass for chip specs (memory, compute, features) |
| `yirage.hardware.HardwareRegistry` | Thread-safe singleton with register/query/import/export |
| `yirage.hardware.detect_current_chip()` | Auto-detect NVIDIA/AMD/Ascend/MetaX/Apple hardware |

### Hardware-Specific Guides

| Platform | Guide | Description |
|----------|-------|-------------|
| **Huawei Ascend NPU** | [Installation Guide](docs/ascend_installation_guide.md) | Complete setup, build, and test instructions |
| **Huawei Ascend NPU** | [Quick Start](docs/ascend_quick_start.md) | Quick API usage examples |
| **MetaX MACA GPU** | [Quick Start](docs/maca_quick_start.md) | MetaX GPU integration 🆕 |

- **[Contributing](CONTRIBUTING.md)** - Contribution guidelines

---

## 🎓 Examples

### Run Benchmarks

```bash
# MPS backend (Apple Silicon)
python benchmark/baselines/pytorch/gated_mlp.py -b 8 --backend mps

# CUDA backend (NVIDIA GPU)
python benchmark/baselines/pytorch/gated_mlp.py -b 8 --backend cuda

# CPU backend
python benchmark/baselines/pytorch/gated_mlp.py -b 8 --backend cpu

# Ascend backend (Huawei NPU) - requires CANN + torch_npu
python benchmark/baselines/pytorch/gated_mlp.py -b 8 --backend ascend

# MACA backend (MetaX GPU) - requires MACA SDK
python benchmark/baselines/pytorch/gated_mlp.py -b 8 --backend maca
```

### Backend Selection

```python
import yirage as yr

# Method 1: Direct specification
mpk = yr.PersistentKernel(
    backend="cpu",
    mode="decode",
    world_size=1,
    mpi_rank=0,
)

# Method 2: With fallback
mpk = yr.PersistentKernel(
    backend="cuda",
    fallback_backends=["mps", "cpu"],  # Auto fallback
    mode="decode",
    world_size=1,
    mpi_rank=0,
)

# Method 3: Query and select
backends = yr.get_available_backends()
best_backend = backends[0]  # Use first available
```

---

## 🏆 Submit Your First Kernel

YiRage lets you **validate optimization results quickly** and then submit kernels to the [gpu-mode](https://github.com/gpu-mode) leaderboard via **popcorn-cli** for community benchmarking on real hardware (A100, H100, etc.).

### Quick Validation Workflow

```bash
# 1. Validate your kernel locally (no GPU required)
python examples/submission.py --validate

# Expected output:
# ✅  NumPy kernel: shape=(256, 256), dtype=uint8
# ⏱   NumPy throughput: 0.312 ms/frame  (0.21 Gpix/s)
# ✅  Torch kernel: shape=(4, 1, 256, 256), device=cpu
# ⏱   Torch throughput: 1.234 ms/batch
# ✅  All validation steps completed.
```

### Submit to Leaderboard (4 steps)

#### 1. Install popcorn-cli

```bash
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
```

#### 2. Register your account

```bash
popcorn-cli register discord
```

#### 3. Set up your project

```bash
# Configure your project with a working example and optional agent skills
popcorn-cli setup
```

#### 4. Submit your kernel

```bash
# Submit the included grayscale example to the grayscale_v2 leaderboard on an A100
popcorn-cli submit --gpu A100 --leaderboard grayscale_v2 --mode leaderboard examples/submission.py
```

> **Tip**: Replace `examples/submission.py` with any file that exports a `solution(input_tensor)` function.  
> See the [popcorn-cli repo](https://github.com/gpu-mode/popcorn-cli) for the full list of available leaderboards and GPUs.

### Writing Your Own Submission

A valid `submission.py` must export a `solution` function with the leaderboard's expected signature:

```python
import torch

def solution(input_tensor: torch.Tensor) -> torch.Tensor:
    """Your optimized kernel implementation."""
    # Example: YiRage-optimized grayscale conversion
    coeffs = torch.tensor([0.299, 0.587, 0.114], device=input_tensor.device)
    return (input_tensor * coeffs[None, :, None, None]).sum(dim=1, keepdim=True)
```

For a complete example with local validation, benchmarking, and YiRage superoptimizer integration, see [`examples/submission.py`](examples/submission.py).

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding a New Backend

1. Implement `BackendInterface`
2. Create `{Backend}KernelConfig`
3. Implement `{Backend}Optimizer`
4. Create `{Backend}SearchStrategy` (optional)
5. Update CMake configuration

See [Ascend Implementation Guide](docs/ascend_implementation_guide.md) for a complete example.

---

## 📄 License

YiRage is licensed under the Apache License 2.0.

**Copyright**:
- YiRage Multi-Backend Extensions: Copyright 2025 Chen Xingqiang
- Original Mirage: Copyright 2023-2024 Carnegie Mellon University

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

---

## 📚 Citation

```bibtex
@software{yirage2025,
  title={YiRage: Yield Revolutionary AGile Engine for Multi-Backend LLM Inference},
  author={Chen, Xingqiang},
  year={2025},
  note={Multi-backend extension for LLM inference optimization},
  url={https://github.com/chenxingqiang/YiRage}
}

```

---

## 🙏 Acknowledgments

YiRage acknowledges CMU Mirage and the broader open-source systems and compiler communities whose work makes multi-backend optimization possible. Comprehensive third-party attribution details are maintained in [NOTICE](NOTICE).

---

## 📞 Contact

- **Issues**: [GitHub Issues](https://github.com/chenxingqiang/YiRage/issues)
- **Author**: Chen Xingqiang
- **Email**: joy6677@outlook.com

---

**YiRage** - Yielding Maximum Performance Across All Hardware 🚀

Copyright 2025 Chen Xingqiang | Apache License 2.0
