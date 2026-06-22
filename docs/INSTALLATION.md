# YiRage Installation Guide

## Native runtime (required)

The Python package **always** depends on the native library `libyirage_runtime` (linked into `yirage.core`). If the extension fails to load, `import yirage` raises `ImportError`. Optional pip extras only add **Python** dependencies (e.g. PyTorch); they are not a substitute for building or shipping the native runtime.

## Quick Start

### From PyPI

```bash
# Default install (wheel includes prebuilt native code when available)
pip install yirage

# Optional Python extras (CUDA-/ROCm-/MPS-oriented stacks, etc.)
pip install yirage[cuda]
pip install yirage[all]
```

### From Source (recommended when developing or selecting backends)

```bash
git clone https://github.com/chenxingqiang/YiRage.git
cd YiRage

# Auto-detect hardware and install
pip install -e .

# Or specify backend explicitly
YIRAGE_BACKEND=cuda pip install -e .     # NVIDIA GPU
YIRAGE_BACKEND=rocm pip install -e .     # AMD GPU
YIRAGE_BACKEND=mps pip install -e .      # Apple Silicon
YIRAGE_BACKEND=ascend pip install -e .   # Huawei NPU
YIRAGE_BACKEND=maca pip install -e .     # MetaX GPU
YIRAGE_BACKEND=cpu pip install -e .      # CPU backend (full native build)
```

## Backend-Specific Installation

### NVIDIA CUDA

```bash
# Prerequisites: CUDA Toolkit 11.8+ or 12.x
# https://developer.nvidia.com/cuda-toolkit

# From PyPI (optional Python extras; native wheel still required)
pip install yirage[cuda]

# From source (full C++ backend)
YIRAGE_BACKEND=cuda pip install -e .

# With specific CUDA path
CUDA_HOME=/usr/local/cuda-12.1 YIRAGE_BACKEND=cuda pip install -e .
```

### AMD ROCm

```bash
# Prerequisites: ROCm 5.x or 6.x
# https://rocm.docs.amd.com/

# Install ROCm PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

# Then install YiRage
YIRAGE_BACKEND=rocm pip install -e .

# With specific ROCm path
ROCM_PATH=/opt/rocm-6.0 YIRAGE_BACKEND=rocm pip install -e .
```

### Apple Silicon (MPS)

```bash
# Prerequisites: macOS 12.3+ with Apple Silicon (M1/M2/M3)

# From PyPI
pip install yirage[mps]

# From source (auto-detected on Apple Silicon)
pip install -e .

# Or explicitly
YIRAGE_BACKEND=mps pip install -e .

# Optional: OpenMP for parallel search
brew install libomp
```

### Huawei Ascend NPU

```bash
# Prerequisites: CANN 7.0+
# https://www.hiascend.com/

# Load Ascend environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Install torch_npu
pip install torch_npu

# Install YiRage
YIRAGE_BACKEND=ascend pip install -e .
```

### MetaX MACA

```bash
# Prerequisites: MACA SDK
# Contact MetaX for SDK access

export MACA_PATH=/opt/maca
YIRAGE_BACKEND=maca pip install -e .
```

### Intel XPU

```bash
# Prerequisites: Intel oneAPI
# https://www.intel.com/oneapi

# Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch

YIRAGE_BACKEND=xpu pip install -e .
```

### CPU backend

Use this when you target CPU execution only; the **full** C++/Rust/Cython build still runs.

```bash
# From PyPI (when a wheel exists for your platform)
pip install yirage

# From source: select CPU in generated config.cmake
YIRAGE_BACKEND=cpu pip install -e .
```

## Installation Options

### pip install Options

```bash
# Editable install (for development)
pip install -e .

# With extras
pip install -e ".[dev]"        # Development tools
pip install -e ".[cuda]"       # CUDA dependencies
pip install -e ".[llm]"        # LLM inference deps
pip install -e ".[all]"        # Everything
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `YIRAGE_BACKEND` | Backend selection | `cuda`, `mps`, `cpu` |
| `USE_CUDA` | Enable CUDA | `ON` / `OFF` |
| `USE_ROCM` | Enable ROCm | `ON` / `OFF` |
| `USE_MPS` | Enable MPS | `ON` / `OFF` |
| `USE_ASCEND` | Enable Ascend | `ON` / `OFF` |
| `USE_MACA` | Enable MACA | `ON` / `OFF` |
| `CUDA_HOME` | CUDA path | `/usr/local/cuda` |
| `ROCM_PATH` | ROCm path | `/opt/rocm` |
| `ASCEND_HOME` | Ascend path | `/usr/local/Ascend/...` |
| `MACA_PATH` | MACA path | `/opt/maca` |
| `SKIP_BUILD` | Skip C++ rebuild | `1` |

### Multiple Backends

```bash
# Enable multiple backends
YIRAGE_BACKEND=cuda,cpu pip install -e .

# Or use individual flags
USE_CUDA=ON USE_CPU=ON pip install -e .
```

## Development Installation

```bash
# Clone repository
git clone https://github.com/chenxingqiang/YiRage.git
cd YiRage

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/python -v

# Code formatting
black python/
ruff check python/
```

## Verify Installation

```python
import yirage as yr

# Check version
print(f"YiRage version: {yr.__version__}")

# List available backends
backends = yr.get_available_backends()
print(f"Available backends: {backends}")

# Check specific backend
if yr.is_backend_available('cuda'):
    print("CUDA backend ready!")
elif yr.is_backend_available('mps'):
    print("MPS backend ready!")
```

## Troubleshooting

### libz3.dylib not found (macOS)

```bash
# Set library path
export DYLD_LIBRARY_PATH="$(python -c 'import z3, os; print(os.path.dirname(z3.__file__) + \"/lib\"')"):$DYLD_LIBRARY_PATH"
```

### CMake not found

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt install cmake

# Or via pip
pip install cmake
```

### CUDA not detected

```bash
# Verify CUDA
nvidia-smi
nvcc --version

# Set CUDA_HOME
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### ROCm not detected

```bash
# Verify ROCm
rocm-smi
hipcc --version

# Set ROCm path
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
```

## Docker Installation

```bash
# CUDA
docker build -t yirage:cuda -f docker/Dockerfile.cuda .
docker run --gpus all -it yirage:cuda

# CPU
docker build -t yirage:cpu -f docker/Dockerfile .
docker run -it yirage:cpu
```

## Platform Support

| Platform | Backends | Status |
|----------|----------|--------|
| Linux x86_64 | CUDA, ROCm, CPU | ✅ Full support |
| Linux aarch64 | Ascend, CPU | ✅ Full support |
| macOS arm64 | MPS, CPU | ✅ Full support |
| macOS x86_64 | CPU | ✅ CPU only |
| Windows x86_64 | CUDA, CPU | 🔧 Experimental |
