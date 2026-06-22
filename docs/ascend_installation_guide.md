# YiRage Ascend NPU Installation Guide

Complete guide for installing, configuring, and testing YiRage on Huawei Ascend NPU systems.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Building YiRage](#building-yirage)
4. [Installation](#installation)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| NPU | Ascend 910/910B/910B2/310P |
| Memory | 16GB+ system RAM |
| HBM | 32GB+ (64GB recommended) |

### Software Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| OS | Ubuntu 20.04/22.04 (aarch64) | ARM64 architecture |
| CANN | 6.0+ (8.0+ recommended) | Huawei AI development toolkit |
| Python | 3.8-3.11 | 3.11 recommended |
| PyTorch | 2.1-2.6 | Must match torch_npu version |
| torch_npu | Matching PyTorch version | Ascend PyTorch adapter |
| triton-ascend | 3.0+ | Triton compiler for Ascend |

---

## Environment Setup

### Step 1: Verify Ascend Hardware

```bash
# Check NPU availability
npu-smi info

# Expected output shows NPU device info:
# +---------------------------+---------------+----------------------------------------------------+
# | NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
# | 4     910B2               | OK            | 113.6       54                0    / 0             |
# +===========================+===============+====================================================+
```

### Step 2: Load CANN Environment

```bash
# Check CANN installation
ls /usr/local/Ascend/

# Load CANN toolkit environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Load driver libraries (CRITICAL - often missed!)
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# Load ATB (Ascend Transformer Boost) if available
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

# Verify CANN version
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg | head -5
```

### Step 3: Verify Python Environment

```bash
# Check Python version
python3 --version  # Should be 3.8-3.11

# Check PyTorch
python3 -c "import torch; print('PyTorch:', torch.__version__)"

# Check torch_npu
python3 -c "import torch_npu; print('torch_npu:', torch_npu.__version__)"

# Check NPU availability in PyTorch
python3 -c "import torch; print('NPU available:', torch.npu.is_available())"
python3 -c "import torch; print('NPU count:', torch.npu.device_count())"
```

### Step 4: Install triton-ascend

```bash
# Install triton-ascend (Triton compiler for Ascend NPU)
pip3 install triton-ascend

# Verify installation
python3 -c "import triton; print('Triton:', triton.__version__)"
```

### Step 5: Install Build Dependencies

```bash
# Install required Python packages
pip3 install z3-solver graphviz cython

# Install CMake >= 3.24 (if system version is too old)
pip3 install "cmake>=3.24"

# Verify CMake version
cmake --version  # Should be 3.24+

# Install Rust (required for formal verifier)
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env

# Verify Rust installation
rustc --version
cargo --version
```

---

## Building YiRage

### Step 1: Clone or Upload YiRage

```bash
# Option A: Clone from GitHub (if network allows)
git clone https://github.com/chenxingqiang/YiRage.git --depth 1
cd YiRage

# Option B: Upload from local machine via rsync
# Run this on your LOCAL machine:
rsync -avz --progress \
  --exclude '.git' \
  --exclude 'build' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.eggs' \
  --exclude '*.egg-info' \
  -e "ssh -p <PORT> -o StrictHostKeyChecking=no" \
  /path/to/YiRage/ \
  root@<HOST>:~/YiRage/
```

### Step 2: Configure for Ascend

```bash
cd ~/YiRage

# Use the Ascend configuration file
cp config.ascend.cmake config.cmake

# Verify configuration
cat config.cmake
# Should show: set(USE_ASCEND ON)
```

### Step 3: Configure Z3

```bash
# Create Z3 CMake configuration
mkdir -p deps/z3/build
Z3_BASE=$(python3 -c "import z3; import os; print(os.path.dirname(z3.__file__))")

cat > deps/z3/build/z3-config.cmake << EOF
set(Z3_FOUND TRUE)
set(Z3_VERSION "4.15.4")
set(Z3_INCLUDE_DIRS "${Z3_BASE}/include")
set(Z3_LIBRARIES "${Z3_BASE}/lib/libz3.so")
set(Z3_CXX_INCLUDE_DIRS "${Z3_BASE}/include")

if(NOT TARGET z3::libz3)
  add_library(z3::libz3 SHARED IMPORTED)
  set_target_properties(z3::libz3 PROPERTIES
    IMPORTED_LOCATION "${Z3_BASE}/lib/libz3.so"
    INTERFACE_INCLUDE_DIRECTORIES "${Z3_BASE}/include"
  )
endif()
EOF

cat > deps/z3/build/Z3Config.cmake << EOF
include("\${CMAKE_CURRENT_LIST_DIR}/z3-config.cmake")
EOF

echo "✅ Z3 configured at: $Z3_BASE"
```

### Step 4: Configure CUTLASS Stub

```bash
# Create CUTLASS stub (for non-CUDA builds)
mkdir -p deps/cutlass/include/cutlass/detail

cat > deps/cutlass/include/cutlass/cutlass.h << 'EOF'
#pragma once
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE inline
#define CUTLASS_DEVICE inline
#endif
namespace cutlass {}
EOF

cat > deps/cutlass/include/cutlass/detail/helper_macros.hpp << 'EOF'
#pragma once
#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE inline
#define CUTLASS_DEVICE inline
#endif
EOF

echo "✅ CUTLASS stub configured"
```

### Step 5: Configure JSON Library

```bash
# Download nlohmann/json if not present
if [ ! -f "deps/json/include/nlohmann/json.hpp" ]; then
    mkdir -p deps/json/include/nlohmann
    curl -sL https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp \
        -o deps/json/include/nlohmann/json.hpp
    echo "✅ JSON library downloaded"
else
    echo "✅ JSON library already present"
fi
```

### Step 6: Run CMake

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DZ3_DIR=$HOME/YiRage/deps/z3/build

# Expected output:
# -- Ascend backend enabled (requires CANN + torch_npu at runtime)
# -- Using Ascend fingerprint (parallel search enabled)
# -- Configuring done
```

### Step 7: Build

```bash
# Build with all available cores
make -j$(nproc)

# Expected output:
# [100%] Built target yirage_runtime
```

---

## Installation

### Install Python Package

```bash
cd ~/YiRage

# Set Z3 directory for pip
export Z3_DIR=$HOME/YiRage/deps/z3/build

# Install in editable mode
pip3 install -e . --no-build-isolation

# Verify installation
python3 -c "import yirage; print('YiRage version:', yirage.__version__)"
```

---

## Testing

### Quick Verification

```bash
# Load environment (add to ~/.bashrc for persistence)
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

cd ~/YiRage

# Run basic Ascend test
python3 tests/ascend/test_triton_integration.py
```

### NPU Compute Test

```python
#!/usr/bin/env python3
"""Quick NPU compute test"""
import torch
import torch_npu

device = torch.device("npu:0")
print(f"NPU Device: {device}")
print(f"NPU Count: {torch.npu.device_count()}")

# MatMul test
a = torch.randn(1024, 1024, dtype=torch.float16, device=device)
b = torch.randn(1024, 1024, dtype=torch.float16, device=device)
c = torch.matmul(a, b)
torch.npu.synchronize()

print(f"✅ MatMul result shape: {c.shape}")
print(f"✅ NPU compute working!")
```

### YiRage Graph Test

```python
#!/usr/bin/env python3
"""YiRage graph creation test"""
import yirage as yr

# Create kernel graph
graph = yr.new_kernel_graph()
X = graph.new_input(dims=(8, 4096), dtype=yr.float16)
W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
O = graph.matmul(X, W)
graph.mark_output(O)

print(f"✅ Graph created: matmul({X.dim}, {W.dim}) -> {O.dim}")
```

### Superoptimize Test

```bash
# Run full optimization search test
python3 tests/ascend/test_superoptimize.py
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. `libhccl.so: cannot open shared object file`

**Cause**: CANN environment not loaded properly.

**Solution**:
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 2. `libascend_hal.so: cannot open shared object file`

**Cause**: Driver libraries not in LD_LIBRARY_PATH.

**Solution**:
```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
```

#### 3. `npu-smi: error while loading shared libraries`

**Cause**: Driver libraries not found.

**Solution**: Same as above, ensure driver path is in LD_LIBRARY_PATH.

#### 4. CMake version too old

**Cause**: System CMake < 3.24.

**Solution**:
```bash
pip3 install "cmake>=3.24"
hash -r  # Refresh PATH cache
cmake --version
```

#### 5. `No module named 'triton'`

**Cause**: triton-ascend not installed.

**Solution**:
```bash
pip3 install triton-ascend
```

#### 6. `undefined symbol: _ZN6yirage...`

**Cause**: Library not compiled correctly or fingerprint mode mismatch.

**Solution**: Rebuild with proper configuration:
```bash
cd build && rm -rf * && cmake .. -DZ3_DIR=... && make -j$(nproc)
cd .. && pip3 install -e . --no-build-isolation --force-reinstall
```

#### 7. `int2 has not been declared`

**Cause**: Missing vector types for non-CUDA builds.

**Solution**: Update to latest YiRage with `vector_types.h` fix.

---

## Environment Variables Summary

Add to `~/.bashrc` for persistence:

```bash
# Ascend CANN environment
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true

# Rust (if installed via rustup)
source $HOME/.cargo/env

# YiRage Z3 configuration
export Z3_DIR=$HOME/YiRage/deps/z3/build
```

---

## Verified Environment

This guide was tested on:

| Component | Version |
|-----------|---------|
| Hardware | Ascend 910B2 |
| CANN | 8.2.RC2 |
| Python | 3.11.10 |
| PyTorch | 2.6.0 |
| torch_npu | 2.6.0.post2 |
| triton-ascend | 3.2.0 |
| YiRage | 0.2.4 |
| CMake | 4.2.0 |
| Rust | 1.91.1 |

---

## Quick Start Script

Save as `setup_yirage_ascend.sh`:

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "YiRage Ascend Setup Script"
echo "=========================================="

# 1. Load environment
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
source $HOME/.cargo/env 2>/dev/null || true

# 2. Install Python dependencies
echo "[1/6] Installing Python dependencies..."
pip3 install -q z3-solver graphviz cython triton-ascend "cmake>=3.24"

# 3. Install Rust if needed
if ! command -v rustc &> /dev/null; then
    echo "[2/6] Installing Rust..."
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    source $HOME/.cargo/env
else
    echo "[2/6] Rust already installed"
fi

cd ~/YiRage

# 4. Configure
echo "[3/6] Configuring build..."
cp config.ascend.cmake config.cmake

# Setup Z3
mkdir -p deps/z3/build
Z3_BASE=$(python3 -c "import z3; import os; print(os.path.dirname(z3.__file__))")
cat > deps/z3/build/z3-config.cmake << EOF
set(Z3_FOUND TRUE)
set(Z3_INCLUDE_DIRS "${Z3_BASE}/include")
set(Z3_LIBRARIES "${Z3_BASE}/lib/libz3.so")
if(NOT TARGET z3::libz3)
  add_library(z3::libz3 SHARED IMPORTED)
  set_target_properties(z3::libz3 PROPERTIES
    IMPORTED_LOCATION "${Z3_BASE}/lib/libz3.so"
    INTERFACE_INCLUDE_DIRECTORIES "${Z3_BASE}/include")
endif()
EOF
cat > deps/z3/build/Z3Config.cmake << 'EOF'
include("${CMAKE_CURRENT_LIST_DIR}/z3-config.cmake")
EOF

# Setup CUTLASS stub
mkdir -p deps/cutlass/include/cutlass
echo '#pragma once
#define CUTLASS_HOST_DEVICE inline
#define CUTLASS_DEVICE inline
namespace cutlass {}' > deps/cutlass/include/cutlass/cutlass.h

# 5. Build
echo "[4/6] Building..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DZ3_DIR=$HOME/YiRage/deps/z3/build
make -j$(nproc)

# 6. Install
echo "[5/6] Installing Python package..."
cd ~/YiRage
export Z3_DIR=$HOME/YiRage/deps/z3/build
pip3 install -e . --no-build-isolation

# 7. Verify
echo "[6/6] Verifying installation..."
python3 -c "import yirage; print('✅ YiRage', yirage.__version__, 'installed successfully!')"
python3 tests/ascend/test_triton_integration.py

echo ""
echo "=========================================="
echo "✅ YiRage Ascend setup complete!"
echo "=========================================="
```

---

## References

- [CANN Official Documentation](https://www.hiascend.com/document)
- [Ascend PyTorch (torch_npu)](https://github.com/Ascend/pytorch)
- [Triton-Ascend](https://github.com/Ascend/triton-ascend)
- [YiRage GitHub](https://github.com/chenxingqiang/YiRage)
