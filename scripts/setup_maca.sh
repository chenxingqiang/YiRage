#!/bin/bash
# YiRage MACA Environment Setup Script
# Tested on MetaX C500 GPU Server

set -e

echo "=========================================="
echo "YiRage MACA Backend Setup"
echo "=========================================="

# 1. Check MACA SDK
echo ""
echo "[1/7] Checking MACA SDK..."
if [ -d "/opt/maca" ]; then
    export MACA_PATH=/opt/maca
    echo "✓ MACA SDK found at $MACA_PATH"
else
    echo "✗ MACA SDK not found at /opt/maca"
    echo "  Please install MACA SDK first"
    exit 1
fi

# Check mxcc compiler
if [ -f "$MACA_PATH/mxgpu_llvm/bin/mxcc" ]; then
    echo "✓ mxcc compiler found"
else
    echo "✗ mxcc compiler not found"
    exit 1
fi

# 2. Set environment variables
echo ""
echo "[2/7] Setting environment variables..."
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export PATH=${MACA_PATH}/mxgpu_llvm/bin:$PATH
echo "✓ Environment variables set"

# 3. Check Python and mcPytorch
echo ""
echo "[3/7] Checking Python environment..."

# Try to activate conda if available
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate base 2>/dev/null || true
fi

python3 -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || {
    echo "✗ PyTorch not found. Please install mcPytorch."
    exit 1
}

# Verify it's mcPytorch
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
if [[ "$TORCH_VER" == *"metax"* ]]; then
    echo "✓ mcPytorch detected"
else
    echo "⚠ Standard PyTorch detected (not mcPytorch)"
    echo "  Some features may not work correctly"
fi

# 4. Install Python dependencies
echo ""
echo "[4/7] Installing Python dependencies..."
pip install -q z3-solver graphviz cython 2>/dev/null || pip install z3-solver graphviz cython
echo "✓ Python dependencies installed"

# 5. Check CMake version
echo ""
echo "[5/7] Checking CMake..."
CMAKE_VER=$(cmake --version 2>/dev/null | head -1 | grep -oP '\d+\.\d+' || echo "0.0")
CMAKE_MAJOR=$(echo $CMAKE_VER | cut -d. -f1)
CMAKE_MINOR=$(echo $CMAKE_VER | cut -d. -f2)

if [ "$CMAKE_MAJOR" -ge 3 ] && [ "$CMAKE_MINOR" -ge 24 ]; then
    echo "✓ CMake version $CMAKE_VER is compatible"
else
    echo "⚠ CMake version too old, installing via pip..."
    pip install "cmake>=3.24"
    echo "✓ CMake installed via pip"
fi

# 6. Setup dependencies
echo ""
echo "[6/7] Setting up YiRage dependencies..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Z3 config
mkdir -p deps/z3/build
Z3_BASE=$(python3 -c "import z3; import os; print(os.path.dirname(z3.__file__))" 2>/dev/null || echo "")
if [ -n "$Z3_BASE" ]; then
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
    echo "✓ Z3 configured"
else
    echo "⚠ Z3 Python package not found, trying system Z3..."
fi

# JSON (if empty)
if [ ! -f "deps/json/include/nlohmann/json.hpp" ]; then
    mkdir -p deps/json/include/nlohmann
    echo "  Downloading nlohmann/json..."
    curl -sL https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp \
        -o deps/json/include/nlohmann/json.hpp || {
        echo "⚠ Could not download json.hpp, trying alternative..."
    }
    
    cat > deps/json/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(nlohmann_json)
add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
EOF
    echo "✓ JSON library configured"
fi

# CUTLASS stub
mkdir -p deps/cutlass/include/cutlass/detail
cat > deps/cutlass/include/cutlass/cutlass.h << 'EOF'
#pragma once
#if defined(__NVCC__) || (defined(__clang__) && (defined(__CUDA__) || defined(__MACA__)))
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE
#define CUTLASS_DEVICE
#endif
namespace cutlass {}
EOF

cat > deps/cutlass/include/cutlass/detail/helper_macros.hpp << 'EOF'
#pragma once
#if defined(__NVCC__) || (defined(__clang__) && (defined(__CUDA__) || defined(__MACA__)))
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE
#define CUTLASS_DEVICE
#endif
EOF
echo "✓ CUTLASS stub configured"

# 7. Create config.cmake
echo ""
echo "[7/7] Creating config.cmake for MACA backend..."
cat > config.cmake << 'EOF'
set(USE_CUDA OFF)
set(USE_MACA ON)
set(USE_CUDNN OFF)
set(USE_CPU ON)
set(USE_ASCEND OFF)
set(USE_NKI OFF)
set(USE_MPS OFF)
EOF
echo "✓ config.cmake created"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Build C++ library:"
echo "     mkdir -p build && cd build"
echo "     cmake .. -DUSE_CUDA=OFF -DUSE_MACA=ON -DCMAKE_BUILD_TYPE=Release"
echo "     make -j\$(nproc)"
echo ""
echo "  2. Install Python package:"
echo "     cd .. && pip install -e ."
echo ""
echo "  3. Test installation:"
echo "     python3 -c \"import yirage; print('Success!')\""
echo ""

