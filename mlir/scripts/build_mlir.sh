#!/bin/bash
#===----------------------------------------------------------------------===//
#
# YiRage MLIR Build Script
#
# This script builds LLVM/MLIR and the YiRage MLIR dialect.
#
# Usage:
#   ./build_mlir.sh                    # Build everything
#   ./build_mlir.sh --llvm-only        # Build only LLVM/MLIR
#   ./build_mlir.sh --yirage-only      # Build only YiRage dialect
#
#===----------------------------------------------------------------------===//

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MLIR_DIR="$(dirname "$SCRIPT_DIR")"
YIRAGE_ROOT="$(dirname "$MLIR_DIR")"
DEPS_DIR="$YIRAGE_ROOT/deps"
LLVM_SRC="$DEPS_DIR/llvm-project"
LLVM_BUILD="$LLVM_SRC/build"
LLVM_INSTALL="$DEPS_DIR/llvm-install"
YIRAGE_MLIR_BUILD="$MLIR_DIR/build"

# Detect number of cores
if [[ "$OSTYPE" == "darwin"* ]]; then
    NPROC=$(sysctl -n hw.ncpu)
else
    NPROC=$(nproc)
fi

# Use half the cores by default to avoid system freeze
BUILD_JOBS=$((NPROC / 2))
if [ $BUILD_JOBS -lt 1 ]; then
    BUILD_JOBS=1
fi

echo "=== YiRage MLIR Build Script ==="
echo "LLVM Source: $LLVM_SRC"
echo "LLVM Build: $LLVM_BUILD"
echo "YiRage MLIR: $MLIR_DIR"
echo "Build jobs: $BUILD_JOBS"
echo ""

build_llvm() {
    echo "=== Building LLVM/MLIR ==="
    
    # Clone if not exists
    if [ ! -d "$LLVM_SRC" ]; then
        echo "Cloning LLVM (release/18.x)..."
        git clone --depth 1 --branch release/18.x \
            https://github.com/llvm/llvm-project.git "$LLVM_SRC"
    fi
    
    mkdir -p "$LLVM_BUILD"
    cd "$LLVM_BUILD"
    
    # Detect architecture
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
        LLVM_TARGETS="AArch64"
    else
        LLVM_TARGETS="X86"
    fi
    
    echo "Configuring LLVM for $LLVM_TARGETS..."
    cmake -G Ninja ../llvm \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS" \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DCMAKE_INSTALL_PREFIX="$LLVM_INSTALL" \
        -DLLVM_ENABLE_RTTI=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
    
    echo "Building MLIR libraries..."
    ninja -j$BUILD_JOBS mlir-headers mlir-libraries mlir-cmake-exports
    
    echo "Installing MLIR..."
    ninja install-mlir-headers install-mlir-libraries
    
    echo "LLVM/MLIR build complete!"
}

build_yirage_mlir() {
    echo "=== Building YiRage MLIR Dialect ==="
    
    mkdir -p "$YIRAGE_MLIR_BUILD"
    cd "$YIRAGE_MLIR_BUILD"
    
    # Find MLIR
    if [ -d "$LLVM_BUILD/lib/cmake/mlir" ]; then
        MLIR_CMAKE_DIR="$LLVM_BUILD/lib/cmake/mlir"
    elif [ -d "$LLVM_INSTALL/lib/cmake/mlir" ]; then
        MLIR_CMAKE_DIR="$LLVM_INSTALL/lib/cmake/mlir"
    else
        echo "Error: MLIR not found. Please build LLVM/MLIR first."
        exit 1
    fi
    
    echo "Using MLIR from: $MLIR_CMAKE_DIR"
    
    cmake -G Ninja "$MLIR_DIR" \
        -DMLIR_DIR="$MLIR_CMAKE_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DYIRAGE_ENABLE_PYTHON=OFF
    
    echo "Building YiRage dialect..."
    ninja -j$BUILD_JOBS
    
    echo "YiRage MLIR dialect build complete!"
    echo "yirage-opt tool: $YIRAGE_MLIR_BUILD/yirage-opt"
}

# Parse arguments
LLVM_ONLY=false
YIRAGE_ONLY=false

for arg in "$@"; do
    case $arg in
        --llvm-only)
            LLVM_ONLY=true
            shift
            ;;
        --yirage-only)
            YIRAGE_ONLY=true
            shift
            ;;
        -j*)
            BUILD_JOBS="${arg#-j}"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --llvm-only     Build only LLVM/MLIR"
            echo "  --yirage-only   Build only YiRage dialect"
            echo "  -jN             Use N parallel jobs"
            exit 0
            ;;
    esac
done

# Build
if [ "$YIRAGE_ONLY" = true ]; then
    build_yirage_mlir
elif [ "$LLVM_ONLY" = true ]; then
    build_llvm
else
    build_llvm
    build_yirage_mlir
fi

echo ""
echo "=== Build Complete ==="
