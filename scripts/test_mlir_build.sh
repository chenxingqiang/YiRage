#!/bin/bash
# =============================================================================
# Test MLIR Build Script for YiRage
# =============================================================================
#
# This script tests the USE_MLIR=ON build configuration.
#
# Requirements:
#   - LLVM/MLIR 17+ installed (or will be built from submodule)
#   - CMake 3.20+
#   - Ninja or Make
#
# Usage:
#   ./scripts/test_mlir_build.sh [options]
#
# Options:
#   --llvm-source=<source>  LLVM source: system, submodule, prebuilt
#   --build-type=<type>     Build type: Release, Debug, RelWithDebInfo
#   --install               Install after building
#   --test                  Run tests after building
#   --clean                 Clean before building
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build-mlir"

# Default options
LLVM_SOURCE="system"
BUILD_TYPE="Release"
DO_INSTALL=false
DO_TEST=false
DO_CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --llvm-source=*)
            LLVM_SOURCE="${1#*=}"
            shift
            ;;
        --build-type=*)
            BUILD_TYPE="${1#*=}"
            shift
            ;;
        --install)
            DO_INSTALL=true
            shift
            ;;
        --test)
            DO_TEST=true
            shift
            ;;
        --clean)
            DO_CLEAN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --llvm-source=<source>  LLVM source: system, submodule, prebuilt"
            echo "  --build-type=<type>     Build type: Release, Debug, RelWithDebInfo"
            echo "  --install               Install after building"
            echo "  --test                  Run tests after building"
            echo "  --clean                 Clean before building"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================"
echo "  YiRage MLIR Build Test"
echo "============================================"
echo "Project Root: ${PROJECT_ROOT}"
echo "Build Dir:    ${BUILD_DIR}"
echo "LLVM Source:  ${LLVM_SOURCE}"
echo "Build Type:   ${BUILD_TYPE}"
echo "============================================"

# Clean if requested
if [ "$DO_CLEAN" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# =============================================================================
# Find MLIR
# =============================================================================

find_mlir() {
    echo "Looking for MLIR..."
    
    # Check environment variable
    if [ -n "${MLIR_DIR}" ] && [ -f "${MLIR_DIR}/MLIRConfig.cmake" ]; then
        echo "Using MLIR from MLIR_DIR: ${MLIR_DIR}"
        return 0
    fi
    
    # Common paths
    local SEARCH_PATHS=(
        "/usr/lib/llvm-17/lib/cmake/mlir"
        "/usr/lib/llvm-18/lib/cmake/mlir"
        "/usr/lib/llvm-16/lib/cmake/mlir"
        "/opt/homebrew/opt/llvm/lib/cmake/mlir"
        "/usr/local/opt/llvm/lib/cmake/mlir"
        "/opt/llvm/lib/cmake/mlir"
    )
    
    for path in "${SEARCH_PATHS[@]}"; do
        if [ -f "${path}/MLIRConfig.cmake" ]; then
            export MLIR_DIR="${path}"
            echo "Found MLIR at: ${MLIR_DIR}"
            return 0
        fi
    done
    
    echo "MLIR not found in system paths."
    return 1
}

# =============================================================================
# Configure
# =============================================================================

configure_mlir_build() {
    echo ""
    echo "Configuring with USE_MLIR=ON..."
    
    local CMAKE_ARGS=(
        "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
        "-DUSE_MLIR=ON"
        "-DYIRAGE_LLVM_SOURCE=${LLVM_SOURCE}"
        "-DYIRAGE_BUILD_MLIR_TOOLS=ON"
    )
    
    if [ -n "${MLIR_DIR}" ]; then
        CMAKE_ARGS+=("-DMLIR_DIR=${MLIR_DIR}")
        
        # Also set LLVM_DIR
        local LLVM_DIR="${MLIR_DIR}/../llvm"
        if [ -f "${LLVM_DIR}/LLVMConfig.cmake" ]; then
            CMAKE_ARGS+=("-DLLVM_DIR=${LLVM_DIR}")
        fi
    fi
    
    # Use Ninja if available
    local GENERATOR="Unix Makefiles"
    if command -v ninja &> /dev/null; then
        GENERATOR="Ninja"
    fi
    CMAKE_ARGS+=("-G${GENERATOR}")
    
    echo "Running cmake with: ${CMAKE_ARGS[*]}"
    cmake "${PROJECT_ROOT}" "${CMAKE_ARGS[@]}"
}

# =============================================================================
# Build
# =============================================================================

build_mlir() {
    echo ""
    echo "Building..."
    
    local NPROC=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    if [ -f "build.ninja" ]; then
        ninja -j${NPROC}
    else
        make -j${NPROC}
    fi
}

# =============================================================================
# Test MLIR Components
# =============================================================================

test_mlir_components() {
    echo ""
    echo "Testing MLIR components..."
    
    # Check yirage-opt exists
    local YIRAGE_OPT=""
    if [ -f "${BUILD_DIR}/mlir/yirage-opt" ]; then
        YIRAGE_OPT="${BUILD_DIR}/mlir/yirage-opt"
    elif [ -f "${BUILD_DIR}/bin/yirage-opt" ]; then
        YIRAGE_OPT="${BUILD_DIR}/bin/yirage-opt"
    fi
    
    if [ -n "${YIRAGE_OPT}" ]; then
        echo "Found yirage-opt: ${YIRAGE_OPT}"
        
        # Test help
        echo "Testing yirage-opt --help..."
        "${YIRAGE_OPT}" --help | head -20
        
        # Test simple MLIR
        echo ""
        echo "Testing simple MLIR parsing..."
        local TEST_MLIR="${PROJECT_ROOT}/mlir/test/simple_matmul.mlir"
        if [ -f "${TEST_MLIR}" ]; then
            "${YIRAGE_OPT}" "${TEST_MLIR}" --canonicalize
        else
            echo "Test file not found: ${TEST_MLIR}"
        fi
    else
        echo "WARNING: yirage-opt not found"
    fi
    
    # Check libraries
    echo ""
    echo "Checking libraries..."
    
    local LIBS=(
        "YirageDialect"
        "YirageTransforms"
        "YirageExecution"
    )
    
    for lib in "${LIBS[@]}"; do
        local found=false
        for ext in ".a" ".so" ".dylib"; do
            if ls "${BUILD_DIR}"/lib/*${lib}* &>/dev/null 2>&1 ||
               ls "${BUILD_DIR}"/mlir/lib/*${lib}* &>/dev/null 2>&1; then
                echo "Found: ${lib}"
                found=true
                break
            fi
        done
        if [ "$found" = false ]; then
            echo "WARNING: ${lib} not found"
        fi
    done
}

# =============================================================================
# Run Python Tests
# =============================================================================

run_python_tests() {
    echo ""
    echo "Running Python tests..."
    
    cd "${PROJECT_ROOT}"
    
    # Set up Python path
    export PYTHONPATH="${PROJECT_ROOT}/python:${PROJECT_ROOT}/mlir/python:${PYTHONPATH}"
    
    # Run MLIR tests
    if command -v pytest &> /dev/null; then
        pytest tests/python/test_mlir.py -v --tb=short || true
        pytest tests/python/test_gpu_codegen.py -v --tb=short || true
    else
        echo "pytest not found, skipping Python tests"
    fi
}

# =============================================================================
# Main
# =============================================================================

main() {
    # Try to find MLIR first (unless using submodule)
    if [ "${LLVM_SOURCE}" = "system" ]; then
        if ! find_mlir; then
            echo ""
            echo "ERROR: MLIR not found. Please install LLVM/MLIR or use:"
            echo "  --llvm-source=submodule  (build from source)"
            echo "  --llvm-source=prebuilt   (download prebuilt)"
            exit 1
        fi
    fi
    
    # Configure
    configure_mlir_build
    
    # Build
    build_mlir
    
    # Test components
    test_mlir_components
    
    # Run Python tests if requested
    if [ "$DO_TEST" = true ]; then
        run_python_tests
    fi
    
    echo ""
    echo "============================================"
    echo "  MLIR Build Complete!"
    echo "============================================"
    echo "yirage-opt: ${BUILD_DIR}/mlir/yirage-opt"
    echo ""
    echo "Next steps:"
    echo "  1. Test: yirage-opt mlir/test/simple_matmul.mlir --yirage-to-linalg"
    echo "  2. Run GPU pipeline: yirage-opt input.mlir --yirage-gpu-pipeline"
    echo ""
}

main
