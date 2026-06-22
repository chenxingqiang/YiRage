#!/bin/bash
# =============================================================================
# YiRage Multi-Backend Installation Script
# =============================================================================
# Detects hardware and installs YiRage with appropriate backend support.
#
# Usage:
#   ./scripts/install.sh [OPTIONS]
#
# Options:
#   --backend=BACKEND    Force specific backend (cuda, rocm, mps, ascend, maca, cpu)
#   --all-backends       Enable all available backends
#   --dev                Include development dependencies
#   --no-build           Skip C++ build (Python deps only)
#   --clean              Clean build before installing
#   --venv               Create and use virtual environment
#   --help               Show this help message
#
# Environment Variables:
#   CUDA_HOME            CUDA installation path
#   ROCM_PATH            ROCm installation path
#   ASCEND_HOME          Ascend toolkit path
#   MACA_PATH            MetaX MACA SDK path
#
# Examples:
#   ./scripts/install.sh                    # Auto-detect and install
#   ./scripts/install.sh --backend=cuda     # Force CUDA backend
#   ./scripts/install.sh --dev --venv       # Dev install with venv
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default options
BACKEND=""
ALL_BACKENDS=false
DEV_INSTALL=false
NO_BUILD=false
CLEAN_BUILD=false
USE_VENV=false

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "${BLUE}=============================================="
    echo -e "  $1"
    echo -e "==============================================${NC}"
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

show_help() {
    head -35 "$0" | tail -30
    exit 0
}

# =============================================================================
# Parse Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend=*)
            BACKEND="${1#*=}"
            shift
            ;;
        --all-backends)
            ALL_BACKENDS=true
            shift
            ;;
        --dev)
            DEV_INSTALL=true
            shift
            ;;
        --no-build)
            NO_BUILD=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --venv)
            USE_VENV=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Hardware Detection
# =============================================================================

detect_hardware() {
    print_header "Hardware Detection"
    
    DETECTED_BACKENDS=""
    
    # Detect OS
    OS_TYPE=$(uname -s)
    ARCH=$(uname -m)
    echo "  OS: $OS_TYPE ($ARCH)"
    
    # NVIDIA CUDA
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
        if [ -n "$CUDA_VERSION" ]; then
            print_step "NVIDIA GPU detected (Driver: $CUDA_VERSION)"
            DETECTED_BACKENDS="$DETECTED_BACKENDS cuda"
            
            # Check CUDA toolkit
            if [ -n "$CUDA_HOME" ]; then
                echo "      CUDA_HOME: $CUDA_HOME"
            elif [ -d "/usr/local/cuda" ]; then
                export CUDA_HOME="/usr/local/cuda"
                echo "      CUDA_HOME: $CUDA_HOME (auto-detected)"
            fi
        fi
    fi
    
    # AMD ROCm
    if command -v rocm-smi &> /dev/null || [ -d "/opt/rocm" ]; then
        print_step "AMD GPU with ROCm detected"
        DETECTED_BACKENDS="$DETECTED_BACKENDS rocm"
        
        if [ -z "$ROCM_PATH" ]; then
            export ROCM_PATH="/opt/rocm"
        fi
        echo "      ROCM_PATH: $ROCM_PATH"
    fi
    
    # Apple MPS
    if [ "$OS_TYPE" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
        print_step "Apple Silicon detected (MPS available)"
        DETECTED_BACKENDS="$DETECTED_BACKENDS mps"
        
        # Get chip info
        CHIP_INFO=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
        echo "      Chip: $CHIP_INFO"
    fi
    
    # Huawei Ascend
    if command -v npu-smi &> /dev/null || [ -d "/usr/local/Ascend" ]; then
        print_step "Huawei Ascend NPU detected"
        DETECTED_BACKENDS="$DETECTED_BACKENDS ascend"
        
        if [ -z "$ASCEND_HOME" ]; then
            export ASCEND_HOME="/usr/local/Ascend/ascend-toolkit/latest"
        fi
        echo "      ASCEND_HOME: $ASCEND_HOME"
    fi
    
    # MetaX MACA
    if [ -d "$MACA_PATH" ] || [ -d "/opt/maca" ]; then
        print_step "MetaX MACA GPU detected"
        DETECTED_BACKENDS="$DETECTED_BACKENDS maca"
        
        if [ -z "$MACA_PATH" ]; then
            export MACA_PATH="/opt/maca"
        fi
        echo "      MACA_PATH: $MACA_PATH"
    fi
    
    # Intel XPU
    if command -v xpu-smi &> /dev/null || [ -d "/opt/intel/oneapi" ]; then
        print_step "Intel XPU detected"
        DETECTED_BACKENDS="$DETECTED_BACKENDS xpu"
    fi
    
    # Google TPU (check for libtpu)
    if [ -f "/usr/share/tpu/tpu_library_path" ] || [ -n "$TPU_NAME" ]; then
        print_step "Google TPU detected"
        DETECTED_BACKENDS="$DETECTED_BACKENDS tpu"
    fi
    
    # CPU (always available)
    print_step "CPU backend (always available)"
    DETECTED_BACKENDS="$DETECTED_BACKENDS cpu"
    
    echo ""
    echo "  Detected backends:$DETECTED_BACKENDS"
}

# =============================================================================
# Environment Setup
# =============================================================================

setup_backend_environment() {
    # This function auto-detects and exports all required environment variables
    # for the detected backends. Call this before building.
    
    print_step "Auto-configuring backend environment variables..."
    
    # =========================================================================
    # CUDA Environment
    # =========================================================================
    if [[ "$DETECTED_BACKENDS" == *"cuda"* ]] && [ -z "$CUDA_HOME" ]; then
        for cuda_path in /usr/local/cuda /opt/cuda /usr/lib/cuda /usr/local/cuda-*; do
            if [ -f "$cuda_path/bin/nvcc" ]; then
                export CUDA_HOME="$cuda_path"
                echo "    CUDA_HOME=$CUDA_HOME (auto-detected)"
                break
            fi
        done
    fi
    
    if [ -n "$CUDA_HOME" ]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
    fi
    
    # =========================================================================
    # Ascend Environment (Huawei NPU)
    # =========================================================================
    if [[ "$DETECTED_BACKENDS" == *"ascend"* ]]; then
        # Find Ascend toolkit
        if [ -z "$ASCEND_HOME_PATH" ]; then
            for ascend_path in \
                "/usr/local/Ascend/ascend-toolkit/latest" \
                "/usr/local/Ascend/nnrt/latest" \
                "/opt/Ascend/ascend-toolkit/latest"; do
                if [ -d "$ascend_path" ]; then
                    export ASCEND_HOME_PATH="$ascend_path"
                    export ASCEND_HOME="$ascend_path"
                    echo "    ASCEND_HOME_PATH=$ASCEND_HOME_PATH (auto-detected)"
                    break
                fi
            done
        fi
        
        if [ -n "$ASCEND_HOME_PATH" ]; then
            # Set OPP path
            if [ -z "$ASCEND_OPP_PATH" ] && [ -d "$ASCEND_HOME_PATH/opp" ]; then
                export ASCEND_OPP_PATH="$ASCEND_HOME_PATH/opp"
                echo "    ASCEND_OPP_PATH=$ASCEND_OPP_PATH"
            fi
            
            # Add lib paths
            for lib_path in \
                "$ASCEND_HOME_PATH/lib64" \
                "$ASCEND_HOME_PATH/aarch64-linux/lib64" \
                "/usr/local/Ascend/driver/lib64" \
                "/usr/local/Ascend/driver/lib64/driver"; do
                if [ -d "$lib_path" ]; then
                    export LD_LIBRARY_PATH="$lib_path:$LD_LIBRARY_PATH"
                fi
            done
            
            # Add bin path
            if [ -d "$ASCEND_HOME_PATH/bin" ]; then
                export PATH="$ASCEND_HOME_PATH/bin:$PATH"
            fi
            
            # Source setenv if available
            if [ -f "$ASCEND_HOME_PATH/bin/setenv.bash" ]; then
                source "$ASCEND_HOME_PATH/bin/setenv.bash" 2>/dev/null || true
            fi
            
            echo "    LD_LIBRARY_PATH updated with Ascend paths"
        fi
    fi
    
    # =========================================================================
    # ROCm Environment (AMD GPU)
    # =========================================================================
    if [[ "$DETECTED_BACKENDS" == *"rocm"* ]]; then
        if [ -z "$ROCM_PATH" ]; then
            for rocm_path in /opt/rocm /opt/rocm-*; do
                if [ -d "$rocm_path" ]; then
                    export ROCM_PATH="$rocm_path"
                    export HIP_PATH="$rocm_path"
                    echo "    ROCM_PATH=$ROCM_PATH (auto-detected)"
                    break
                fi
            done
        fi
        
        if [ -n "$ROCM_PATH" ]; then
            export PATH="$ROCM_PATH/bin:$PATH"
            export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH"
        fi
    fi
    
    # =========================================================================
    # MACA Environment (MetaX GPU)
    # =========================================================================
    if [[ "$DETECTED_BACKENDS" == *"maca"* ]]; then
        if [ -z "$MACA_PATH" ]; then
            for maca_path in /opt/maca /usr/local/maca; do
                if [ -d "$maca_path" ]; then
                    export MACA_PATH="$maca_path"
                    echo "    MACA_PATH=$MACA_PATH (auto-detected)"
                    break
                fi
            done
        fi
        
        if [ -n "$MACA_PATH" ]; then
            export PATH="$MACA_PATH/bin:$PATH"
            export LD_LIBRARY_PATH="$MACA_PATH/lib:$LD_LIBRARY_PATH"
        fi
    fi
    
    # =========================================================================
    # Intel oneAPI (XPU)
    # =========================================================================
    if [[ "$DETECTED_BACKENDS" == *"xpu"* ]]; then
        if [ -z "$ONEAPI_ROOT" ]; then
            for oneapi_path in /opt/intel/oneapi /usr/local/intel/oneapi; do
                if [ -d "$oneapi_path" ]; then
                    export ONEAPI_ROOT="$oneapi_path"
                    echo "    ONEAPI_ROOT=$ONEAPI_ROOT (auto-detected)"
                    # Source setvars if available
                    if [ -f "$oneapi_path/setvars.sh" ]; then
                        source "$oneapi_path/setvars.sh" --force 2>/dev/null || true
                    fi
                    break
                fi
            done
        fi
    fi
    
    # Clean up duplicate paths in LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | awk '!seen[$0]++' | tr '\n' ':' | sed 's/:$//')
}

setup_environment() {
    print_header "Environment Setup"
    
    # Check Python
    PYTHON_VERSION=$(python3 --version 2>&1)
    print_step "Python: $PYTHON_VERSION"
    
    # Create venv if requested
    if [ "$USE_VENV" = true ]; then
        if [ ! -d ".venv" ]; then
            print_step "Creating virtual environment..."
            python3 -m venv .venv
        fi
        source .venv/bin/activate
        print_step "Activated: $(which python)"
        
        # Upgrade pip
        pip install --upgrade pip
    fi
    
    # Check cmake
    if ! command -v cmake &> /dev/null; then
        print_warn "CMake not found. Installing via pip..."
        pip install cmake
    fi
    print_step "CMake: $(cmake --version | head -1)"
    
    # Auto-configure backend environment
    setup_backend_environment
}

# =============================================================================
# Install Python Dependencies
# =============================================================================

install_python_deps() {
    print_header "Installing Python Dependencies"
    
    # Core dependencies
    print_step "Installing core dependencies..."
    pip install numpy torch z3-solver graphviz tqdm cython cmake
    
    # Backend-specific Python packages
    for backend in $INSTALL_BACKENDS; do
        case $backend in
            cuda)
                print_step "Installing CUDA Python deps..."
                pip install triton 2>/dev/null || print_warn "triton not available"
                ;;
            rocm)
                print_step "Installing ROCm Python deps..."
                print_warn "For ROCm PyTorch, install manually:"
                echo "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.0"
                ;;
            mps)
                print_step "MPS: Using PyTorch MPS backend (built-in)"
                ;;
            ascend)
                print_step "Installing Ascend Python deps..."
                
                # Check if torch_npu is already installed
                if python3 -c "import torch_npu" 2>/dev/null; then
                    TORCH_NPU_VER=$(python3 -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null)
                    print_step "torch_npu already installed: $TORCH_NPU_VER"
                else
                    # Try to install torch_npu
                    # First check torch version for compatibility
                    TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null)
                    
                    if [ -n "$TORCH_VER" ]; then
                        echo "    Detected PyTorch version: $TORCH_VER"
                        
                        # torch_npu version should match torch version
                        # Try pip install first
                        if pip install torch_npu 2>/dev/null; then
                            print_step "torch_npu installed successfully"
                        else
                            print_warn "torch_npu not available via pip"
                            echo ""
                            echo "    torch_npu installation options:"
                            echo "    1. From Huawei official source:"
                            echo "       pip install torch-npu -i https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/ascend-repo/simple/"
                            echo ""
                            echo "    2. Match PyTorch version ($TORCH_VER):"
                            echo "       pip install torch-npu==$TORCH_VER"
                            echo ""
                            echo "    3. From source (for custom builds):"
                            echo "       git clone https://gitee.com/ascend/pytorch.git"
                            echo "       cd pytorch && pip install -e ."
                            echo ""
                        fi
                    else
                        print_warn "PyTorch not installed, install PyTorch first before torch_npu"
                    fi
                fi
                ;;
            xpu)
                print_step "Installing Intel XPU Python deps..."
                pip install intel-extension-for-pytorch 2>/dev/null || print_warn "Intel extension not available"
                ;;
            tpu)
                print_step "Installing TPU Python deps..."
                pip install jax jaxlib 2>/dev/null || print_warn "JAX not available"
                ;;
        esac
    done
    
    # Development dependencies
    if [ "$DEV_INSTALL" = true ]; then
        print_step "Installing development dependencies..."
        pip install pytest pytest-cov pytest-asyncio black isort mypy ruff pre-commit
        pip install ray transformers accelerate tensorboard
    fi
}

# =============================================================================
# Build C++ Extensions
# =============================================================================

build_cpp() {
    print_header "Building C++ Extensions"
    
    # Clean if requested
    if [ "$CLEAN_BUILD" = true ]; then
        print_step "Cleaning previous build..."
        rm -rf build/ dist/ *.egg-info python/yirage/*.so
    fi
    
    # Set Z3 library path
    Z3_LIB_PATH=$(python -c "import z3; import os; print(os.path.dirname(z3.__file__) + '/lib')" 2>/dev/null || echo "")
    if [ -n "$Z3_LIB_PATH" ]; then
        export DYLD_LIBRARY_PATH="$Z3_LIB_PATH:$DYLD_LIBRARY_PATH"
        export LD_LIBRARY_PATH="$Z3_LIB_PATH:$LD_LIBRARY_PATH"
        print_step "Z3 library path: $Z3_LIB_PATH"
    fi
    
    # Configure CMake flags based on backends
    CMAKE_FLAGS=""
    
    for backend in $INSTALL_BACKENDS; do
        case $backend in
            cuda)
                CMAKE_FLAGS="$CMAKE_FLAGS -DYIRAGE_BACKEND_CUDA=ON"
                if [ -n "$CUDA_HOME" ]; then
                    CMAKE_FLAGS="$CMAKE_FLAGS -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME"
                fi
                ;;
            rocm)
                CMAKE_FLAGS="$CMAKE_FLAGS -DYIRAGE_BACKEND_ROCM=ON"
                if [ -n "$ROCM_PATH" ]; then
                    CMAKE_FLAGS="$CMAKE_FLAGS -DROCM_PATH=$ROCM_PATH"
                fi
                ;;
            mps)
                CMAKE_FLAGS="$CMAKE_FLAGS -DYIRAGE_BACKEND_MPS=ON"
                ;;
            ascend)
                CMAKE_FLAGS="$CMAKE_FLAGS -DYIRAGE_BACKEND_ASCEND=ON"
                if [ -n "$ASCEND_HOME" ]; then
                    CMAKE_FLAGS="$CMAKE_FLAGS -DASCEND_HOME=$ASCEND_HOME"
                fi
                ;;
            maca)
                CMAKE_FLAGS="$CMAKE_FLAGS -DYIRAGE_BACKEND_MACA=ON"
                if [ -n "$MACA_PATH" ]; then
                    CMAKE_FLAGS="$CMAKE_FLAGS -DMACA_PATH=$MACA_PATH"
                fi
                ;;
            xpu)
                CMAKE_FLAGS="$CMAKE_FLAGS -DYIRAGE_BACKEND_XPU=ON"
                ;;
            tpu)
                CMAKE_FLAGS="$CMAKE_FLAGS -DYIRAGE_BACKEND_TPU=ON"
                ;;
            cpu)
                CMAKE_FLAGS="$CMAKE_FLAGS -DYIRAGE_BACKEND_CPU=ON"
                ;;
        esac
    done
    
    print_step "CMake flags: $CMAKE_FLAGS"
    
    # Build
    print_step "Running pip install -e ..."
    CMAKE_ARGS="$CMAKE_FLAGS" pip install -e . --no-build-isolation -v
    
    print_step "Build complete!"
}

# =============================================================================
# Verify Installation
# =============================================================================

verify_installation() {
    print_header "Verifying Installation"
    
    # Set library paths for verification
    Z3_LIB_PATH=$(python -c "import z3; import os; print(os.path.dirname(z3.__file__) + '/lib')" 2>/dev/null || echo "")
    export DYLD_LIBRARY_PATH="$Z3_LIB_PATH:$DYLD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$Z3_LIB_PATH:$LD_LIBRARY_PATH"
    
    python -c "
import sys
print(f'  Python: {sys.version.split()[0]}')

try:
    import yirage as yr
    print(f'  YiRage: {yr.__version__}')
    backends = yr.get_available_backends()
    print(f'  Backends: {backends}')
    
    if not backends:
        print('  ⚠ No backends available (C++ bindings may need rebuild)')
except ImportError as e:
    print(f'  YiRage: Not available')
    print(f'    Error: {e}')

try:
    import torch
    print(f'  PyTorch: {torch.__version__}')
    
    if torch.cuda.is_available():
        print(f'    CUDA: {torch.cuda.get_device_name(0)}')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f'    MPS: Available')
except ImportError:
    print('  PyTorch: Not installed')

try:
    import ray
    print(f'  Ray: {ray.__version__}')
except ImportError:
    pass
"
}

# =============================================================================
# Generate Activation Script
# =============================================================================

generate_activate_script() {
    print_header "Generating Activation Script"
    
    Z3_LIB_PATH=$(python -c "import z3; import os; print(os.path.dirname(z3.__file__) + '/lib')" 2>/dev/null || echo "")
    
    cat > "$PROJECT_ROOT/activate_yirage.sh" << EOF
#!/bin/bash
# YiRage Environment Activation Script
# Source this file: source activate_yirage.sh

# Project root
export YIRAGE_HOME="$PROJECT_ROOT"

# Virtual environment
if [ -d "\$YIRAGE_HOME/.venv" ]; then
    source "\$YIRAGE_HOME/.venv/bin/activate"
fi

# Z3 library path
export DYLD_LIBRARY_PATH="$Z3_LIB_PATH:\$DYLD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$Z3_LIB_PATH:\$LD_LIBRARY_PATH"

# Backend-specific paths
EOF

    for backend in $INSTALL_BACKENDS; do
        case $backend in
            cuda)
                echo "export CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"" >> "$PROJECT_ROOT/activate_yirage.sh"
                echo "export PATH=\"\$CUDA_HOME/bin:\$PATH\"" >> "$PROJECT_ROOT/activate_yirage.sh"
                ;;
            rocm)
                echo "export ROCM_PATH=\"${ROCM_PATH:-/opt/rocm}\"" >> "$PROJECT_ROOT/activate_yirage.sh"
                echo "export PATH=\"\$ROCM_PATH/bin:\$PATH\"" >> "$PROJECT_ROOT/activate_yirage.sh"
                ;;
            ascend)
                echo "export ASCEND_HOME=\"${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}\"" >> "$PROJECT_ROOT/activate_yirage.sh"
                echo "source \"\$ASCEND_HOME/bin/setenv.bash\" 2>/dev/null || true" >> "$PROJECT_ROOT/activate_yirage.sh"
                ;;
        esac
    done
    
    cat >> "$PROJECT_ROOT/activate_yirage.sh" << 'EOF'

# Python path
export PYTHONPATH="$YIRAGE_HOME/python:$PYTHONPATH"

echo "YiRage environment activated!"
echo "  YIRAGE_HOME: $YIRAGE_HOME"
echo "  Python: $(which python)"
EOF

    chmod +x "$PROJECT_ROOT/activate_yirage.sh"
    print_step "Created: activate_yirage.sh"
}

# =============================================================================
# Main Installation Flow
# =============================================================================

main() {
    print_header "YiRage Multi-Backend Installation"
    echo "  Project: $PROJECT_ROOT"
    echo ""
    
    # Detect hardware
    detect_hardware
    
    # Determine which backends to install
    if [ -n "$BACKEND" ]; then
        # User specified backend
        INSTALL_BACKENDS="$BACKEND"
        echo ""
        print_step "Using specified backend: $BACKEND"
    elif [ "$ALL_BACKENDS" = true ]; then
        # Install all detected backends
        INSTALL_BACKENDS="$DETECTED_BACKENDS"
        echo ""
        print_step "Installing all detected backends"
    else
        # Use detected backends (default)
        INSTALL_BACKENDS="$DETECTED_BACKENDS"
        echo ""
        print_step "Installing detected backends"
    fi
    
    echo "  Backends to install:$INSTALL_BACKENDS"
    echo ""
    
    # Setup environment
    setup_environment
    
    # Install Python dependencies
    install_python_deps
    
    # Build C++ if not skipped
    if [ "$NO_BUILD" = false ]; then
        build_cpp
    else
        print_warn "Skipping C++ build (--no-build)"
    fi
    
    # Verify installation
    verify_installation
    
    # Generate activation script
    generate_activate_script
    
    # Done
    print_header "Installation Complete!"
    echo ""
    echo "  To activate the environment:"
    echo "    source activate_yirage.sh"
    echo ""
    echo "  To run tests:"
    echo "    pytest tests/python -v"
    echo ""
    echo "  Installed backends:$INSTALL_BACKENDS"
    echo ""
}

# Run main
main
