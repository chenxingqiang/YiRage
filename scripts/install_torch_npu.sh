#!/bin/bash
# =============================================================================
# torch_npu Installation Helper for Huawei Ascend NPU
# =============================================================================
# This script helps install torch_npu with proper version matching.
#
# Usage:
#   ./scripts/install_torch_npu.sh [OPTIONS]
#
# Options:
#   --force         Force reinstall even if already installed
#   --source        Install from source (Gitee)
#   --check         Only check current installation status
#   --help          Show this help message
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() { echo -e "${GREEN}[✓]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
print_error() { echo -e "${RED}[✗]${NC} $1"; }
print_info() { echo -e "${BLUE}[i]${NC} $1"; }

# Parse arguments
FORCE_INSTALL=false
INSTALL_FROM_SOURCE=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE_INSTALL=true; shift ;;
        --source) INSTALL_FROM_SOURCE=true; shift ;;
        --check) CHECK_ONLY=true; shift ;;
        --help) head -20 "$0" | tail -15; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo -e "${BLUE}=============================================="
echo -e "  torch_npu Installation Helper"
echo -e "==============================================${NC}"

# =============================================================================
# Step 1: Check Prerequisites
# =============================================================================

echo ""
print_info "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
print_step "Python: $PYTHON_VERSION"

# Check PyTorch
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
if [ -z "$TORCH_VERSION" ]; then
    print_error "PyTorch not installed. Install PyTorch first:"
    echo "    pip install torch"
    exit 1
fi
print_step "PyTorch: $TORCH_VERSION"

# Extract base version (e.g., 2.1.0 from 2.1.0+cpu)
TORCH_BASE_VERSION=$(echo "$TORCH_VERSION" | sed 's/+.*//')

# Check CANN/Ascend toolkit
ASCEND_HOME="${ASCEND_HOME_PATH:-${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}}"
if [ ! -d "$ASCEND_HOME" ]; then
    print_error "CANN toolkit not found at: $ASCEND_HOME"
    echo "    Install CANN from: https://www.hiascend.com/software/cann"
    exit 1
fi
print_step "CANN: $ASCEND_HOME"

# Check npu-smi
if command -v npu-smi &> /dev/null; then
    NPU_INFO=$(npu-smi info -l 2>/dev/null | head -5)
    print_step "NPU Driver: Available"
    echo "$NPU_INFO" | sed 's/^/    /'
else
    print_warn "npu-smi not found (NPU driver may not be loaded)"
fi

# =============================================================================
# Step 2: Check Current torch_npu Status
# =============================================================================

echo ""
print_info "Checking torch_npu status..."

TORCH_NPU_VERSION=$(python3 -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null)
if [ -n "$TORCH_NPU_VERSION" ]; then
    print_step "torch_npu installed: $TORCH_NPU_VERSION"
    
    # Check if NPU is accessible
    NPU_AVAILABLE=$(python3 -c "import torch; import torch_npu; print(torch.npu.is_available())" 2>/dev/null)
    if [ "$NPU_AVAILABLE" = "True" ]; then
        NPU_COUNT=$(python3 -c "import torch; import torch_npu; print(torch.npu.device_count())" 2>/dev/null)
        NPU_NAME=$(python3 -c "import torch; import torch_npu; print(torch.npu.get_device_name(0))" 2>/dev/null)
        print_step "NPU available: $NPU_COUNT device(s) - $NPU_NAME"
    else
        print_warn "torch_npu installed but NPU not accessible"
    fi
    
    if [ "$CHECK_ONLY" = true ] || [ "$FORCE_INSTALL" = false ]; then
        echo ""
        print_step "torch_npu is ready!"
        exit 0
    fi
else
    print_warn "torch_npu not installed"
fi

if [ "$CHECK_ONLY" = true ]; then
    exit 1
fi

# =============================================================================
# Step 3: Install torch_npu
# =============================================================================

echo ""
print_info "Installing torch_npu..."

if [ "$INSTALL_FROM_SOURCE" = true ]; then
    # Install from source (Gitee)
    echo "    Installing from source (Gitee)..."
    
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    git clone --depth 1 https://gitee.com/ascend/pytorch.git
    cd pytorch
    
    pip install -e . || {
        print_error "Source installation failed"
        rm -rf "$TEMP_DIR"
        exit 1
    }
    
    rm -rf "$TEMP_DIR"
    print_step "torch_npu installed from source"
else
    # Try multiple installation methods
    
    # Method 1: Huawei Ascend Repository
    echo "    Trying Huawei Ascend Repository..."
    if pip install torch-npu -i https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/ascend-repo/simple/ 2>/dev/null; then
        print_step "torch_npu installed from Huawei Repository"
    else
        # Method 2: Direct pip install
        echo "    Trying direct pip install..."
        if pip install torch-npu 2>/dev/null; then
            print_step "torch_npu installed from PyPI"
        else
            # Method 3: Version-matched install
            echo "    Trying version-matched install (torch_npu==$TORCH_BASE_VERSION)..."
            if pip install "torch-npu==$TORCH_BASE_VERSION" 2>/dev/null; then
                print_step "torch_npu $TORCH_BASE_VERSION installed"
            else
                print_error "All installation methods failed"
                echo ""
                echo "    Manual installation options:"
                echo "    1. Visit: https://www.hiascend.com/software/cann/community"
                echo "    2. Download matching torch_npu wheel for:"
                echo "       - PyTorch: $TORCH_VERSION"
                echo "       - CANN: $(cat $ASCEND_HOME/version.info 2>/dev/null | head -1 || echo 'unknown')"
                echo "       - Python: $(python3 -c 'import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")')"
                echo "    3. Install: pip install torch_npu-*.whl"
                echo ""
                echo "    Or try source installation:"
                echo "       $0 --source"
                exit 1
            fi
        fi
    fi
fi

# =============================================================================
# Step 4: Verify Installation
# =============================================================================

echo ""
print_info "Verifying installation..."

TORCH_NPU_VERSION=$(python3 -c "import torch_npu; print(torch_npu.__version__)" 2>/dev/null)
if [ -n "$TORCH_NPU_VERSION" ]; then
    print_step "torch_npu: $TORCH_NPU_VERSION"
    
    NPU_AVAILABLE=$(python3 -c "import torch; import torch_npu; print(torch.npu.is_available())" 2>/dev/null)
    if [ "$NPU_AVAILABLE" = "True" ]; then
        print_step "NPU is accessible!"
        
        # Run simple test
        python3 -c "
import torch
import torch_npu

x = torch.randn(2, 3).npu()
y = torch.randn(3, 4).npu()
z = torch.mm(x, y)
torch.npu.synchronize()
print('  Simple matmul test: PASSED')
" && print_step "Compute test: PASSED"
    else
        print_warn "torch_npu installed but NPU not accessible"
        echo "    Check: npu-smi info"
    fi
else
    print_error "Installation verification failed"
    exit 1
fi

echo ""
echo -e "${GREEN}torch_npu installation complete!${NC}"
