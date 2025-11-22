#!/bin/bash
# YiRage Multi-Backend Implementation Validation Script
# Usage: bash scripts/validate_multi_backend.sh

set -e  # Exit on error

echo "=========================================="
echo "YiRage Multi-Backend Validation"
echo "=========================================="
echo ""

YIRAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$YIRAGE_ROOT"

ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1 (MISSING)"
        ((ERRORS++))
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1/"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1/ (MISSING)"
        ((WARNINGS++))
        return 1
    fi
}

echo "[1] Checking Backend Headers..."
check_file "include/yirage/backend/backend_interface.h"
check_file "include/yirage/backend/backend_registry.h"
check_file "include/yirage/backend/backends.h"
check_file "include/yirage/backend/cuda_backend.h"
check_file "include/yirage/backend/cpu_backend.h"
check_file "include/yirage/backend/mps_backend.h"
check_file "include/yirage/backend/triton_backend.h"
check_file "include/yirage/backend/nki_backend.h"
check_file "include/yirage/backend/cudnn_backend.h"
check_file "include/yirage/backend/mkl_backend.h"
echo ""

echo "[2] Checking Backend Sources..."
check_file "src/backend/backend_utils.cc"
check_file "src/backend/backend_registry.cc"
check_file "src/backend/backends.cc"
check_file "src/backend/cuda_backend.cc"
check_file "src/backend/cpu_backend.cc"
check_file "src/backend/mps_backend.cc"
check_file "src/backend/triton_backend.cc"
check_file "src/backend/nki_backend.cc"
check_file "src/backend/cudnn_backend.cc"
check_file "src/backend/mkl_backend.cc"
echo ""

echo "[3] Checking Kernel Config Headers..."
check_file "include/yirage/kernel/common/kernel_interface.h"
check_file "include/yirage/kernel/cuda/cuda_kernel_config.h"
check_file "include/yirage/kernel/cpu/cpu_kernel_config.h"
check_file "include/yirage/kernel/mps/mps_kernel_config.h"
check_file "include/yirage/kernel/triton/triton_kernel_config.h"
check_file "include/yirage/kernel/nki/nki_kernel_config.h"
check_file "include/yirage/kernel/cudnn/cudnn_kernel_config.h"
check_file "include/yirage/kernel/mkl/mkl_kernel_config.h"
echo ""

echo "[4] Checking Kernel Optimizer Sources..."
check_file "src/kernel/common/kernel_factory.cc"
check_file "src/kernel/cuda/cuda_optimizer.cc"
check_file "src/kernel/cpu/cpu_optimizer.cc"
check_file "src/kernel/mps/mps_optimizer.cc"
check_file "src/kernel/triton/triton_optimizer.cc"
check_file "src/kernel/nki/nki_optimizer.cc"
check_file "src/kernel/cudnn/cudnn_optimizer.cc"
check_file "src/kernel/mkl/mkl_optimizer.cc"
echo ""

echo "[5] Checking Search Strategy Headers..."
check_file "include/yirage/search/common/search_strategy.h"
check_file "include/yirage/search/backend_strategies/cuda_strategy.h"
check_file "include/yirage/search/backend_strategies/cpu_strategy.h"
check_file "include/yirage/search/backend_strategies/mps_strategy.h"
check_file "include/yirage/search/backend_strategies/triton_strategy.h"
check_file "include/yirage/search/backend_strategies/nki_strategy.h"
echo ""

echo "[6] Checking Search Strategy Sources..."
check_file "src/search/common/search_strategy_factory.cc"
check_file "src/search/backend_strategies/cuda_strategy.cc"
check_file "src/search/backend_strategies/cpu_strategy.cc"
check_file "src/search/backend_strategies/mps_strategy.cc"
check_file "src/search/backend_strategies/triton_strategy.cc"
check_file "src/search/backend_strategies/nki_strategy.cc"
echo ""

echo "[7] Checking Python API..."
check_file "python/yirage/backend_api.py"
check_file "python/yirage/__init__.py"
echo ""

echo "[8] Checking Build System..."
check_file "config.cmake"
check_file "CMakeLists.txt"
check_file "setup.py"
echo ""

echo "[9] Checking Documentation..."
check_file "MULTI_BACKEND_README.md"
check_file "QUICKSTART_MULTI_BACKEND.md"
check_file "MULTI_BACKEND_INDEX.md"
check_file "ALL_BACKENDS_STATUS.md"
check_file "COMPLETE_BACKEND_IMPLEMENTATION.md"
check_file "FINAL_IMPLEMENTATION_OVERVIEW.md"
check_file "VALIDATION_REPORT.md"
check_file "docs/ypk/multi_backend_design.md"
check_file "docs/ypk/backend_usage.md"
check_file "docs/ypk/BACKEND_KERNEL_OPTIMIZATION_DESIGN.md"
echo ""

echo "[10] Checking Tests & Examples..."
check_file "tests/backend/test_backend_registry.cc"
check_file "demo/backend_selection_demo.py"
echo ""

echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo -e "Errors:   ${RED}$ERRORS${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ All critical files present!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Build: pip install -e . -v"
    echo "  2. Test: python demo/backend_selection_demo.py"
    exit 0
else
    echo -e "${RED}✗ Validation failed with $ERRORS errors${NC}"
    exit 1
fi

