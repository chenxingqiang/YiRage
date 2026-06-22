#!/bin/bash
# =============================================================================
# YiRage PyPI Build Script
# =============================================================================
#
# Build and publish YiRage to PyPI.
#
# Usage:
#   ./scripts/build_pypi.sh [OPTIONS]
#
# Options:
#   --backend=BACKEND    Build for specific backend (cuda, mps, cpu, all)
#   --test               Upload to TestPyPI instead of PyPI
#   --no-upload          Build only, don't upload
#   --clean              Clean before build
#   --help               Show this help
#
# Environment Variables:
#   TWINE_USERNAME       PyPI username (or use __token__)
#   TWINE_PASSWORD       PyPI password or API token
#
# Examples:
#   ./scripts/build_pypi.sh                    # Build CPU version
#   ./scripts/build_pypi.sh --backend=cuda    # Build CUDA version
#   ./scripts/build_pypi.sh --test            # Upload to TestPyPI
#
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults
BACKEND="cpu"
TEST_PYPI=false
NO_UPLOAD=false
CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backend=*)
            BACKEND="${1#*=}"
            shift
            ;;
        --test)
            TEST_PYPI=true
            shift
            ;;
        --no-upload)
            NO_UPLOAD=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help|-h)
            head -30 "$0" | tail -25
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}=============================================="
echo -e "  YiRage PyPI Build"
echo -e "==============================================${NC}"
echo "  Backend: $BACKEND"
echo "  TestPyPI: $TEST_PYPI"
echo ""

# Clean
if [ "$CLEAN" = true ]; then
    echo -e "${GREEN}[+]${NC} Cleaning..."
    rm -rf build/ dist/ *.egg-info python/*.egg-info
fi

# Check dependencies
echo -e "${GREEN}[+]${NC} Checking build dependencies..."
pip install --quiet build twine

# Set backend environment
export YIRAGE_BACKEND="$BACKEND"
echo -e "${GREEN}[+]${NC} Setting YIRAGE_BACKEND=$BACKEND"

# Build sdist and wheel
echo -e "${GREEN}[+]${NC} Building source distribution and wheel..."
python -m build

# List built files
echo -e "${GREEN}[+]${NC} Built packages:"
ls -la dist/

# Check packages
echo -e "${GREEN}[+]${NC} Checking packages..."
twine check dist/*

# Upload
if [ "$NO_UPLOAD" = false ]; then
    if [ "$TEST_PYPI" = true ]; then
        echo -e "${GREEN}[+]${NC} Uploading to TestPyPI..."
        twine upload --repository testpypi dist/*
        echo ""
        echo -e "${GREEN}Done!${NC} Install with:"
        echo "  pip install --index-url https://test.pypi.org/simple/ yirage"
    else
        echo -e "${YELLOW}[!]${NC} Uploading to PyPI..."
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            twine upload dist/*
            echo ""
            echo -e "${GREEN}Done!${NC} Install with:"
            echo "  pip install yirage"
        else
            echo "Upload cancelled."
        fi
    fi
else
    echo -e "${GREEN}[+]${NC} Build complete (no upload)."
fi

echo ""
echo -e "${BLUE}Package ready in dist/${NC}"
