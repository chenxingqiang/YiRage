#!/bin/bash
# YiRage Renaming Script
# Renames all yirage -> yirage and mi -> yr

set -e

YIRAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$YIRAGE_ROOT"

echo "=========================================="
echo "YiRage Project Renaming"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Rename directory: yirage/ -> yirage/"
echo "  2. Update all file contents: yirage -> yirage"
echo "  3. Update all imports: mi -> yr"
echo "  4. Rename CMake project"
echo "  5. Update setup.py package name"
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Counter
RENAMED_DIRS=0
UPDATED_FILES=0

# Step 1: Rename directories
echo -e "${BLUE}[Step 1] Renaming directories...${NC}"

if [ -d "include/yirage" ]; then
    mv include/yirage include/yirage
    echo -e "${GREEN}✓${NC} Renamed: include/yirage -> include/yirage"
    ((RENAMED_DIRS++))
fi

if [ -d "python/yirage" ]; then
    mv python/yirage python/yirage
    echo -e "${GREEN}✓${NC} Renamed: python/yirage -> python/yirage"
    ((RENAMED_DIRS++))
fi

echo ""

# Step 2: Update file contents - namespace and include paths
echo -e "${BLUE}[Step 2] Updating namespace and include paths...${NC}"

# Find all C++ and header files
find include src -type f \( -name "*.h" -o -name "*.cc" -o -name "*.cpp" -o -name "*.cu" -o -name "*.cuh" \) | while read file; do
    # Update namespace
    sed -i.bak 's/namespace yirage/namespace yirage/g' "$file"
    
    # Update include paths
    sed -i.bak 's|#include "yirage/|#include "yirage/|g' "$file"
    sed -i.bak 's|#include <yirage/|#include <yirage/|g' "$file"
    
    # Update using namespace
    sed -i.bak 's/using namespace yirage/using namespace yirage/g' "$file"
    
    # Update comments and documentation
    sed -i.bak 's/YiRage/YiRage/g' "$file"
    
    # Remove backup
    rm -f "${file}.bak"
    
    echo -e "${GREEN}✓${NC} Updated: $file"
    ((UPDATED_FILES++))
done

echo ""

# Step 3: Update Python files
echo -e "${BLUE}[Step 3] Updating Python files...${NC}"

find python -type f -name "*.py" | while read file; do
    # Update imports from yirage to yirage
    sed -i.bak 's/from yirage/from yirage/g' "$file"
    sed -i.bak 's/import yirage/import yirage/g' "$file"
    
    # Update as yr to as yr
    sed -i.bak 's/ as yr/ as yr/g' "$file"
    
    # Update yirage. to yirage.
    sed -i.bak 's/yirage\./yirage\./g' "$file"
    
    # Update docstrings and comments
    sed -i.bak 's/YiRage/YiRage/g' "$file"
    
    rm -f "${file}.bak"
    
    echo -e "${GREEN}✓${NC} Updated: $file"
    ((UPDATED_FILES++))
done

echo ""

# Step 4: Update CMake files
echo -e "${BLUE}[Step 4] Updating CMake files...${NC}"

if [ -f "CMakeLists.txt" ]; then
    sed -i.bak 's/project(YIRAGE/project(YIRAGE/g' CMakeLists.txt
    sed -i.bak 's/yirage_runtime/yirage_runtime/g' CMakeLists.txt
    sed -i.bak 's/YIRAGE_/YIRAGE_/g' CMakeLists.txt
    sed -i.bak 's/YiRage/YiRage/g' CMakeLists.txt
    rm -f CMakeLists.txt.bak
    echo -e "${GREEN}✓${NC} Updated: CMakeLists.txt"
    ((UPDATED_FILES++))
fi

echo ""

# Step 5: Update setup.py
echo -e "${BLUE}[Step 5] Updating setup.py...${NC}"

if [ -f "setup.py" ]; then
    sed -i.bak 's/name="yirage-project"/name="yirage"/g' setup.py
    sed -i.bak 's/"yirage/"yirage/g' setup.py
    sed -i.bak 's/YiRage/YiRage/g' setup.py
    sed -i.bak 's/yirage_path/yirage_path/g' setup.py
    sed -i.bak 's/yirage_runtime/yirage_runtime/g' setup.py
    rm -f setup.py.bak
    echo -e "${GREEN}✓${NC} Updated: setup.py"
    ((UPDATED_FILES++))
fi

echo ""

# Step 6: Update Python package version
echo -e "${BLUE}[Step 6] Updating Python version file...${NC}"

if [ -f "python/yirage/version.py" ]; then
    # Keep version.py as is, just update comments
    sed -i.bak 's/YiRage/YiRage/g' python/yirage/version.py
    rm -f python/yirage/version.py.bak
    echo -e "${GREEN}✓${NC} Updated: python/yirage/version.py"
fi

echo ""

# Step 7: Update documentation
echo -e "${BLUE}[Step 7] Updating documentation...${NC}"

find . -type f -name "*.md" | while read file; do
    # Update import examples: import yirage as yr -> import yirage as yr
    sed -i.bak 's/import yirage as yr/import yirage as yr/g' "$file"
    sed -i.bak 's/import yirage/import yirage/g' "$file"
    
    # Update yr. -> yr.
    sed -i.bak 's/ mi\./ yr\./g' "$file"
    sed -i.bak 's/^mi\./yr\./g' "$file"
    
    # Update namespace references
    sed -i.bak 's/yirage::/yirage::/g' "$file"
    
    # Update YiRage -> YiRage (project name)
    sed -i.bak 's/YiRage Persistent Kernel/YiRage Persistent Kernel/g' "$file"
    sed -i.bak 's/# YiRage/# YiRage/g' "$file"
    
    # But keep original YiRage attribution
    sed -i.bak 's/based on Mirage by CMU/based on Mirage by CMU/g' "$file"
    sed -i.bak 's/Original Mirage Copyright/Original Mirage Copyright/g' "$file"
    
    rm -f "${file}.bak"
    
    echo -e "${GREEN}✓${NC} Updated: $file"
    ((UPDATED_FILES++))
done

echo ""

# Step 8: Update config files
echo -e "${BLUE}[Step 8] Updating config files...${NC}"

for file in config.cmake pyproject.toml MANIFEST.in; do
    if [ -f "$file" ]; then
        sed -i.bak 's/yirage/yirage/g' "$file"
        sed -i.bak 's/YiRage/YiRage/g' "$file"
        rm -f "${file}.bak"
        echo -e "${GREEN}✓${NC} Updated: $file"
        ((UPDATED_FILES++))
    fi
done

echo ""

# Step 9: Update conda environment
echo -e "${BLUE}[Step 9] Updating conda files...${NC}"

if [ -f "conda/yirage.yml" ]; then
    mv conda/yirage.yml conda/yirage.yml
    sed -i.bak 's/name: yirage/name: yirage/g' conda/yirage.yml
    sed -i.bak 's/yirage/yirage/g' conda/yirage.yml
    rm -f conda/yirage.yml.bak
    echo -e "${GREEN}✓${NC} Renamed and updated: conda/yirage.yml"
    ((UPDATED_FILES++))
fi

echo ""

# Step 10: Update Python Cython files
echo -e "${BLUE}[Step 10] Updating Cython files...${NC}"

find python -type f \( -name "*.pyx" -o -name "*.pxd" \) | while read file; do
    sed -i.bak 's/yirage/yirage/g' "$file"
    sed -i.bak 's/from yirage/from yirage/g' "$file"
    rm -f "${file}.bak"
    echo -e "${GREEN}✓${NC} Updated: $file"
    ((UPDATED_FILES++))
done

echo ""

# Summary
echo "=========================================="
echo -e "${GREEN}Renaming completed!${NC}"
echo "=========================================="
echo "  Directories renamed:  $RENAMED_DIRS"
echo "  Files updated:        $UPDATED_FILES"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff"
echo "  2. Test build: pip install -e . -v"
echo "  3. Commit: git commit -am 'refactor: Rename YiRage to YiRage'"
echo "=========================================="

