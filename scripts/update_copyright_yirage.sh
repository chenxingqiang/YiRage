#!/bin/bash
# YiRage Copyright Update Script
# Updates copyright headers for all YiRage-created files

set -e

YIRAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$YIRAGE_ROOT"

echo "=========================================="
echo "YiRage Copyright Update"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# Define copyright headers
read -r -d '' NEW_CPP_COPYRIGHT << 'EOF' || true
/* Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This file is part of YiRage (Yi Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */
EOF

read -r -d '' NEW_PYTHON_COPYRIGHT << 'EOF' || true
# Copyright 2025 Chen Xingqiang (YiRage Project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is part of YiRage (Yi Revolutionary AGile Engine),
# a derivative work based on Mirage by CMU.
# Original Mirage Copyright 2023-2024 CMU.
EOF

read -r -d '' MODIFIED_COPYRIGHT << 'EOF' || true
/* Original Copyright 2023-2024 CMU
 * Modifications Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified for YiRage (Yi Revolutionary AGile Engine).
 */
EOF

# List of completely new files (use YiRage copyright)
declare -a NEW_FILES=(
    # Backend layer
    "include/yirage/backend/backend_interface.h"
    "include/yirage/backend/backend_registry.h"
    "include/yirage/backend/backends.h"
    "include/yirage/backend/cuda_backend.h"
    "include/yirage/backend/cpu_backend.h"
    "include/yirage/backend/mps_backend.h"
    "include/yirage/backend/triton_backend.h"
    "include/yirage/backend/nki_backend.h"
    "include/yirage/backend/cudnn_backend.h"
    "include/yirage/backend/mkl_backend.h"
    "src/backend/backend_utils.cc"
    "src/backend/backend_registry.cc"
    "src/backend/backends.cc"
    "src/backend/cuda_backend.cc"
    "src/backend/cpu_backend.cc"
    "src/backend/mps_backend.cc"
    "src/backend/mps_backend_complete.cc"
    "src/backend/triton_backend.cc"
    "src/backend/nki_backend.cc"
    "src/backend/cudnn_backend.cc"
    "src/backend/mkl_backend.cc"
    
    # Kernel layer
    "include/yirage/kernel/common/kernel_interface.h"
    "include/yirage/kernel/cuda/cuda_kernel_config.h"
    "include/yirage/kernel/cpu/cpu_kernel_config.h"
    "include/yirage/kernel/mps/mps_kernel_config.h"
    "include/yirage/kernel/triton/triton_kernel_config.h"
    "include/yirage/kernel/nki/nki_kernel_config.h"
    "include/yirage/kernel/cudnn/cudnn_kernel_config.h"
    "include/yirage/kernel/mkl/mkl_kernel_config.h"
    "src/kernel/common/kernel_factory.cc"
    "src/kernel/cuda/cuda_optimizer.cc"
    "src/kernel/cpu/cpu_optimizer.cc"
    "src/kernel/mps/mps_optimizer.cc"
    "src/kernel/triton/triton_optimizer.cc"
    "src/kernel/nki/nki_optimizer.cc"
    "src/kernel/cudnn/cudnn_optimizer.cc"
    "src/kernel/mkl/mkl_optimizer.cc"
    
    # Search layer
    "include/yirage/search/common/search_strategy.h"
    "include/yirage/search/backend_strategies/cuda_strategy.h"
    "include/yirage/search/backend_strategies/cpu_strategy.h"
    "include/yirage/search/backend_strategies/mps_strategy.h"
    "include/yirage/search/backend_strategies/triton_strategy.h"
    "include/yirage/search/backend_strategies/nki_strategy.h"
    "src/search/common/search_strategy_factory.cc"
    "src/search/backend_strategies/cuda_strategy.cc"
    "src/search/backend_strategies/cpu_strategy.cc"
    "src/search/backend_strategies/mps_strategy.cc"
    "src/search/backend_strategies/triton_strategy.cc"
    "src/search/backend_strategies/nki_strategy.cc"
    
    # Python & Tests
    "python/yirage/backend_api.py"
    "tests/backend/test_backend_registry.cc"
    "demo/backend_selection_demo.py"
)

# List of modified files (use dual copyright)
declare -a MODIFIED_FILES=(
    "include/yirage/type.h"
    "include/yirage/config.h"
    "python/yirage/__init__.py"
)

# Update new files
echo -e "${BLUE}[1] Updating new files with YiRage copyright...${NC}"
for file in "${NEW_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        # Create temporary file with new copyright
        tmp_file=$(mktemp)
        
        # Add new copyright
        echo "$NEW_CPP_COPYRIGHT" > "$tmp_file"
        echo "" >> "$tmp_file"
        
        # Add original content (skip old copyright)
        sed '1,/\*\//d' "$file" >> "$tmp_file"
        
        # Replace original file
        mv "$tmp_file" "$file"
        echo -e "${GREEN}✓${NC} $file"
    fi
done

# Update modified files
echo ""
echo -e "${BLUE}[2] Updating modified files with dual copyright...${NC}"
for file in "${MODIFIED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        tmp_file=$(mktemp)
        
        # Add dual copyright
        echo "$MODIFIED_COPYRIGHT" > "$tmp_file"
        echo "" >> "$tmp_file"
        
        # Add original content (skip old copyright)
        sed '1,/\*\//d' "$file" >> "$tmp_file"
        
        # Replace original file
        mv "$tmp_file" "$file"
        echo -e "${GREEN}✓${NC} $file"
    fi
done

echo ""
echo "=========================================="
echo "Copyright update completed!"
echo "  New files:      ${#NEW_FILES[@]}"
echo "  Modified files: ${#MODIFIED_FILES[@]}"
echo "=========================================="

