# YiRage License Strategy - 许可证策略

## 项目命名

**原项目**: YiRage (by CMU)  
**派生项目**: **YiRage**

### YiRage 可能的英文全名：

1. **YiRage** - "Yi" (易/亿) + "Rage" (Revolutionary AGile Engine)
   - 全称：**Yi Revolutionary AGile Engine**
   - 寓意：易用的革命性敏捷引擎

2. **YiRage** - "Yi" (一/亿) + "Rage" (Rapid AGgregate Engine)
   - 全称：**Yi Rapid AGgregate Engine**  
   - 寓意：快速聚合引擎

3. **YiRage** - "Yi" (易/翼) + "Rage" (Runtime Accelerated GPU Engine)
   - 全称：**Yi Runtime Accelerated GPU Engine**
   - 寓意：运行时加速GPU引擎

4. **YiRage** - "Yi" (艺) + "Rage" (Refined Architecture for GPU Execution)
   - 全称：**Yi Refined Architecture for GPU Execution**
   - 寓意：精炼的GPU执行架构

**推荐**: **Yi Revolutionary AGile Engine (YiRage)**  
**简称**: YiRage

---

## License 兼容策略

### 原项目 License

**YiRage**: Apache License 2.0  
**Copyright**: 2023-2024 CMU

### 派生项目 License 策略

根据 Apache License 2.0 的要求，派生作品应当：

1. **保留原始版权声明**（Apache 2.0 第 4(a) 条）
2. **添加派生作品的版权**
3. **明确标注修改**（Apache 2.0 第 4(b) 条）
4. **保持相同许可证**（Apache 2.0）

### 推荐的版权声明格式

#### 对于新增文件（完全由您创建）

```cpp
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
```

#### 对于修改的现有文件（基于 CMU 代码）

```cpp
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
 */
```

---

## 全局修改策略

### 文件分类

#### 类别 A: 完全新增的文件（由我们创建）✅ 使用新版权

**Backend 层** (10 个文件):
```
include/yirage/backend/backend_interface.h
include/yirage/backend/backend_registry.h
include/yirage/backend/backends.h
include/yirage/backend/cuda_backend.h
include/yirage/backend/cpu_backend.h
include/yirage/backend/mps_backend.h
include/yirage/backend/triton_backend.h
include/yirage/backend/nki_backend.h
include/yirage/backend/cudnn_backend.h
include/yirage/backend/mkl_backend.h

src/backend/backend_utils.cc
src/backend/backend_registry.cc
src/backend/backends.cc
src/backend/cuda_backend.cc
src/backend/cpu_backend.cc
src/backend/mps_backend.cc
src/backend/mps_backend_complete.cc
src/backend/triton_backend.cc
src/backend/nki_backend.cc
src/backend/cudnn_backend.cc
src/backend/mkl_backend.cc
```

**Kernel 优化层** (16 个文件):
```
include/yirage/kernel/common/kernel_interface.h
include/yirage/kernel/cuda/cuda_kernel_config.h
include/yirage/kernel/cpu/cpu_kernel_config.h
include/yirage/kernel/mps/mps_kernel_config.h
include/yirage/kernel/triton/triton_kernel_config.h
include/yirage/kernel/nki/nki_kernel_config.h
include/yirage/kernel/cudnn/cudnn_kernel_config.h
include/yirage/kernel/mkl/mkl_kernel_config.h

src/kernel/common/kernel_factory.cc
src/kernel/cuda/cuda_optimizer.cc
src/kernel/cpu/cpu_optimizer.cc
src/kernel/mps/mps_optimizer.cc
src/kernel/triton/triton_optimizer.cc
src/kernel/nki/nki_optimizer.cc
src/kernel/cudnn/cudnn_optimizer.cc
src/kernel/mkl/mkl_optimizer.cc
```

**Search 策略层** (12 个文件):
```
include/yirage/search/common/search_strategy.h
include/yirage/search/backend_strategies/cuda_strategy.h
include/yirage/search/backend_strategies/cpu_strategy.h
include/yirage/search/backend_strategies/mps_strategy.h
include/yirage/search/backend_strategies/triton_strategy.h
include/yirage/search/backend_strategies/nki_strategy.h

src/search/common/search_strategy_factory.cc
src/search/backend_strategies/cuda_strategy.cc
src/search/backend_strategies/cpu_strategy.cc
src/search/backend_strategies/mps_strategy.cc
src/search/backend_strategies/triton_strategy.cc
src/search/backend_strategies/nki_strategy.cc
```

**Python & 测试** (3 个文件):
```
python/yirage/backend_api.py
tests/backend/test_backend_registry.cc
demo/backend_selection_demo.py
```

**总计**: 52 个完全新增文件 → 使用 YiRage 版权

#### 类别 B: 修改的现有文件 ✅ 使用双版权

```
include/yirage/type.h              (修改了 BackendType enum)
include/yirage/config.h            (修改了多后端配置)
python/yirage/__init__.py          (添加了导入)
config.cmake                       (添加了后端选项)
CMakeLists.txt                     (添加了后端编译)
setup.py                           (修改了 get_backend_macros)
```

**总计**: 6 个修改文件 → 使用双版权声明

---

## 批量修改方案

### 方案 1: 使用脚本批量修改（推荐）

创建 `scripts/update_copyright.sh`:

```bash
#!/bin/bash
# Update copyright for YiRage project

NEW_COPYRIGHT_FULL='/* Copyright 2025 Chen Xingqiang (YiRage Project)
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
 */'

DUAL_COPYRIGHT='/* Original Copyright 2023-2024 CMU
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
 */'

# 新增文件列表
NEW_FILES=(
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
    # ... (所有 52 个新文件)
)

# 修改文件列表
MODIFIED_FILES=(
    "include/yirage/type.h"
    "include/yirage/config.h"
    "python/yirage/__init__.py"
    "config.cmake"
    "CMakeLists.txt"
    "setup.py"
)

echo "Updating copyright for YiRage project..."

# 处理新增文件
for file in "${NEW_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        # 替换现有的 CMU 版权为 YiRage 版权
        sed -i.bak "1,/\*\//c\\
$NEW_COPYRIGHT_FULL" "$file"
        echo "✓ Updated: $file"
    fi
done

# 处理修改文件
for file in "${MODIFIED_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        # 替换为双版权声明
        sed -i.bak "1,/\*\//c\\
$DUAL_COPYRIGHT" "$file"
        echo "✓ Updated: $file"
    fi
done

echo "Done! Copyright updated for YiRage project."
```

### 方案 2: 手动创建标准版权文件

创建 `COPYRIGHT_YIRAGE.txt`:
```
YiRage (Yi Revolutionary AGile Engine)
Copyright 2025 Chen Xingqiang

This project is a derivative work based on YiRage by Carnegie Mellon University.
Original Mirage Copyright 2023-2024 CMU.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 项目重命名策略

### 推荐命名方案

**项目名**: YiRage  
**英文全称**: **Yi Revolutionary AGile Engine**  
**中文名**: 易锐智算引擎

**解释**:
- **Yi** (易): 简单易用
- **Revolutionary**: 革命性的多后端架构
- **AGile**: 敏捷高效
- **Engine**: 推理引擎

**Slogan**: "Revolutionizing LLM Inference Across All Hardware"

### 备选命名

| 名称 | 全称 | 寓意 |
|------|------|------|
| YiRage | Yi Revolutionary AGile Engine | 革命性敏捷引擎 |
| YiRage | Yi Rapid Acceleration Gear Engine | 快速加速齿轮引擎 |
| YiRage | Yi Runtime Adaptive GPU Engine | 运行时自适应GPU引擎 |
| YiRage | Yi Resourceful Architecture for GPUs Everywhere | 无处不在的GPU架构 |

---

## 具体修改方案

### 步骤 1: 创建 YiRage 版权头模板

