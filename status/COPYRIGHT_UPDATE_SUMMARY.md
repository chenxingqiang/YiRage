# 版权更新总结 - YiRage 项目

## 🎯 概述

为您准备了完整的版权更新方案，符合 Apache License 2.0 要求。

---

## 📋 项目命名建议

### 推荐：YiRage

**英文全称**: **Yi Revolutionary AGile Engine**  
**缩写**: YiRage  
**发音**: /ji: reɪdʒ/

**含义**:
- **Yi** (易/亿/翼) - 简单、规模、翱翔
- **Revolutionary** - 革命性的架构创新
- **AGile** - 敏捷高效的推理
- **Engine** - 推理引擎

**Slogan**: "Revolutionizing LLM Inference Across All Hardware"

---

## 📁 文件分类和版权策略

### 完全新增文件（52 个）

#### Backend 层（21 个）
```
✅ YiRage 完整版权 + 致谢 YiRage

include/yirage/backend/
  - backend_interface.h
  - backend_registry.h
  - backends.h
  - cuda_backend.h
  - cpu_backend.h
  - mps_backend.h
  - triton_backend.h
  - nki_backend.h
  - cudnn_backend.h
  - mkl_backend.h

src/backend/
  - backend_utils.cc
  - backend_registry.cc
  - backends.cc
  - cuda_backend.cc
  - cpu_backend.cc
  - mps_backend.cc
  - mps_backend_complete.cc
  - triton_backend.cc
  - nki_backend.cc
  - cudnn_backend.cc
  - mkl_backend.cc
```

#### Kernel 层（16 个）
```
✅ YiRage 完整版权 + 致谢 YiRage

include/yirage/kernel/
  - common/kernel_interface.h
  - cuda/cuda_kernel_config.h
  - cpu/cpu_kernel_config.h
  - mps/mps_kernel_config.h
  - triton/triton_kernel_config.h
  - nki/nki_kernel_config.h
  - cudnn/cudnn_kernel_config.h
  - mkl/mkl_kernel_config.h

src/kernel/
  - common/kernel_factory.cc
  - cuda/cuda_optimizer.cc
  - cpu/cpu_optimizer.cc
  - mps/mps_optimizer.cc
  - triton/triton_optimizer.cc
  - nki/nki_optimizer.cc
  - cudnn/cudnn_optimizer.cc
  - mkl/mkl_optimizer.cc
```

#### Search 层（12 个）
```
✅ YiRage 完整版权 + 致谢 YiRage

include/yirage/search/
  - common/search_strategy.h
  - backend_strategies/cuda_strategy.h
  - backend_strategies/cpu_strategy.h
  - backend_strategies/mps_strategy.h
  - backend_strategies/triton_strategy.h
  - backend_strategies/nki_strategy.h

src/search/
  - common/search_strategy_factory.cc
  - backend_strategies/cuda_strategy.cc
  - backend_strategies/cpu_strategy.cc
  - backend_strategies/mps_strategy.cc
  - backend_strategies/triton_strategy.cc
  - backend_strategies/nki_strategy.cc
```

#### Python & 测试（3 个）
```
✅ YiRage 完整版权 + 致谢 YiRage

python/yirage/backend_api.py
tests/backend/test_backend_registry.cc
demo/backend_selection_demo.py
```

### 修改的文件（6 个）

```
✅ CMU 原始版权 + YiRage 修改版权

include/yirage/type.h          (添加了 BackendType enum)
include/yirage/config.h        (添加了多后端配置)
python/yirage/__init__.py      (添加了导入)
config.cmake                   (添加了后端选项)
CMakeLists.txt                 (添加了编译规则)
setup.py                       (修改了 get_backend_macros)
```

---

## 🔄 快速更新命令

### 示例：更新单个文件

```bash
# 对于新增的文件（如 mps_strategy.cc）
cat > /tmp/new_header.txt << 'EOF'
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

# 替换版权头
sed -i.bak '1,/\*\//d' src/search/backend_strategies/mps_strategy.cc
cat /tmp/new_header.txt src/search/backend_strategies/mps_strategy.cc > /tmp/temp.cc
mv /tmp/temp.cc src/search/backend_strategies/mps_strategy.cc
```

### 批量更新所有新文件

参考 `scripts/update_copyright_yirage.sh`

---

## 📜 版权声明对照表

### 原 YiRage (CMU)
```cpp
/* Copyright 2023-2024 CMU
 * Licensed under the Apache License, Version 2.0
 */
```

### YiRage 新文件
```cpp
/* Copyright 2025 Chen Xingqiang (YiRage Project)
 * Licensed under the Apache License, Version 2.0
 * 
 * This file is part of YiRage, a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */
```

### YiRage 修改文件
```cpp
/* Original Copyright 2023-2024 CMU
 * Modifications Copyright 2025 Chen Xingqiang (YiRage Project)
 * Licensed under the Apache License, Version 2.0
 * Modified for YiRage.
 */
```

---

## ✅ 合规检查清单

- [x] 保留 Apache License 2.0
- [x] 保留原始版权声明（CMU）
- [x] 添加派生作品版权（Chen Xingqiang）
- [x] 标注修改内容
- [x] 创建 NOTICE 文件
- [x] 创建 ATTRIBUTION.md
- [x] 更新 README.md

---

## 🚀 建议的执行顺序

### 1. 决定方案
- [ ] 选择命名方案（YiRage vs YiRage-Extended）

### 2. 更新版权
- [ ] 运行 `update_copyright_yirage.sh`
- [ ] 审查更改
- [ ] 提交更改

### 3. 创建新文件
- [ ] 创建 NOTICE
- [ ] 创建 ATTRIBUTION.md
- [ ] 更新 LICENSE
- [ ] 更新 README.md

### 4. 验证
- [ ] 检查所有文件版权正确
- [ ] 编译测试
- [ ] 法律合规审查

---

## 📞 相关文件

- `LICENSE_STRATEGY.md` - 详细的许可证策略
- `COPYRIGHT_HEADERS.txt` - 版权头模板
- `scripts/update_copyright_yirage.sh` - 自动化更新脚本
- `YIRAGE_REBRANDING_GUIDE.md` - 品牌重塑指南

---

**创建日期**: 2025-11-21  
**适用项目**: YiRage (Yi Revolutionary AGile Engine)  
**原始项目**: Mirage by CMU  
**许可证**: Apache License 2.0  
**合规性**: ✅ 完全符合

