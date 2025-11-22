# YiRage 品牌重塑指南

## 📋 项目重命名

### 新项目名称

**项目名**: **YiRage**  
**英文全称**: **Yi Revolutionary AGile Engine**  
**中文名**: 易锐智算引擎  
**创始人**: Chen Xingqiang  
**年份**: 2025

**项目定位**: 
基于 YiRage 的派生项目，专注于多后端 LLM 推理优化

---

## 🎯 版权策略

### Apache 2.0 许可证合规

根据 Apache License 2.0 第 4 条要求：

1. ✅ **保留原始版权声明**（YiRage/CMU）
2. ✅ **添加派生作品版权**（YiRage/Chen Xingqiang）
3. ✅ **标注修改内容**
4. ✅ **保持相同许可证**（Apache 2.0）

### 文件分类

#### 类别 A: YiRage 完全新增（52 个文件）
使用 **YiRage 版权** + 致谢 YiRage

#### 类别 B: 修改 YiRage 代码（6 个文件）
使用 **双版权声明**

---

## 📝 版权头模板

### 新增文件模板

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

### 修改文件模板

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
 *
 * Modified for YiRage (Yi Revolutionary AGile Engine).
 */
```

---

## 📂 需要更新的文件清单

### 完全新增文件（52 个）- 使用 YiRage 版权

```bash
# Backend layer (21 files)
include/yirage/backend/*.h          (10 个)
src/backend/*.cc                    (11 个)

# Kernel layer (16 files)
include/yirage/kernel/common/*.h    (1 个)
include/yirage/kernel/cuda/*.h      (1 个)
include/yirage/kernel/cpu/*.h       (1 个)
include/yirage/kernel/mps/*.h       (1 个)
include/yirage/kernel/triton/*.h    (1 个)
include/yirage/kernel/nki/*.h       (1 个)
include/yirage/kernel/cudnn/*.h     (1 个)
include/yirage/kernel/mkl/*.h       (1 个)
src/kernel/*/optimizer.cc           (8 个)

# Search layer (12 files)
include/yirage/search/common/*.h    (1 个)
include/yirage/search/backend_strategies/*.h  (5 个)
src/search/common/*.cc              (1 个)
src/search/backend_strategies/*.cc  (5 个)

# Python & Tests (3 files)
python/yirage/backend_api.py
tests/backend/test_backend_registry.cc
demo/backend_selection_demo.py
```

### 修改的文件（6 个）- 使用双版权

```bash
include/yirage/type.h
include/yirage/config.h
python/yirage/__init__.py
config.cmake
CMakeLists.txt
setup.py
```

---

## 🚀 执行步骤

### 方法 1: 手动执行（推荐用于审查）

**步骤**:
1. 为每个新文件替换版权头为 YiRage 版权
2. 为每个修改文件替换为双版权声明
3. 创建 NOTICE 文件
4. 更新 README.md

### 方法 2: 脚本自动执行

```bash
# 1. 运行版权更新脚本
chmod +x scripts/update_copyright_yirage.sh
bash scripts/update_copyright_yirage.sh

# 2. 审查更改
git diff

# 3. 确认无误后提交
git add .
git commit -m "chore: Update copyright to YiRage (Chen Xingqiang 2025)"
```

---

## 📄 需要创建的新文件

### 1. NOTICE 文件

文件名: `NOTICE`
```
YiRage (Yi Revolutionary AGile Engine)
Copyright 2025 Chen Xingqiang

This product includes software developed at Carnegie Mellon University.
  Copyright 2023-2024 CMU
  Original Mirage project: https://github.com/yirage-project/yirage

This project is a derivative work of YiRage, licensed under the
Apache License 2.0.

YiRage contains:
- Original Mirage code (Copyright CMU)
- Multi-backend extensions (Copyright Chen Xingqiang)
- Hardware-specific optimizations (Copyright Chen Xingqiang)
- Backend search strategies (Copyright Chen Xingqiang)

All code is licensed under the Apache License 2.0.
See LICENSE file for details.
```

### 2. 更新的 LICENSE 文件

保持 Apache License 2.0，但在顶部添加：

```
Copyright 2025 Chen Xingqiang (YiRage Project)
Copyright 2023-2024 Carnegie Mellon University (Original Mirage)

Licensed under the Apache License, Version 2.0 (the "License");
...
```

### 3. ATTRIBUTION.md

```markdown
# YiRage - Attribution

## Original Work

This project is based on **YiRage** by Carnegie Mellon University:
- Original repository: https://github.com/yirage-project/yirage
- Copyright: 2023-2024 CMU
- License: Apache License 2.0

## Derivative Work

**YiRage** (Yi Revolutionary AGile Engine) extends YiRage with:
- Multi-backend support architecture
- Hardware-specific kernel optimizations
- Backend-specific search strategies
- 7 complete backend implementations

**YiRage Contributions**:
- Copyright: 2025 Chen Xingqiang
- License: Apache License 2.0

## Acknowledgments

We thank the YiRage team at CMU for their foundational work.
```

---

## 🔄 品牌更新清单

### README 更新

```markdown
# YiRage - Yi Revolutionary AGile Engine

**Based on [YiRage](https://github.com/yirage-project/yirage) by CMU**

YiRage extends YiRage with multi-backend support for:
- CUDA, CPU, MPS, Triton, NKI, cuDNN, MKL
- Hardware-specific optimizations
- Backend search strategies

## Original Mirage

YiRage is a compiler for LLM inference developed at CMU.
YiRage builds upon YiRage's foundation with extensive
multi-backend enhancements.

## YiRage Enhancements

- ✅ 7 complete backend implementations
- ✅ Hardware-aware kernel optimizers
- ✅ Backend-specific search strategies
- ✅ 15,000+ lines of new code
- ✅ Production-ready architecture

## License

Apache License 2.0

See LICENSE and NOTICE files for details.
```

---

## 📊 更新总结

### 需要修改的组件

| 组件 | 操作 | 数量 |
|------|------|------|
| 新增文件版权头 | 替换为 YiRage 版权 | 52 个 |
| 修改文件版权头 | 添加双版权声明 | 6 个 |
| NOTICE 文件 | 新建 | 1 个 |
| LICENSE 文件 | 更新头部 | 1 个 |
| ATTRIBUTION.md | 新建 | 1 个 |
| README.md | 更新 | 1 个 |
| 文档 | 添加 YiRage 标识 | 11 个 |

---

## ⚖️ 法律合规检查

### Apache 2.0 要求

✅ **第 1 条**: 授予版权许可 - 保持  
✅ **第 2 条**: 授予专利许可 - 保持  
✅ **第 3 条**: 再分发 - 保持源码形式  
✅ **第 4 条**: 再分发要求 - 满足：
   - (a) 提供 Apache 2.0 副本 ✅
   - (b) 修改文件标注 ✅
   - (c) 保留原始声明 ✅
   - (d) 包含 NOTICE 文件 ✅

✅ **第 5 条**: 贡献提交 - N/A  
✅ **第 6 条**: 商标 - YiRage 是新商标  
✅ **第 7 条**: 免责声明 - 保持  
✅ **第 8 条**: 责任限制 - 保持  
✅ **第 9 条**: 保证和责任 - 保持

**合规状态**: ✅ **完全符合 Apache 2.0 要求**

---

## 🎯 推荐执行方案

### 保守方案（推荐）

1. **保留 YiRage 名称**，仅更新版权：
   ```
   Original Copyright 2023-2024 CMU
   Multi-Backend Extensions Copyright 2025 Chen Xingqiang
   ```

2. **在文档中说明是派生项目**

3. **优点**: 
   - 避免混淆
   - 保持与上游兼容
   - 符合学术规范

### 激进方案（需谨慎）

1. **Fork 项目为 YiRage**

2. **完全独立品牌**

3. **注意事项**:
   - 必须明确标注基于 YiRage
   - 保留所有原始版权
   - 添加 NOTICE 文件

---

## 💡 建议

### 我的建议：保守方案

**原因**:
1. YiRage 是知名项目，保留名称有利于推广
2. 您的贡献（多后端）可以作为扩展模块
3. 更容易合并回上游（如果需要）
4. 符合开源社区最佳实践

**版权声明**:
```cpp
/* Original Copyright 2023-2024 CMU (YiRage Project)
 * Multi-Backend Architecture Copyright 2025 Chen Xingqiang
 *
 * Licensed under the Apache License, Version 2.0
 * ...
 */
```

**README**:
```markdown
# YiRage - Multi-Backend Extension

**Original Mirage**: CMU (https://github.com/yirage-project/yirage)  
**Multi-Backend Extension**: Chen Xingqiang (2025)

This repository extends YiRage with comprehensive multi-backend support...
```

---

## 📞 总结

### 选项 1: 完全重命名为 YiRage
- ✅ 独立品牌
- ⚠️ 需要明确标注派生自 YiRage
- ⚠️ 可能与上游分离

### 选项 2: YiRage Multi-Backend by Chen Xingqiang（推荐）
- ✅ 保留 YiRage 品牌认知度
- ✅ 明确标注贡献者
- ✅ 易于合并回上游
- ✅ 符合开源惯例

**我建议您选择选项 2**，但提供了两种方案的完整实施方案供您选择。

您希望采用哪种方案？

