# YiRage 版权更新 - 快速指南

**项目名**: YiRage ✅  
**全称**: Yield Revolutionary AGile Engine ✅  
**确认**: 用户已确认 ✅

---

## 🎯 版权更新方案

### 标准版权头

**用于所有新增文件（52 个）**:
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
 * This file is part of YiRage (Yield Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */
```

---

## 📋 文件清单（58 个需要更新）

### ✅ 类别 A: 新增文件（52 个）- YiRage 完整版权

#### Backend 层（21 个文件）
```bash
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

#### Kernel 层（16 个文件）
```bash
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

#### Search 层（12 个文件）
```bash
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

#### Python & 测试（3 个文件）
```bash
python/yirage/backend_api.py
tests/backend/test_backend_registry.cc
demo/backend_selection_demo.py
```

### ✅ 类别 B: 修改文件（6 个）- 双版权声明

```bash
include/yirage/type.h
include/yirage/config.h
python/yirage/__init__.py
config.cmake
CMakeLists.txt
setup.py
```

---

## 🚀 执行方式

### 方式 1: 自动批量更新（推荐）

```bash
# 1. 给脚本执行权限
chmod +x scripts/update_copyright_yirage.sh

# 2. 运行更新脚本
bash scripts/update_copyright_yirage.sh

# 3. 检查结果
git diff | head -100

# 4. 确认无误后
git add .
git commit -m "chore: Update copyright to YiRage (Chen Xingqiang 2025)"
```

### 方式 2: 手动示例（单个文件）

```bash
# 示例：更新 mps_strategy.cc
FILE="src/search/backend_strategies/mps_strategy.cc"

# 创建新版权头
cat > /tmp/yirage_header.txt << 'EOF'
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
 * This file is part of YiRage (Yield Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */
EOF

# 删除旧版权头（到第一个 */）
sed -i.bak '1,/\*\//d' "$FILE"

# 添加新版权头
cat /tmp/yirage_header.txt "$FILE" > /tmp/temp.cc
mv /tmp/temp.cc "$FILE"

echo "✓ Updated: $FILE"
```

---

## 📄 需要创建的附加文件

### 1. NOTICE 文件

**文件名**: `NOTICE`  
**内容**:
```
YiRage (Yield Revolutionary AGile Engine)
Copyright 2025 Chen Xingqiang

This product includes software developed at Carnegie Mellon University.
  YiRage Project
  Copyright 2023-2024 CMU
  https://github.com/yirage-project/yirage

YiRage is a derivative work that extends YiRage with:
- Multi-backend support architecture (7 backends)
- Hardware-specific kernel optimizations
- Backend-specific search strategies
- Comprehensive documentation and examples

Licensed under the Apache License, Version 2.0.
See LICENSE file for full license text.
```

### 2. ATTRIBUTION.md

**文件名**: `ATTRIBUTION.md`  
**内容**:
```markdown
# YiRage - Attribution and Acknowledgments

## Original Work

**YiRage** - A Multi-Level Superoptimizer for Tensor Programs  
- Developed at: Carnegie Mellon University
- Copyright: 2023-2024 CMU
- License: Apache License 2.0
- Repository: https://github.com/yirage-project/yirage

## Derivative Work

**YiRage** (Yield Revolutionary AGile Engine)  
- Developer: Chen Xingqiang (陈星强)
- Copyright: 2025
- License: Apache License 2.0

YiRage extends YiRage with comprehensive multi-backend support:
- 7 complete backend implementations (CUDA, CPU, MPS, Triton, NKI, cuDNN, MKL)
- Hardware-aware kernel optimizers (13,700+ lines)
- Backend-specific search strategies
- Complete documentation system

## Acknowledgments

We thank the YiRage team at Carnegie Mellon University for creating
the foundational superoptimizer framework that YiRage builds upon.

## License

Both YiRage and YiRage are licensed under the Apache License 2.0,
which allows for derivative works under the same license.
```

### 3. 更新 README.md

在文件顶部添加：
```markdown
# YiRage - Yield Revolutionary AGile Engine

**Based on [YiRage](https://github.com/yirage-project/yirage) by CMU**

YiRage (Yield Revolutionary AGile Engine) extends YiRage with comprehensive
multi-backend support, enabling LLM inference optimization across diverse
hardware platforms.

## YiRage Enhancements

- ✅ **7 Backend Implementations**: CUDA, CPU, MPS, Triton, NKI, cuDNN, MKL
- ✅ **Hardware-Aware Optimizers**: 42+ optimization methods
- ✅ **Search Strategies**: 5 independent + 2 reusable strategies
- ✅ **13,700+ Lines**: Production-ready code
- ✅ **Complete Documentation**: 11 detailed guides

## Original Mirage

YiRage is a compiler and runtime for LLM inference developed at CMU.
For the original YiRage project, see: https://github.com/yirage-project/yirage

## License

Apache License 2.0

Copyright 2025 Chen Xingqiang (YiRage Project)  
Original Mirage Copyright 2023-2024 CMU
```

---

## ✅ 执行检查清单

- [ ] 运行 `update_copyright_yirage.sh` 更新 58 个文件
- [ ] 创建 `NOTICE` 文件
- [ ] 创建 `ATTRIBUTION.md` 文件
- [ ] 更新 `README.md` 顶部
- [ ] 更新 `LICENSE` 文件头部
- [ ] 更新所有文档中的项目名称
- [ ] 提交 git commit

---

## 📞 相关文档

- `LICENSE_STRATEGY.md` - 许可证详细策略
- `COPYRIGHT_HEADERS.txt` - 版权头模板
- `COPYRIGHT_UPDATE_SUMMARY.md` - 更新总结
- `scripts/update_copyright_yirage.sh` - 自动化脚本

---

**准备就绪！您说一声就可以执行批量更新！** 🚀

