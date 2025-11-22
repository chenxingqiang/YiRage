# YiRage 重命名计划

**确认**: ✅ 用户已确认  
**执行**: 准备就绪

---

## 🎯 重命名规则

### 规则 1: 目录名
```
include/yirage/  →  include/yirage/
python/yirage/   →  python/yirage/
conda/yirage.yml →  conda/yirage.yml
```

### 规则 2: 命名空间
```cpp
namespace yirage  →  namespace yirage
using namespace yirage  →  using namespace yirage
yirage::backend  →  yirage::backend
```

### 规则 3: Include 路径
```cpp
#include "yirage/..."  →  #include "yirage/..."
#include <yirage/...>  →  #include <yirage/...>
```

### 规则 4: Python 导入
```python
import yirage as yr  →  import yirage as yr
from yirage import   →  from yirage import
yr.function()        →  yr.function()
```

### 规则 5: CMake 项目名
```cmake
project(YIRAGE ...)    →  project(YIRAGE ...)
yirage_runtime         →  yirage_runtime
YIRAGE_SRCS           →  YIRAGE_SRCS
```

### 规则 6: Python 包名
```python
name="yirage-project"  →  name="yirage"
packages=["yirage"]    →  packages=["yirage"]
```

### 规则 7: 注释和文档
```
YiRage (项目名)  →  YiRage
yirage (代码)    →  yirage
```

### 规则 8: 保留原始归属
```
# 保持不变（归属声明）
"based on Mirage by CMU"
"Original Mirage Copyright"
```

---

## 📂 重命名影响范围

### 目录重命名（2 个）
```
✅ include/yirage/  →  include/yirage/
✅ python/yirage/   →  python/yirage/
```

### 文件重命名（1 个）
```
✅ conda/yirage.yml  →  conda/yirage.yml
```

### 文件内容更新（估计 200+ 个文件）

#### C++ 文件（~150 个）
- 所有 `.h` 头文件
- 所有 `.cc` 源文件
- 所有 `.cu` CUDA 文件
- 所有 `.cuh` CUDA 头文件

#### Python 文件（~20 个）
- 所有 `.py` 文件
- 所有 `.pyx` Cython 文件
- 所有 `.pxd` Cython 定义

#### 构建文件（~10 个）
- CMakeLists.txt
- setup.py
- pyproject.toml
- config.cmake
- MANIFEST.in

#### 文档文件（~20 个）
- 所有 `.md` 文件
- 所有 `.rst` 文件

---

## 🔍 详细替换规则

### 在 C++ 代码中

| 原文本 | 替换为 | 示例 |
|--------|--------|------|
| `namespace yirage` | `namespace yirage` | `namespace yirage {` |
| `yirage::` | `yirage::` | `yirage::backend::BackendInterface` |
| `#include "yirage/` | `#include "yirage/` | `#include "yirage/type.h"` |
| `using namespace yirage` | `using namespace yirage` | - |
| `YiRage` (注释) | `YiRage` | `// YiRage backend` |

**保留** (归属声明):
- `"based on Mirage by CMU"`
- `"Original Mirage Copyright"`
- `"derived from YiRage"`

### 在 Python 代码中

| 原文本 | 替换为 | 示例 |
|--------|--------|------|
| `import yirage as yr` | `import yirage as yr` | - |
| `import yirage` | `import yirage` | - |
| `from yirage` | `from yirage` | `from yirage import backend_api` |
| `yirage.` | `yirage.` | `yirage.get_available_backends()` |
| `yr.` | `yr.` | `yr.PersistentKernel()` |

**保留** (归属):
- docstring 中的 YiRage 归属

### 在 CMake 中

| 原文本 | 替换为 |
|--------|--------|
| `project(YIRAGE` | `project(YIRAGE` |
| `yirage_runtime` | `yirage_runtime` |
| `YIRAGE_SRCS` | `YIRAGE_SRCS` |
| `YIRAGE_LINK_LIBS` | `YIRAGE_LINK_LIBS` |
| `YIRAGE_INCLUDE_DIRS` | `YIRAGE_INCLUDE_DIRS` |

### 在 setup.py 中

| 原文本 | 替换为 |
|--------|--------|
| `name="yirage-project"` | `name="yirage"` |
| `yirage_path` | `yirage_path` |
| `yirage.%s` | `yirage.%s` |

---

## ⚠️ 注意事项

### 不要替换的内容

1. **Git 历史中的内容** - 保持不变
2. **LICENSE 文件中的归属** - "based on Mirage by CMU"
3. **NOTICE 文件中的归属** - 原始 YiRage 版权
4. **ATTRIBUTION.md** - 原始项目名称
5. **URL 引用** - `github.com/yirage-project/yirage`

### 需要手动检查的文件

```
LICENSE
NOTICE
ATTRIBUTION.md
README.md (归属部分)
CITATION.bib (如果有)
```

---

## 🚀 执行命令

### 完整重命名流程

```bash
# 1. 备份当前状态
git add .
git commit -m "checkpoint: Before renaming to YiRage"

# 2. 运行重命名脚本
bash scripts/rename_to_yirage.sh

# 3. 检查更改
git diff --stat
git diff | head -200

# 4. 测试编译
pip install -e . -v

# 5. 测试导入
python -c "import yirage as yr; print(yr.get_available_backends())"

# 6. 确认无误后提交
git add .
git commit -m "refactor: Rename YiRage to YiRage

- Rename namespace: yirage -> yirage
- Rename import: mi -> yr
- Rename directories: include/yirage -> include/yirage
- Update all file contents
- Preserve original YiRage attribution"
```

---

## 📊 预期影响

### 文件变更统计（估计）
```
目录重命名:    2 个
文件重命名:    1 个
内容更新:      ~200 个文件
总变更行数:    ~5,000 行
```

### 编译影响
```
✅ 需要重新编译
✅ 需要重新安装 Python 包
✅ 旧的导入方式将失效
```

---

## ✅ 重命名后的使用方式

### Python 使用
```python
# 新的导入方式
import yirage as yr

# 查询后端
backends = yr.get_available_backends()

# 创建 kernel
ypk = yr.PersistentKernel(
    backend="cuda",
    ...
)

# 使用优化器
from yirage.kernel.cuda import CUDAOptimizer
config = CUDAOptimizer.optimize(...)
```

### C++ 使用
```cpp
#include "yirage/backend/backends.h"

using namespace yirage;

auto& registry = backend::BackendRegistry::get_instance();
```

---

## 🎯 准备就绪！

**脚本已创建**: ✅ `scripts/rename_to_yirage.sh`  
**计划已完成**: ✅ 本文档  
**备份建议**: ✅ Git commit checkpoint

**执行命令**:
```bash
bash scripts/rename_to_yirage.sh
```

需要我现在执行重命名吗？

