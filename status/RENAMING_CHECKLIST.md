# YiRage 重命名执行清单

**执行前检查**: ✅ 准备就绪

---

## 📋 执行前确认

### 重命名范围
- [x] 目录名: `yirage/` → `yirage/`
- [x] 命名空间: `namespace yirage` → `namespace yirage`
- [x] Include 路径: `#include "yirage/` → `#include "yirage/`
- [x] Python 导入: `import yirage as yr` → `import yirage as yr`
- [x] CMake 项目: `project(YIRAGE)` → `project(YIRAGE)`
- [x] 包名: `yirage-project` → `yirage`

### 保留内容
- [x] Git 历史
- [x] LICENSE 中的 YiRage 归属
- [x] 文档中的 "based on Mirage by CMU"
- [x] NOTICE 中的原始版权

---

## 🚀 执行步骤

### 1. 创建 Checkpoint ✅
```bash
git add .
git commit -m "checkpoint: Before renaming to YiRage"
```

### 2. 执行重命名
```bash
bash scripts/rename_to_yirage.sh
```

### 3. 验证结果
```bash
# 检查目录是否重命名成功
ls -la include/ | grep yirage
ls -la python/ | grep yirage

# 检查代码更新
git diff --stat

# 检查关键文件
grep "namespace yirage" include/yirage/type.h
grep "import yirage" python/yirage/__init__.py
```

### 4. 测试编译
```bash
pip install -e . -v
```

### 5. 测试导入
```python
import yirage as yr
print(yr.get_available_backends())
```

### 6. 提交更改
```bash
git add .
git commit -m "refactor: Rename YiRage to YiRage

- Directories: include/yirage -> include/yirage
- Namespace: yirage -> yirage  
- Python import: mi -> yr
- Package name: yirage-project -> yirage
- Preserve original YiRage attribution"
```

---

## 📊 预期更改

### 目录结构变化
```
Before:                      After:
include/yirage/       →      include/yirage/
python/yirage/        →      python/yirage/
conda/yirage.yml      →      conda/yirage.yml
```

### 代码示例变化

**C++ 代码**:
```cpp
// Before
#include "yirage/backend/backends.h"
namespace yirage {
  backend::BackendRegistry::get_instance();
}

// After
#include "yirage/backend/backends.h"
namespace yirage {
  backend::BackendRegistry::get_instance();
}
```

**Python 代码**:
```python
# Before
import yirage as yr
backends = yr.get_available_backends()

# After
import yirage as yr
backends = yr.get_available_backends()
```

**CMake**:
```cmake
# Before
project(YIRAGE LANGUAGES C CXX CUDA)
add_library(yirage_runtime ${YIRAGE_SRCS})

# After
project(YIRAGE LANGUAGES C CXX CUDA)
add_library(yirage_runtime ${YIRAGE_SRCS})
```

---

## ⚠️ 重要提醒

### 破坏性更改

这是一个**破坏性更改**，会导致：
- ❌ 旧的导入方式失效
- ❌ 需要重新编译
- ❌ 需要重新安装
- ✅ 但所有功能保持不变

### 兼容性

如果需要保持兼容，可以在 `python/yirage/__init__.py` 添加：
```python
# Backward compatibility alias
import sys
sys.modules['yirage'] = sys.modules['yirage']
```

---

## ✅ 执行确认

**准备状态**: ✅ 脚本已创建  
**Git 状态**: 检查中...  
**备份建议**: 创建 checkpoint commit  

**执行命令**:
```bash
cd /Users/xingqiangchen/yirage
bash scripts/rename_to_yirage.sh
```

---

**等待您的确认指令！** 🎯

需要我现在执行重命名吗？

