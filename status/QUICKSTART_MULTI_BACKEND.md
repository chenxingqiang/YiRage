# YiRage 多后端支持 - 快速开始

## 🚀 5 分钟上手

### 1. 配置后端

编辑 `config.cmake`:
```cmake
set(USE_CUDA ON)   # NVIDIA GPU
set(USE_CPU ON)    # 通用 CPU
set(USE_MPS OFF)   # Apple Silicon (仅 macOS)
```

### 2. 编译安装

```bash
cd yirage
pip install -e . -v
```

### 3. 查询后端

```python
import yirage as yr

# 查看可用后端
print(yr.get_available_backends())
# ['cuda', 'cpu']

# 检查 CUDA
if yr.is_backend_available('cuda'):
    print("CUDA ready!")
```

### 4. 使用后端

```python
# 创建 PersistentKernel
ypk = yr.PersistentKernel(
    mode="decode",
    backend="cuda",              # 指定后端
    fallback_backends=["cpu"],   # 备用
    # ... 其他参数
)
```

## 📚 更多信息

- **完整文档**: [MULTI_BACKEND_README.md](MULTI_BACKEND_README.md)
- **使用指南**: [docs/ypk/backend_usage.md](docs/ypk/backend_usage.md)
- **设计文档**: [docs/ypk/multi_backend_design.md](docs/ypk/multi_backend_design.md)
- **实现总结**: [IMPLEMENTATION_COMPLETE_SUMMARY.md](IMPLEMENTATION_COMPLETE_SUMMARY.md)

## 🎯 支持的后端

| 后端 | 状态 | 硬件 |
|------|------|------|
| CUDA | ✅ 完整支持 | NVIDIA GPU |
| CPU | ✅ 完整支持 | x86/ARM CPU |
| MPS | ⚠️ 基础支持 | Apple Silicon |

## 💡 快速示例

### 查询后端信息
```python
info = yr.get_backend_info('cuda')
print(f"设备数: {info.get('device_count', 0)}")
```

### 列出所有后端
```python
yr.list_backends(verbose=True)
```

### 设置默认后端
```python
yr.set_default_backend('cuda')
```

## 🔧 故障排除

**问题**: 后端不可用
```python
# 检查编译了哪些后端
backends = yr.get_available_backends()
print(f"已编译: {backends}")
```

**问题**: CUDA 找不到
```bash
# 检查 CUDA
nvidia-smi

# 检查环境变量
echo $CUDA_HOME
```

## 📞 获取帮助

- GitHub Issues: https://github.com/yirage-project/yirage/issues
- Slack: https://join.slack.com/t/yiragesystem/...

---

**快速链接**:
- [完整 README](MULTI_BACKEND_README.md)
- [详细文档](docs/ypk/)
- [示例代码](demo/backend_selection_demo.py)





