# YiRage YPK 多后端支持设计文档

## 1. 设计目标

为 YiRage Persistent Kernel (YPK) 添加对 PyTorch 多种后端的支持，实现：
- 支持 PyTorch 的所有主要后端
- 编译时可选择多个后端同时编译
- 运行时可动态选择后端执行
- 保持向后兼容性
- 最小化代码修改

## 2. 支持的后端列表

参考 `torch.backends`，计划支持以下后端：

| 后端类型 | 优先级 | 说明 |
|---------|--------|------|
| CUDA | P0 | 已支持，NVIDIA GPU |
| CPU | P0 | CPU 执行 |
| MPS | P1 | Apple Metal Performance Shaders |
| CUDNN | P1 | CUDA Deep Neural Network library |
| MKL | P2 | Intel Math Kernel Library |
| MKLDNN | P2 | Intel oneDNN |
| OpenMP | P2 | OpenMP 并行 |
| NKI | P2 | 已支持，AWS Neuron |
| Triton | P2 | 已在 Python 层支持 |
| cuSPARSELt | P3 | CUDA 稀疏矩阵 |
| MHA | P3 | Multi-Head Attention |
| NNPACK | P3 | Neural Network PACKage |
| opt_einsum | P3 | 优化的 einsum |
| Xeon | P3 | Intel Xeon 优化 |

## 3. 架构设计

### 3.1 后端抽象层架构

```
┌─────────────────────────────────────────────────┐
│           Python API Layer                      │
│   PersistentKernel(backend="cuda", ...)        │
└────────────────┬────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────┐
│         Backend Manager (C++)                   │
│  - Backend Registration                         │
│  - Backend Selection                            │
│  - Capability Query                             │
└────────────────┬────────────────────────────────┘
                 │
     ┌───────────┴───────────┬──────────┬─────────┐
     │                       │          │         │
┌────▼──────┐  ┌─────▼──────┐  ┌──▼───┐  ┌──▼───┐
│CUDA       │  │CPU         │  │MPS   │  │ ...  │
│Backend    │  │Backend     │  │Backend│  │      │
└───────────┘  └────────────┘  └──────┘  └──────┘
```

### 3.2 核心接口定义

#### BackendInterface (C++)
```cpp
class BackendInterface {
public:
    virtual ~BackendInterface() = default;
    
    // 后端信息
    virtual BackendType get_type() const = 0;
    virtual std::string get_name() const = 0;
    virtual bool is_available() const = 0;
    
    // 编译相关
    virtual bool compile(const CompileContext& ctx) = 0;
    virtual std::string get_compile_flags() const = 0;
    
    // 运行时相关
    virtual void* allocate_memory(size_t size) = 0;
    virtual void free_memory(void* ptr) = 0;
    virtual void synchronize() = 0;
    
    // 能力查询
    virtual size_t get_max_memory() const = 0;
    virtual size_t get_max_shared_memory() const = 0;
    virtual bool supports_data_type(DataType dt) const = 0;
};
```

#### BackendRegistry (C++)
```cpp
class BackendRegistry {
public:
    static BackendRegistry& instance();
    
    void register_backend(std::unique_ptr<BackendInterface> backend);
    BackendInterface* get_backend(BackendType type);
    BackendInterface* get_backend(const std::string& name);
    
    std::vector<BackendType> get_available_backends() const;
    bool is_backend_available(BackendType type) const;
    
private:
    std::unordered_map<BackendType, std::unique_ptr<BackendInterface>> backends_;
    std::unordered_map<std::string, BackendType> name_to_type_;
};
```

### 3.3 后端类型扩展

修改 `include/yirage/type.h`:

```cpp
enum BackendType {
  // GPU Backends
  BT_CUDA = 0,
  BT_MPS = 1,
  BT_CUDNN = 2,
  BT_CUSPARSELT = 3,
  
  // CPU Backends
  BT_CPU = 10,
  BT_MKL = 11,
  BT_MKLDNN = 12,
  BT_OPENMP = 13,
  BT_XEON = 14,
  
  // Specialized Backends
  BT_NKI = 20,
  BT_TRITON = 21,
  BT_MHA = 22,
  BT_NNPACK = 23,
  BT_OPT_EINSUM = 24,
  
  BT_UNKNOWN = 999,
};

// Backend metadata
struct BackendInfo {
    BackendType type;
    std::string name;
    std::string display_name;
    bool requires_gpu;
    std::vector<std::string> required_libs;
};
```

## 4. 编译系统改造

### 4.1 config.cmake 改造

从互斥选择改为多选：

```cmake
# 后端选项 - 可以多选
set(USE_CUDA ON)
set(USE_CPU ON)
set(USE_MPS OFF)
set(USE_CUDNN OFF)
set(USE_MKL OFF)
set(USE_MKLDNN OFF)
set(USE_OPENMP ON)
set(USE_NKI OFF)
set(USE_TRITON ON)
set(USE_CUSPARSELT OFF)
set(USE_MHA OFF)
set(USE_NNPACK OFF)
set(USE_OPT_EINSUM OFF)
set(USE_XEON OFF)

# 其他选项
set(BUILD_CPP_EXAMPLES OFF)
set(USE_FORMAL_VERIFIER OFF)
```

### 4.2 CMakeLists.txt 改造

```cmake
# 检查并启用后端
if(USE_CUDA)
    add_compile_definitions(YIRAGE_BACKEND_CUDA_ENABLED)
    # CUDA specific setup
endif()

if(USE_CPU)
    add_compile_definitions(YIRAGE_BACKEND_CPU_ENABLED)
    # CPU specific setup
endif()

if(USE_MPS)
    add_compile_definitions(YIRAGE_BACKEND_MPS_ENABLED)
    # MPS specific setup
endif()

# ... 其他后端
```

### 4.3 setup.py 改造

```python
def get_backend_macros(config_file):
    """读取 config.cmake 并返回所有启用的后端宏"""
    backend_flags = {
        "USE_CUDA": None,
        "USE_CPU": None,
        "USE_MPS": None,
        "USE_CUDNN": None,
        "USE_MKL": None,
        "USE_MKLDNN": None,
        "USE_OPENMP": None,
        "USE_NKI": None,
        "USE_TRITON": None,
        # ... 其他后端
    }
    
    pattern = re.compile(r'^\s*set\s*\(\s*([A-Z_]+)\s+(ON|OFF)\s*\)')
    
    with open(config_file, 'r') as f:
        for line in f:
            match = pattern.match(line)
            if match:
                var, val = match.groups()
                if var in backend_flags:
                    backend_flags[var] = (val.upper() == "ON")
    
    macros = []
    # 为每个启用的后端添加宏
    for flag_name, enabled in backend_flags.items():
        if enabled:
            backend_name = flag_name.replace("USE_", "")
            macros.append((f"YIRAGE_BACKEND_{backend_name}_ENABLED", None))
    
    # 至少需要一个后端
    if not any(backend_flags.values()):
        raise ValueError("At least one backend must be enabled in config.cmake")
    
    return macros
```

## 5. 配置文件改造

### 5.1 config.h 改造

移除互斥检查，改为按需配置：

```cpp
namespace yirage {
namespace config {

// 每个后端的配置
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
    namespace cuda {
        size_t const MAX_DMEM_SIZE = (size_t)2 * 1024 * 1024 * 1024;    // 2 GB
        size_t const MAX_SMEM_SIZE = 96 * 1024;                         // 96 KB
    }
#endif

#ifdef YIRAGE_BACKEND_CPU_ENABLED
    namespace cpu {
        size_t const MAX_DMEM_SIZE = (size_t)64 * 1024 * 1024 * 1024;   // 64 GB
        size_t const MAX_SMEM_SIZE = (size_t)32 * 1024 * 1024;          // 32 MB
    }
#endif

#ifdef YIRAGE_BACKEND_MPS_ENABLED
    namespace mps {
        size_t const MAX_DMEM_SIZE = (size_t)16 * 1024 * 1024 * 1024;   // 16 GB
        size_t const MAX_SMEM_SIZE = (size_t)64 * 1024;                 // 64 KB
    }
#endif

// ... 其他后端配置

} // namespace config
} // namespace yirage
```

## 6. Python API 改造

### 6.1 PersistentKernel 扩展

```python
class PersistentKernel:
    def __init__(
        self,
        mode: str,
        backend: str = "cuda",  # 新增：后端选择
        fallback_backends: list = None,  # 新增：fallback 后端列表
        world_size: int = 1,
        mpi_rank: int = 0,
        # ... 其他参数
    ):
        # 验证后端是否可用
        available_backends = self._get_available_backends()
        if backend not in available_backends:
            if fallback_backends:
                for fb in fallback_backends:
                    if fb in available_backends:
                        backend = fb
                        break
                else:
                    raise ValueError(f"No available backend from {[backend] + fallback_backends}")
            else:
                raise ValueError(f"Backend '{backend}' not available. Available: {available_backends}")
        
        self.backend = backend
        # ... 初始化
    
    @staticmethod
    def _get_available_backends():
        """查询编译时启用的后端"""
        return core.get_available_backends()
```

### 6.2 新增后端查询 API

```python
def get_available_backends() -> list:
    """返回所有可用的后端"""
    return core.get_available_backends()

def is_backend_available(backend: str) -> bool:
    """检查指定后端是否可用"""
    return backend in get_available_backends()

def get_backend_info(backend: str) -> dict:
    """获取后端详细信息"""
    return core.get_backend_info(backend)
```

## 7. 实现路线图

### Phase 1: 基础架构 (Week 1-2)
- [ ] 扩展 BackendType 枚举
- [ ] 实现 BackendInterface 抽象类
- [ ] 实现 BackendRegistry 单例
- [ ] 修改 config.cmake 支持多后端
- [ ] 修改 CMakeLists.txt 支持多后端编译
- [ ] 修改 setup.py 支持多后端宏定义

### Phase 2: 核心后端实现 (Week 3-4)
- [ ] 重构 CUDA Backend (已有代码)
- [ ] 实现 CPU Backend
- [ ] 实现 MPS Backend (macOS)
- [ ] 实现 CUDNN Backend
- [ ] 重构 NKI Backend (已有代码)

### Phase 3: Python 绑定 (Week 5)
- [ ] 更新 Cython 绑定
- [ ] 实现 Python 后端查询 API
- [ ] 更新 PersistentKernel 类
- [ ] 更新 KNGraph.superoptimize

### Phase 4: 扩展后端 (Week 6-8)
- [ ] 实现 MKL Backend
- [ ] 实现 MKLDNN Backend
- [ ] 实现 OpenMP Backend
- [ ] 实现 Triton Backend (集成现有代码)
- [ ] 实现其他后端

### Phase 5: 测试与文档 (Week 9-10)
- [ ] 单元测试
- [ ] 集成测试
- [ ] 性能测试
- [ ] 文档更新
- [ ] 示例代码

## 8. 向后兼容性

- 默认后端保持为 CUDA
- 旧的 config.cmake 配置仍然有效
- Python API 默认参数保持不变
- 编译宏 `YIRAGE_BACKEND_USE_CUDA` 自动映射到 `YIRAGE_BACKEND_CUDA_ENABLED`

## 9. 测试策略

### 9.1 单元测试
- 每个 Backend 实现的独立测试
- BackendRegistry 的注册和查询测试
- 编译系统的多后端测试

### 9.2 集成测试
- 多后端同时编译测试
- 运行时后端切换测试
- Fallback 机制测试

### 9.3 性能测试
- 各后端性能基准测试
- 后端切换开销测试

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 破坏现有 CUDA 代码 | 高 | 保持向后兼容，充分测试 |
| 编译时间增加 | 中 | 支持选择性编译后端 |
| 二进制文件过大 | 中 | 模块化设计，动态加载 |
| 某些后端库缺失 | 低 | 编译时检测，优雅降级 |

## 11. 依赖库

| 后端 | 必需库 | 可选库 |
|------|--------|--------|
| CUDA | CUDA Toolkit, cuBLAS | CUDNN, cuSPARSELt |
| CPU | - | OpenMP |
| MPS | Metal (macOS only) | - |
| MKL | Intel MKL | - |
| MKLDNN | oneDNN | - |
| NKI | AWS Neuron SDK | - |
| Triton | PyTorch, Triton | - |

## 12. 文档更新

需要更新的文档：
- INSTALL.md - 添加各后端安装说明
- README.md - 添加多后端特性说明
- API 文档 - 添加后端选择 API
- 教程 - 添加多后端使用示例

---

**文档版本**: v1.0  
**创建日期**: 2025-11-21  
**最后更新**: 2025-11-21

