# YiRage Multi-Backend Support

## 🎯 概述

YiRage 现在支持多种硬件后端，每个后端都有专门优化的 kernel 实现和搜索策略，以充分发挥不同硬件架构的性能潜力。

## 📊 支持的后端

| 后端 | 状态 | 硬件 | 优化重点 |
|------|------|------|----------|
| **CUDA** | ✅ 完整支持 | NVIDIA GPU | Tensor Core, Warp 优化, 共享内存 |
| **CPU** | ✅ 基础支持 | x86/ARM CPU | SIMD, OpenMP, Cache blocking |
| **MPS** | ⚠️ 骨架实现 | Apple Silicon | Metal Shaders, Tile 优化 |
| **Triton** | 🔄 集成中 | NVIDIA GPU | 编译器优化, 自动调优 |
| **NKI** | 🔄 迁移中 | AWS Neuron | 专用指令, 数据流 |
| 其他 | 📋 计划中 | - | - |

## 🏗️ 架构设计

### 分层架构

```
┌─────────────────────────────────────────┐
│         Python API Layer                │
│  - Backend Selection                    │
│  - Configuration                        │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│      Backend Manager (C++)              │
│  - Backend Registry                     │
│  - Factory Pattern                      │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴────────┬─────────┐
        │                 │         │
┌───────▼────────┐ ┌─────▼───┐ ┌──▼────┐
│ CUDA Backend   │ │CPU      │ │MPS    │
│ - Kernel Impl  │ │Backend  │ │Backend│
│ - Search       │ │         │ │       │
│ - Optimization │ │         │ │       │
└────────────────┘ └─────────┘ └───────┘
```

### 关键组件

#### 1. 后端抽象层
- **BackendInterface**: 统一后端接口
- **BackendRegistry**: 后端注册管理
- **BackendFactory**: 后端工厂

#### 2. Kernel 优化层
- **KernelConfig**: 通用配置基类
- **KernelExecutor**: Kernel 执行器
- **KernelOptimizer**: 后端特定优化器

#### 3. 搜索策略层
- **SearchStrategy**: 搜索策略接口
- **CandidateGenerator**: 候选配置生成
- **PerformanceEvaluator**: 性能评估

## 📁 目录结构

```
yirage/
├── include/yirage/
│   ├── backend/                    # 后端抽象层
│   │   ├── backend_interface.h
│   │   ├── backend_registry.h
│   │   ├── cuda_backend.h
│   │   ├── cpu_backend.h
│   │   └── mps_backend.h
│   │
│   ├── kernel/
│   │   ├── common/                 # 通用 Kernel 接口
│   │   │   └── kernel_interface.h
│   │   ├── cuda/                   # CUDA 专用
│   │   │   └── cuda_kernel_config.h
│   │   ├── cpu/                    # CPU 专用 (计划)
│   │   └── mps/                    # MPS 专用 (计划)
│   │
│   └── search/
│       ├── common/                 # 通用搜索接口
│       │   └── search_strategy.h
│       └── backend_strategies/     # 后端策略 (计划)
│           ├── cuda_strategy.h
│           ├── cpu_strategy.h
│           └── mps_strategy.h
│
├── src/
│   ├── backend/                    # 后端实现
│   │   ├── backend_registry.cc
│   │   ├── cuda_backend.cc
│   │   ├── cpu_backend.cc
│   │   └── mps_backend.cc
│   │
│   ├── kernel/
│   │   ├── cuda/                   # CUDA Kernels
│   │   │   ├── kernels/            # 当前实现
│   │   │   └── optimized/          # 优化版本 (计划)
│   │   ├── cpu/                    # CPU Kernels (计划)
│   │   └── mps/                    # MPS Shaders (计划)
│   │
│   └── search/
│       ├── search.cc               # 搜索调度
│       └── backend_strategies/     # 策略实现 (计划)
│
├── python/yirage/
│   ├── backend_api.py              # 后端查询 API
│   └── kernel.py                   # Kernel 图 API (扩展中)
│
├── docs/ypk/
│   ├── multi_backend_design.md               # 设计文档
│   ├── backend_usage.md                      # 使用指南
│   ├── BACKEND_KERNEL_OPTIMIZATION_DESIGN.md # Kernel 优化设计
│   └── BACKEND_OPTIMIZATION_SUMMARY.md       # 实现总结
│
└── tests/backend/
    └── test_backend_registry.cc    # 后端测试
```

## 🚀 快速开始

### 1. 配置编译选项

编辑 `config.cmake`:

```cmake
# 启用需要的后端
set(USE_CUDA ON)
set(USE_CPU ON)
set(USE_MPS OFF)      # macOS only
set(USE_OPENMP ON)
set(USE_TRITON ON)
```

### 2. 编译安装

```bash
cd yirage
pip install -e . -v
```

### 3. 查询可用后端

```python
import yirage as yr

# 列出所有可用后端
backends = yr.get_available_backends()
print(f"Available backends: {backends}")

# 检查特定后端
if yr.is_backend_available('cuda'):
    print("CUDA is available")

# 获取后端详细信息
info = yr.get_backend_info('cuda')
print(info)
```

### 4. 使用特定后端

```python
# 创建 PersistentKernel 时指定后端
ypk = yr.PersistentKernel(
    mode="decode",
    backend="cuda",  # 指定后端
    fallback_backends=["cpu"],  # 备用后端
    # ... 其他参数
)

# 或者为 Kernel Graph 指定后端
graph = yr.new_kernel_graph()
graph.superoptimize(
    backend="cuda",
    backend_config={
        "use_tensor_core": True,
        "max_warps": 32
    }
)
```

## 📖 文档

### 设计文档
- **[多后端设计](docs/ypk/multi_backend_design.md)** - 完整的架构设计
- **[Kernel 优化设计](docs/ypk/BACKEND_KERNEL_OPTIMIZATION_DESIGN.md)** - Kernel 层优化架构
- **[实现总结](docs/ypk/BACKEND_OPTIMIZATION_SUMMARY.md)** - 实现状态和计划

### 使用指南
- **[后端使用指南](docs/ypk/backend_usage.md)** - 详细的使用说明和示例

### 实现文档
- **[多后端实现总结](docs/ypk/MULTI_BACKEND_IMPLEMENTATION_SUMMARY.md)** - 文件清单和修改记录
- **[变更日志](CHANGELOG_MULTI_BACKEND.md)** - 详细的变更记录

## 🔧 后端特性对比

### CUDA Backend
- ✅ Tensor Core 支持
- ✅ 共享内存优化
- ✅ Warp 级优化
- ✅ CUTLASS 集成
- ✅ 多设备支持
- ⚠️ 需要 NVIDIA GPU 和 CUDA Toolkit

### CPU Backend
- ✅ OpenMP 并行
- ✅ SIMD 向量化 (SSE/AVX/AVX512)
- ✅ Cache blocking
- ✅ 跨平台支持
- ⚠️ 性能低于 GPU (适合小模型或 Debug)

### MPS Backend
- ⚠️ 骨架实现
- 📋 Metal shader 优化
- 📋 统一内存利用
- 📋 Apple Silicon 专用指令
- ⚠️ 仅支持 macOS 12.3+

## 🎯 性能优化策略

### CUDA 优化
1. **Warp 级优化**: 最大化 warp 利用率
2. **共享内存**: Swizzling 避免 bank conflict
3. **Tensor Core**: 自动选择最优 MMA 配置
4. **Memory Coalescing**: 优化全局内存访问模式

### CPU 优化
1. **Cache Blocking**: 根据 L1/L2/L3 cache 优化 tile 大小
2. **SIMD**: AVX2/AVX512 向量化
3. **OpenMP**: 多线程并行
4. **数据预取**: 优化内存访问延迟

### MPS 优化
1. **Threadgroup 优化**: 最大化 GPU 占用率
2. **Tile 内存**: 高效利用 threadgroup 内存
3. **访问模式**: 优化内存访问 coalescing

## 🧪 测试

### 单元测试

```bash
# 编译测试
cd yirage/build
make test_backend_registry

# 运行测试
./tests/backend/test_backend_registry
```

### Python 测试

```bash
cd yirage
python demo/backend_selection_demo.py
```

### 性能基准

```python
import yirage as yr
import time

def benchmark_backend(backend):
    # 创建和编译
    ypk = yr.PersistentKernel(backend=backend, ...)
    ypk.compile()
    
    # 预热
    for _ in range(10):
        ypk()
    
    # 测试
    start = time.time()
    for _ in range(100):
        ypk()
    end = time.time()
    
    return (end - start) / 100

# 比较各后端
for backend in yr.get_available_backends():
    latency = benchmark_backend(backend)
    print(f"{backend}: {latency*1000:.2f} ms")
```

## 📊 实现状态

### ✅ 已完成
- [x] 后端抽象层设计和实现
- [x] CUDA 后端完整支持
- [x] CPU 后端基础支持
- [x] MPS 后端骨架
- [x] 后端注册和查询机制
- [x] Python API 集成
- [x] 构建系统多后端支持
- [x] 核心 Kernel 接口设计
- [x] 搜索策略接口设计
- [x] CUDA Kernel 配置和优化器设计
- [x] 完整文档

### 🔄 进行中
- [ ] CUDA 优化 Kernel 实现
- [ ] CPU 优化 Kernel 实现
- [ ] MPS 完整实现
- [ ] 后端特定搜索策略实现

### 📋 计划中
- [ ] Triton 后端集成
- [ ] NKI 后端迁移
- [ ] CUDNN 后端
- [ ] MKL/MKLDNN 后端
- [ ] 自动调优系统
- [ ] 性能分析工具
- [ ] 混合精度支持

## 🤝 贡献

### 添加新后端

1. **实现 BackendInterface**
   ```cpp
   class MyBackend : public BackendInterface {
       // 实现所有虚函数
   };
   
   REGISTER_BACKEND(MyBackend);
   ```

2. **创建 Kernel 配置**
   ```cpp
   struct MyKernelConfig : public KernelConfig {
       // 后端特定配置
   };
   ```

3. **实现搜索策略**
   ```cpp
   class MySearchStrategy : public SearchStrategy {
       // 实现搜索逻辑
   };
   ```

4. **添加到构建系统**
   - 在 `config.cmake` 添加 `USE_MY_BACKEND`
   - 在 `CMakeLists.txt` 添加编译规则

5. **更新文档和测试**

参考现有的 `CUDABackend` 或 `CPUBackend` 实现。

## 📞 支持

- **问题反馈**: [GitHub Issues](https://github.com/yirage-project/yirage/issues)
- **讨论**: [Slack Channel](https://join.slack.com/t/yiragesystem/shared_invite/...)
- **文档**: [docs/ypk/](docs/ypk/)

## 📄 许可证

Apache License 2.0

---

**维护者**: YiRage Team  
**版本**: 1.0.0-alpha  
**最后更新**: 2025-11-21





