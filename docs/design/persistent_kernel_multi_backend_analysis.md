# Persistent Kernel 多后端适配分析

## 实现状态 (Implementation Status)

**更新日期: 2025-12-31**

多后端抽象层**已完整实现**，共 **18 个文件**，**7195 行代码**。

### 已完成组件

| 组件 | 文件 | 代码行数 | 状态 |
|------|------|---------|------|
| 核心接口 | `pk_backend_interface.h` | 548 | ✅ 已完成 |
| 运行时适配器 | `pk_runtime_adapter.h/.cc` | 1056 | ✅ 已完成 |
| 工具函数 | `pk_utils.h` | 415 | ✅ 已完成 |
| CUDA 后端 | `cuda_pk_backend.h/.cc` | 783 | ✅ 已完成 |
| CPU 后端 | `cpu_pk_backend.h/.cc` | 772 | ✅ 已完成 |
| Ascend 后端 | `ascend_pk_backend.h/.cc` | 766 | ✅ 已完成 |
| MACA 后端 | `maca_pk_backend.h/.cc` | 772 | ✅ 已完成 |
| 后端工厂 | `pk_backend_factory.cc` | 171 | ✅ 已完成 |
| Cython 绑定 | `pk_backend.pxd/.pyx` | 733 | ✅ 已完成 |
| C++ 测试 | `test_pk_backends.cc` | 530 | ✅ 已完成 |
| Python 测试 | `test_pk_backends.py` | 477 | ✅ 已完成 |

### 新增文件结构

```
include/persistent_kernel/
├── pk_backend_interface.h           # 核心接口定义 (548 行)
├── pk_runtime_adapter.h             # 运行时适配器 (397 行)
├── pk_utils.h                       # 工具函数 (415 行)
└── backends/
    ├── pk_backends.h                # 统一引入头文件 (172 行)
    ├── cuda_pk_backend.h            # CUDA 后端头文件 (190 行)
    ├── cpu_pk_backend.h             # CPU 后端头文件 (192 行)
    ├── ascend_pk_backend.h          # Ascend 后端头文件 (184 行)
    └── maca_pk_backend.h            # MACA 后端头文件 (185 行)

src/persistent_kernel/
├── cuda_pk_backend.cc               # CUDA 实现 (593 行)
├── cpu_pk_backend.cc                # CPU 实现 (580 行)
├── ascend_pk_backend.cc             # Ascend 实现 (582 行)
├── maca_pk_backend.cc               # MACA 实现 (587 行)
├── pk_backend_factory.cc            # 工厂实现 (171 行)
└── pk_runtime_adapter.cc            # 运行时适配器实现 (659 行)

python/yirage/_cython/
├── pk_backend.pxd                   # Cython 声明 (195 行)
└── pk_backend.pyx                   # Cython 实现 (538 行)

tests/
├── persistent_kernel/
│   └── test_pk_backends.cc          # C++ 测试 (530 行)
└── python/
    └── test_pk_backends.py          # Python 测试 (477 行)
```

### 核心功能

1. **多后端抽象接口** (`PKBackendInterface`)
   - 后端类型: CUDA, CPU, MPS, ASCEND, MACA, TRITON, NKI
   - 执行模式: OFFLINE, ONLINE, ONEPASS, EAGER, GRAPH, STREAMING
   - 数据类型: FP32, FP16, BF16, INT8, INT4, FP8

2. **内存管理接口** (`PKMemoryAllocator`)
   - allocate / free
   - copy_h2d / copy_d2h / copy_d2d / copy_h2d_async
   - memset, get_total_memory, get_free_memory

3. **原子操作接口** (`PKAtomicOps`)
   - fetch_add_u64/u32, fetch_sub_u64/u32
   - compare_exchange_u64/u32
   - store_release_u64, load_acquire_u64
   - memory_fence, thread_fence

4. **任务执行器接口** (`PKTaskExecutor`)
   - supports_task, execute, get_shared_memory_size
   - 支持所有 LLM 推理任务类型

5. **模式适配器** (`PKModeAdapter`)
   - PKOfflineModeAdapter: 批量处理
   - PKOnlineModeAdapter: 流式生成
   - PKOnepassModeAdapter: 单次前向
   - PKEagerModeAdapter: 即时执行
   - PKGraphModeAdapter: 图编译执行

6. **批量管理器** (`PKBatchManager`)
   - 请求队列管理
   - 页内存分配
   - KV Cache 管理

7. **工具函数** (`pk_utils.h`)
   - 后端选择和优先级
   - 配置构建器 (PKRuntimeConfigBuilder)
   - 性能分析器 (PKProfiler)
   - 日志系统 (PK_DEBUG/INFO/WARN/ERROR)

### C++ 使用示例

```cpp
#include "persistent_kernel/backends/pk_backends.h"

using namespace yirage::persistent_kernel;

// 1. 创建后端
auto backend = create_pk_backend(PKBackendType::CUDA, 0);
if (!backend || !backend->is_available()) {
    backend = create_pk_backend(PKBackendType::CPU, 0);
}

// 2. 检查能力
auto caps = backend->get_capabilities();
std::cout << "Backend: " << backend->get_display_name() << std::endl;
std::cout << "TMA: " << caps.supports_tma << std::endl;
std::cout << "Tensor Cores: " << caps.supports_tensor_cores << std::endl;

// 3. 配置运行时
auto config = PKRuntimeConfigBuilder()
    .mode(PKMode::ONLINE)
    .workers(8)
    .max_seq_length(4096)
    .batch_config(16, 64)
    .build();

// 4. 初始化并执行
if (backend->initialize(config)) {
    // 创建模式适配器
    auto adapter = create_mode_adapter(backend.get(), PKMode::ONLINE);
    adapter->initialize(config);
    
    // 执行推理
    while (adapter->step()) {
        // 处理输出...
    }
    
    adapter->finalize();
    backend->finalize();
}
```

### Python 使用示例

```python
from yirage._cython import pk_backend

# 获取最佳后端
backend = pk_backend.get_best_backend()
print(f"Using: {backend.get_display_name()}")

# 检查能力
caps = backend.get_capabilities()
print(f"Supports TMA: {caps.supports_tma}")
print(f"Supported modes: {[pk_backend.mode_to_name(m) for m in caps.supported_modes]}")

# 初始化
config = pk_backend.PyPKRuntimeConfig()
config.mode = pk_backend.PyPKMode.ONLINE
config.num_workers = 4

if backend.initialize(config):
    # 使用后端...
    backend.synchronize()
    backend.finalize()
```

---

## 当前状态

**Persistent Kernel 完全绑定 CUDA**，以下是具体问题：

### 1. CUDA 专用代码统计

| 文件 | CUDA 依赖项数量 | 主要依赖 |
|------|----------------|----------|
| `persistent_kernel.cuh` | 28+ | `__device__`, `__global__`, atomics |
| `runtime_header.h` | 5+ | `cudaStream_t`, `cuda_runtime.h` |
| `tasks/*.cuh` | 500+ | 全部是 CUDA kernels |
| `tma.cuh` | 100+ | Hopper TMA 专用 |

### 2. Mode 实现分析

当前 Mode 通过编译时宏实现：

```cpp
// persistent_kernel.cuh - Line 167
#ifdef MODE_OFFLINE
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
    // OFFLINE 模式: 批量处理所有请求
    // ~150 行 CUDA 设备代码
}
#endif

// persistent_kernel.cuh - Line 325
#ifdef MODE_ONLINE  
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
    // ONLINE 模式: 单请求处理
    // ~25 行 CUDA 设备代码
}
#endif
```

### 3. 硬件架构分支

```cpp
// persistent_kernel.cuh - Line 32-38
#if defined(YIRAGE_GRACE_HOPPER)
#include "tasks/hopper/task_header.cuh"      // SM90+ 专用
#elif defined(YIRAGE_GRACE_BLACKWELL)
#include "tasks/blackwell/task_header.cuh"   // SM100+ 专用
#else
#include "tasks/ampere/task_header.cuh"      // SM80 默认
#endif
```

---

## 需要改造的核心组件

### 1. RuntimeConfig 结构体

**当前** (`runtime_header.h:251-266`):
```cpp
struct RuntimeConfig {
  // ... 通用字段 ...
  
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  int *prompt_length;     // Mode 专用
  int *request_ids;       // Mode 专用
#endif
  
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
  cudaStream_t worker_stream, scheduler_stream;  // CUDA 专用
#endif
};
```

**需要改为**:
```cpp
struct RuntimeConfig {
  // 通用字段
  void* device_memory;
  size_t memory_size;
  
  // Mode 配置 (运行时)
  PKMode mode;
  bool mode_offline_enabled;
  bool mode_online_enabled;
  
  // 后端抽象句柄
  void* backend_context;  // 后端特定上下文
  void* stream_handle;    // 通用流/队列句柄
};
```

### 2. 设备函数抽象

**当前**:
```cpp
__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc,
                  RuntimeConfig const &runtime_config);
```

**需要改为**:
```cpp
// 后端无关的任务接口
class TaskExecutor {
public:
  virtual void execute(
    const TaskDesc& desc,
    const RuntimeConfig& config,
    void* shared_memory,
    int thread_id
  ) = 0;
};

// CUDA 实现
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
class CudaTaskExecutor : public TaskExecutor {
  __device__ void execute(...) override;
};
#endif

// CPU 实现
class CpuTaskExecutor : public TaskExecutor {
  void execute(...) override;  // 普通 C++ 函数
};
```

### 3. 内存管理抽象

**当前** (`persistent_kernel.cuh`):
```cpp
template <typename T>
__host__ T *gpu_malloc(size_t size) {
  T *ptr;
  CUDA_CHECK(cudaMalloc(&ptr, size));
  return ptr;
}

template <typename T>
__host__ void gpu_free(T *ptr) {
  CUDA_CHECK(cudaFree(ptr));
}
```

**需要改为**:
```cpp
// 后端无关的内存接口
class MemoryAllocator {
public:
  virtual void* allocate(size_t size) = 0;
  virtual void free(void* ptr) = 0;
  virtual void copy_h2d(void* dst, const void* src, size_t size) = 0;
  virtual void copy_d2h(void* dst, const void* src, size_t size) = 0;
};

// 后端实现
class CudaAllocator : public MemoryAllocator { ... };
class CpuAllocator : public MemoryAllocator { ... };
class AscendAllocator : public MemoryAllocator { ... };
```

### 4. 原子操作抽象

**当前** (`mpk_atoms.cuh`):
```cpp
__device__ __forceinline__ unsigned long long int
    atom_add_release_gpu_u64(unsigned long long int *addr,
                              unsigned long long int val) {
  // CUDA PTX 原子操作
  asm volatile("atom.add.release.gpu.u64 %0, [%1], %2;"
               : "=l"(ret)
               : "l"(addr), "l"(val));
  return ret;
}
```

**需要改为**:
```cpp
// 后端无关的原子操作接口
template<typename T>
class AtomicOps {
public:
  virtual T fetch_add(T* addr, T val) = 0;
  virtual T compare_exchange(T* addr, T expected, T desired) = 0;
  virtual void store_release(T* addr, T val) = 0;
  virtual T load_acquire(T* addr) = 0;
};

// CUDA 实现 (使用 PTX)
// CPU 实现 (使用 std::atomic)
// Ascend 实现 (使用 ACL 原子操作)
```

---

## 各后端适配工作量

### CUDA (已有)
- ✅ MODE_OFFLINE
- ✅ MODE_ONLINE  
- ✅ MODE_ONEPASS
- ⚠️ 需要重构为接口实现

### CPU (需新建)
| 组件 | 工作量 | 说明 |
|------|--------|------|
| MemoryAllocator | 1天 | malloc/free |
| TaskExecutor | 3天 | 线程池实现 |
| AtomicOps | 1天 | std::atomic |
| Mode 支持 | 2天 | EAGER, GRAPH |
| **总计** | **~7天** | |

### Ascend NPU (需新建)
| 组件 | 工作量 | 说明 |
|------|--------|------|
| MemoryAllocator | 2天 | ACL 内存接口 |
| TaskExecutor | 5天 | ACL kernel 调用 |
| AtomicOps | 2天 | ACL 原子操作 |
| Mode 支持 | 3天 | OFFLINE, ONLINE, GRAPH |
| **总计** | **~12天** | |

### MACA (需新建)
| 组件 | 工作量 | 说明 |
|------|--------|------|
| MemoryAllocator | 2天 | MACA 内存接口 |
| TaskExecutor | 5天 | MACA kernel 适配 |
| AtomicOps | 2天 | MACA 原子操作 |
| Mode 支持 | 3天 | OFFLINE, ONLINE, ONEPASS |
| **总计** | **~12天** | |

### MPS (需新建)
| 组件 | 工作量 | 说明 |
|------|--------|------|
| MemoryAllocator | 2天 | Metal Buffer |
| TaskExecutor | 5天 | Metal Compute Shader |
| AtomicOps | 2天 | Metal 原子操作 |
| Mode 支持 | 2天 | EAGER, GRAPH |
| **总计** | **~11天** | |

---

## 改造路线图

### Phase 1: 接口抽象 (1周)

```
include/persistent_kernel/
├── backend_interface.h     # 后端接口定义
├── memory_interface.h      # 内存抽象
├── atomic_interface.h      # 原子操作抽象
├── task_interface.h        # 任务执行抽象
└── mode_interface.h        # Mode 运行时选择
```

### Phase 2: CUDA 重构 (1周)

将现有 CUDA 代码重构为接口实现：
```
src/persistent_kernel/cuda/
├── cuda_backend.cu
├── cuda_memory.cu
├── cuda_atomic.cu
└── cuda_tasks.cu
```

### Phase 3: CPU 后端 (1周)

```
src/persistent_kernel/cpu/
├── cpu_backend.cc
├── cpu_memory.cc
├── cpu_atomic.cc
└── cpu_tasks.cc
```

### Phase 4: 其他后端 (按需)

```
src/persistent_kernel/ascend/
src/persistent_kernel/maca/
src/persistent_kernel/mps/
```

---

## 关键改动文件

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `persistent_kernel.cuh` | 重构 | 拆分为接口+CUDA实现 |
| `runtime_header.h` | 重构 | 移除CUDA依赖 |
| `mpk_atoms.cuh` | 重构 | 抽象原子操作 |
| `tasks/*.cuh` | 重构 | 每个任务需要多后端实现 |
| Python `persistent_kernel.py` | 更新 | 支持后端选择 |

---

## 编译系统改动

```cmake
# CMakeLists.txt 新增选项
option(YIRAGE_PK_ENABLE_CUDA "Enable CUDA persistent kernel" ON)
option(YIRAGE_PK_ENABLE_CPU "Enable CPU persistent kernel" ON)
option(YIRAGE_PK_ENABLE_ASCEND "Enable Ascend persistent kernel" OFF)
option(YIRAGE_PK_ENABLE_MACA "Enable MACA persistent kernel" OFF)
option(YIRAGE_PK_ENABLE_MPS "Enable MPS persistent kernel" OFF)

# 运行时 Mode 选择 (替代编译时宏)
option(YIRAGE_PK_RUNTIME_MODE "Enable runtime mode selection" ON)
```

---

## 总结

**当前 Persistent Kernel 完全绑定 CUDA**，需要：

1. **接口抽象层** - 解耦硬件依赖
2. **运行时 Mode 选择** - 替代编译时宏
3. **每个后端的完整实现** - 内存、原子操作、任务执行
4. **任务注册系统** - 按后端注册任务实现

**预计总工作量**: 6-8 周（1人）
