# Backend配置机制对比分析：CUDA vs MPS vs CPU

## 为什么需要这些配置信息？

这些配置参数是**kernel优化搜索空间**的核心，直接决定了：

1. **并行度** (Grid/Block Dimensions)
   - Grid dimensions: 决定启动多少个thread blocks
   - Block dimensions: 每个block内有多少个threads
   - 直接影响GPU/CPU的利用率和性能

2. **内存访问模式** (Shared Memory配置)
   - 每个backend的shared memory大小不同
   - 影响数据缓存策略和访存效率

3. **循环优化** (Forloop Ranges)
   - 循环展开程度
   - 影响指令级并行度和寄存器使用

4. **搜索效率** (Search Threads)
   - 多线程并行搜索最优配置
   - 平衡搜索时间和搜索质量

---

## CUDA Backend：动态运行时查询

### 🔍 信息获取方式

CUDA通过**运行时API动态查询GPU硬件**：

```c++
// src/utils/cuda_helper.cu
size_t get_max_shared_mem() {
  int device;
  cudaGetDevice(&device);
  cudaDeviceProp deviceProps;
  cudaGetDeviceProperties(&deviceProps, device);
  return deviceProps.sharedMemPerBlock;
}
```

```python
# python/yirage/utils.py
def get_configurations_from_gpu(rank):
    props = torch.cuda.get_device_properties(rank)
    sm_cnt = props.multi_processor_count  # 查询SM数量
    # 根据SM数量动态确定worker和scheduler配置
    if sm_cnt >= 160:
        worker = 144
    elif sm_cnt >= 132:
        worker = 128
    # ...
```

### 📊 CUDA能查询的硬件信息

通过 `cudaDeviceProp` 可以获取：

| 属性 | 说明 | 示例 (RTX 3090) |
|------|------|-----------------|
| `multiProcessorCount` | SM数量 | 82 |
| `sharedMemPerBlock` | 每个block的shared memory | 48 KB |
| `maxThreadsPerBlock` | 每个block最大线程数 | 1024 |
| `maxThreadsDim[3]` | Block维度限制 | (1024, 1024, 64) |
| `maxGridSize[3]` | Grid维度限制 | (2³¹-1, 65535, 65535) |
| `warpSize` | Warp大小 | 32 |
| `computeCapability` | 计算能力 | 8.6 |
| `hasTensorCores` | 是否有Tensor Core | Yes |
| `memoryBusWidth` | 内存总线宽度 | 384-bit |

### 🎯 CUDA的配置生成策略

```cpp
// src/search/backend_strategies/cuda_strategy.cc
CUDASearchStrategy::CUDASearchStrategy(int compute_capability)
    : compute_capability_(compute_capability),
      has_tensor_cores_(compute_capability >= 70) {}  // ← 根据compute capability决定

std::vector<CandidateConfig>
CUDASearchStrategy::generate_candidates(kernel::Graph const &graph) {
  // 动态生成warp配置
  auto warp_configs = generate_warp_configs(m * n);
  
  // 如果有Tensor Core，生成Tensor Core配置
  if (has_tensor_cores_) {
    tc_configs = generate_tensor_core_configs(m, n, k);
  }
  
  // 动态生成grid/block配置
  auto grid_block_configs = generate_grid_block_configs(m, n);
  
  // 尝试不同的shared memory layouts
  for (auto layout : {ROW_MAJOR, SWIZZLED}) {
    // ...
  }
}
```

### 📐 CUDA的动态配置逻辑

```cpp
// src/kernel/cuda/cuda_optimizer.cc
if (config.use_tensor_core) {
    // Tensor Core配置 (Ampere架构)
    int tile_m = config.mma_m * 4; // 64
    int tile_n = config.mma_n * 4; // 32
    int tile_k = config.mma_k * 2; // 32
    
    config.block_dim_x = 32;  // Warp size
    config.block_dim_y = 4;   // 4 warps
    config.num_warps = 4;
} else {
    // 传统CUDA cores配置
    int block_size = 256;
    int tile_size = 32;
    config.block_dim_x = block_size;
    config.num_warps = (block_size + 31) / 32;
}
```

### 🔧 CUDA的静态常量配置

```cpp
// include/yirage/config.h
#ifdef YIRAGE_BACKEND_CUDA_ENABLED
namespace cuda {
size_t const MAX_DMEM_SIZE = (size_t)2 * 1024 * 1024 * 1024;    // 2 GB
size_t const MAX_SMEM_SIZE = 96 * 1024;                         // 96 KB (Ampere+)
}
#endif

// 通用常量
size_t const MAX_NUM_THREADBLOCKS_PER_KERNEL = 4096;
int const NUM_THREADS_PER_WARP = 32;
int const MAX_NUM_WARP_GROUPS = 4;
```

---

## MPS Backend：静态配置文件

### 🔍 信息获取方式

MPS使用**静态配置文件** (`python/yirage/mps_config.py`)：

```python
def get_mps_search_config():
    # 动态获取CPU核心数（用于搜索并行度）
    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))
    
    return {
        # 静态定义的硬件限制
        "max_num_threadblock_graph_op": 6,
        "max_num_kernel_graph_op": 4,
        
        # 静态定义的搜索空间
        "grid_dims_to_explore": [
            (32, 1, 1),   # SIMD-aligned
            (64, 1, 1),
            (128, 1, 1),
            # ...
        ],
        
        "block_dims_to_explore": [
            (32, 1, 1),   # 1 SIMD group
            (64, 1, 1),   # 2 SIMD groups
            # ...
        ],
    }
```

### ⚠️ MPS的限制与解决方案

**MPS无法完全查询GPU硬件信息**，原因：

1. **Metal API限制**：Metal没有类似 `cudaDeviceProp` 的直接API
2. **权限问题**：需要通过 `MTLDevice` 对象，但Python层不直接暴露
3. **版本差异**：M1/M2/M3规格不同，但Metal API返回的信息有限

**但可以查询系统内存大小**：

```python
# python/yirage/mps_config.py
def get_system_memory_size():
    # 通过 sysctl 命令获取
    result = subprocess.run(['sysctl', '-n', 'hw.memsize'], ...)
    return int(result.stdout.strip())  # 返回字节数

def get_mps_memory_config():
    total_mem = get_system_memory_size()
    # 使用75%的内存，为系统预留空间
    return int(total_mem * 0.75)
```

**不同Mac型号的内存配置**：

| 型号 | 内存选项 |
|------|---------|
| M1/M2/M3 | 8GB, 16GB, 24GB |
| M1/M2/M3 Pro | 16GB, 18GB, 32GB, 36GB |
| M1/M2/M3 Max | 32GB, 48GB, 64GB, 96GB, 128GB |
| M1/M2/M3 Ultra | 64GB, 128GB, 192GB, 512GB |

### 🛠 MPS可以间接获取的信息

```python
# 通过system_profiler (macOS命令行工具)
def get_apple_gpu_info():
    result = subprocess.run(
        ['system_profiler', 'SPDisplaysDataType'],
        capture_output=True, text=True
    )
    # 解析输出来判断是M1/M2/M3及Pro/Max变体
```

但这种方式：
- ❌ 不够可靠（依赖系统命令）
- ❌ 信息有限（只能知道芯片型号）
- ❌ 无法获取实时GPU状态

### 📐 MPS的静态配置基础

基于Apple官方文档的硬件规格：

| 规格 | 所有M系列共通 | 来源 |
|------|--------------|------|
| Threadgroup memory | 32 KB | Metal Feature Set Tables |
| SIMD width | 32 | Metal Programming Guide |
| Max threads/threadgroup | 1024 | Metal Programming Guide |
| 统一内存 | Yes | Apple Silicon架构 |

```cpp
// include/yirage/config.h
#ifdef YIRAGE_BACKEND_MPS_ENABLED
namespace mps {
size_t const MAX_DMEM_SIZE = (size_t)16 * 1024 * 1024 * 1024;   // 16 GB (统一内存)
size_t const MAX_SMEM_SIZE = 64 * 1024;                         // 64 KB (threadgroup)
                                                                 // ↑ 这里配置的是64KB，但实际是32KB！
}
#endif
```

**⚠️ 发现问题**：`config.h` 中定义的 `MAX_SMEM_SIZE = 64KB` 与实际规格不符！

---

## CPU Backend：基于缓存的静态配置

### 🔍 信息获取方式

CPU同样使用**静态配置**，但基于通用CPU架构特性：

```python
def get_cpu_search_config():
    cpu_count = multiprocessing.cpu_count()
    search_threads = max(4, int(cpu_count * 0.75))
    
    return {
        # CPU特定的配置
        "max_num_threadblock_graph_op": 5,
        "max_num_kernel_graph_op": 3,
        
        # 基于缓存层次的配置
        "grid_dims_to_explore": [
            (8, 1, 1),    # L1 cache友好
            (16, 1, 1),   # L2 cache友好
            (32, 1, 1),   # 缓存行对齐
            (64, 1, 1),
        ],
    }
```

### 📊 CPU配置的设计依据

```cpp
#ifdef YIRAGE_BACKEND_CPU_ENABLED
namespace cpu {
size_t const MAX_DMEM_SIZE = (size_t)64 * 1024 * 1024 * 1024;   // 64 GB (系统RAM)
size_t const MAX_SMEM_SIZE = (size_t)32 * 1024 * 1024;          // 32 MB (L3 cache)
}
#endif
```

基于通用CPU架构：
- **L1 Cache**: 32-64 KB (per core)
- **L2 Cache**: 256 KB - 1 MB (per core)
- **L3 Cache**: 8-64 MB (shared)
- **Cache Line**: 64 bytes
- **Vector Width**: AVX2 (256-bit), AVX-512 (512-bit), NEON (128-bit)

---

## 三种Backend配置对比总结

| 维度 | CUDA | MPS | CPU |
|------|------|-----|-----|
| **配置方式** | 🟢 动态查询 + 静态常量 | 🟡 静态配置 | 🟡 静态配置 |
| **硬件感知** | 🟢 完全感知 (运行时API) | 🔴 有限感知 (无直接API) | 🟢 部分感知 (CPU核心数) |
| **配置生成** | 🟢 动态生成候选配置 | 🟡 预定义配置空间 | 🟡 预定义配置空间 |
| **自适应性** | 🟢 强（不同GPU自动适配）| 🟡 中（基于CPU核心适配）| 🟡 中（基于CPU核心适配）|
| **信息来源** | `cudaDeviceProp` API | Apple官方文档 + 经验 | CPU架构知识 + 经验 |
| **准确性** | 🟢 100%准确 | 🟢 基于官方规格 | 🟢 基于通用架构 |
| **灵活性** | 🟢 高（Tensor Core检测等）| 🔴 低（固定配置）| 🔴 低（固定配置）|

---

## 为什么CUDA可以动态查询，MPS不行？

### CUDA的优势

1. **成熟的生态系统**
   - CUDA已发展15年+，API非常完善
   - `cudaDeviceProp` 提供40+个硬件属性
   - 文档详尽，社区成熟

2. **统一的硬件抽象**
   - 所有NVIDIA GPU遵循相同的编程模型
   - SM、Warp、Shared Memory等概念统一
   - 不同代次主要是规模差异

3. **运行时可查询**
   ```c++
   cudaGetDeviceProperties(&props, device);
   // 立即获得所有硬件信息
   ```

### MPS的局限

1. **Metal API设计哲学不同**
   - Metal更底层，更接近硬件
   - 假设程序员了解目标硬件
   - 没有提供全面的"自描述"API

2. **Apple硬件多样性较小**
   - M1/M2/M3虽然性能不同，但架构相似
   - Apple可以假设开发者针对特定芯片优化
   - 不需要像CUDA那样支持上百种GPU

3. **需要通过MTLDevice查询**
   ```objc
   // Objective-C/Swift代码
   let device = MTLCreateSystemDefaultDevice()
   device.maxThreadsPerThreadgroup  // 可以查询部分信息
   ```
   但在Python层，这些信息不易获取

---

## 配置参数的实际影响

### 示例：Grid/Block Dimensions选择

#### CUDA场景 (RTX 3090)
```python
# CUDA自动根据问题规模和GPU能力生成
- SM数量: 82
- 每个SM最大block数: 16
- 理论最大并行block: 82 * 16 = 1312

# 自动选择：
- Grid: (256, 1, 1) → 256个blocks
- Block: (256, 1, 1) → 256个threads/block
- 总线程: 65536个并行
```

#### MPS场景 (M3 Max)
```python
# 静态配置，基于经验：
- GPU核心: 40 (M3 Max)
- SIMD width: 32

# 预定义选择：
- Grid: (128, 1, 1) → 128个threadgroups
- Block: (256, 1, 1) → 256个threads (8 SIMD groups)
- 总线程: 32768个并行
```

### 示例：Shared Memory使用

#### CUDA
```c++
// 运行时查询，动态分配
size_t smem = get_max_shared_mem();  // 96 KB (Ampere)
if (smem >= 96*1024) {
    // 使用大tile
    tile_size = 128;
} else {
    // 使用小tile
    tile_size = 64;
}
```

#### MPS
```python
# 静态假设
# 已知所有M系列都是32KB threadgroup memory
# 代码直接使用固定的tile size
tile_size = 64  # 保守选择
```

---

## 改进建议

### 1. 修正config.h中的MPS配置

```cpp
#ifdef YIRAGE_BACKEND_MPS_ENABLED
namespace mps {
// 编译时上限：64GB (覆盖大多数Mac配置)
// 运行时应该动态查询系统内存
size_t const MAX_DMEM_SIZE = (size_t)64 * 1024 * 1024 * 1024;   // 64 GB (upper limit)
size_t const MAX_SMEM_SIZE = 32 * 1024;  // ← 已修正为32 KB
}
#endif
```

**设计决策**：
- **编译时常量**: 设为64GB，能覆盖绝大多数Mac配置
- **运行时检查**: Python层动态查询实际内存大小
- **安全策略**: 使用75%的系统内存，为OS和其他进程预留空间

### 2. MPS增加运行时检测（可选）

如果要实现类似CUDA的动态查询，需要：

```python
# 通过PyObjC调用Metal API
import Metal
device = Metal.MTLCreateSystemDefaultDevice()

# 可以查询的信息
info = {
    'name': device.name,  # "Apple M3 Max"
    'max_threads_per_threadgroup': device.maxThreadsPerThreadgroup.width,
    'threadgroup_memory_length': device.maxThreadgroupMemoryLength,
    'recommended_max_working_set_size': device.recommendedMaxWorkingSetSize,
}
```

但这需要：
- 添加对PyObjC的依赖
- 处理不同macOS版本的兼容性
- 当前的静态配置已经足够好

### 3. 统一配置接口

```python
# 统一的配置获取接口
def get_backend_config(backend: str) -> dict:
    if backend == "cuda":
        return get_cuda_config_from_runtime()  # 动态查询
    elif backend == "mps":
        return get_mps_search_config()  # 静态配置
    elif backend == "cpu":
        return get_cpu_search_config()  # 静态配置
```

---

## 结论

### 为什么需要这些配置？

1. **定义搜索空间** - 确定哪些kernel配置值得尝试
2. **保证正确性** - 不超过硬件限制（内存、线程数等）
3. **提升效率** - 减少无效配置的搜索，加快优化速度
4. **跨平台适配** - 不同硬件架构需要不同的优化策略

### CUDA vs MPS的关键区别

| 特性 | CUDA | MPS |
|------|------|-----|
| **信息获取** | 运行时API查询 | 基于官方规格的静态配置 |
| **适应性** | 自动适配所有NVIDIA GPU | 手动为M系列定制 |
| **复杂度** | 高（需处理各种GPU） | 低（M系列硬件较统一）|
| **准确性** | 100%（直接读硬件）| 高（基于官方文档）|

### 当前MPS配置的优点

✅ **简单可靠** - 基于Apple官方规格，不依赖复杂的运行时查询  
✅ **足够准确** - M系列芯片架构统一，静态配置已经很好  
✅ **易于维护** - 配置清晰，易于理解和修改  
✅ **跨版本兼容** - 适用于M1/M2/M3及未来M系列

### 需要修复的问题

#### 1. ✅ Threadgroup Memory (已修复)

```cpp
// include/yirage/config.h 第61行
size_t const MAX_SMEM_SIZE = 64 * 1024;  // ❌ 错误：应该是32KB
// 改为：
size_t const MAX_SMEM_SIZE = 32 * 1024;  // ✅ 正确
```

#### 2. ✅ 设备内存配置 (已优化)

**问题**：硬编码 16GB 不能适配所有Mac配置

**解决方案**：两层设计

**C++ 编译时（config.h）**：
```cpp
// 设置较大的上限，覆盖大多数配置
size_t const MAX_DMEM_SIZE = (size_t)64 * 1024 * 1024 * 1024;  // 64 GB
```

**Python 运行时（mps_config.py）**：
```python
def get_mps_memory_config():
    """动态查询系统内存，使用75%作为安全值"""
    total_mem = get_system_memory_size()  # 通过 sysctl 查询
    return int(total_mem * 0.75)  # 为系统预留25%
```

**内存配置对比**：

| Mac型号 | 系统内存 | 可用内存(75%) | 旧配置(16GB) | 改进 |
|---------|---------|--------------|--------------|------|
| M1 (8GB) | 8 GB | 6 GB | ❌ 超出 | ✅ 正确 |
| M1 (16GB) | 16 GB | 12 GB | ✅ 正确 | ✅ 正确 |
| M1 Pro (32GB) | 32 GB | 24 GB | ❌ 限制 | ✅ 充分利用 |
| M1 Max (64GB) | 64 GB | 48 GB | ❌ 严重限制 | ✅ 充分利用 |
| M2 Ultra (192GB) | 192 GB | 144 GB | ❌ 严重限制 | ✅ 充分利用 |
| M3 Max (128GB) | 128 GB | 96 GB | ❌ 严重限制 | ✅ 充分利用 |

**为什么使用75%？**

- ✅ 为macOS系统进程预留内存
- ✅ 为其他应用预留空间  
- ✅ 防止内存碎片导致的分配失败
- ✅ 保持系统响应性

**总体评价**：当前MPS配置设计合理，符合Apple Silicon的实际情况，通过动态内存查询实现了真正的硬件适配。

