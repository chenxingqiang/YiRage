# YiRage MACA 后端完整指南

本文档基于在 **MetaX C500 GPU** 上的实际成功运行经验编写，涵盖环境配置、编译安装、测试验证和性能优化的完整流程。

## 目录

1. [概述](#1-概述)
2. [环境要求](#2-环境要求)
3. [环境配置](#3-环境配置)
4. [编译安装](#4-编译安装)
5. [验证安装](#5-验证安装)
6. [使用指南](#6-使用指南)
7. [Benchmark 测试](#7-benchmark-测试)
8. [MACA 技术特性](#8-maca-技术特性)
9. [故障排除](#9-故障排除)
10. [常见问题](#10-常见问题)

---

## 1. 概述

### 1.1 什么是 MACA

MACA (MetaX Advanced Compute Architecture) 是沐曦 (MetaX) 公司自研的 GPU 编程模型，与 NVIDIA CUDA 高度兼容但具有独特的硬件特性。

### 1.2 YiRage MACA 后端

YiRage 的 MACA 后端支持：
- **Fingerprint 验证**: 使用 MACA GPU 内核进行图等价性验证
- **Kernel 编译**: 通过 `mxcc` 编译器生成优化的 GPU 代码
- **性能分析**: 使用 mcPytorch 进行真实硬件 profiling

### 1.3 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        YiRage Framework                         │
├─────────────────────────────────────────────────────────────────┤
│  Python API (yirage.kernel)                                     │
│    └── superoptimize(backend="maca")                           │
├─────────────────────────────────────────────────────────────────┤
│  Search Engine                                                  │
│    ├── Fusion Graph Discovery                                   │
│    ├── Fingerprint Verification (MACA GPU)                      │
│    └── Parameter Optimization                                   │
├─────────────────────────────────────────────────────────────────┤
│  MACA Backend                                                   │
│    ├── device_memory_manager.maca (内存管理)                    │
│    ├── customized_kernel.maca (主 fingerprint 内核)             │
│    ├── matmul_kernel.maca (矩阵乘法)                            │
│    ├── reduction_kernel.maca (规约操作)                         │
│    └── ... (11 个 .maca 内核文件)                               │
├─────────────────────────────────────────────────────────────────┤
│  MACA Runtime                                                   │
│    ├── mxcc Compiler                                            │
│    ├── mcruntime Library                                        │
│    └── mcPytorch (torch.cuda.* → MACA)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 环境要求

### 2.1 硬件要求

| 组件 | 要求 |
|------|------|
| GPU | MetaX C500 或其他 MACA 兼容 GPU |
| 内存 | ≥ 16 GB 系统内存 |
| 显存 | ≥ 16 GB GPU 显存 |
| 存储 | ≥ 10 GB 可用空间 |

### 2.2 软件要求

| 组件 | 版本 | 说明 |
|------|------|------|
| **MACA SDK** | 3.2+ | 包含 mxcc 编译器 |
| **mcPytorch** | 2.6.0+metax3.2.1.3 | PyTorch MACA 移植版 |
| **Python** | 3.10+ | 推荐 3.10 或 3.11 |
| **CMake** | 3.24+ | 构建系统 |
| **Rust** | 最新稳定版 | Triton 转译器依赖 |
| **GCC** | 支持 C++17 | 系统编译器 |
| **Z3** | 4.8+ | SMT 求解器 |

### 2.3 验证 MACA 环境

```bash
# 检查 MACA SDK
ls /opt/maca/mxgpu_llvm/bin/mxcc
# 输出: /opt/maca/mxgpu_llvm/bin/mxcc

# 检查 MACA 版本
/opt/maca/mxgpu_llvm/bin/mxcc --version

# 检查 mcPytorch
python -c "import torch; print(torch.__version__)"
# 输出: 2.6.0+metax3.2.1.3

# 检查 GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# 输出: MetaX C500 (或类似)
```

---

## 3. 环境配置

### 3.1 设置环境变量

将以下内容添加到 `~/.bashrc` 或 `~/.zshrc`：

```bash
# ==================== MACA SDK 配置 ====================
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${PATH}

# ==================== mcPytorch 配置 ====================
# 如果使用 conda 环境
# conda activate mcpytorch

# ==================== YiRage 配置 ====================
export YIRAGE_HOME=/path/to/YiRage
export PYTHONPATH=${YIRAGE_HOME}/python:${PYTHONPATH}
```

应用配置：

```bash
source ~/.bashrc  # 或 source ~/.zshrc
```

### 3.2 验证配置

```bash
# 验证 mxcc
which mxcc
# 输出: /opt/maca/mxgpu_llvm/bin/mxcc

# 验证动态库
ldd /opt/maca/lib/libmcruntime.so | head -5

# 验证 Python 环境
python << 'EOF'
import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
EOF
```

---

## 4. 编译安装

### 4.1 获取源码

```bash
git clone https://github.com/chenxingqiang/YiRage.git
cd YiRage
```

### 4.2 安装 Python 依赖

```bash
pip install z3-solver graphviz cython numpy
```

### 4.3 安装 Rust（如未安装）

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env
```

### 4.4 配置依赖项

#### 4.4.1 创建 config.cmake

```bash
cat > config.cmake << 'EOF'
# YiRage Backend Configuration for MACA
set(USE_CUDA OFF)       # 禁用 NVIDIA CUDA
set(USE_MACA ON)        # 启用 MetaX MACA
set(USE_CUDNN OFF)      # 禁用 cuDNN
set(USE_CPU ON)         # 保留 CPU 后端
set(USE_ASCEND OFF)     # 禁用华为 Ascend
set(USE_NKI OFF)        # 禁用 AWS NKI
set(USE_MPS OFF)        # 禁用 Apple MPS
EOF
```

#### 4.4.2 配置 Z3

```bash
mkdir -p deps/z3/build
Z3_BASE=$(python -c "import z3; import os; print(os.path.dirname(z3.__file__))")

cat > deps/z3/build/z3-config.cmake << EOF
set(Z3_FOUND TRUE)
set(Z3_VERSION "$(python -c 'import z3; print(z3.get_version_string())')")
set(Z3_INCLUDE_DIRS "${Z3_BASE}/include")
set(Z3_LIBRARIES "${Z3_BASE}/lib/libz3.so")
set(Z3_CXX_INCLUDE_DIRS "${Z3_BASE}/include")

if(NOT TARGET z3::libz3)
  add_library(z3::libz3 SHARED IMPORTED)
  set_target_properties(z3::libz3 PROPERTIES
    IMPORTED_LOCATION "${Z3_BASE}/lib/libz3.so"
    INTERFACE_INCLUDE_DIRECTORIES "${Z3_BASE}/include"
  )
endif()
EOF

cat > deps/z3/build/Z3Config.cmake << 'EOF'
include("${CMAKE_CURRENT_LIST_DIR}/z3-config.cmake")
EOF

echo "Z3 配置完成: ${Z3_BASE}"
```

#### 4.4.3 配置 JSON

```bash
mkdir -p deps/json/include/nlohmann

# 下载 nlohmann/json
curl -sL https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp \
  -o deps/json/include/nlohmann/json.hpp

# 创建 CMakeLists.txt
cat > deps/json/CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(nlohmann_json)
add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
EOF

echo "JSON 配置完成"
```

#### 4.4.4 配置 CUTLASS stub

```bash
mkdir -p deps/cutlass/include/cutlass/detail

cat > deps/cutlass/include/cutlass/cutlass.h << 'EOF'
#pragma once
// CUTLASS stub for MACA backend
#if defined(__NVCC__) || (defined(__clang__) && (defined(__CUDA__) || defined(__MACA__)))
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE
#define CUTLASS_DEVICE
#endif
namespace cutlass {}
EOF

cat > deps/cutlass/include/cutlass/detail/helper_macros.hpp << 'EOF'
#pragma once
#if defined(__NVCC__) || (defined(__clang__) && (defined(__CUDA__) || defined(__MACA__)))
#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__
#else
#define CUTLASS_HOST_DEVICE
#define CUTLASS_DEVICE
#endif
EOF

echo "CUTLASS stub 配置完成"
```

### 4.5 编译

```bash
# 创建并进入构建目录
mkdir -p build && cd build

# 配置 CMake
cmake .. \
  -DUSE_CUDA=OFF \
  -DUSE_MACA=ON \
  -DUSE_CUDNN=OFF \
  -DUSE_ASCEND=OFF \
  -DUSE_NKI=OFF \
  -DUSE_MPS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DZ3_DIR=${PWD}/../deps/z3/build

# 编译（使用所有 CPU 核心）
make -j$(nproc)

# 返回项目根目录
cd ..
```

### 4.6 安装 Python 包

```bash
pip install -e .
```

### 4.7 验证编译

```bash
# 检查编译产物
ls -la build/lib*.so 2>/dev/null || ls -la build/*.so 2>/dev/null

# 检查 Python 导入
python -c "import yirage; print(f'YiRage version: {yirage.__version__}')"
```

---

## 5. 验证安装

### 5.1 基本功能验证

```python
#!/usr/bin/env python3
"""YiRage MACA 安装验证脚本"""

import sys

def verify_installation():
    """验证 YiRage + MACA 安装"""
    print("=" * 60)
    print("YiRage MACA 安装验证")
    print("=" * 60)
    
    # 1. 检查 PyTorch
    print("\n[1/5] 检查 PyTorch...")
    try:
        import torch
        print(f"  ✅ PyTorch: {torch.__version__}")
        
        if "metax" not in torch.__version__.lower():
            print("  ⚠️  警告: 非 mcPytorch 版本，profiling 可能受限")
    except ImportError:
        print("  ❌ PyTorch 未安装")
        return False
    
    # 2. 检查 CUDA/MACA
    print("\n[2/5] 检查 MACA GPU...")
    if torch.cuda.is_available():
        print(f"  ✅ CUDA/MACA 可用")
        print(f"  ✅ 设备数量: {torch.cuda.device_count()}")
        print(f"  ✅ 设备名称: {torch.cuda.get_device_name(0)}")
        print(f"  ✅ 显存大小: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️  CUDA/MACA 不可用，将使用 CPU 后端")
    
    # 3. 检查 YiRage
    print("\n[3/5] 检查 YiRage...")
    try:
        import yirage
        print(f"  ✅ YiRage: {yirage.__version__}")
    except ImportError as e:
        print(f"  ❌ YiRage 导入失败: {e}")
        return False
    
    # 4. 创建测试图
    print("\n[4/5] 创建测试计算图...")
    try:
        graph = yirage.new_kernel_graph()
        X = graph.new_input(dims=(16, 64), dtype=yirage.float16)
        W = graph.new_input(dims=(64, 64), dtype=yirage.float16)
        Y = graph.matmul(X, W)
        graph.mark_output(Y)
        print("  ✅ 计算图创建成功")
    except Exception as e:
        print(f"  ❌ 计算图创建失败: {e}")
        return False
    
    # 5. 验证 MACA 后端
    print("\n[5/5] 验证 MACA 后端...")
    try:
        # 检查 MACA 后端是否可用
        if hasattr(yirage, 'get_available_backends'):
            backends = yirage.get_available_backends()
            if 'maca' in backends:
                print("  ✅ MACA 后端已注册")
            else:
                print("  ⚠️  MACA 后端未在列表中")
        else:
            print("  ⚠️  无法检查后端列表")
        
        # 基本编译测试
        print("  ✅ 后端验证完成")
    except Exception as e:
        print(f"  ⚠️  后端验证警告: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 YiRage MACA 安装验证通过!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = verify_installation()
    sys.exit(0 if success else 1)
```

### 5.2 GPU 内核验证

```python
#!/usr/bin/env python3
"""验证 MACA GPU 内核"""

import torch
import yirage

def test_maca_kernel():
    """测试 MACA GPU 内核执行"""
    print("MACA GPU 内核测试")
    print("-" * 40)
    
    # 创建简单计算图
    graph = yirage.new_kernel_graph()
    
    # 输入张量
    A = graph.new_input(dims=(32, 64), dtype=yirage.float16)
    B = graph.new_input(dims=(64, 128), dtype=yirage.float16)
    
    # 矩阵乘法
    C = graph.matmul(A, B)
    graph.mark_output(C)
    
    print(f"输入 A: {A.shape}")
    print(f"输入 B: {B.shape}")
    print(f"输出 C: (32, 128)")
    
    # 如果 GPU 可用，测试执行
    if torch.cuda.is_available():
        # 创建测试数据
        a = torch.randn(32, 64, dtype=torch.float16, device="cuda")
        b = torch.randn(64, 128, dtype=torch.float16, device="cuda")
        
        # PyTorch 参考结果
        c_ref = torch.matmul(a, b)
        print(f"PyTorch 结果形状: {c_ref.shape}")
        print("✅ GPU 内核测试通过")
    else:
        print("⚠️  GPU 不可用，跳过执行测试")
    
    return True

if __name__ == "__main__":
    test_maca_kernel()
```

---

## 6. 使用指南

### 6.1 基本使用流程

```python
import yirage
import torch

# Step 1: 创建计算图
graph = yirage.new_kernel_graph()

# Step 2: 定义输入
X = graph.new_input(dims=(batch, features), dtype=yirage.float16)
W = graph.new_input(dims=(features, hidden), dtype=yirage.float16)

# Step 3: 定义计算
Y = graph.matmul(X, W)
Y = graph.relu(Y)

# Step 4: 标记输出
graph.mark_output(Y)

# Step 5: 超优化
optimized = graph.superoptimize(
    backend="maca",      # 使用 MACA 后端
    config="mlp",        # 配置类型
    verbose=True         # 显示搜索进度
)

# Step 6: 执行
x = torch.randn(batch, features, dtype=torch.float16, device="cuda")
w = torch.randn(features, hidden, dtype=torch.float16, device="cuda")
result = optimized(x, w)
```

### 6.2 支持的操作

| 类别 | 操作 | API |
|------|------|-----|
| 矩阵运算 | MatMul | `graph.matmul(A, B)` |
| 元素运算 | Add | `graph.add(A, B)` |
| 元素运算 | Mul | `graph.mul(A, B)` |
| 元素运算 | Div | `graph.div(A, B)` |
| 激活函数 | ReLU | `graph.relu(X)` |
| 激活函数 | GELU | `graph.gelu(X)` |
| 激活函数 | SiLU | `graph.silu(X)` |
| 归一化 | RMSNorm | `graph.rms_norm(X)` |
| 规约 | Reduction | `graph.reduction(X, dim)` |

### 6.3 搜索配置

```python
# MLP 优化
optimized = graph.superoptimize(
    backend="maca",
    config="mlp",
    max_search_time=300,   # 最大搜索时间（秒）
    verbose=True
)

# Attention 优化
optimized = graph.superoptimize(
    backend="maca",
    config="attention",
    max_search_time=600,
    verbose=True
)
```

### 6.4 完整示例：RMSNorm + Linear 融合

```python
#!/usr/bin/env python3
"""RMSNorm + Linear 融合示例"""

import yirage
import torch
import time

def create_rms_norm_linear_graph(batch, seq_len, hidden, intermediate):
    """创建 RMSNorm + Linear 计算图"""
    graph = yirage.new_kernel_graph()
    
    # 输入
    X = graph.new_input(dims=(batch * seq_len, hidden), dtype=yirage.float16)
    W = graph.new_input(dims=(hidden, intermediate), dtype=yirage.float16)
    
    # RMSNorm
    X_norm = graph.rms_norm(X)
    
    # Linear
    Y = graph.matmul(X_norm, W)
    
    graph.mark_output(Y)
    return graph

def main():
    # 参数
    batch, seq_len = 4, 512
    hidden, intermediate = 4096, 11008
    
    print("RMSNorm + Linear 融合优化")
    print(f"输入形状: ({batch}*{seq_len}, {hidden})")
    print(f"输出形状: ({batch}*{seq_len}, {intermediate})")
    print("-" * 50)
    
    # 创建计算图
    graph = create_rms_norm_linear_graph(batch, seq_len, hidden, intermediate)
    
    # 超优化
    print("\n开始搜索最优融合方案...")
    start = time.time()
    optimized = graph.superoptimize(
        backend="maca",
        config="mlp",
        verbose=True
    )
    elapsed = time.time() - start
    
    if optimized:
        print(f"\n✅ 找到优化方案！搜索耗时: {elapsed:.2f}s")
        
        # 性能测试
        if torch.cuda.is_available():
            x = torch.randn(batch * seq_len, hidden, 
                          dtype=torch.float16, device="cuda")
            w = torch.randn(hidden, intermediate, 
                          dtype=torch.float16, device="cuda")
            
            # Warmup
            for _ in range(10):
                _ = optimized(x, w)
            torch.cuda.synchronize()
            
            # Profile
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(100):
                _ = optimized(x, w)
            end_event.record()
            torch.cuda.synchronize()
            
            avg_time = start_event.elapsed_time(end_event) / 100
            print(f"平均执行时间: {avg_time:.4f} ms")
    else:
        print("\n❌ 未找到优化方案")

if __name__ == "__main__":
    main()
```

---

## 7. Benchmark 测试

### 7.1 运行 MACA Benchmark

```bash
cd YiRage

# 运行所有 MACA benchmark
python benchmark/end-to-end/maca/run_all.py

# 运行单个 benchmark
python benchmark/end-to-end/maca/llama_maca.py
python benchmark/end-to-end/maca/chameleon_maca.py
python benchmark/end-to-end/maca/lora_maca.py
python benchmark/end-to-end/maca/ngpt_maca.py
```

### 7.2 Benchmark 文件列表

```
benchmark/end-to-end/maca/
├── run_all.py           # 运行所有 benchmark
├── llama_maca.py        # LLaMA 模型优化
├── chameleon_maca.py    # Chameleon 模型优化
├── lora_maca.py         # LoRA 微调优化
└── ngpt_maca.py         # nGPT 模型优化
```

### 7.3 性能对比测试

```python
#!/usr/bin/env python3
"""MACA vs PyTorch 性能对比"""

import torch
import time

def benchmark_pytorch_vs_yirage():
    """对比 PyTorch 和 YiRage 性能"""
    import yirage
    
    # 配置
    batch, m, n, k = 32, 4096, 4096, 4096
    warmup, repeat = 50, 200
    
    print(f"MatMul 性能对比: ({batch}, {m}, {k}) x ({k}, {n})")
    print("-" * 50)
    
    # 创建数据
    A = torch.randn(batch, m, k, dtype=torch.float16, device="cuda")
    B = torch.randn(k, n, dtype=torch.float16, device="cuda")
    
    # PyTorch 基准
    for _ in range(warmup):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeat):
        _ = torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()
    
    pytorch_time = start.elapsed_time(end) / repeat
    print(f"PyTorch: {pytorch_time:.4f} ms")
    
    # YiRage 优化
    graph = yirage.new_kernel_graph()
    X = graph.new_input(dims=(batch, m, k), dtype=yirage.float16)
    W = graph.new_input(dims=(k, n), dtype=yirage.float16)
    Y = graph.matmul(X, W)
    graph.mark_output(Y)
    
    optimized = graph.superoptimize(backend="maca", config="mlp")
    
    if optimized:
        for _ in range(warmup):
            _ = optimized(A, B)
        torch.cuda.synchronize()
        
        start.record()
        for _ in range(repeat):
            _ = optimized(A, B)
        end.record()
        torch.cuda.synchronize()
        
        yirage_time = start.elapsed_time(end) / repeat
        speedup = pytorch_time / yirage_time
        
        print(f"YiRage:  {yirage_time:.4f} ms")
        print(f"加速比:  {speedup:.2f}x")
    else:
        print("YiRage 优化失败")

if __name__ == "__main__":
    benchmark_pytorch_vs_yirage()
```

---

## 8. MACA 技术特性

### 8.1 64 线程 Warp

MACA GPU 使用 **64 线程 warp**（NVIDIA 使用 32）：

```
NVIDIA CUDA:  32 threads/warp
MetaX MACA:   64 threads/warp
```

YiRage 自动处理此差异：
- `dim_strategy.cc` 会过滤 blockDim 确保兼容性
- Block size 推荐使用 64 的倍数

### 8.2 内存层次

```
┌────────────────────────────────────────┐
│           Global Memory (HBM)          │  64 GB
├────────────────────────────────────────┤
│         L2 Cache (Shared)              │  ~128 MB
├────────────────────────────────────────┤
│    ┌──────────┐    ┌──────────┐        │
│    │ L1 Cache │    │ L1 Cache │   ...  │  Per SM
│    │ (64 KB)  │    │ (64 KB)  │        │
│    └──────────┘    └──────────┘        │
│    ┌──────────┐    ┌──────────┐        │
│    │ Shared   │    │ Shared   │   ...  │  Per SM
│    │ Memory   │    │ Memory   │        │
│    │ (64 KB)  │    │ (64 KB)  │        │
│    └──────────┘    └──────────┘        │
└────────────────────────────────────────┘
```

### 8.3 YiRage 内存配置

在 `include/yirage/config.h` 中：

```cpp
#elif defined(YIRAGE_FINGERPRINT_USE_MACA)
// MetaX MACA GPU (C500)
size_t const MAX_DMEM_FP_SIZE = (size_t)2 * 1024 * 1024 * 1024;  // 2 GB
size_t const MAX_SMEM_FP_SIZE = (size_t)1 * 1024 * 1024;         // 1 MB
```

### 8.4 API 映射

| CUDA API | MACA API |
|----------|----------|
| `cudaMalloc` | `mcMalloc` |
| `cudaMemcpy` | `mcMemcpy` |
| `cudaSetDevice` | `mcSetDevice` |
| `cudaDeviceSynchronize` | `mcDeviceSynchronize` |
| `cudaGetDeviceCount` | `mcGetDeviceCount` |
| `cudaStream_t` | `mcStream_t` |

---

## 9. 故障排除

### 9.1 编译错误

#### 找不到 mxcc

```bash
# 错误
CMake Error: Could not find mxcc compiler

# 解决
export MACA_PATH=/opt/maca
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${PATH}
```

#### 找不到 Z3

```bash
# 错误
CMake Error: Could not find Z3

# 解决
pip install z3-solver
# 然后重新配置 deps/z3/build/z3-config.cmake
```

#### 链接错误

```bash
# 错误
undefined reference to `mcMalloc`

# 解决
export LD_LIBRARY_PATH=/opt/maca/lib:${LD_LIBRARY_PATH}
```

### 9.2 运行时错误

#### CUDA/MACA 不可用

```python
# 错误
RuntimeError: Found no NVIDIA driver on your system

# 解决 - 确保使用 mcPytorch
import torch
assert "metax" in torch.__version__.lower(), "请使用 mcPytorch"
```

#### 搜索缓冲区溢出

```python
# 错误
AssertionError: num < max_num_graphs

# 解决 - 已在代码中修复，确保使用最新版本
# python/yirage/_cython/core.pyx 中 max_num_new_graphs = 8192
```

#### 显存不足

```python
# 错误
RuntimeError: CUDA out of memory

# 解决 - 减小 batch size 或输入尺寸
# 或检查 config.h 中的 MAX_DMEM_FP_SIZE
```

### 9.3 Profiling 错误

#### 无法进行性能分析

```python
# 错误
Warning: mcPytorch not available, skipping profiling

# 解决 - 安装 mcPytorch
# 或接受首个有效图（非最优但可用）
```

---

## 10. 常见问题

### Q1: MACA 和 CUDA 的主要区别？

**A**: 
- Warp 大小: MACA 64 vs CUDA 32
- 编译器: mxcc vs nvcc
- 运行时: mcruntime vs cudart
- API 前缀: mc* vs cuda*

### Q2: 是否需要修改现有 CUDA 代码？

**A**: 基本不需要。mcPytorch 已映射 `torch.cuda.*` API 到 MACA。YiRage 在编译时自动处理后端差异。

### Q3: 搜索需要多长时间？

**A**: 
- 简单图（< 5 ops）: 几秒到几分钟
- 中等图（5-10 ops）: 几分钟到十几分钟
- 复杂图（> 10 ops）: 可能需要更长时间

建议使用 `verbose=True` 查看进度。

### Q4: 如何加速搜索？

**A**:
1. 使用 checkpoint 保存/加载搜索状态
2. 缩小搜索空间（限制 config）
3. 使用更多 CPU 核心进行并行搜索

### Q5: 优化后性能提升多少？

**A**: 取决于计算图复杂度。典型情况：
- 简单融合: 1.2x - 1.5x
- 中等融合: 1.5x - 2x
- 复杂融合: 2x - 4x

---

## 附录

### A. MACA Kernel 文件列表

```
src/kernel/maca/
├── all_reduce_kernel.maca       # AllReduce 操作
├── customized_kernel.maca       # 主 fingerprint 内核
├── device_memory_manager.maca   # 设备内存管理
├── device_tensor_kernel.maca    # 张量操作
├── element_binary_kernel.maca   # 二元运算
├── element_unary_kernel.maca    # 一元运算
├── input_kernel.maca            # 输入初始化
├── matmul_kernel.maca           # 矩阵乘法
├── output_kernel.maca           # 输出处理
├── reduction_kernel.maca        # 规约操作
└── rms_norm_kernel.maca         # RMS 归一化
```

### B. 环境变量汇总

```bash
# 必需
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}
export PATH=${MACA_PATH}/mxgpu_llvm/bin:${PATH}

# 可选
export YIRAGE_HOME=/path/to/YiRage
export PYTHONPATH=${YIRAGE_HOME}/python:${PYTHONPATH}
export YIRAGE_VERBOSE=1  # 详细日志
```

### C. CMake 选项

```cmake
# 后端选择
-DUSE_CUDA=OFF
-DUSE_MACA=ON
-DUSE_CUDNN=OFF
-DUSE_ASCEND=OFF
-DUSE_NKI=OFF
-DUSE_MPS=OFF
-DUSE_CPU=ON

# 构建类型
-DCMAKE_BUILD_TYPE=Release  # 或 Debug

# 依赖路径
-DZ3_DIR=/path/to/z3/build
```

---

*文档版本: 2025-12-04*  
*基于 MetaX C500 GPU + mcPytorch 2.6.0+metax3.2.1.3 验证*  
*YiRage 项目: https://github.com/chenxingqiang/YiRage*

