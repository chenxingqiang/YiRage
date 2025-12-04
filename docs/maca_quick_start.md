# YiRage MACA Backend Quick Start Guide

本文档基于在 **MetaX C500 GPU** 上的实际成功运行经验编写。

## 1. 环境要求

### 硬件
- MetaX C500 GPU (或其他 MACA 兼容 GPU)

### 软件
- **MACA SDK**: `/opt/maca` (包含 `mxcc` 编译器)
- **mcPytorch**: PyTorch 2.6.0+metax3.2.1.3 或兼容版本
- **Python**: 3.10+
- **CMake**: 3.24+
- **Rust**: 最新稳定版
- **GCC**: 支持 C++17

## 2. 环境配置

### 2.1 设置环境变量

```bash
# MACA SDK 路径
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH}

# 可选：添加 mxcc 到 PATH
export PATH=${MACA_PATH}/mxgpu_llvm/bin:$PATH
```

### 2.2 验证 mcPytorch

```python
import torch
print(f"PyTorch: {torch.__version__}")  # 应显示 2.6.0+metax3.2.1.3
print(f"CUDA available: {torch.cuda.is_available()}")  # True
print(f"Device: {torch.cuda.get_device_name(0)}")  # MetaX C500
```

### 2.3 验证 MACA 编译器

```bash
which mxcc
# 应输出: /opt/maca/mxgpu_llvm/bin/mxcc
```

## 3. 编译 YiRage

### 3.1 安装依赖

```bash
# Python 依赖
pip install z3-solver graphviz cython

# Rust (如未安装)
curl https://sh.rustup.rs -sSf | sh -s -- -y
source $HOME/.cargo/env

# CMake (如版本过低)
pip install "cmake>=3.24"
```

### 3.2 配置 config.cmake

创建 `config.cmake` 文件：

```cmake
set(USE_CUDA OFF)
set(USE_MACA ON)
set(USE_CUDNN OFF)
set(USE_CPU ON)
set(USE_ASCEND OFF)
set(USE_NKI OFF)
set(USE_MPS OFF)
```

### 3.3 设置依赖

```bash
cd YiRage

# Z3 配置 (使用 pip 安装的 z3)
mkdir -p deps/z3/build
Z3_BASE=$(python -c "import z3; import os; print(os.path.dirname(z3.__file__))")

cat > deps/z3/build/z3-config.cmake << EOF
set(Z3_FOUND TRUE)
set(Z3_VERSION "4.15.4")
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

cat > deps/z3/build/Z3Config.cmake << EOF
include("\${CMAKE_CURRENT_LIST_DIR}/z3-config.cmake")
EOF

# JSON 配置 (如 deps/json 为空)
mkdir -p deps/json/include/nlohmann
curl -sL https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp \
  -o deps/json/include/nlohmann/json.hpp

cat > deps/json/CMakeLists.txt << EOF
cmake_minimum_required(VERSION 3.10)
project(nlohmann_json)
add_library(nlohmann_json INTERFACE)
add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
target_include_directories(nlohmann_json INTERFACE \${CMAKE_CURRENT_SOURCE_DIR}/include)
EOF

# CUTLASS stub (如 deps/cutlass 为空)
mkdir -p deps/cutlass/include/cutlass/detail
cat > deps/cutlass/include/cutlass/cutlass.h << 'EOF'
#pragma once
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
```

### 3.4 编译

```bash
mkdir -p build && cd build

cmake .. \
  -DUSE_CUDA=OFF \
  -DUSE_MACA=ON \
  -DUSE_CUDNN=OFF \
  -DUSE_ASCEND=OFF \
  -DUSE_NKI=OFF \
  -DUSE_MPS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DZ3_DIR=${PWD}/../deps/z3/build

make -j$(nproc)
```

### 3.5 安装 Python 包

```bash
cd ..
pip install -e .
```

## 4. 验证安装

```python
import yirage
import torch

print(f"YiRage: {yirage.__version__}")
print(f"Device: {torch.cuda.get_device_name(0)}")

# 创建简单图
graph = yirage.new_kernel_graph()
X = graph.new_input(dims=(16, 64), dtype=yirage.float16)
W = graph.new_input(dims=(64, 64), dtype=yirage.float16)
Y = graph.matmul(X, W)
graph.mark_output(Y)

print("✅ YiRage + MACA ready!")
```

## 5. 使用示例

### 5.1 基本优化

```python
import yirage
import torch

# 创建计算图
graph = yirage.new_kernel_graph()
X = graph.new_input(dims=(32, 64), dtype=yirage.float16)
W = graph.new_input(dims=(64, 64), dtype=yirage.float16)
B = graph.new_input(dims=(32, 64), dtype=yirage.float16)

# 定义操作: Y = ReLU(X @ W + B)
XW = graph.matmul(X, W)
XWB = graph.add(XW, B)
Y = graph.relu(XWB)
graph.mark_output(Y)

# 搜索最优融合方案 (首次运行需要几分钟)
print("Searching for optimal fusion...")
optimized = graph.superoptimize(
    backend="maca",    # 使用 MACA 后端
    config="mlp",      # MLP 配置
    verbose=False      # 设为 True 可查看搜索进度
)

if optimized:
    print(f"Found optimized graph!")
    
    # 准备输入
    x = torch.randn(32, 64, dtype=torch.float16, device="cuda")
    w = torch.randn(64, 64, dtype=torch.float16, device="cuda")
    b = torch.randn(32, 64, dtype=torch.float16, device="cuda")
    
    # 运行优化后的图
    result = optimized(x, w, b)
    print(f"Output shape: {result.shape}")
```

### 5.2 性能对比

```python
import torch
import time

# PyTorch 基准
def pytorch_mlp(x, w, b):
    return torch.relu(torch.matmul(x, w) + b)

# 准备数据
x = torch.randn(64, 128, dtype=torch.float16, device="cuda")
w = torch.randn(128, 128, dtype=torch.float16, device="cuda")
b = torch.randn(64, 128, dtype=torch.float16, device="cuda")

# 使用 CUDA events 精确计时
def profile(func, warmup=20, repeat=100):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(repeat):
        func()
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / repeat

pytorch_time = profile(lambda: pytorch_mlp(x, w, b))
print(f"PyTorch time: {pytorch_time:.4f} ms")

# YiRage 优化后 (假设 optimized 已创建)
# yirage_time = profile(lambda: optimized(x, w, b))
# print(f"YiRage time: {yirage_time:.4f} ms")
# print(f"Speedup: {pytorch_time / yirage_time:.2f}x")
```

## 6. MACA 特性说明

### 6.1 64 线程 Warp

MACA GPU 使用 **64 线程 warp**（NVIDIA 使用 32）。YiRage 自动处理此差异：

```python
# 搜索配置会自动适配 64 线程 warp
optimized = graph.superoptimize(backend="maca", ...)
```

### 6.2 搜索时间

- **首次搜索**: 需要几分钟（搜索融合方案）
- **后续运行**: 可使用 checkpoint 加速
- **搜索状态**: `verbose=True` 可查看进度

### 6.3 支持的操作

- MatMul (矩阵乘法)
- Add, Sub, Mul, Div (元素运算)
- ReLU, GELU, SiLU (激活函数)
- RMSNorm, LayerNorm (归一化)
- Reduction (规约操作)

## 7. 故障排除

### 7.1 找不到 mcruntime

```bash
export LD_LIBRARY_PATH=/opt/maca/lib:$LD_LIBRARY_PATH
```

### 7.2 mxcc 编译错误

确保 MACA SDK 版本与 mcPytorch 兼容。

### 7.3 搜索缓冲区溢出

如果出现 `num < max_num_graphs` 错误，搜索找到的图太多。这通常不影响结果。

### 7.4 profiling 失败

确保使用 mcPytorch（不是标准 PyTorch）：
```python
import torch
assert "metax" in torch.__version__.lower()
```

## 8. 参考

- [MACA SDK 文档](https://www.metax-tech.com/)
- [YiRage GitHub](https://github.com/chenxingqiang/YiRage)
- [mcPytorch 文档](https://www.metax-tech.com/pytorch)

---

*文档版本: 2025-12-04*
*基于 MetaX C500 GPU 验证*
