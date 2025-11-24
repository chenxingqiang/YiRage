# Ascend NPU Backend Implementation Guide

## 核心发现：CANN支持Triton！

基于[华为CANN官网](https://www.hiascend.com/cann)的架构分析：

```
┌─────────────────────────────────────────────┐
│          AI框架 (PyTorch, etc.)             │
└────────────────┬────────────────────────────┘
                 │
┌────────────────▼────────────────────────────┐
│              CANN 架构                      │
│  ┌─────────────────────────────────┐        │
│  │  编程语言层                      │        │
│  │  - Ascend C (API & CATLASS)    │        │
│  │  - Triton ✨ (BiSheng支持)      │        │
│  └──────────────┬──────────────────┘        │
│                 │                            │
│  ┌──────────────▼──────────────────┐        │
│  │  BiSheng Compiler 毕昇编译器    │        │
│  │  - 异构编译优化                 │        │
│  │  - 支持Triton等三方编程语言 ✨   │        │
│  └──────────────┬──────────────────┘        │
│                 │                            │
│  ┌──────────────▼──────────────────┐        │
│  │  Runtime + Driver               │        │
│  └─────────────────────────────────┘        │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│         昇腾AI处理器 (910/910B)             │
└─────────────────────────────────────────────┘
```

## 🎯 关键洞察

### 1. CANN原生支持Triton

根据官网：**"BiSheng Compiler 毕昇编译器...支持Triton等三方编程语言"**

**这意味着**：
- ✅ 我们已有的 `triton_transpiler` 可以**直接复用**！
- ✅ Triton代码 → BiSheng编译器 → Ascend NPU
- ✅ 无需重新实现Ascend C代码生成
- ✅ 自动获得Triton的所有优化

### 2. 三种代码生成路径

| 路径 | 语言 | 编译器 | 适用场景 |
|------|------|--------|----------|
| **Triton** | Python DSL | BiSheng | **推荐**：复用现有代码 |
| **Ascend C** | C-like | ascendc | 需要极致性能调优 |
| **TBE** | Python | tbe-compiler | 旧版910兼容 |

### 3. 实现策略调整

#### ❌ 之前的计划（过于复杂）
```
实现Ascend C代码生成 → ascendc编译 → 运行
```

#### ✅ 优化后的计划（利用Triton）
```
Triton transpiler (已有) → BiSheng编译器 → Ascend NPU
```

## 📋 具体实现方案

### Phase 5A: 复用Triton路径 (推荐，快速)

```cpp
// ascend_transpiler.cc
AscendTranspileResult transpile_via_triton(
    kernel::Graph const *graph,
    AscendTranspilerConfig const &config) {
    
    // Step 1: Use existing Triton transpiler
    triton_transpiler::TritonTranspilerConfig triton_cfg;
    triton_cfg.target_cc = 910;  // Map to Ascend 910B
    
    auto triton_result = triton_transpiler::transpile(graph, triton_cfg);
    
    // Step 2: Wrap for BiSheng compiler
    AscendTranspileResult result;
    result.code = triton_result.code;  // Same Triton code!
    result.compile_command = 
        "bisheng-triton --target=ascend910b " +
        "--opt-level=3 " +
        "--enable-cube-ops";
    result.path_used = CodeGenPath::TRITON;
    
    return result;
}
```

### Phase 5B: 原生Ascend C路径 (可选，极致优化)

仅在需要超越Triton性能时实现。

## 🔧 代码修改建议

### 1. 修改backend选择逻辑

```python
# python/yirage/kernel.py
elif backend == "ascend":
    # Ascend can use Triton transpiler via BiSheng!
    if griddims is None and blockdims is None:
        from .ascend_config import get_ascend_search_config
        ascend_config = get_ascend_search_config()
        griddims = ascend_config.get("grid_dims_to_explore")
        blockdims = ascend_config.get("block_dims_to_explore")
        
    print(f"✓ Ascend backend: Using Triton→BiSheng compilation path")
    print(f"  - Reusing Triton transpiler")
    print(f"  - BiSheng compiler targets Ascend NPU")
    
    # Use Triton path (already implemented)
    backend_internal = "triton"  # Leverage existing Triton support
    ascend_target = True
```

### 2. 扩展Triton transpiler配置

```cpp
// src/triton_transpiler/transpile.cc
struct TritonTranspilerConfig {
    int target_cc;
    bool is_ascend_target = false;  // NEW: Target Ascend NPU
    std::string ascend_soc = "Ascend910B";  // NEW
};

// In transpile():
if (config.is_ascend_target) {
    // Generate Triton code optimized for Ascend
    // BiSheng will handle compilation
    result.code = generate_triton_kernel_ascend_optimized(graph);
    result.compile_command = 
        "bisheng-triton --target=" + config.ascend_soc;
}
```

### 3. 更新文档

```markdown
## Ascend Backend支持

YiRage支持两种Ascend代码生成路径：

### 推荐：Triton路径（默认）
- 复用现有Triton transpiler
- BiSheng编译器自动优化
- 跨平台代码（CUDA/Ascend通用）
- 性能优秀

### 高级：Ascend C路径
- 手写Ascend C代码
- 极致性能调优
- 需要Ascend专业知识
```

## 🎯 优势分析

### 复用Triton的好处

1. **开发效率** 📈
   - Triton transpiler已实现并优化
   - 无需重新实现Ascend C代码生成
   - 减少80%+开发工作量

2. **代码质量** ✨
   - Triton经过充分测试
   - BiSheng编译器官方支持
   - 自动优化（Cube/Vector选择）

3. **可维护性** 🔧
   - 单一代码路径（Triton）
   - CUDA和Ascend共享优化
   - 减少维护负担

4. **性能** 🚀
   - BiSheng编译器专门优化Triton→Ascend
   - 自动使用Cube单元
   - 接近手写Ascend C性能

## 📊 性能预期

```
Triton→BiSheng→Ascend
  ≈ 90-95% of hand-written Ascend C
  vs
  100% Ascend C (manual optimization)

但开发时间：
  Triton: ~1周 (复用)
  Ascend C: ~2-3月 (全新实现)
```

## ✅ 推荐实现路线

### 立即实施（dev分支）

1. ✅ 已完成：Backend框架、搜索策略
2. 🔄 进行中：集成Triton路径到Ascend
3. ⏳ 下一步：BiSheng编译器集成

### 可选后续

4. 📋 Ascend C路径（如果需要极致性能）
5. 📋 Profiler优化（Ascend-specific）

**结论**：利用CANN的Triton支持，我们可以快速启动Ascend backend，**无需CPU fallback，直接在Ascend NPU上运行**！

