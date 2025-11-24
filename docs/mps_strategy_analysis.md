# MPS搜索策略分析与优化建议

## 📊 当前实现分析

### ✅ 做得好的地方

1. **正确的基础架构** ✅
   - 正确识别SIMD width = 32
   - threadgroup memory = 32KB (准确)
   - 线程数限制 32-1024 (正确)

2. **多维度评估** ✅
   ```cpp
   float score = 0.4f * gpu_util_score + 
                 0.3f * memory_score +
                 0.3f * tg_memory_score;
   ```
   权重分配合理

3. **基础验证** ✅
   - 检查threadgroup大小是否为SIMD width的倍数
   - 检查tile大小有效性

### ❌ 存在的问题和改进机会

---

## 🔴 问题1: Threadgroup配置生成过于简单

### 当前实现
```cpp
std::vector<int> MPSSearchStrategy::generate_threadgroup_configs(
    size_t problem_size) {
  std::vector<int> configs;
  int simd_width = 32;
  for (int mult = 4; mult <= 32; mult *= 2) {  // 只尝试4, 8, 16, 32倍
    int size = simd_width * mult;
    if (size <= 1024) {
      configs.push_back(size);
    }
  }
  return configs;  // 只有4个候选: 128, 256, 512, 1024
}
```

### 问题
- **搜索空间太小**: 只有4个候选值
- **忽略问题特性**: 不考虑实际problem_size
- **缺少细粒度**: 跳过了64, 96, 160, 192, 224, 288, 320...等有效值

### 改进方案
```cpp
std::vector<int> generate_threadgroup_configs(size_t problem_size) {
  std::vector<int> configs;
  int simd_width = 32;
  
  // 基于problem_size动态调整范围
  int min_mult = (problem_size < 1024) ? 2 : 4;      // 小问题用更小的threadgroup
  int max_mult = (problem_size > 1048576) ? 32 : 16; // 大问题用更大的threadgroup
  
  // 生成更多候选值（所有SIMD width的倍数）
  for (int mult = min_mult; mult <= max_mult; mult++) {
    int size = simd_width * mult;
    if (size >= 32 && size <= 1024) {
      configs.push_back(size);
    }
  }
  
  // 特别添加一些经验优化的值
  std::vector<int> special = {64, 96, 128, 192, 256, 320, 512};
  for (int s : special) {
    if (s % simd_width == 0 && s >= 32 && s <= 1024) {
      if (std::find(configs.begin(), configs.end(), s) == configs.end()) {
        configs.push_back(s);
      }
    }
  }
  
  std::sort(configs.begin(), configs.end());
  return configs;  // 现在有10-20个候选值
}
```

**改进效果**: 搜索空间从4个增加到10-20个，覆盖更细粒度的配置

---

## 🔴 问题2: Tile配置不够灵活

### 当前实现
```cpp
std::vector<std::tuple<int, int, int>>
MPSSearchStrategy::generate_tile_configs(int m, int n, int k) {
  std::vector<std::tuple<int, int, int>> configs;
  std::vector<int> tile_sizes = {16, 32, 48, 64};  // 固定4个值
  
  for (int tile : tile_sizes) {
    configs.emplace_back(
        std::min(m, tile),
        std::min(n, tile),
        std::min(k, tile));
  }
  return configs;  // 只有4个候选
}
```

### 问题
- **tile大小单一**: 所有维度使用相同tile
- **不考虑threadgroup memory**: 32KB限制没有体现
- **忽略矩阵形状**: 不考虑m, n, k的相对大小
- **缺少大tile**: 最大才64，对于大矩阵不够

### 改进方案
```cpp
std::vector<std::tuple<int, int, int>>
generate_tile_configs(int m, int n, int k) {
  std::vector<std::tuple<int, int, int>> configs;
  
  // Threadgroup memory = 32KB = 32768 bytes
  const size_t tg_memory = 32 * 1024;
  const size_t fp16_size = 2;  // float16
  
  // 尝试不同的tile大小组合
  std::vector<int> tile_m_sizes = {16, 32, 48, 64, 96, 128};
  std::vector<int> tile_n_sizes = {16, 32, 48, 64, 96, 128};
  std::vector<int> tile_k_sizes = {8, 16, 24, 32, 48, 64};
  
  for (int tm : tile_m_sizes) {
    for (int tn : tile_n_sizes) {
      for (int tk : tile_k_sizes) {
        // 跳过超出维度的配置
        if (tm > m || tn > n || tk > k) continue;
        
        // 计算需要的threadgroup memory
        // A: tm x tk, B: tk x tn, C: tm x tn
        size_t memory_needed = (tm * tk + tk * tn + tm * tn) * fp16_size;
        
        // 确保不超过threadgroup memory (留20%余量)
        if (memory_needed > tg_memory * 0.8) continue;
        
        // 偏好平衡的tile配置
        float balance_score = 1.0f - std::abs(
            static_cast<float>(tm * tn) / (tk * tk) - 1.0f);
        
        if (balance_score > 0.5f) {  // 只保留相对平衡的配置
          configs.emplace_back(
              std::min(m, tm),
              std::min(n, tn),
              std::min(k, tk));
        }
      }
    }
  }
  
  // 至少返回一个有效配置
  if (configs.empty()) {
    configs.emplace_back(
        std::min(m, 32),
        std::min(n, 32),
        std::min(k, 32));
  }
  
  // 去重
  std::sort(configs.begin(), configs.end());
  configs.erase(std::unique(configs.begin(), configs.end()), configs.end());
  
  return configs;  // 现在可能有几十个候选
}
```

**改进效果**: 
- 考虑了实际内存限制
- tile不再强制正方形
- 自动适应矩阵形状

---

## 🔴 问题3: GPU利用率评估过于简化

### 当前实现
```cpp
float MPSSearchStrategy::evaluate_gpu_utilization(
    kernel::mps::MPSKernelConfig const &config) {
  int total_threads = config.get_total_blocks() *
                     config.threads_per_threadgroup;
  
  // 假设每个GPU核心可以处理~1024线程
  int ideal_threads = gpu_cores_ * 1024;
  
  float utilization = std::min(1.0f, 
      static_cast<float>(total_threads) / ideal_threads);
  
  return utilization;
}
```

### 问题
- **1024倍数是错误的**: Apple GPU不是这样工作的
- **未考虑GPU变体**: M1基础版(8核) vs M3 Max(40核)差异巨大
- **忽略occupancy**: 不考虑实际并发threadgroup数
- **缺少带宽考虑**: 统一内存架构的特性未体现

### 改进方案
```cpp
float evaluate_gpu_utilization(kernel::mps::MPSKernelConfig const &config) {
  // Apple GPU架构特点：
  // - 每个GPU核心是一个完整的计算单元
  // - 每个核心可以并发执行多个threadgroup
  // - 统一内存架构，内存访问共享
  
  int num_threadgroups = config.get_total_blocks();
  int threads_per_tg = config.threads_per_threadgroup;
  int num_simd_groups = (threads_per_tg + 31) / 32;
  
  // 估算每个GPU核心可以并发的threadgroup数
  // 基于threadgroup memory使用情况
  size_t tg_memory_used = (config.tile_m * config.tile_k +
                           config.tile_k * config.tile_n +
                           config.tile_m * config.tile_n) * sizeof(float);
  size_t tg_memory_available = 32 * 1024;
  int max_concurrent_tg_per_core = 
      std::max(1, static_cast<int>(tg_memory_available / tg_memory_used));
  
  // M1/M2/M3有不同的并发能力
  // 根据GPU family调整
  float concurrency_factor = 1.0f;
  switch (config.gpu_family) {
    case 7:  // M1
      concurrency_factor = 4.0f;  // 每核心约4个并发threadgroup
      break;
    case 8:  // M2
      concurrency_factor = 6.0f;  // 改进的调度
      break;
    case 9:  // M3
      concurrency_factor = 8.0f;  // 更好的并发性
      break;
    default:
      concurrency_factor = 4.0f;
  }
  
  max_concurrent_tg_per_core = std::min(
      max_concurrent_tg_per_core,
      static_cast<int>(concurrency_factor));
  
  // 理想情况下，有足够的threadgroup填满所有GPU核心
  int ideal_threadgroups = gpu_cores_ * max_concurrent_tg_per_core;
  
  // 计算利用率
  float utilization = std::min(1.0f,
      static_cast<float>(num_threadgroups) / ideal_threadgroups);
  
  // 奖励使用合理大小的threadgroup (192-512最优)
  float size_bonus = 1.0f;
  if (threads_per_tg >= 192 && threads_per_tg <= 512) {
    size_bonus = 1.1f;
  } else if (threads_per_tg < 64 || threads_per_tg > 768) {
    size_bonus = 0.9f;
  }
  
  return utilization * size_bonus;
}
```

**改进效果**: 更准确反映Apple Silicon的并发特性

---

## 🔴 问题4: 内存效率评估不够深入

### 当前实现
```cpp
float MPSSearchStrategy::evaluate_memory_efficiency(
    kernel::mps::MPSKernelConfig const &config) {
  float pattern_score = 1.0f;
  
  switch (config.access_pattern) {
  case kernel::mps::MemoryPattern::COALESCED:
    pattern_score = 1.0f; // Best
    break;
  case kernel::mps::MemoryPattern::TILED:
    pattern_score = 0.85f; // Good
    break;
  case kernel::mps::MemoryPattern::STRIDED:
    pattern_score = 0.7f; // Acceptable
    break;
  }
  
  return pattern_score;
}
```

### 问题
- **仅考虑pattern**: 忽略实际数据大小和访问次数
- **未利用统一内存**: Apple Silicon的统一内存架构特性
- **缺少带宽估算**: 不考虑内存带宽利用率
- **忽略缓存**: GPU L1/L2缓存影响

### 改进方案
```cpp
float evaluate_memory_efficiency(kernel::mps::MPSKernelConfig const &config) {
  // 基础pattern分数
  float pattern_score = 1.0f;
  switch (config.access_pattern) {
  case kernel::mps::MemoryPattern::COALESCED:
    pattern_score = 1.0f;
    break;
  case kernel::mps::MemoryPattern::TILED:
    pattern_score = 0.90f;  // Tiled其实很适合Apple GPU
    break;
  case kernel::mps::MemoryPattern::STRIDED:
    pattern_score = 0.75f;
    break;
  }
  
  // 统一内存架构奖励：大tile可以更好利用带宽
  size_t tile_size = config.tile_m * config.tile_n * config.tile_k;
  float bandwidth_score = 1.0f;
  if (tile_size >= 16384) {  // 16K elements
    bandwidth_score = 1.1f;  // 大数据传输更高效
  } else if (tile_size < 1024) {
    bandwidth_score = 0.9f;  // 小数据传输有overhead
  }
  
  // Threadgroup memory重用率
  // 数据在threadgroup memory中停留越久越好
  float reuse_factor = static_cast<float>(config.tile_k) /
                      std::sqrt(config.tile_m * config.tile_n);
  float reuse_score = std::min(1.0f, reuse_factor);
  
  // M系列不同的内存带宽
  float bandwidth_multiplier = 1.0f;
  switch (config.gpu_family) {
  case 7:  // M1: 68.25 GB/s
    bandwidth_multiplier = 0.9f;
    break;
  case 8:  // M2: 100 GB/s
    bandwidth_multiplier = 1.0f;
    break;
  case 9:  // M3: 100+ GB/s
    bandwidth_multiplier = 1.05f;
    break;
  }
  
  return (pattern_score * 0.4f +
          bandwidth_score * 0.3f +
          reuse_score * 0.3f) * bandwidth_multiplier;
}
```

**改进效果**: 考虑了统一内存架构和实际带宽特性

---

## 🔴 问题5: Threadgroup Memory评估计算错误

### 当前实现
```cpp
float MPSSearchStrategy::evaluate_threadgroup_memory(
    kernel::mps::MPSKernelConfig const &config) {
  // 计算有问题！
  size_t required_memory = (config.tile_m * config.tile_k +
                           config.tile_k * config.tile_n +
                           config.tile_m * config.tile_n) *
                          sizeof(float);  // ← 这里应该是sizeof(float16)
                          
  float memory_ratio = static_cast<float>(required_memory) /
                      config.threadgroup_memory_size;
  
  // 评分逻辑也有问题
  float score = 1.0f - std::abs(1.0f - memory_ratio);
  return std::max(0.0f, score);
}
```

### 问题
- **数据类型错误**: 使用`sizeof(float)`而不是实际的float16
- **评分逻辑不合理**: 使用越接近100%越好？实际上70-80%最优
- **未考虑中间结果**: 可能需要额外临时空间
- **缺少安全边界**: 没有留余量给系统使用

### 改进方案
```cpp
float evaluate_threadgroup_memory(kernel::mps::MPSKernelConfig const &config) {
  // 使用正确的数据类型大小
  const size_t dtype_size = 2;  // float16 = 2 bytes
  
  // 计算实际需要的memory
  // Matrix A: tile_m x tile_k
  // Matrix B: tile_k x tile_n  
  // Matrix C: tile_m x tile_n (accumulation)
  size_t memory_a = config.tile_m * config.tile_k * dtype_size;
  size_t memory_b = config.tile_k * config.tile_n * dtype_size;
  size_t memory_c = config.tile_m * config.tile_n * sizeof(float);  // C用float累加
  
  // 可能需要的临时空间 (如reduce操作)
  size_t temp_memory = config.threads_per_threadgroup * sizeof(float);
  
  size_t total_required = memory_a + memory_b + memory_c + temp_memory;
  
  // Apple Silicon: 32KB threadgroup memory
  const size_t tg_memory = 32 * 1024;
  
  // 如果超出限制，直接返回0
  if (total_required > tg_memory) {
    return 0.0f;
  }
  
  // 计算利用率
  float utilization = static_cast<float>(total_required) / tg_memory;
  
  // 最优范围：60-80% (留余量给系统，但也不要浪费)
  float score = 1.0f;
  if (utilization >= 0.60f && utilization <= 0.80f) {
    score = 1.0f;  // 理想范围
  } else if (utilization > 0.80f && utilization <= 0.95f) {
    score = 0.9f - (utilization - 0.80f) * 2.0f;  // 超过80%开始惩罚
  } else if (utilization < 0.60f) {
    score = 0.7f + utilization * 0.5f;  // 利用率太低也不好
  } else {
    score = 0.5f;  // 超过95%，太危险
  }
  
  // 如果tile配置能整除threadgroup，给予奖励
  int threads_needed = (config.tile_m * config.tile_n + 31) / 32 * 32;
  if (threads_needed == config.threads_per_threadgroup) {
    score *= 1.1f;
  }
  
  return std::min(1.0f, score);
}
```

**改进效果**: 
- 修正计算错误
- 更合理的评分逻辑
- 考虑实际工程约束

---

## 🔴 问题6: GPU核心数检测不准确

### 当前实现
```cpp
int MPSOptimizer::get_gpu_core_count() {
  int family = detect_gpu_family();
  
  switch (family) {
  case 7:  // M1
    return 8;   // M1 has 7-8 GPU cores (base)
  case 8:  // M2
    return 10;  // M2 has 8-10 GPU cores (base)
  case 9:  // M3
    return 10;  // M3 has 10+ GPU cores (base)
  default:
    return 8;
  }
}
```

### 问题
- **忽略Pro/Max/Ultra变体**: M3 Max有40核！
- **返回值不准确**: 只区分了M1/M2/M3基础版
- **缺少运行时检测**: 应该尝试查询实际GPU核心数

### 改进方案
```cpp
int get_gpu_core_count() {
#ifdef __APPLE__
  // 尝试通过Metal API获取实际GPU核心数
  // 这需要添加Metal framework依赖
  // 暂时使用sysctl和模型识别
  
  char model[256];
  size_t len = sizeof(model);
  if (sysctlbyname("hw.model", model, &len, NULL, 0) == 0) {
    std::string model_str(model);
    
    // M1系列
    if (model_str.find("Mac13,") != std::string::npos) {
      if (model_str.find("Mac13,2") != std::string::npos) {
        return 8;   // M1 Pro (14/16 cores)
      } else if (model_str.find("Mac13,1") != std::string::npos) {
        return 32;  // M1 Max (24/32 cores)
      }
      return 7;  // M1 (7/8 cores)
    }
    
    // M2系列
    if (model_str.find("Mac14,") != std::string::npos) {
      if (model_str.find("Mac14,5") != std::string::npos ||
          model_str.find("Mac14,9") != std::string::npos) {
        return 19;  // M2 Pro (16/19 cores)
      } else if (model_str.find("Mac14,6") != std::string::npos) {
        return 38;  // M2 Max (30/38 cores)
      }
      return 10;  // M2 (8/10 cores)
    }
    
    // M3系列
    if (model_str.find("Mac15,") != std::string::npos) {
      if (model_str.find("Mac15,3") != std::string::npos ||
          model_str.find("Mac15,6") != std::string::npos) {
        return 18;  // M3 Pro (14/18 cores)
      } else if (model_str.find("Mac15,7") != std::string::npos ||
          model_str.find("Mac15,11") != std::string::npos) {
        return 40;  // M3 Max (30/40 cores)
      }
      return 10;  // M3 (10 cores)
    }
  }
  
  // 备选：尝试使用IOKit查询
  // TODO: 添加IOKit实现
  
  // 默认值
  return 10;
#else
  return 0;
#endif
}
```

**改进效果**: 能够识别Pro/Max变体，更准确的GPU核心数

---

## 🟢 额外优化建议

### 1. 添加Adaptive搜索策略

```cpp
// 根据初步结果动态调整搜索空间
std::vector<CandidateConfig> generate_adaptive_candidates(
    kernel::Graph const &graph,
    std::vector<CandidateConfig> const &initial_results) {
  
  // 分析top 10%配置的特征
  auto top_configs = get_top_k_configs(initial_results, 
                                       initial_results.size() / 10);
  
  // 提取共同特征
  int avg_threadgroup = 0;
  int avg_tile_m = 0, avg_tile_n = 0, avg_tile_k = 0;
  
  for (auto const &config : top_configs) {
    auto *mps_cfg = static_cast<kernel::mps::MPSKernelConfig*>(
        config.config.get());
    avg_threadgroup += mps_cfg->threads_per_threadgroup;
    avg_tile_m += mps_cfg->tile_m;
    avg_tile_n += mps_cfg->tile_n;
    avg_tile_k += mps_cfg->tile_k;
  }
  
  int n = top_configs.size();
  avg_threadgroup /= n;
  avg_tile_m /= n;
  avg_tile_n /= n;
  avg_tile_k /= n;
  
  // 在最优值附近生成更多候选
  return generate_fine_grained_candidates(
      avg_threadgroup, avg_tile_m, avg_tile_n, avg_tile_k);
}
```

### 2. 考虑Dynamic Caching (M3+特性)

```cpp
// M3引入了Dynamic Caching - GPU内存可以按需分配给不同任务
float evaluate_for_m3_dynamic_caching(
    kernel::mps::MPSKernelConfig const &config) {
  
  if (config.gpu_family < 9) {  // M3 = family 9
    return 1.0f;  // 不影响旧芯片
  }
  
  // M3的Dynamic Caching允许更灵活的内存使用
  // 偏好较大的threadgroup以充分利用这个特性
  float bonus = 1.0f;
  if (config.threads_per_threadgroup >= 256) {
    bonus = 1.15f;  // M3在大threadgroup上表现更好
  }
  
  return bonus;
}
```

### 3. 实现Profile-guided优化

```cpp
// 记录实际运行结果，用于future优化
class MPSProfiler {
  struct ProfileEntry {
    MPSKernelConfig config;
    float actual_time_ms;
    float estimated_score;
    size_t problem_size;
  };
  
  std::vector<ProfileEntry> history_;
  
public:
  void record_result(MPSKernelConfig const &cfg, 
                    float time, float score, size_t size) {
    history_.push_back({cfg, time, score, size});
  }
  
  // 为相似问题提供建议配置
  MPSKernelConfig suggest_config(size_t problem_size) {
    // 查找相似问题的最佳配置
    auto similar = find_similar_problems(problem_size);
    return get_best_config(similar);
  }
};
```

---

## 📊 与CUDA策略对比

| 特性 | CUDA策略 | MPS策略(当前) | MPS策略(建议) |
|------|----------|---------------|---------------|
| **候选数量** | 几百个 | ~16个 | ~100-200个 |
| **Threadgroup配置** | 4-32 warps | 4种大小 | 10-20种大小 |
| **Tile配置** | 多种Tensor Core配置 | 4种固定tile | 几十种动态tile |
| **GPU变体识别** | 通过compute capability | 基础识别M1/M2/M3 | 识别Pro/Max/Ultra |
| **内存优化** | Shared memory考虑充分 | 基础考虑 | 统一内存优化 |
| **Occupancy** | 详细计算 | 简化计算 | 改进计算 |
| **Adaptive搜索** | 无 | 无 | 建议添加 |

---

## 🎯 优先级改进路线图

### 阶段1: 基础修复 (立即)
1. ✅ 修正threadgroup memory计算（数据类型）
2. ✅ 改进tile配置生成（考虑内存限制）
3. ✅ 增加threadgroup候选数量

### 阶段2: 准确性提升 (短期)
1. ⏳ 改进GPU核心数检测（识别Pro/Max/Ultra）
2. ⏳ 优化GPU利用率评估（实际并发特性）
3. ⏳ 增强内存效率评估（统一内存架构）

### 阶段3: 高级优化 (中期)
1. 📋 实现adaptive搜索策略
2. 📋 添加M3 Dynamic Caching支持
3. 📋 实现profile-guided优化

### 阶段4: 生产就绪 (长期)
1. 📋 添加Metal API直接查询
2. 📋 实现运行时auto-tuning
3. 📋 建立benchmark数据库

---

## 💡 总结

当前MPS搜索策略**基础正确但需要大幅改进**：

### ✅ 优势
- 正确理解Apple Silicon基础架构
- 多维度评估框架合理
- 基础验证逻辑正确

### ❌ 不足
- 搜索空间太小（16个候选 vs CUDA的几百个）
- GPU特性理解不够深入（忽略Pro/Max/Ultra差异）
- 统一内存架构优势未充分利用
- 评估指标有计算错误

### 🎯 改进效果预期
实施上述改进后：
- **搜索空间**: 16个 → 100-200个候选
- **准确性**: +30-50% 性能提升
- **适配性**: 自动适应所有M系列变体
- **效率**: Profile-guided优化持续改进

**建议优先实施阶段1和阶段2的改进！**

