# YiRage + Ray/RLlib 闭环集成设计

## 核心问题分析

### 问题 1: 缺乏闭环
- 单向数据流：搜索产生数据给 RL，但 RL 决策无法影响搜索
- 验证需要 GPU：YiRage fingerprint 验证必须在 GPU 上执行
- RL 策略更新与搜索执行脱节

### 问题 2: 缺乏层级化设计 (关键!)
- 搜索是**多层次**的：
  - **Level 1**: 硬件配置搜索 (grid_dim, block_dim, etc.)
  - **Level 2**: µGraph 搜索 (在配置约束下)
- 配置**控制**图搜索空间
- 原设计将两者混为一体

### 解决方案：层级化闭环

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      层级化闭环架构                                      │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Level 1: Config Policy (RL 策略 1)                                 │ │
│  │   观察: 目标图特征 + 硬件能力                                        │ │
│  │   动作: 选择 grid_dim, block_dim, forloop_range, etc.              │ │
│  │   奖励: 基于 Level 2 搜索的整体结果                                  │ │
│  └────────────────────────────┬───────────────────────────────────────┘ │
│                               │                                          │
│                    配置 + 约束传递给 Level 2                             │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Level 2: Graph Policy (RL 策略 2 / 传统搜索)                       │ │
│  │   输入: 目标图 + 配置约束                                            │ │
│  │   约束:                                                              │ │
│  │     • 可用 imap 由 grid_dim 决定                                    │ │
│  │     • 可用 frange 由 forloop_range 决定                             │ │
│  │     • 最大算子数由资源约束决定                                       │ │
│  │   动作: 在约束内构建 µGraph                                          │ │
│  │   奖励: GPU 验证结果 + 性能                                          │ │
│  └────────────────────────────┬───────────────────────────────────────┘ │
│                               │                                          │
│                               ▼                                          │
│                    ┌──────────────────────┐                              │
│                    │ GPU Verification     │                              │
│                    │ (Fingerprint + Prof) │                              │
│                    └──────────┬───────────┘                              │
│                               │                                          │
│                    verified, latency_ms                                  │
│                               │                                          │
│                    ┌──────────▼──────────┐                               │
│                    │ Reward Computation  │                               │
│                    │                      │                               │
│                    │ reward_l1 → Policy 1 │                              │
│                    │ reward_l2 → Policy 2 │                              │
│                    └─────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 精巧设计：三层协作架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Ray Cluster                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Layer 1: RLlib Trainer (CPU)                                       │ │
│  │ ┌──────────────────────────────────────────────────────────────┐   │ │
│  │ │ PPO/SAC Algorithm                                            │   │ │
│  │ │  • Policy Network (决策哪个配置)                              │   │ │
│  │ │  • Value Network (评估配置价值)                               │   │ │
│  │ │  • Experience Replay Buffer                                   │   │ │
│  │ └──────────────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────┬─────────────────────────────────────┘ │
│                                 │                                        │
│                        actions (配置选择)                                │
│                                 │                                        │
│                                 ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Layer 2: YiRageSearchEnv (Python, 桥接层)                          │ │
│  │ ┌──────────────────────────────────────────────────────────────┐   │ │
│  │ │ • action → 解析配置 (grid_dim, block_dim, imaps, etc.)       │   │ │
│  │ │ • 调用 C++ RLSearchContext 生成候选 kernel graph              │   │ │
│  │ │ • 路由到 GPU 进行验证 (通过 C++ 或 Ray Verifier Pool)         │   │ │
│  │ │ • 收集 fingerprint 结果，计算 reward                          │   │ │
│  │ │ • 构造 next_obs 返回给 RLlib                                  │   │ │
│  │ └──────────────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────┬─────────────────────────────────────┘ │
│                                 │                                        │
│                     verify_request (kernel_graph)                        │
│                                 │                                        │
│                                 ▼                                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Layer 3: YiRage C++ Core + GPU Verification                        │ │
│  │ ┌────────────────────────────────────────────────────────────────┐ │ │
│  │ │ RLSearchContext (C++)                                          │ │ │
│  │ │  • apply_action() → 修改 kernel/threadblock graph              │ │ │
│  │ │  • verify() → GPU fingerprint verification                     │ │ │
│  │ │  • profile() → GPU performance profiling                       │ │ │
│  │ │  • get_state() → 返回搜索状态给 Python                         │ │ │
│  │ └────────────────────────────────────────────────────────────────┘ │ │
│  │                                OR                                   │ │
│  │ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐           │ │
│  │ │ GPUVerifier 0  │ │ GPUVerifier 1  │ │ GPUVerifier 2  │ (Ray)     │ │
│  │ │  GPU:0         │ │  GPU:1         │ │  GPU:0         │           │ │
│  │ └────────────────┘ └────────────────┘ └────────────────┘           │ │
│  └──────────────────────────────┬─────────────────────────────────────┘ │
│                                 │                                        │
│                    (verified, latency_ms, fingerprint)                   │
│                                 │                                        │
│                                 ▼                                        │
│                          Reward Computation                              │
│                                 │                                        │
│                                 ▼                                        │
│                      (obs, reward, done, info)                           │
│                                 │                                        │
│                          返回给 RLlib                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 文件结构

```
python/yirage/rl/
├── __init__.py                    # Module entry point
├── env/
│   ├── __init__.py
│   ├── yirage_env.py              # 核心 Gymnasium 环境 (闭环桥接)
│   ├── action_space.py            # 动作空间定义
│   ├── observation.py             # 观察空间和编码
│   └── reward.py                  # 奖励计算
├── verifier/
│   ├── __init__.py
│   ├── gpu_verifier.py            # GPU Verifier (Ray Actor)
│   └── verifier_pool.py           # Verifier 池管理
├── training/
│   ├── __init__.py
│   ├── trainer.py                 # 训练入口
│   └── callbacks.py               # RLlib 回调
└── models/
    ├── __init__.py
    ├── graph_encoder.py           # 图编码网络
    └── policy_network.py          # 策略网络 (Action Masking)

include/search/
└── rl_interface.h                 # C++ RL 接口定义

src/search/
└── rl_interface.cc                # C++ RL 接口实现

python/yirage/_cython/
├── rl_core.pxd                    # Cython 声明
└── rl_core.pyx                    # Cython 绑定
```

---

## 核心组件详解

### 1. YiRageSearchEnv (闭环桥接层)

```python
class YiRageSearchEnv(gym.Env):
    """
    核心闭环环境
    
    step() 流程:
    1. RL policy 输出 action (配置选择)
    2. Env 解码 action 为 SearchConfig
    3. 路由到 C++ RLSearchContext (如果可用) 或 Python fallback
    4. C++ 调用 GPU 进行 fingerprint 验证
    5. C++ 返回验证结果和性能数据
    6. Env 计算 reward
    7. Env 返回 observation 给 RL policy
    
    闭环关键: GPU 验证结果直接影响 RL 训练
    """
    
    def step(self, action):
        # 使用 C++ 上下文形成真正闭环
        if self._use_cpp_context:
            return self._step_with_cpp_context(action, ...)
        else:
            return self._step_with_python_fallback(action, ...)
```

### 2. RLSearchContext (C++ 核心)

```cpp
class RLSearchContext {
public:
    // 应用 RL 动作 (闭环入口)
    bool apply_action(int action_type, const RLConfig& config);
    
    // GPU 验证 (闭环关键!)
    VerifyResult verify();  // 调用 GPU fingerprint 验证
    
    // GPU Profiling (性能信号)
    ProfileResult profile(int warmup, int iters);
    
    // 状态获取 (用于构造 observation)
    SearchState get_state() const;
};
```

### 3. RewardComputer (多目标奖励)

```python
class RewardComputer:
    """
    多目标奖励计算
    
    闭环关键: GPU 验证结果直接影响奖励
    
    reward = validity_weight * validity_reward
           + performance_weight * performance_reward  # 来自 GPU profiling
           + efficiency_weight * efficiency_reward
           + exploration_weight * exploration_reward
    """
    
    def compute(self, verify_result, profile_result, ...):
        # GPU 验证结果 → 有效性奖励
        if verify_result.verified:
            validity_reward = 1.0
        else:
            validity_reward = -0.1
        
        # GPU profiling → 性能奖励
        if verify_result.verified and profile_result:
            performance_reward = log(best_latency / current_latency)
        
        return weighted_sum(...)
```

---

## 闭环流程详解

### 完整训练流程

```python
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from yirage.rl import YiRageSearchEnv, EnvConfig, train_rl_search

# 1. 配置环境
env_config = EnvConfig(
    target_graph_json=target_graph,
    backend="cuda",
    num_gpus=4,
    max_search_depth=50,
)

# 2. 配置 RLlib
config = (
    PPOConfig()
    .environment(
        env=YiRageSearchEnv,
        env_config=vars(env_config),
    )
    .framework("torch")
    .resources(
        num_gpus=0,  # Trainer 在 CPU
        num_cpus_per_worker=1,
    )
    .rollouts(
        num_rollout_workers=8,  # CPU workers
    )
)

# 3. 训练 (闭环自动形成)
results = train_rl_search(config)

# 4. 使用训练好的策略搜索
policy = load_trained_policy(results["best_checkpoint"])
kernels = search_with_policy(policy, new_target_graph)
```

### 单步闭环细节

```
┌─────────────────────────────────────────────────────────────────┐
│ Episode Step N                                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. RL Policy 输出 action                                        │
│     action = [action_type=3, grid_x=2, block_x=3, ...]          │
│                        │                                         │
│                        ▼                                         │
│  2. YiRageSearchEnv 解码                                         │
│     config = SearchConfig(                                       │
│         grid_dim=(16, 64, 1),                                   │
│         block_dim=(256, 1, 1),                                  │
│         operator="MATMUL",                                       │
│         ...                                                      │
│     )                                                            │
│                        │                                         │
│                        ▼                                         │
│  3. C++ RLSearchContext.apply_action()                          │
│     → 修改内部 kernel graph                                      │
│     → 更新搜索状态                                               │
│                        │                                         │
│                        ▼                                         │
│  4. action_type == FINISH? → C++ verify()                       │
│     ┌─────────────────────────────────────────────┐              │
│     │ GPU Fingerprint Verification                │              │
│     │  • 编译 kernel                              │              │
│     │  • 运行随机输入                             │              │
│     │  • 比较输出 fingerprint                     │              │
│     │  • 返回 verified=True/False                 │              │
│     └─────────────────────────────────────────────┘              │
│                        │                                         │
│                        ▼                                         │
│  5. verified? → C++ profile()                                    │
│     ┌─────────────────────────────────────────────┐              │
│     │ GPU Performance Profiling                   │              │
│     │  • Warmup iterations                        │              │
│     │  • Timed execution                          │              │
│     │  • 返回 latency_ms                          │              │
│     └─────────────────────────────────────────────┘              │
│                        │                                         │
│                        ▼                                         │
│  6. RewardComputer 计算 reward                                   │
│     reward = f(verified, latency_ms, depth, novelty)             │
│                        │                                         │
│                        ▼                                         │
│  7. ObservationEncoder 编码新状态                                │
│     obs = encode(updated_state)                                  │
│                        │                                         │
│                        ▼                                         │
│  8. 返回 (obs, reward, done, truncated, info)                   │
│     → RLlib 收集经验                                             │
│     → 更新 Policy (闭环完成!)                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 关键设计决策

### 1. 双路径架构

```python
if self._use_cpp_context:
    # 路径 A: C++ 原生闭环 (推荐)
    # - 所有操作在 C++ 中完成
    # - GPU 验证直接调用
    # - 最高效率
    return self._step_with_cpp_context(action)
else:
    # 路径 B: Python + Ray 闭环 (备选)
    # - 通过 Ray Actor 调用 GPU
    # - 更灵活但开销更大
    return self._step_with_python_fallback(action)
```

### 2. GPU 资源管理

- **C++ 路径**: 直接使用 CUDA context
- **Ray 路径**: 通过 `VerifierPool` 管理多 GPU

### 3. 异步验证支持

```python
# 批量异步验证 (提高吞吐)
futures = [pool.verify_async(kernel) for kernel in batch]
results = ray.get(futures)
```

---

## 实施路线

### Phase 1: 核心闭环 (当前)
- [x] YiRageSearchEnv 实现
- [x] Action/Observation 空间定义
- [x] Reward 计算器
- [x] C++ RLSearchContext 接口
- [x] Cython 绑定

### Phase 2: GPU 集成
- [ ] 完成 C++ verify() 的实际 GPU 调用
- [ ] 完成 C++ profile() 的实际 GPU 调用
- [ ] Ray GPUVerifier Actor 实现

### Phase 3: RLlib 集成
- [ ] PPO/SAC 配置
- [ ] 回调和监控
- [ ] 检查点管理

### Phase 4: 优化
- [ ] 批量验证
- [ ] 异步执行
- [ ] 课程学习

---

## 总结

本设计实现了真正的 RL-YiRage 闭环：

1. **RL → YiRage**: 通过 `apply_action()` 将 RL 决策传递给 C++ 搜索核心
2. **YiRage → GPU**: 通过 `verify()` 和 `profile()` 在 GPU 上验证和评估
3. **GPU → RL**: 通过 `RewardComputer` 将 GPU 结果转化为训练信号

关键洞察：
- **GPU 验证是闭环核心** - 提供真实的正确性和性能信号
- **C++ 作为桥梁** - 连接 Python RL 和底层 CUDA 操作
- **Ray 提供可扩展性** - 分布式验证和训练
