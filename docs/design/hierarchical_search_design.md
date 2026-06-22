# YiRage 层级化搜索设计

## 核心洞察

搜索空间实际上是**层级化**的：

1. **Level 1: 硬件配置搜索 (Config Search)**
   - 决定硬件执行参数
   - 约束下层 µGraph 的可行解空间

2. **Level 2: µGraph 搜索 (Graph Search)**
   - 在给定配置约束下搜索最优图结构
   - 受上层配置参数的严格约束

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       层级化搜索架构                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Level 1: Config Search (RL Policy 1)                               │ │
│  │                                                                     │ │
│  │  观察: 目标图特征、硬件能力、历史性能                                │ │
│  │  动作: 选择硬件配置参数                                              │ │
│  │       ┌──────────────────────────────────────────────┐              │ │
│  │       │ HardwareConfig                               │              │ │
│  │       │  • grid_dim: (x, y, z)                      │              │ │
│  │       │  • block_dim: (x, y, z)                     │              │ │
│  │       │  • forloop_range: int                       │              │ │
│  │       │  • reduction_dimx: int                      │              │ │
│  │       │  • shared_memory_size: int                  │              │ │
│  │       │  • num_registers: int                       │              │ │
│  │       └──────────────────────────────────────────────┘              │ │
│  │                                                                     │ │
│  │  输出: 配置 + 约束边界                                               │ │
│  └────────────────────────────┬───────────────────────────────────────┘ │
│                               │                                          │
│                    ┌──────────▼──────────┐                               │
│                    │ 搜索空间约束计算    │                               │
│                    │ (Search Space       │                               │
│                    │  Constraint)        │                               │
│                    └──────────┬──────────┘                               │
│                               │                                          │
│                    配置约束传递给 Level 2                                │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Level 2: µGraph Search (RL Policy 2 / Traditional Search)         │ │
│  │                                                                     │ │
│  │  输入: 目标图 + 配置约束                                             │ │
│  │  约束:                                                               │ │
│  │       • 可用的 imap 选项 (由 grid_dim 决定)                         │ │
│  │       • 可用的 omap 选项 (由 block_dim 决定)                        │ │
│  │       • 可用的 frange 值 (由 forloop_range 约束)                    │ │
│  │       • tensor 分割方式 (由 grid/block 决定)                        │ │
│  │                                                                     │ │
│  │  动作: 在约束内构建 µGraph                                           │ │
│  │       • 选择 KN 算子                                                 │ │
│  │       • 选择 TB 算子                                                 │ │
│  │       • 选择输入/输出映射 (在约束范围内)                              │ │
│  │                                                                     │ │
│  │  输出: 完整的 kernel graph                                          │ │
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
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 配置如何约束 µGraph 搜索

### 约束传递机制

```python
class SearchSpaceConstraints:
    """
    从 Level 1 配置计算 Level 2 搜索空间约束
    """
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        
    def get_valid_imaps(self) -> List[Tuple[int, int, int]]:
        """
        imap 选项由 grid_dim 决定
        
        imap[i] ∈ {-1, 0, 1} 表示:
          -1: 不映射到此维度
           0: 映射到 blockIdx.x
           1: 映射到 blockIdx.y
        
        但只有当 grid_dim 的对应维度 > 1 时，映射才有意义
        """
        valid_imaps = []
        
        for ix in [-1, 0, 1]:
            for iy in [-1, 0, 1]:
                for iz in [-1, 0, 1]:
                    # 只有 grid_dim > 1 的维度才能映射
                    if ix == 0 and self.config.grid_dim[0] <= 1:
                        continue
                    if iy == 1 and self.config.grid_dim[1] <= 1:
                        continue
                    if iz == 2 and self.config.grid_dim[2] <= 1:
                        continue
                    valid_imaps.append((ix, iy, iz))
        
        return valid_imaps
    
    def get_valid_franges(self) -> List[int]:
        """
        frange 必须是 forloop_range 的因子
        """
        fr = self.config.forloop_range
        return [f for f in [1, 2, 4, 8, 16, 32] if fr % f == 0]
    
    def get_max_operators(self) -> int:
        """
        算子数量受 shared memory 和 register 约束
        """
        # 简化计算：更多资源允许更多算子
        sm_limit = self.config.shared_memory_size // 4096  # 每算子约 4KB
        reg_limit = self.config.num_registers // 32  # 每算子约 32 regs
        return min(sm_limit, reg_limit, 20)  # 上限 20
    
    def get_tensor_tile_sizes(self, tensor_shape: List[int]) -> List[List[int]]:
        """
        tensor 分块大小由 grid_dim 和 block_dim 决定
        
        tensor 的每个维度可以按 grid_dim 分割（跨 block）
        或按 block_dim 分割（block 内）
        """
        tiles = []
        for i, dim in enumerate(tensor_shape):
            # 可以按 grid 维度分割
            grid_tiles = [dim // self.config.grid_dim[j] 
                          for j in range(3) if self.config.grid_dim[j] > 1]
            # 可以按 block 维度分割
            block_tiles = [dim // self.config.block_dim[j]
                           for j in range(3) if self.config.block_dim[j] > 1]
            tiles.append(list(set(grid_tiles + block_tiles + [dim])))
        return tiles
```

### 示例：配置如何影响搜索

```
配置 1: grid_dim=(64, 1, 1), block_dim=(128, 1, 1), forloop_range=16

  约束:
  - imap 只能使用 x 维度: (-1, *, *), (0, *, *)
  - frange 只能是 [1, 2, 4, 8, 16]
  - tensor 只能沿一个维度分块
  
配置 2: grid_dim=(8, 8, 1), block_dim=(128, 4, 1), forloop_range=32

  约束:
  - imap 可以使用 x 和 y: (-1, *, *), (0, *, *), (*, 1, *)
  - frange 只能是 [1, 2, 4, 8, 16, 32]
  - tensor 可以沿两个维度分块
  - 更多算子组合可能
```

---

## Level 1: 硬件配置搜索 (Config Policy)

### 观察空间

```python
class ConfigObservationSpace:
    """
    Level 1 观察: 目标图特征 + 硬件能力
    """
    
    observation_space = spaces.Dict({
        # 目标计算图特征
        "target_graph_features": spaces.Box(-np.inf, np.inf, (64,)),
        # - 输入/输出 tensor 形状统计
        # - 算子类型分布
        # - 计算/内存比率
        
        # 硬件能力
        "hardware_features": spaces.Box(0, 1, (16,)),
        # - SM 数量
        # - 最大 shared memory
        # - 最大 registers
        # - warp size
        # - compute capability
        
        # 历史性能 (用于学习)
        "history_features": spaces.Box(-np.inf, np.inf, (32,)),
        # - 之前配置的性能
        # - 成功率统计
    })
```

### 动作空间

```python
class ConfigActionSpace:
    """
    Level 1 动作: 选择硬件配置
    """
    
    action_space = spaces.Dict({
        # Grid 维度选择
        "grid_x": spaces.Discrete(len(GRID_CHOICES)),  # [1, 2, 4, 8, ..., 128]
        "grid_y": spaces.Discrete(len(GRID_CHOICES)),
        "grid_z": spaces.Discrete(len(GRID_CHOICES)),
        
        # Block 维度选择
        "block_x": spaces.Discrete(len(BLOCK_CHOICES)),  # [32, 64, 128, 256, ...]
        "block_y": spaces.Discrete(len(BLOCK_CHOICES)),
        "block_z": spaces.Discrete(len(BLOCK_CHOICES)),
        
        # Forloop 配置
        "forloop_range": spaces.Discrete(len(FRANGE_CHOICES)),  # [1, 2, 4, 8, ...]
        "reduction_dimx": spaces.Discrete(len(RDIM_CHOICES)),
        
        # 内存配置
        "shared_memory_tier": spaces.Discrete(4),  # 小/中/大/最大
        "register_pressure": spaces.Discrete(3),  # 低/中/高
    })
```

### 奖励设计

```python
class ConfigReward:
    """
    Level 1 奖励: 基于下层搜索的整体结果
    """
    
    def compute(self, 
                config: HardwareConfig,
                level2_results: List[GraphSearchResult]) -> float:
        """
        配置的奖励 = 该配置下找到的最佳 kernel 性能
        
        这样 Level 1 学会选择"好的"配置
        即那些能让 Level 2 找到好 kernel 的配置
        """
        
        if not level2_results:
            # 配置太差，Level 2 找不到任何有效 kernel
            return -1.0
        
        best_result = min(level2_results, key=lambda r: r.latency_ms)
        
        if not best_result.verified:
            return -0.5
        
        # 性能奖励
        speedup = self.baseline_latency / best_result.latency_ms
        perf_reward = np.log(speedup + 1)
        
        # 搜索效率奖励 (更少的 Level 2 步数找到好结果)
        efficiency_reward = -0.01 * best_result.search_steps
        
        return perf_reward + efficiency_reward
```

---

## Level 2: µGraph 搜索 (Graph Policy)

### 输入：配置约束

```python
class ConstrainedGraphSearch:
    """
    Level 2 搜索: 在 Level 1 配置约束下搜索 µGraph
    """
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.constraints = SearchSpaceConstraints(config)
        
        # 动态构建约束后的动作空间
        self.valid_imaps = self.constraints.get_valid_imaps()
        self.valid_franges = self.constraints.get_valid_franges()
        self.max_ops = self.constraints.get_max_operators()
```

### 观察空间

```python
class GraphObservationSpace:
    """
    Level 2 观察: 当前图状态 + 配置约束
    """
    
    observation_space = spaces.Dict({
        # 当前 µGraph 状态
        "current_graph_embedding": spaces.Box(-np.inf, np.inf, (128,)),
        "num_kn_operators": spaces.Box(0, 50, (1,)),
        "num_tb_operators": spaces.Box(0, 50, (1,)),
        "search_level": spaces.Discrete(2),  # KN or TB
        
        # 配置约束 (从 Level 1 传来)
        "config_embedding": spaces.Box(-np.inf, np.inf, (32,)),
        
        # 有效动作掩码 (基于约束)
        "valid_imap_mask": spaces.MultiBinary(len(ALL_IMAPS)),
        "valid_frange_mask": spaces.MultiBinary(len(ALL_FRANGES)),
        "valid_operator_mask": spaces.MultiBinary(len(ALL_OPS)),
        
        # 剩余资源
        "remaining_ops": spaces.Box(0, 50, (1,)),
    })
```

### 动作空间 (受约束)

```python
class ConstrainedGraphActionSpace:
    """
    Level 2 动作: 在约束内构建图
    """
    
    def __init__(self, constraints: SearchSpaceConstraints):
        self.constraints = constraints
        
        # 动作空间根据约束动态调整
        self.action_space = spaces.Dict({
            "action_type": spaces.Discrete(4),  # ADD_KN, CREATE_TB, ADD_TB, FINISH
            
            # 算子选择
            "operator": spaces.Discrete(len(ALL_OPS)),
            "input_0": spaces.Discrete(MAX_TENSORS),
            "input_1": spaces.Discrete(MAX_TENSORS),
            
            # imap 选择 (从约束后的有效选项中选)
            "imap_idx": spaces.Discrete(len(constraints.get_valid_imaps())),
            
            # frange 选择 (从约束后的有效选项中选)
            "frange_idx": spaces.Discrete(len(constraints.get_valid_franges())),
        })
    
    def decode_action(self, action: dict) -> GraphAction:
        """解码动作，确保符合约束"""
        
        # imap 从约束后的列表中选
        valid_imaps = self.constraints.get_valid_imaps()
        imap = valid_imaps[action["imap_idx"] % len(valid_imaps)]
        
        # frange 同理
        valid_franges = self.constraints.get_valid_franges()
        frange = valid_franges[action["frange_idx"] % len(valid_franges)]
        
        return GraphAction(
            action_type=action["action_type"],
            operator=ALL_OPS[action["operator"]],
            inputs=[action["input_0"], action["input_1"]],
            imap=imap,
            frange=frange,
        )
```

---

## 层级化搜索流程

### 完整流程

```python
class HierarchicalSearch:
    """
    层级化搜索: Level 1 (Config) → Level 2 (Graph)
    """
    
    def __init__(self):
        self.config_policy = ConfigPolicy()  # RL Policy 1
        self.graph_policy = GraphPolicy()    # RL Policy 2 (或传统搜索)
        self.verifier_pool = VerifierPool()
    
    def search(self, target_graph: str) -> List[KernelResult]:
        """
        执行层级化搜索
        """
        results = []
        
        # 多次尝试不同配置
        for episode in range(num_config_episodes):
            
            # ═══════════════════════════════════════════
            # Level 1: 选择硬件配置
            # ═══════════════════════════════════════════
            config_obs = self._get_config_observation(target_graph)
            config_action = self.config_policy.get_action(config_obs)
            hardware_config = self._decode_config_action(config_action)
            
            # 计算搜索空间约束
            constraints = SearchSpaceConstraints(hardware_config)
            
            # ═══════════════════════════════════════════
            # Level 2: 在约束下搜索 µGraph
            # ═══════════════════════════════════════════
            graph_search = ConstrainedGraphSearch(hardware_config)
            
            for step in range(max_graph_steps):
                graph_obs = graph_search.get_observation()
                
                # 动作受约束
                graph_action = self.graph_policy.get_action(
                    graph_obs, 
                    valid_action_mask=constraints.get_action_mask()
                )
                
                # 应用动作 (约束保证合法)
                graph_search.apply_action(graph_action)
                
                if graph_search.is_complete():
                    break
            
            # ═══════════════════════════════════════════
            # GPU 验证
            # ═══════════════════════════════════════════
            kernel_graph = graph_search.get_kernel_graph()
            verify_result = self.verifier_pool.verify(kernel_graph)
            
            if verify_result.verified:
                profile_result = self.verifier_pool.profile(kernel_graph)
                results.append(KernelResult(
                    config=hardware_config,
                    kernel_graph=kernel_graph,
                    latency_ms=profile_result.latency_ms,
                ))
            
            # ═══════════════════════════════════════════
            # 奖励反馈
            # ═══════════════════════════════════════════
            
            # Level 2 奖励: 验证结果 + 性能
            graph_reward = self._compute_graph_reward(verify_result, profile_result)
            self.graph_policy.update(graph_reward)
            
            # Level 1 奖励: 整体搜索结果
            config_reward = self._compute_config_reward(results)
            self.config_policy.update(config_reward)
        
        return sorted(results, key=lambda r: r.latency_ms)
```

### 两级策略的训练

```python
class HierarchicalTrainer:
    """
    层级化策略训练
    
    两种训练模式:
    1. 联合训练: Level 1 和 Level 2 同时训练
    2. 分层训练: 先固定 Level 1，训练 Level 2；再训练 Level 1
    """
    
    def train_joint(self, target_graphs: List[str]):
        """
        联合训练
        
        每个 episode:
        1. Level 1 选配置
        2. Level 2 搜索图
        3. GPU 验证
        4. 两级都更新
        """
        for epoch in range(num_epochs):
            for target in target_graphs:
                # Level 1 选配置
                config = self.config_policy.sample_config(target)
                
                # Level 2 在配置下搜索
                constraints = SearchSpaceConstraints(config)
                for _ in range(graph_steps):
                    action = self.graph_policy.sample_action(constraints)
                    self.graph_env.step(action)
                
                # 验证和奖励
                result = self.verify()
                
                # 更新两个策略
                self.graph_policy.learn(result)
                self.config_policy.learn(result)
    
    def train_hierarchical(self, target_graphs: List[str]):
        """
        分层训练 (更稳定)
        
        Phase 1: 用固定配置训练 Level 2 (图策略)
        Phase 2: 用训练好的 Level 2 训练 Level 1 (配置策略)
        """
        
        # Phase 1: 训练 Level 2
        print("Phase 1: Training Graph Policy...")
        for config in CANONICAL_CONFIGS:  # 预定义的典型配置
            constraints = SearchSpaceConstraints(config)
            for epoch in range(graph_epochs):
                self._train_graph_policy_with_config(constraints)
        
        # Phase 2: 训练 Level 1
        print("Phase 2: Training Config Policy...")
        for epoch in range(config_epochs):
            for target in target_graphs:
                config = self.config_policy.sample_config(target)
                
                # Level 2 现在是固定的 (或者继续微调)
                result = self._run_graph_search(config)
                
                # 只更新 Level 1
                self.config_policy.learn(result)
```

---

## 与 Ray 的集成

### 分布式层级化搜索

```python
@ray.remote(num_cpus=1)
class ConfigSearchWorker:
    """
    Level 1 Worker: 搜索配置空间
    """
    
    def __init__(self, config_policy, graph_policy, verifier_pool):
        self.config_policy = config_policy
        self.graph_policy = graph_policy
        self.verifier_pool = verifier_pool
    
    def search_config(self, target_graph: str) -> List[KernelResult]:
        """
        探索一个配置并运行 Level 2 搜索
        """
        # Level 1: 采样配置
        config = self.config_policy.sample()
        constraints = SearchSpaceConstraints(config)
        
        # Level 2: 在约束下搜索 (可能有多个 graph episode)
        results = []
        for _ in range(graph_episodes_per_config):
            kernel = self._run_graph_search(constraints)
            verify_result = self.verifier_pool.verify(kernel)
            if verify_result.verified:
                results.append(kernel)
        
        return results


class DistributedHierarchicalSearch:
    """
    分布式层级化搜索协调器
    """
    
    def __init__(self, num_workers: int):
        self.workers = [
            ConfigSearchWorker.remote(...)
            for _ in range(num_workers)
        ]
    
    def parallel_search(self, target_graph: str) -> List[KernelResult]:
        """
        并行探索配置空间
        
        每个 worker 探索不同的配置
        """
        futures = [
            worker.search_config.remote(target_graph)
            for worker in self.workers
        ]
        
        all_results = ray.get(futures)
        
        # 合并结果
        results = []
        for worker_results in all_results:
            results.extend(worker_results)
        
        return sorted(results, key=lambda r: r.latency_ms)
```

---

## 总结

### 关键改进

1. **层级化设计**: Level 1 (配置) → Level 2 (图)
2. **自上而下约束**: 配置决定图搜索空间
3. **两级奖励**: 
   - Level 1: 基于整体搜索结果
   - Level 2: 基于单次验证结果
4. **分布式**: 并行探索不同配置

### 与现有设计的区别

| 方面 | 原设计 | 新设计 |
|------|--------|--------|
| 搜索结构 | 扁平化 | 层级化 |
| 配置选择 | 与图混合 | 独立 Level 1 |
| 约束传递 | 无 | 显式约束 |
| RL 策略 | 1 个 | 2 个 (可选联合) |
| 动作空间 | 固定 | 动态 (受约束) |
