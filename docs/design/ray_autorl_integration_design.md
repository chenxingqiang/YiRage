# YiRage + Ray AutoRL Integration Design

## Executive Summary

This document outlines the architectural design for integrating YiRage's kernel superoptimization search with Ray's AutoRL framework to enable:

1. **Distributed Search** - Scale search across multiple nodes/GPUs
2. **RL-Guided Search** - Use reinforcement learning to learn optimal search strategies
3. **Hyperparameter Optimization** - Auto-tune search configurations
4. **Transfer Learning** - Reuse learned policies across similar workloads

---

## 1. Current YiRage Search Architecture Analysis

### 1.1 Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    YiRage Search Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────────┐   │
│  │ Computation  │───▶│ KernelGraph   │───▶│ CandidateConfig │   │
│  │    Graph     │    │  Generator    │    │   Generation    │   │
│  └──────────────┘    └───────────────┘    └─────────────────┘   │
│                              │                     │             │
│                              ▼                     ▼             │
│                      ┌───────────────┐    ┌─────────────────┐   │
│                      │  DimStrategy  │    │   Verifier      │   │
│                      │  (Candidates) │    │ (Fingerprint)   │   │
│                      └───────────────┘    └─────────────────┘   │
│                              │                     │             │
│                              ▼                     ▼             │
│                      ┌───────────────┐    ┌─────────────────┐   │
│                      │   Profiler    │───▶│  Best Graph     │   │
│                      │   (Backend)   │    │   Selection     │   │
│                      └───────────────┘    └─────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Search State Space

The current search operates over:

| Dimension | Description | Cardinality |
|-----------|-------------|-------------|
| `grid_dim` | GPU grid dimensions (x,y,z) | ~100-1000 |
| `block_dim` | Block dimensions (x,y,z) | ~50-200 |
| `imap` | Input tensor mapping | ~10-50 |
| `omap` | Output tensor mapping | ~10-50 |
| `fmap` | Forloop dimension mapping | ~5-20 |
| `frange` | Forloop range | ~10-20 |
| `operator_sequence` | Operator ordering | Combinatorial |

**Total Search Space**: ~10^8 to 10^15 configurations

### 1.3 Current Search Strategies

```cpp
enum class Strategy {
  GREEDY,       // Local best at each step
  BEAM,         // Beam search with width k
  GENETIC,      // Genetic algorithm
  REINFORCEMENT // RL-based (placeholder)
};
```

---

## 2. Ray AutoRL Integration Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Ray AutoRL + YiRage Integration                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         Ray Cluster                                 │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │ │
│  │  │   Head Node │  │ Worker Node │  │ Worker Node │  ...            │ │
│  │  │  (Driver)   │  │   (Actor)   │  │   (Actor)   │                 │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                     YiRage RL Search Layer                          ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              ││
│  │  │ RL Policy    │  │ Environment  │  │ Experience   │              ││
│  │  │ (PPO/SAC)    │  │ (SearchEnv)  │  │ Replay       │              ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘              ││
│  └─────────────────────────────────────────────────────────────────────┘│
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                     YiRage Core Search                              ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              ││
│  │  │ KernelGraph  │  │  DimStrategy │  │  Verifier    │              ││
│  │  │  Generator   │  │              │  │              │              ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘              ││
│  └─────────────────────────────────────────────────────────────────────┘│
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Design

#### 2.2.1 YiRageSearchEnv (Gymnasium Environment)

```python
class YiRageSearchEnv(gymnasium.Env):
    """
    RL Environment wrapping YiRage's kernel search.
    
    State: Current search context (graph structure, explored nodes, metrics)
    Action: Next configuration choice (operator, dimensions, mappings)
    Reward: Performance improvement + verification success
    """
    
    observation_space = spaces.Dict({
        "graph_embedding": spaces.Box(...),      # Graph neural network embedding
        "search_depth": spaces.Discrete(100),    # Current search depth
        "explored_fraction": spaces.Box(0, 1),   # Fraction of space explored
        "best_perf_so_far": spaces.Box(...),     # Best performance found
        "hardware_features": spaces.Box(...),    # Target hardware features
    })
    
    action_space = spaces.Dict({
        "operator_type": spaces.Discrete(N_OPS),
        "grid_dim_x": spaces.Discrete(MAX_GRID),
        "grid_dim_y": spaces.Discrete(MAX_GRID),
        "block_dim": spaces.Discrete(MAX_BLOCK),
        "imap_choice": spaces.Discrete(N_IMAPS),
        "frange_choice": spaces.Discrete(N_FRANGES),
    })
```

#### 2.2.2 Ray RLlib Integration

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Register YiRage environment
register_env("yirage_search", lambda config: YiRageSearchEnv(config))

# Configure PPO for kernel search
config = (
    PPOConfig()
    .environment("yirage_search")
    .framework("torch")
    .training(
        gamma=0.99,
        lr=3e-4,
        train_batch_size=4096,
        model={
            "custom_model": "YiRageGraphTransformer",
            "custom_model_config": {
                "graph_encoder": "GAT",
                "hidden_dim": 256,
                "num_heads": 8,
            }
        }
    )
    .resources(num_gpus=1)
    .rollouts(num_rollout_workers=8)
)
```

---

## 3. Detailed Component Specifications

### 3.1 State Representation

#### 3.1.1 Graph Embedding

```python
class KernelGraphEncoder(nn.Module):
    """
    Encode kernel graph structure for RL policy.
    Uses Graph Attention Network (GAT) for permutation invariance.
    """
    
    def __init__(self, hidden_dim=256, num_layers=4):
        self.node_encoder = NodeEncoder(hidden_dim)
        self.edge_encoder = EdgeEncoder(hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=8)
            for _ in range(num_layers)
        ])
        self.pooling = GlobalAttentionPooling(hidden_dim)
    
    def forward(self, graph):
        # Encode nodes (operators)
        node_features = self.node_encoder(graph.node_types, graph.node_dims)
        
        # Encode edges (data flow)
        edge_features = self.edge_encoder(graph.edge_types)
        
        # Graph attention layers
        for gat in self.gat_layers:
            node_features = gat(node_features, graph.edge_index, edge_features)
        
        # Global pooling for graph-level representation
        return self.pooling(node_features, graph.batch)
```

#### 3.1.2 Hardware Feature Encoding

```python
HARDWARE_FEATURES = {
    "cuda": {
        "warp_size": 32,
        "sm_count": 108,  # H100
        "shared_mem_per_sm": 228 * 1024,
        "tensor_core_available": True,
    },
    "maca": {
        "warp_size": 64,
        "sm_count": 64,  # C500
        "shared_mem_per_sm": 128 * 1024,
        "tensor_core_available": True,
    },
    "ascend": {
        "ai_core_count": 32,
        "cube_unit_size": 16,
        "l1_buffer_size": 1024 * 1024,
        "vector_unit_available": True,
    },
}
```

### 3.2 Action Space Design

#### 3.2.1 Hierarchical Action Space

```python
class HierarchicalActionSpace:
    """
    Two-level action space matching YiRage's search hierarchy.
    
    Level 1 (Kernel): Choose operator type, input tensors
    Level 2 (Threadblock): Choose grid/block dims, mappings
    """
    
    # Level 1: Kernel-level actions
    kernel_actions = spaces.Dict({
        "action_type": spaces.Discrete(3),  # [ADD_OP, CREATE_TB, FINISH]
        "operator": spaces.Discrete(len(KN_OPERATORS)),
        "input_indices": spaces.MultiDiscrete([MAX_TENSORS] * MAX_INPUTS),
    })
    
    # Level 2: Threadblock-level actions
    tb_actions = spaces.Dict({
        "action_type": spaces.Discrete(3),  # [ADD_TB_OP, CREATE_OUTPUT, RETURN]
        "operator": spaces.Discrete(len(TB_OPERATORS)),
        "grid_dim": spaces.MultiDiscrete([MAX_GRID] * 3),
        "block_dim": spaces.MultiDiscrete([MAX_BLOCK] * 3),
        "imap": spaces.MultiDiscrete([N_IMAP_CHOICES] * MAX_INPUTS),
        "omap": spaces.Discrete(N_OMAP_CHOICES),
        "frange": spaces.Discrete(N_FRANGE_CHOICES),
    })
```

### 3.3 Reward Design

```python
class YiRageRewardFunction:
    """
    Multi-objective reward combining performance, validity, and efficiency.
    """
    
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        self.alpha = alpha  # Performance weight
        self.beta = beta    # Validity weight
        self.gamma = gamma  # Efficiency weight
    
    def compute_reward(self, state, action, next_state, info):
        # Performance reward (normalized speedup)
        perf_reward = 0.0
        if info.get("profiled"):
            baseline_perf = info["baseline_latency_ms"]
            actual_perf = info["kernel_latency_ms"]
            speedup = baseline_perf / max(actual_perf, 1e-6)
            perf_reward = np.log(speedup + 1)  # Log scale for stability
        
        # Validity reward
        validity_reward = 1.0 if info.get("verified") else -0.5
        
        # Efficiency reward (penalize long search paths)
        search_depth = next_state["search_depth"]
        efficiency_reward = -0.01 * search_depth
        
        # Early termination bonus
        if info.get("found_optimal"):
            efficiency_reward += 2.0
        
        total_reward = (
            self.alpha * perf_reward +
            self.beta * validity_reward +
            self.gamma * efficiency_reward
        )
        
        return total_reward
```

---

## 4. Distributed Search with Ray

### 4.1 Search Workers

```python
@ray.remote(num_gpus=1)
class YiRageSearchWorker:
    """
    Distributed search worker for parallel kernel exploration.
    """
    
    def __init__(self, worker_id: int, backend: str = "cuda"):
        self.worker_id = worker_id
        self.backend = backend
        self.search_engine = KernelGraphGenerator(backend=backend)
        self.profiler = KernelProfiler(backend=backend)
    
    def explore_subspace(
        self,
        computation_graph: dict,
        config_partition: List[dict],
        policy_weights: Optional[dict] = None,
    ) -> List[SearchResult]:
        """
        Explore a partition of the search space.
        
        Args:
            computation_graph: Target computation graph
            config_partition: Subset of configurations to explore
            policy_weights: Optional RL policy for guided search
        
        Returns:
            List of valid configurations with performance metrics
        """
        results = []
        
        for config in config_partition:
            # Generate candidate graph
            candidate = self.search_engine.generate_with_config(
                computation_graph, config
            )
            
            # Verify correctness
            if not self.search_engine.verify(candidate):
                continue
            
            # Profile performance
            latency = self.profiler.profile(candidate)
            
            results.append(SearchResult(
                config=config,
                graph=candidate,
                latency=latency,
                worker_id=self.worker_id,
            ))
        
        return results
```

### 4.2 Coordinator

```python
class RaySearchCoordinator:
    """
    Coordinates distributed search across Ray workers.
    """
    
    def __init__(self, num_workers: int = 8, backend: str = "cuda"):
        ray.init()
        
        self.workers = [
            YiRageSearchWorker.remote(i, backend)
            for i in range(num_workers)
        ]
        
        self.result_aggregator = ResultAggregator()
        self.policy_trainer = RLPolicyTrainer()
    
    def parallel_search(
        self,
        computation_graph: KNGraph,
        search_config: GeneratorConfig,
        use_rl_policy: bool = True,
    ) -> OptimizedKernel:
        """
        Execute distributed search with optional RL guidance.
        """
        # Generate configuration space
        config_space = self.generate_config_space(search_config)
        
        # Partition space across workers
        partitions = np.array_split(config_space, len(self.workers))
        
        # Get current policy weights (if using RL)
        policy_weights = None
        if use_rl_policy:
            policy_weights = self.policy_trainer.get_weights()
        
        # Launch parallel exploration
        futures = [
            worker.explore_subspace.remote(
                computation_graph.to_dict(),
                partition.tolist(),
                policy_weights,
            )
            for worker, partition in zip(self.workers, partitions)
        ]
        
        # Collect results
        all_results = ray.get(futures)
        
        # Aggregate and select best
        best_result = self.result_aggregator.select_best(all_results)
        
        # Update RL policy with experience
        if use_rl_policy:
            self.policy_trainer.update(all_results)
        
        return best_result.to_optimized_kernel()
```

---

## 5. RL Training Pipeline

### 5.1 Training Loop

```python
class YiRageRLTrainer:
    """
    RL training loop for kernel search policy.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize Ray RLlib trainer
        self.trainer = PPO(
            env="yirage_search",
            config={
                "framework": "torch",
                "num_workers": config.num_workers,
                "train_batch_size": config.batch_size,
                "sgd_minibatch_size": config.minibatch_size,
                "num_sgd_iter": config.num_epochs,
                "lr": config.learning_rate,
                "gamma": config.gamma,
                "lambda": config.gae_lambda,
                "clip_param": config.clip_param,
                "model": {
                    "custom_model": "YiRageGraphTransformer",
                },
            }
        )
        
        # Curriculum learning scheduler
        self.curriculum = CurriculumScheduler(config.curriculum_stages)
    
    def train(self, num_iterations: int = 1000):
        """
        Train the search policy.
        """
        for i in range(num_iterations):
            # Update curriculum (increase difficulty over time)
            env_config = self.curriculum.get_current_config(i)
            self.trainer.workers.foreach_worker(
                lambda w: w.foreach_env(lambda e: e.set_config(env_config))
            )
            
            # Train step
            result = self.trainer.train()
            
            # Log metrics
            self.log_metrics(i, result)
            
            # Checkpoint
            if i % self.config.checkpoint_interval == 0:
                self.trainer.save(f"checkpoints/yirage_rl_{i}")
    
    def log_metrics(self, iteration: int, result: dict):
        print(f"Iteration {iteration}:")
        print(f"  Episode reward mean: {result['episode_reward_mean']:.2f}")
        print(f"  Episode length mean: {result['episode_len_mean']:.1f}")
        print(f"  Valid kernels found: {result['custom_metrics']['valid_kernels']}")
        print(f"  Best speedup: {result['custom_metrics']['best_speedup']:.2f}x")
```

### 5.2 Curriculum Learning

```python
class CurriculumScheduler:
    """
    Gradually increase search difficulty during training.
    """
    
    STAGES = [
        # Stage 1: Simple single-operator graphs
        {"max_operators": 2, "search_space_fraction": 0.1},
        
        # Stage 2: Multi-operator without fusion
        {"max_operators": 4, "search_space_fraction": 0.3},
        
        # Stage 3: Operator fusion enabled
        {"max_operators": 6, "search_space_fraction": 0.5, "enable_fusion": True},
        
        # Stage 4: Full search space
        {"max_operators": 10, "search_space_fraction": 1.0, "enable_fusion": True},
    ]
    
    def get_current_config(self, iteration: int) -> dict:
        stage_idx = min(iteration // 250, len(self.STAGES) - 1)
        return self.STAGES[stage_idx]
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (2-3 weeks)

| Task | Description | Files to Modify/Create |
|------|-------------|------------------------|
| 1.1 | Create YiRageSearchEnv | `python/yirage/rl/search_env.py` |
| 1.2 | Implement state encoding | `python/yirage/rl/encoders.py` |
| 1.3 | Define action space | `python/yirage/rl/action_space.py` |
| 1.4 | Implement reward function | `python/yirage/rl/rewards.py` |

### Phase 2: Ray Integration (2-3 weeks)

| Task | Description | Files to Modify/Create |
|------|-------------|------------------------|
| 2.1 | Ray worker implementation | `python/yirage/rl/workers.py` |
| 2.2 | Search coordinator | `python/yirage/rl/coordinator.py` |
| 2.3 | RLlib model integration | `python/yirage/rl/models.py` |
| 2.4 | Distributed profiling | `python/yirage/rl/distributed_profiler.py` |

### Phase 3: Training Pipeline (2-3 weeks)

| Task | Description | Files to Modify/Create |
|------|-------------|------------------------|
| 3.1 | Training loop | `python/yirage/rl/trainer.py` |
| 3.2 | Curriculum learning | `python/yirage/rl/curriculum.py` |
| 3.3 | Hyperparameter tuning | `python/yirage/rl/tune_config.py` |
| 3.4 | Checkpoint management | `python/yirage/rl/checkpoints.py` |

### Phase 4: Integration & Testing (2 weeks)

| Task | Description | Files to Modify/Create |
|------|-------------|------------------------|
| 4.1 | Integrate with superoptimize() | `python/yirage/kernel.py` |
| 4.2 | Add CLI interface | `scripts/train_rl_search.py` |
| 4.3 | Benchmarking suite | `benchmark/rl_search_benchmark.py` |
| 4.4 | Documentation | `docs/rl_search_guide.md` |

---

## 7. API Design

### 7.1 Python API

```python
import yirage as yr
from yirage.rl import RLSearchConfig, train_search_policy

# Option 1: Use pre-trained policy for optimization
graph = yr.new_kernel_graph()
X = graph.new_input(dims=(8, 4096), dtype=yr.float16)
W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
O = graph.matmul(X, W)
graph.mark_output(O)

# Use RL-guided search
optimized = graph.superoptimize(
    backend="cuda",
    search_strategy="rl",  # NEW: Use RL policy
    rl_config=RLSearchConfig(
        policy_checkpoint="pretrained/yirage_rl_v1",
        max_search_time=60,
        num_workers=4,
    )
)

# Option 2: Train custom policy for specific workload
train_search_policy(
    workload_graphs=[graph1, graph2, graph3],
    backend="cuda",
    output_dir="custom_policy",
    num_iterations=1000,
    num_workers=8,
)
```

### 7.2 Configuration Schema

```python
@dataclass
class RLSearchConfig:
    """Configuration for RL-guided kernel search."""
    
    # Policy configuration
    policy_checkpoint: Optional[str] = None  # Pre-trained policy path
    policy_type: str = "ppo"  # "ppo", "sac", "ddpg"
    
    # Search parameters
    max_search_time: float = 300.0  # Max time in seconds
    max_candidates: int = 10000  # Max candidates to evaluate
    exploration_fraction: float = 0.1  # Random exploration ratio
    
    # Distributed execution
    num_workers: int = 4
    gpus_per_worker: float = 1.0
    
    # Performance targets
    target_speedup: Optional[float] = None  # Early stop if achieved
    baseline_comparison: bool = True  # Compare with baseline search
```

---

## 8. Performance Considerations

### 8.1 Overhead Analysis

| Component | Overhead | Mitigation |
|-----------|----------|------------|
| Graph encoding | ~1-5ms | Cache embeddings, batch encoding |
| Policy inference | ~0.1-1ms | GPU inference, batch predictions |
| Ray communication | ~0.5-2ms | Minimize data transfer, use refs |
| Verification | ~10-100ms | Fingerprint caching, early pruning |
| Profiling | ~100-1000ms | Adaptive profiling, sampling |

### 8.2 Expected Benefits

| Metric | Baseline Search | RL-Guided Search | Improvement |
|--------|-----------------|------------------|-------------|
| Search time (avg) | 300s | 60s | 5x faster |
| Configurations explored | 100,000 | 5,000 | 20x fewer |
| Valid kernels found | 10 | 50 | 5x more |
| Best kernel quality | 1.0x | 1.2x | 20% better |

---

## 9. Future Extensions

1. **Meta-Learning**: Learn to quickly adapt policy to new workloads
2. **Multi-Objective RL**: Optimize for latency, memory, and power simultaneously
3. **Neural Architecture Search**: Jointly optimize kernel structure and parameters
4. **Online Learning**: Continuously improve policy during deployment
5. **Cross-Backend Transfer**: Transfer learned policies between CUDA/MACA/Ascend

---

## 10. Dependencies

```txt
# requirements-rl.txt
ray[rllib]>=2.9.0
gymnasium>=0.29.0
torch>=2.0.0
torch-geometric>=2.4.0
numpy>=1.24.0
wandb>=0.16.0  # Optional: experiment tracking
```

---

## Appendix A: File Structure

```
python/yirage/rl/
├── __init__.py
├── action_space.py      # Action space definitions
├── checkpoints.py       # Model checkpointing
├── coordinator.py       # Distributed search coordination
├── curriculum.py        # Curriculum learning
├── distributed_profiler.py  # Distributed kernel profiling
├── encoders.py          # State encoders (graph, hardware)
├── models.py            # Neural network models
├── rewards.py           # Reward functions
├── search_env.py        # Gymnasium environment
├── trainer.py           # RL training loop
├── tune_config.py       # Hyperparameter tuning
└── workers.py           # Ray remote workers

scripts/
├── train_rl_search.py   # CLI for training
└── eval_rl_search.py    # CLI for evaluation

benchmark/
└── rl_search_benchmark.py  # Benchmarking suite

docs/
├── rl_search_guide.md   # User guide
└── rl_search_api.md     # API reference
```

---

*Document Version: 1.0*  
*Author: Chen Xingqiang*  
*Date: December 2024*
