# YiRage 分布式搜索架构设计

## 设计原则

1. **C++ 层优先** - 搜索效率提升从底层开始
2. **CPU 分布式优先** - Ray 主要用于分布式 CPU 搜索协调
3. **数据驱动** - 底层暴露完整的搜索反馈数据
4. **RL 算法后置** - 基于底层数据反馈构建高级搜索

---

## 分层架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│ Layer 4: RL 策略层 (Python)           [Phase 4 - 后续]                   │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ • 搜索策略学习 (PPO/SAC)                                            │ │
│ │ • 基于 Layer 3 的 SearchTrajectory 数据训练                         │ │
│ │ • 输出: 配置选择概率分布                                            │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 3: 搜索反馈收集层 (C++ with Python bindings)  [Phase 2]           │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ • SearchTrajectoryCollector - 收集搜索轨迹                          │ │
│ │ • SearchMetrics - 暴露搜索统计数据                                  │ │
│ │ • CandidateEvaluationLog - 候选评估日志                             │ │
│ │ • 输出: JSON/Protobuf 格式的训练数据                                │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Ray 分布式协调层 (Python)    [Phase 1b]                        │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ • 配置空间分区 (Configuration Space Partitioning)                   │ │
│ │ • 多进程/多节点 Worker 调度                                         │ │
│ │ • 结果聚合与最优选择                                                │ │
│ │ • CPU-only, 不依赖 GPU                                              │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│ Layer 1: C++ 搜索核心优化层          [Phase 1a - 最优先]                │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ • DistributedSearchConfig - 分布式搜索配置                          │ │
│ │ • SearchPartition - 搜索空间分区                                    │ │
│ │ • PartitionedGenerator - 分区化的图生成器                           │ │
│ │ • SearchFeedback - 搜索过程回调钩子                                 │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1a: C++ 搜索核心优化

### 1.1 新增数据结构

```cpp
// include/search/distributed/search_partition.h

namespace yirage {
namespace search {

/**
 * 搜索空间分区 - 用于分布式搜索
 */
struct SearchPartition {
  int partition_id;
  int total_partitions;
  
  // 配置空间范围
  std::vector<dim3> grid_dim_range;
  std::vector<dim3> block_dim_range;
  std::vector<int3> imap_range;
  std::vector<int3> omap_range;
  std::vector<int> frange_range;
  
  // 分区元数据
  size_t estimated_candidates;
  
  // 从完整配置空间创建分区
  static std::vector<SearchPartition> 
  create_partitions(GeneratorConfig const& config, int num_partitions);
};

/**
 * 搜索反馈数据 - 用于收集训练数据
 */
struct SearchFeedback {
  // 候选配置
  struct CandidateInfo {
    int candidate_id;
    dim3 grid_dim;
    dim3 block_dim;
    std::vector<int3> imaps;
    int3 omap;
    int frange;
    
    // 评估结果
    bool verified;
    double fingerprint_time_ms;
    double estimated_performance;
    
    // 搜索上下文
    int search_depth;
    int operator_count;
  };
  
  std::vector<CandidateInfo> candidates;
  
  // 搜索统计
  int total_states_explored;
  int valid_graphs_found;
  double search_time_seconds;
  
  // 转换为 JSON
  json to_json() const;
};

/**
 * 搜索回调接口 - 用于外部监控和数据收集
 */
class SearchCallback {
public:
  virtual ~SearchCallback() = default;
  
  // 状态探索回调
  virtual void on_state_explored(SearchContext const& ctx, int depth) {}
  
  // 候选生成回调
  virtual void on_candidate_generated(
    dim3 grid_dim, dim3 block_dim, 
    std::vector<int3> const& imaps, int3 omap, int frange) {}
  
  // 验证结果回调
  virtual void on_verification_result(
    kernel::Graph const& graph, bool verified, 
    double fingerprint_time_ms) {}
  
  // 有效图发现回调
  virtual void on_valid_graph_found(
    kernel::Graph const& graph, 
    double estimated_performance) {}
  
  // 搜索完成回调
  virtual void on_search_completed(SearchFeedback const& feedback) {}
};

} // namespace search
} // namespace yirage
```

### 1.2 分区化搜索生成器

```cpp
// include/search/distributed/partitioned_generator.h

namespace yirage {
namespace search {

/**
 * 分区化内核图生成器
 * 支持分布式搜索，每个 Worker 处理一个分区
 */
class PartitionedKernelGraphGenerator {
public:
  PartitionedKernelGraphGenerator(
    kernel::Graph const& computation_graph,
    GeneratorConfig const& config,
    SearchPartition const& partition,
    SearchCallback* callback = nullptr);
  
  // 只搜索指定分区
  void generate_kernel_graphs_for_partition();
  
  // 获取搜索反馈
  SearchFeedback get_feedback() const;
  
  // 获取生成的图
  std::vector<json> const& get_generated_graphs() const;
  
private:
  SearchPartition partition_;
  SearchCallback* callback_;
  SearchFeedback feedback_;
  
  // 检查配置是否在分区范围内
  bool is_in_partition(dim3 grid_dim, dim3 block_dim) const;
};

} // namespace search
} // namespace yirage
```

### 1.3 C 接口扩展

```cpp
// include/search/search_c.h (扩展)

namespace yirage {
namespace search_c {

// 现有接口保持不变...

// ========== 新增分布式搜索接口 ==========

/**
 * 创建搜索分区
 * @param num_partitions 分区数量
 * @param config 搜索配置 (JSON 字符串)
 * @return 分区数组 (JSON 字符串)
 */
char* create_search_partitions(
  int num_partitions,
  char const* config_json);

/**
 * 执行分区搜索
 * @param input_graph 输入计算图
 * @param partition_json 分区配置 (JSON)
 * @param config_json 搜索配置 (JSON)
 * @param collect_feedback 是否收集反馈数据
 * @param feedback_json 输出: 反馈数据 (JSON)
 * @return 生成的图数量
 */
int search_partition(
  kernel::Graph const* input_graph,
  char const* partition_json,
  char const* config_json,
  bool collect_feedback,
  char** feedback_json,  // 输出参数
  int max_num_graphs,
  kernel::Graph** new_graphs);

/**
 * 获取搜索进度
 * 用于长时间运行的搜索任务
 */
struct SearchProgress {
  int states_explored;
  int valid_graphs;
  double elapsed_seconds;
  double estimated_remaining_seconds;
};

SearchProgress get_search_progress(int search_handle);

/**
 * 异步搜索接口
 * 返回句柄，可以查询进度或取消
 */
int start_async_search(
  kernel::Graph const* input_graph,
  char const* config_json);

bool cancel_search(int search_handle);

} // namespace search_c
} // namespace yirage
```

---

## Phase 1b: Ray 分布式协调层 (Python)

### 2.1 轻量级 Ray Worker

```python
# python/yirage/distributed/worker.py

import ray
from typing import List, Dict, Any, Optional
import json

@ray.remote
class SearchWorker:
    """
    轻量级搜索 Worker
    主要在 CPU 上运行，调用 C++ 搜索核心
    """
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        # 导入 C++ 绑定
        from yirage.core import (
            search_partition,
            create_search_partitions,
        )
        self._search_partition = search_partition
    
    def search(
        self,
        graph_json: str,
        partition_json: str,
        config_json: str,
        collect_feedback: bool = True,
    ) -> Dict[str, Any]:
        """
        执行分区搜索
        
        Args:
            graph_json: 输入图 (JSON 序列化)
            partition_json: 分区配置
            config_json: 搜索配置
            collect_feedback: 是否收集反馈
            
        Returns:
            {
                "graphs": [...],  # 发现的有效图
                "feedback": {...},  # 搜索反馈数据
            }
        """
        result = self._search_partition(
            graph_json,
            partition_json,
            config_json,
            collect_feedback,
        )
        
        return {
            "worker_id": self.worker_id,
            "graphs": result["graphs"],
            "feedback": result["feedback"] if collect_feedback else None,
        }
```

### 2.2 分布式协调器

```python
# python/yirage/distributed/coordinator.py

import ray
from typing import List, Dict, Any, Optional
import json
import time

class DistributedSearchCoordinator:
    """
    分布式搜索协调器
    使用 Ray 协调多个 CPU Worker 进行并行搜索
    """
    
    def __init__(
        self,
        num_workers: int = None,
        ray_address: Optional[str] = None,
    ):
        """
        Args:
            num_workers: Worker 数量，默认为 CPU 核心数
            ray_address: Ray 集群地址，None 则使用本地
        """
        import os
        self.num_workers = num_workers or os.cpu_count()
        
        # 初始化 Ray
        if not ray.is_initialized():
            if ray_address:
                ray.init(address=ray_address)
            else:
                ray.init()
        
        # 创建 Workers (CPU-only)
        self.workers = [
            SearchWorker.remote(i) 
            for i in range(self.num_workers)
        ]
        
        # 搜索状态
        self.all_feedback: List[Dict] = []
        self.all_graphs: List[Dict] = []
    
    def parallel_search(
        self,
        computation_graph: Any,
        config: Optional[Dict] = None,
        collect_feedback: bool = True,
    ) -> Dict[str, Any]:
        """
        执行并行分布式搜索
        
        Args:
            computation_graph: 输入计算图
            config: 搜索配置
            collect_feedback: 是否收集反馈数据
            
        Returns:
            {
                "graphs": [...],
                "best_graph": ...,
                "feedback": [...],  # 所有 Worker 的反馈
                "statistics": {...},
            }
        """
        from yirage.core import create_search_partitions
        
        start_time = time.time()
        
        # 序列化输入图
        graph_json = self._serialize_graph(computation_graph)
        config_json = json.dumps(config or {})
        
        # 创建分区
        partitions_json = create_search_partitions(
            self.num_workers,
            config_json,
        )
        partitions = json.loads(partitions_json)
        
        print(f"Created {len(partitions)} search partitions")
        print(f"Dispatching to {self.num_workers} workers...")
        
        # 分发任务到 Workers
        futures = [
            worker.search.remote(
                graph_json,
                json.dumps(partition),
                config_json,
                collect_feedback,
            )
            for worker, partition in zip(self.workers, partitions)
        ]
        
        # 收集结果
        results = ray.get(futures)
        
        # 聚合结果
        all_graphs = []
        all_feedback = []
        
        for result in results:
            all_graphs.extend(result["graphs"])
            if result["feedback"]:
                all_feedback.append(result["feedback"])
        
        elapsed = time.time() - start_time
        
        # 统计
        total_states = sum(
            fb.get("total_states_explored", 0) 
            for fb in all_feedback
        )
        
        print(f"Search completed in {elapsed:.2f}s")
        print(f"Total states explored: {total_states}")
        print(f"Valid graphs found: {len(all_graphs)}")
        
        # 保存反馈数据供 RL 训练使用
        self.all_feedback = all_feedback
        self.all_graphs = all_graphs
        
        return {
            "graphs": all_graphs,
            "best_graph": self._select_best(all_graphs),
            "feedback": all_feedback,
            "statistics": {
                "total_workers": self.num_workers,
                "total_states": total_states,
                "valid_graphs": len(all_graphs),
                "elapsed_seconds": elapsed,
            },
        }
    
    def get_training_data(self) -> Dict[str, Any]:
        """
        获取 RL 训练数据
        从收集的反馈中提取训练样本
        """
        training_samples = []
        
        for feedback in self.all_feedback:
            for candidate in feedback.get("candidates", []):
                sample = {
                    # 状态特征
                    "state": {
                        "grid_dim": candidate["grid_dim"],
                        "block_dim": candidate["block_dim"],
                        "search_depth": candidate["search_depth"],
                        "operator_count": candidate["operator_count"],
                    },
                    # 动作
                    "action": {
                        "imaps": candidate["imaps"],
                        "omap": candidate["omap"],
                        "frange": candidate["frange"],
                    },
                    # 奖励信号
                    "reward": {
                        "verified": candidate["verified"],
                        "performance": candidate.get("estimated_performance", 0),
                    },
                }
                training_samples.append(sample)
        
        return {
            "samples": training_samples,
            "num_samples": len(training_samples),
        }
    
    def save_training_data(self, filepath: str):
        """保存训练数据到文件"""
        data = self.get_training_data()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {data['num_samples']} training samples to {filepath}")
    
    def _serialize_graph(self, graph: Any) -> str:
        """序列化计算图"""
        # TODO: 实现图序列化
        return json.dumps({})
    
    def _select_best(self, graphs: List[Dict]) -> Optional[Dict]:
        """选择最佳图"""
        if not graphs:
            return None
        # TODO: 基于性能估计选择
        return graphs[0]
    
    def shutdown(self):
        """关闭 Workers"""
        for worker in self.workers:
            ray.kill(worker)
        self.workers = []
```

---

## Phase 2: 搜索反馈数据收集

### 3.1 C++ 反馈收集器实现

```cpp
// src/search/distributed/feedback_collector.cc

namespace yirage {
namespace search {

/**
 * 默认反馈收集器实现
 */
class FeedbackCollector : public SearchCallback {
public:
  FeedbackCollector() : feedback_() {}
  
  void on_state_explored(SearchContext const& ctx, int depth) override {
    feedback_.total_states_explored++;
  }
  
  void on_candidate_generated(
      dim3 grid_dim, dim3 block_dim,
      std::vector<int3> const& imaps, int3 omap, int frange) override {
    
    SearchFeedback::CandidateInfo info;
    info.candidate_id = feedback_.candidates.size();
    info.grid_dim = grid_dim;
    info.block_dim = block_dim;
    info.imaps = imaps;
    info.omap = omap;
    info.frange = frange;
    info.verified = false;
    
    current_candidate_ = info;
  }
  
  void on_verification_result(
      kernel::Graph const& graph, bool verified,
      double fingerprint_time_ms) override {
    
    current_candidate_.verified = verified;
    current_candidate_.fingerprint_time_ms = fingerprint_time_ms;
    
    feedback_.candidates.push_back(current_candidate_);
  }
  
  void on_valid_graph_found(
      kernel::Graph const& graph,
      double estimated_performance) override {
    
    feedback_.valid_graphs_found++;
    
    // 更新最后一个候选的性能估计
    if (!feedback_.candidates.empty()) {
      feedback_.candidates.back().estimated_performance = estimated_performance;
    }
  }
  
  void on_search_completed(SearchFeedback const& /* unused */) override {
    // 计算搜索时间由外部设置
  }
  
  SearchFeedback const& get_feedback() const { return feedback_; }
  
private:
  SearchFeedback feedback_;
  SearchFeedback::CandidateInfo current_candidate_;
};

} // namespace search
} // namespace yirage
```

### 3.2 反馈数据 Python 绑定

```python
# python/yirage/distributed/feedback.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json

@dataclass
class CandidateInfo:
    """候选配置信息"""
    candidate_id: int
    grid_dim: tuple
    block_dim: tuple
    imaps: List[tuple]
    omap: tuple
    frange: int
    
    verified: bool = False
    fingerprint_time_ms: float = 0.0
    estimated_performance: float = 0.0
    search_depth: int = 0
    operator_count: int = 0

@dataclass
class SearchFeedback:
    """搜索反馈数据"""
    partition_id: int = 0
    candidates: List[CandidateInfo] = field(default_factory=list)
    
    total_states_explored: int = 0
    valid_graphs_found: int = 0
    search_time_seconds: float = 0.0
    
    @classmethod
    def from_json(cls, json_str: str) -> "SearchFeedback":
        """从 JSON 解析"""
        data = json.loads(json_str)
        
        candidates = [
            CandidateInfo(**c) for c in data.get("candidates", [])
        ]
        
        return cls(
            partition_id=data.get("partition_id", 0),
            candidates=candidates,
            total_states_explored=data.get("total_states_explored", 0),
            valid_graphs_found=data.get("valid_graphs_found", 0),
            search_time_seconds=data.get("search_time_seconds", 0.0),
        )
    
    def to_training_samples(self) -> List[Dict[str, Any]]:
        """转换为 RL 训练样本"""
        samples = []
        
        for i, candidate in enumerate(self.candidates):
            # 状态: 当前搜索上下文
            state = {
                "search_depth": candidate.search_depth,
                "operator_count": candidate.operator_count,
                "grid_dim": list(candidate.grid_dim),
                "block_dim": list(candidate.block_dim),
            }
            
            # 动作: 配置选择
            action = {
                "imaps": [list(m) for m in candidate.imaps],
                "omap": list(candidate.omap),
                "frange": candidate.frange,
            }
            
            # 奖励: 基于验证结果和性能
            reward = self._compute_reward(candidate)
            
            # 下一状态
            next_state = None
            if i + 1 < len(self.candidates):
                next_candidate = self.candidates[i + 1]
                next_state = {
                    "search_depth": next_candidate.search_depth,
                    "operator_count": next_candidate.operator_count,
                }
            
            samples.append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": next_state is None,
            })
        
        return samples
    
    def _compute_reward(self, candidate: CandidateInfo) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 验证通过奖励
        if candidate.verified:
            reward += 1.0
            
            # 性能奖励 (归一化)
            if candidate.estimated_performance > 0:
                reward += 1.0 / candidate.estimated_performance
        else:
            # 验证失败惩罚
            reward -= 0.5
        
        # 效率奖励 (搜索深度越浅越好)
        reward -= 0.01 * candidate.search_depth
        
        return reward
```

---

## Phase 3: 与 kernel.py 集成

### 4.1 更新 superoptimize 接口

```python
# 在 python/yirage/kernel.py 中添加

def superoptimize(
    self,
    # ... 现有参数 ...
    
    # 新增分布式搜索参数
    distributed: bool = False,
    num_workers: int = None,
    ray_address: str = None,
    collect_feedback: bool = False,
    feedback_output: str = None,
):
    """
    优化内核图
    
    Args:
        distributed: 是否使用分布式搜索
        num_workers: Worker 数量 (默认 = CPU 核心数)
        ray_address: Ray 集群地址
        collect_feedback: 是否收集 RL 训练数据
        feedback_output: 反馈数据输出路径
    """
    if distributed:
        from yirage.distributed import DistributedSearchCoordinator
        
        coordinator = DistributedSearchCoordinator(
            num_workers=num_workers,
            ray_address=ray_address,
        )
        
        result = coordinator.parallel_search(
            computation_graph=self.cygraph,
            config={
                "griddims": griddims,
                "blockdims": blockdims,
                "imaps": imaps,
                "omaps": omaps,
                "fmaps": fmaps,
                "franges": franges,
            },
            collect_feedback=collect_feedback,
        )
        
        # 保存反馈数据
        if collect_feedback and feedback_output:
            coordinator.save_training_data(feedback_output)
        
        # 返回最佳图
        best_graph_json = result["best_graph"]
        if best_graph_json:
            return self._from_json(best_graph_json)
        return None
    
    else:
        # 使用现有的本地搜索逻辑
        # ...
```

---

## 实施计划

### Phase 1a: C++ 核心 (Week 1-2)

| 任务 | 文件 | 描述 |
|------|------|------|
| 1.1 | `include/search/distributed/search_partition.h` | 分区数据结构 |
| 1.2 | `include/search/distributed/search_feedback.h` | 反馈数据结构 |
| 1.3 | `include/search/distributed/search_callback.h` | 回调接口 |
| 1.4 | `src/search/distributed/partitioned_generator.cc` | 分区搜索实现 |
| 1.5 | `src/search/search_c.cc` | 扩展 C 接口 |

### Phase 1b: Ray 协调 (Week 2-3)

| 任务 | 文件 | 描述 |
|------|------|------|
| 2.1 | `python/yirage/distributed/__init__.py` | 模块入口 |
| 2.2 | `python/yirage/distributed/worker.py` | Ray Worker |
| 2.3 | `python/yirage/distributed/coordinator.py` | 协调器 |
| 2.4 | Cython bindings | 更新 Python 绑定 |

### Phase 2: 数据反馈 (Week 3-4)

| 任务 | 文件 | 描述 |
|------|------|------|
| 3.1 | `src/search/distributed/feedback_collector.cc` | 反馈收集器 |
| 3.2 | `python/yirage/distributed/feedback.py` | Python 反馈接口 |
| 3.3 | 更新 `kernel.py` | 集成分布式搜索 |

### Phase 3: RL 集成 (Week 5+) - 后续

基于 Phase 1-2 收集的数据构建

---

## 使用示例

```python
import yirage as yr

# 创建计算图
graph = yr.new_kernel_graph()
X = graph.new_input(dims=(8, 4096), dtype=yr.float16)
W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
O = graph.matmul(X, W)
graph.mark_output(O)

# === 方法 1: 本地分布式搜索 (多 CPU 核心) ===
optimized = graph.superoptimize(
    backend="cuda",
    distributed=True,
    num_workers=8,  # 使用 8 个 CPU 核心
    collect_feedback=True,
    feedback_output="search_feedback.json",  # 保存训练数据
)

# === 方法 2: 集群分布式搜索 ===
optimized = graph.superoptimize(
    backend="cuda",
    distributed=True,
    ray_address="ray://cluster-head:10001",
    num_workers=64,  # 跨节点 64 个 Worker
    collect_feedback=True,
)

# === 后续: 使用收集的数据训练 RL 策略 ===
# from yirage.rl import train_from_feedback
# train_from_feedback("search_feedback.json", output="rl_policy.pt")
```

---

## 依赖

```txt
# requirements-distributed.txt
ray>=2.9.0  # 核心分布式框架 (CPU-only 即可)
# 注意: 不需要 ray[rllib]，那是 RL 阶段的依赖
```

---

*Document Version: 2.0*  
*重新设计: 优先 C++ 层优化 + CPU 分布式搜索*
