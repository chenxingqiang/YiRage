# RL 模型特征设计：从 µGraph 到模型输入

## 核心问题

当前设计的问题：
1. **特征来源不对** - 当前 `graph_embedding` 在 Python 层简单计算，没有从 C++ µGraph 获取真实特征
2. **模型不独立** - 完全依赖 RLlib，没有独立的模型保存/加载机制
3. **特征不完整** - 缺少关键的图结构特征

## 解决方案：从底层到模型的特征流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     特征流：从 µGraph 到 RL 模型                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ C++ Layer: µGraph Feature Extraction                               │ │
│  │                                                                     │ │
│  │  KernelGraph/ThreadblockGraph                                       │ │
│  │       │                                                             │ │
│  │       ▼                                                             │ │
│  │  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │ GraphFeatureExtractor (C++)                                 │   │ │
│  │  │   • 算子特征: 类型、数量、连接模式                           │   │ │
│  │  │   • Tensor 特征: 形状、dtype、内存布局                       │   │ │
│  │  │   • 图结构特征: 深度、宽度、关键路径                         │   │ │
│  │  │   • 配置特征: grid/block dims, imaps, omaps                  │   │ │
│  │  │   • 性能预测特征: 理论 FLOPS、内存带宽利用率                  │   │ │
│  │  └─────────────────────────────────────────────────────────────┘   │ │
│  │       │                                                             │ │
│  │       ▼ JSON/Protobuf                                               │ │
│  └───────┬────────────────────────────────────────────────────────────┘ │
│          │                                                               │
│          ▼                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Python Layer: Feature Processing                                   │ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │ FeatureProcessor (Python)                                   │   │ │
│  │  │   • 解析 C++ 传来的特征                                      │   │ │
│  │  │   • 归一化和标准化                                           │   │ │
│  │  │   • 构建图神经网络输入 (节点/边特征)                          │   │ │
│  │  │   • 缓存和批处理                                             │   │ │
│  │  └─────────────────────────────────────────────────────────────┘   │ │
│  │       │                                                             │ │
│  │       ▼                                                             │ │
│  └───────┬────────────────────────────────────────────────────────────┘ │
│          │                                                               │
│          ▼                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ RL Model Layer                                                      │ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │ PolicyNetwork (PyTorch)                                     │   │ │
│  │  │                                                              │   │ │
│  │  │   graph_features ──▶ GraphEncoder ──┐                       │   │ │
│  │  │   config_features ─▶ ConfigEncoder ─┼──▶ FusionLayer ──▶ π(a|s)│ │
│  │  │   history_features ▶ HistoryEncoder ┘                       │   │ │
│  │  │                                                              │   │ │
│  │  │   Model Save/Load:                                          │   │ │
│  │  │     • save_checkpoint(path)                                  │   │ │
│  │  │     • load_checkpoint(path)                                  │   │ │
│  │  │     • export_onnx(path)                                      │   │ │
│  │  └─────────────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 1. C++ 层：µGraph 特征提取

### 特征定义

```cpp
// include/search/graph_features.h

struct OperatorFeatures {
    int op_type;           // 算子类型 ID
    int num_inputs;        // 输入数量
    int num_outputs;       // 输出数量
    float flops;           // 理论 FLOPS
    float memory_access;   // 内存访问量
};

struct TensorFeatures {
    std::vector<int> dims;       // 形状
    int dtype;                   // 数据类型
    size_t size_bytes;           // 大小
    int memory_level;            // 内存层级 (register/shared/global)
};

struct GraphStructureFeatures {
    int num_operators;           // 算子数量
    int num_tensors;             // tensor 数量
    int graph_depth;             // 图深度
    int graph_width;             // 图宽度
    int critical_path_length;    // 关键路径长度
    float parallelism_degree;    // 并行度
};

struct ConfigFeatures {
    int grid_dim[3];
    int block_dim[3];
    int forloop_range;
    int reduction_dimx;
    float occupancy;             // SM 占用率
    float shared_mem_usage;      // Shared memory 使用率
    float register_usage;        // 寄存器使用率
};

struct PerformancePredictionFeatures {
    float theoretical_flops;     // 理论峰值
    float memory_bandwidth;      // 内存带宽利用率
    float arithmetic_intensity;  // 计算密度
    float estimated_latency;     // 预估延迟
};

// 完整的 µGraph 特征
struct MuGraphFeatures {
    std::vector<OperatorFeatures> operators;
    std::vector<TensorFeatures> tensors;
    std::vector<std::pair<int, int>> edges;  // 算子连接关系
    GraphStructureFeatures structure;
    ConfigFeatures config;
    PerformancePredictionFeatures perf_prediction;
    
    // 序列化
    std::string to_json() const;
    static MuGraphFeatures from_json(const std::string& json);
};
```

### C++ 特征提取接口

```cpp
// include/search/graph_feature_extractor.h

class GraphFeatureExtractor {
public:
    /**
     * 从 KernelGraph 提取特征
     */
    static MuGraphFeatures extract_from_kernel_graph(
        const kernel::KNGraph* kn_graph,
        const HardwareConfig& config
    );
    
    /**
     * 从 ThreadblockGraph 提取特征
     */
    static MuGraphFeatures extract_from_tb_graph(
        const threadblock::TBGraph* tb_graph,
        const HardwareConfig& config
    );
    
    /**
     * 从完整的搜索上下文提取特征
     */
    static MuGraphFeatures extract_from_context(
        const SearchContext& context,
        const HardwareConfig& config
    );
    
private:
    // 算子特征提取
    static OperatorFeatures extract_op_features(const kernel::KNOperator* op);
    
    // Tensor 特征提取
    static TensorFeatures extract_tensor_features(const kernel::DTensor* tensor);
    
    // 结构特征计算
    static GraphStructureFeatures compute_structure_features(
        const std::vector<OperatorFeatures>& ops,
        const std::vector<std::pair<int, int>>& edges
    );
    
    // 性能预测
    static PerformancePredictionFeatures predict_performance(
        const MuGraphFeatures& features,
        const HardwareConfig& config
    );
};
```

### C 接口 (用于 Python 绑定)

```cpp
extern "C" {
    // 提取特征并返回 JSON
    char* extract_graph_features(
        void* search_context,
        const char* config_json
    );
    
    // 释放返回的字符串
    void free_feature_string(char* str);
}
```

---

## 2. Python 层：特征处理

### FeatureProcessor

```python
# python/yirage/rl/features/processor.py

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import json

@dataclass
class OperatorFeature:
    op_type: int
    num_inputs: int
    num_outputs: int
    flops: float
    memory_access: float

@dataclass
class TensorFeature:
    dims: List[int]
    dtype: int
    size_bytes: int
    memory_level: int

@dataclass
class MuGraphFeature:
    """从 C++ 层接收的完整 µGraph 特征"""
    operators: List[OperatorFeature]
    tensors: List[TensorFeature]
    edges: List[Tuple[int, int]]
    
    # 结构特征
    num_operators: int
    num_tensors: int
    graph_depth: int
    graph_width: int
    critical_path_length: int
    parallelism_degree: float
    
    # 配置特征
    grid_dim: Tuple[int, int, int]
    block_dim: Tuple[int, int, int]
    forloop_range: int
    occupancy: float
    shared_mem_usage: float
    register_usage: float
    
    # 性能预测特征
    theoretical_flops: float
    memory_bandwidth: float
    arithmetic_intensity: float
    estimated_latency: float
    
    @classmethod
    def from_json(cls, json_str: str) -> "MuGraphFeature":
        """从 C++ JSON 解析特征"""
        data = json.loads(json_str)
        # ... 解析逻辑
        return cls(...)
    
    @classmethod
    def from_cpp_context(cls, context) -> "MuGraphFeature":
        """从 C++ 上下文提取特征"""
        json_str = context.extract_features()  # 调用 C++ 接口
        return cls.from_json(json_str)


class FeatureProcessor:
    """
    处理从 C++ 层获取的 µGraph 特征
    
    功能:
    1. 解析 C++ 传来的特征
    2. 归一化和标准化
    3. 构建 GNN 输入
    4. 特征缓存
    """
    
    def __init__(
        self,
        feature_dim: int = 64,
        normalize: bool = True,
        cache_size: int = 1000,
    ):
        self.feature_dim = feature_dim
        self.normalize = normalize
        
        # 归一化统计量
        self._stats = {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
        
        # 特征缓存
        from collections import OrderedDict
        self._cache = OrderedDict()
        self._cache_size = cache_size
    
    def process(self, features: MuGraphFeature) -> Dict[str, np.ndarray]:
        """
        处理 µGraph 特征为模型输入
        
        Returns:
            {
                "node_features": [num_nodes, node_dim],
                "edge_index": [2, num_edges],
                "global_features": [global_dim],
            }
        """
        # 构建节点特征 (算子 + tensor)
        node_features = self._build_node_features(features)
        
        # 构建边
        edge_index = self._build_edge_index(features)
        
        # 构建全局特征
        global_features = self._build_global_features(features)
        
        if self.normalize:
            node_features = self._normalize(node_features, "node")
            global_features = self._normalize(global_features, "global")
        
        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "global_features": global_features,
        }
    
    def _build_node_features(self, features: MuGraphFeature) -> np.ndarray:
        """构建节点特征矩阵"""
        num_nodes = features.num_operators + features.num_tensors
        node_dim = 16  # 每个节点的特征维度
        
        node_features = np.zeros((num_nodes, node_dim), dtype=np.float32)
        
        # 算子节点
        for i, op in enumerate(features.operators):
            node_features[i, 0] = 1.0  # 是算子
            node_features[i, 1] = op.op_type / 20.0  # 算子类型 (归一化)
            node_features[i, 2] = op.num_inputs / 4.0
            node_features[i, 3] = op.num_outputs / 4.0
            node_features[i, 4] = np.log(op.flops + 1) / 20.0
            node_features[i, 5] = np.log(op.memory_access + 1) / 20.0
        
        # Tensor 节点
        offset = features.num_operators
        for i, tensor in enumerate(features.tensors):
            idx = offset + i
            node_features[idx, 0] = 0.0  # 是 tensor
            node_features[idx, 6] = len(tensor.dims) / 4.0
            if tensor.dims:
                node_features[idx, 7] = np.log(np.prod(tensor.dims) + 1) / 30.0
                node_features[idx, 8] = np.log(max(tensor.dims) + 1) / 15.0
            node_features[idx, 9] = tensor.dtype / 10.0
            node_features[idx, 10] = tensor.memory_level / 3.0
        
        return node_features
    
    def _build_edge_index(self, features: MuGraphFeature) -> np.ndarray:
        """构建边索引 (用于 GNN)"""
        if not features.edges:
            return np.zeros((2, 0), dtype=np.int64)
        
        edges = np.array(features.edges, dtype=np.int64).T
        return edges
    
    def _build_global_features(self, features: MuGraphFeature) -> np.ndarray:
        """构建全局特征向量"""
        global_features = np.zeros(32, dtype=np.float32)
        
        # 结构特征
        global_features[0] = features.num_operators / 20.0
        global_features[1] = features.num_tensors / 20.0
        global_features[2] = features.graph_depth / 10.0
        global_features[3] = features.graph_width / 10.0
        global_features[4] = features.critical_path_length / 10.0
        global_features[5] = features.parallelism_degree
        
        # 配置特征
        global_features[6] = features.grid_dim[0] / 128.0
        global_features[7] = features.grid_dim[1] / 128.0
        global_features[8] = features.grid_dim[2] / 128.0
        global_features[9] = features.block_dim[0] / 1024.0
        global_features[10] = features.block_dim[1] / 32.0
        global_features[11] = features.forloop_range / 64.0
        global_features[12] = features.occupancy
        global_features[13] = features.shared_mem_usage
        global_features[14] = features.register_usage
        
        # 性能预测特征
        global_features[15] = np.log(features.theoretical_flops + 1) / 30.0
        global_features[16] = features.memory_bandwidth
        global_features[17] = np.log(features.arithmetic_intensity + 1) / 10.0
        global_features[18] = np.log(features.estimated_latency + 1) / 10.0
        
        return global_features
    
    def _normalize(self, features: np.ndarray, key: str) -> np.ndarray:
        """归一化特征"""
        # 简单的 min-max 归一化
        min_val = features.min(axis=0, keepdims=True)
        max_val = features.max(axis=0, keepdims=True)
        
        range_val = max_val - min_val
        range_val[range_val == 0] = 1.0  # 避免除零
        
        return (features - min_val) / range_val
    
    def save_stats(self, path: str):
        """保存归一化统计量"""
        np.savez(path, **self._stats)
    
    def load_stats(self, path: str):
        """加载归一化统计量"""
        data = np.load(path)
        self._stats = {k: data[k] for k in data.files}
```

---

## 3. RL 模型层：设计和保存

### 模型架构

```python
# python/yirage/rl/models/search_policy.py

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path

class GraphEncoder(nn.Module):
    """
    图编码器：将 µGraph 特征编码为嵌入向量
    
    支持两种模式:
    1. MLP: 简单的特征聚合
    2. GNN: 图神经网络 (需要 torch_geometric)
    """
    
    def __init__(
        self,
        node_dim: int = 16,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_layers: int = 3,
        use_gnn: bool = False,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gnn = use_gnn
        
        if use_gnn:
            self._build_gnn(num_layers)
        else:
            self._build_mlp(num_layers)
    
    def _build_mlp(self, num_layers: int):
        """构建 MLP 编码器"""
        layers = [nn.Linear(self.node_dim, self.hidden_dim), nn.ReLU()]
        
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            ])
        
        layers.append(nn.Linear(self.hidden_dim, self.output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 全局池化后的处理
        self.output_layer = nn.Linear(self.output_dim, self.output_dim)
    
    def _build_gnn(self, num_layers: int):
        """构建 GNN 编码器"""
        try:
            from torch_geometric.nn import GCNConv, global_mean_pool
            
            self.convs = nn.ModuleList([
                GCNConv(self.node_dim if i == 0 else self.hidden_dim, self.hidden_dim)
                for i in range(num_layers)
            ])
            
            self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
            self.pool = global_mean_pool
            
        except ImportError:
            print("torch_geometric not found, falling back to MLP")
            self.use_gnn = False
            self._build_mlp(num_layers)
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_dim]
            edge_index: [2, num_edges] (for GNN)
            batch: [num_nodes] batch assignment (for GNN)
        
        Returns:
            Graph embedding: [batch_size, output_dim]
        """
        if self.use_gnn and edge_index is not None:
            return self._forward_gnn(node_features, edge_index, batch)
        else:
            return self._forward_mlp(node_features, batch)
    
    def _forward_mlp(
        self,
        node_features: torch.Tensor,
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """MLP forward"""
        # 节点嵌入
        x = self.mlp(node_features)
        
        # 全局池化
        if batch is None:
            # 单个图
            graph_embedding = x.mean(dim=0, keepdim=True)
        else:
            # 批量图
            from torch_scatter import scatter_mean
            graph_embedding = scatter_mean(x, batch, dim=0)
        
        return self.output_layer(graph_embedding)
    
    def _forward_gnn(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """GNN forward"""
        x = node_features
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        
        if batch is None:
            graph_embedding = x.mean(dim=0, keepdim=True)
        else:
            graph_embedding = self.pool(x, batch)
        
        return self.output_layer(graph_embedding)


class ConfigEncoder(nn.Module):
    """配置编码器：编码硬件配置和约束"""
    
    def __init__(self, input_dim: int = 32, output_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, config_features: torch.Tensor) -> torch.Tensor:
        return self.net(config_features)


class SearchPolicyNetwork(nn.Module):
    """
    搜索策略网络
    
    输入: µGraph 特征 (从 C++ 层获取) + 配置特征 + 历史特征
    输出: 动作概率分布
    
    支持:
    - 模型保存/加载
    - ONNX 导出
    - 与 RLlib 集成
    """
    
    def __init__(
        self,
        graph_dim: int = 128,
        config_dim: int = 64,
        history_dim: int = 32,
        hidden_dim: int = 256,
        action_dim: int = 64,
        use_gnn: bool = False,
    ):
        super().__init__()
        
        self.graph_dim = graph_dim
        self.config_dim = config_dim
        self.history_dim = history_dim
        
        # 编码器
        self.graph_encoder = GraphEncoder(
            node_dim=16,
            output_dim=graph_dim,
            use_gnn=use_gnn,
        )
        
        self.config_encoder = ConfigEncoder(
            input_dim=32,
            output_dim=config_dim,
        )
        
        self.history_encoder = nn.Sequential(
            nn.Linear(history_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )
        
        # 融合层
        fusion_dim = graph_dim + config_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # 策略头
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        
        # 价值头
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        graph_features: Dict[str, torch.Tensor],
        config_features: torch.Tensor,
        history_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            graph_features: {
                "node_features": [num_nodes, node_dim],
                "edge_index": [2, num_edges],
                "global_features": [batch, global_dim],
            }
            config_features: [batch, config_dim]
            history_features: [batch, history_dim]
            action_mask: [batch, action_dim] binary mask
        
        Returns:
            (action_logits, value)
        """
        # 图编码
        graph_emb = self.graph_encoder(
            graph_features["node_features"],
            graph_features.get("edge_index"),
            graph_features.get("batch"),
        )
        
        # 配置编码
        config_emb = self.config_encoder(config_features)
        
        # 历史编码
        history_emb = self.history_encoder(history_features)
        
        # 融合
        fused = torch.cat([graph_emb, config_emb, history_emb], dim=-1)
        hidden = self.fusion(fused)
        
        # 策略和价值
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        
        # 应用动作掩码
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), float("-inf"))
        
        return logits, value
    
    def get_action(
        self,
        graph_features: Dict[str, torch.Tensor],
        config_features: torch.Tensor,
        history_features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Returns:
            (action, log_prob, value)
        """
        logits, value = self.forward(
            graph_features, config_features, history_features, action_mask
        )
        
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        log_prob = torch.log(probs.gather(1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)
        
        return action, log_prob, value.squeeze(-1)
    
    # =====================================
    # 模型保存和加载
    # =====================================
    
    def save(self, path: str, include_optimizer: bool = False, optimizer=None):
        """
        保存模型检查点
        
        Args:
            path: 保存路径
            include_optimizer: 是否保存优化器状态
            optimizer: 优化器 (如果 include_optimizer=True)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": {
                "graph_dim": self.graph_dim,
                "config_dim": self.config_dim,
                "history_dim": self.history_dim,
            },
        }
        
        if include_optimizer and optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "SearchPolicyNetwork":
        """
        加载模型
        
        Args:
            path: 检查点路径
            device: 目标设备
        
        Returns:
            加载的模型
        """
        checkpoint = torch.load(path, map_location=device)
        
        config = checkpoint["model_config"]
        model = cls(
            graph_dim=config["graph_dim"],
            config_dim=config["config_dim"],
            history_dim=config["history_dim"],
        )
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        print(f"Model loaded from {path}")
        return model
    
    def export_onnx(self, path: str, batch_size: int = 1):
        """
        导出 ONNX 模型
        
        Args:
            path: ONNX 文件路径
            batch_size: 批大小
        """
        # 创建虚拟输入
        dummy_node_features = torch.randn(10, 16)
        dummy_config = torch.randn(batch_size, 32)
        dummy_history = torch.randn(batch_size, self.history_dim)
        
        # 简化版导出 (不包含图结构)
        # 完整版需要更复杂的处理
        torch.onnx.export(
            self,
            (
                {"node_features": dummy_node_features, "global_features": dummy_config},
                dummy_config,
                dummy_history,
            ),
            path,
            opset_version=14,
            input_names=["graph", "config", "history"],
            output_names=["logits", "value"],
        )
        
        print(f"ONNX model exported to {path}")
```

---

## 4. 完整集成

### 在环境中使用

```python
# 更新 hierarchical_env.py

class ConstrainedGraphEnv(gym.Env):
    def __init__(self, ...):
        ...
        # 特征处理器
        self.feature_processor = FeatureProcessor()
        
        # C++ 上下文 (用于提取特征)
        self._cpp_context = None
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        获取观察 - 从 C++ µGraph 提取特征
        """
        if self._cpp_context is not None:
            # 从 C++ 提取特征
            features_json = self._cpp_context.extract_features()
            mu_features = MuGraphFeature.from_json(features_json)
            processed = self.feature_processor.process(mu_features)
            
            return {
                "node_features": processed["node_features"],
                "edge_index": processed["edge_index"],
                "global_features": processed["global_features"],
                "config_embedding": self.constraints.encode(),
            }
        else:
            # Fallback: 简单特征
            return self._get_simple_observation()
```

### 训练和保存

```python
from yirage.rl.models.search_policy import SearchPolicyNetwork
from yirage.rl.features.processor import FeatureProcessor

# 创建模型
policy = SearchPolicyNetwork(
    graph_dim=128,
    config_dim=64,
    use_gnn=True,
)

# 训练...
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        # 特征来自 C++ µGraph
        graph_features = batch["graph_features"]
        config_features = batch["config_features"]
        history = batch["history"]
        
        # 前向传播
        logits, value = policy(graph_features, config_features, history)
        
        # 计算损失和更新
        loss = compute_ppo_loss(...)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 定期保存
    if epoch % 100 == 0:
        policy.save(
            f"checkpoints/policy_epoch_{epoch}.pt",
            include_optimizer=True,
            optimizer=optimizer,
        )

# 最终保存
policy.save("checkpoints/policy_final.pt")

# 导出 ONNX (用于部署)
policy.export_onnx("checkpoints/policy.onnx")
```

### 加载和推理

```python
# 加载模型
policy = SearchPolicyNetwork.load("checkpoints/policy_final.pt")

# 推理
with torch.no_grad():
    # 从 C++ 获取 µGraph 特征
    features = cpp_context.extract_features()
    mu_features = MuGraphFeature.from_json(features)
    processed = feature_processor.process(mu_features)
    
    # 获取动作
    action, log_prob, value = policy.get_action(
        graph_features=processed,
        config_features=config_embedding,
        history_features=history,
        deterministic=True,  # 推理时使用确定性策略
    )
```

---

## 总结

关键改进:
1. **特征从 C++ 底层获取**: `GraphFeatureExtractor` 从 µGraph 提取完整特征
2. **特征处理流水线**: `FeatureProcessor` 处理、归一化、构建 GNN 输入
3. **独立的模型设计**: `SearchPolicyNetwork` 支持 GNN/MLP 编码器
4. **完整的保存/加载**: `save()`, `load()`, `export_onnx()`
