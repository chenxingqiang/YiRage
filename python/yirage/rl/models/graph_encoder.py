# Copyright 2025 Chen Xingqiang (YiRage Project)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Graph encoder networks for kernel graph representation.

Encodes kernel/threadblock graphs into fixed-size embeddings
for the RL policy network.
"""

from typing import Optional, Dict, Any
import numpy as np


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None


class SimpleGraphEncoder:
    """
    Simple feature-based graph encoder (no GNN).

    Uses hand-crafted features from graph structure.
    Works without PyTorch.
    """

    def __init__(self, output_dim: int = 128):
        self.output_dim = output_dim

    def encode(self, graph_json: str) -> np.ndarray:
        """
        Encode graph to feature vector.

        Args:
            graph_json: JSON string of kernel graph

        Returns:
            Feature vector of shape (output_dim,)
        """
        import json

        features = np.zeros(self.output_dim, dtype=np.float32)

        try:
            graph = json.loads(graph_json)
        except:
            return features

        # Basic graph statistics
        operators = graph.get("operators", [])
        tensors = graph.get("tensors", [])

        features[0] = len(operators) / 20.0
        features[1] = len(tensors) / 20.0

        # Operator type distribution
        op_types = {}
        for op in operators:
            op_type = op.get("type", "unknown")
            op_types[op_type] = op_types.get(op_type, 0) + 1

        # Encode op type counts
        known_ops = [
            "matmul",
            "add",
            "mul",
            "div",
            "exp",
            "silu",
            "reduction",
            "rms_norm",
            "softmax",
        ]
        for i, op_type in enumerate(known_ops):
            if i + 2 < self.output_dim:
                features[i + 2] = op_types.get(op_type, 0) / 10.0

        # Tensor dimension statistics
        dims = []
        for tensor in tensors:
            tensor_dims = tensor.get("dims", [])
            dims.extend(tensor_dims)

        if dims:
            features[20] = np.mean(dims) / 4096.0
            features[21] = np.std(dims) / 4096.0 if len(dims) > 1 else 0
            features[22] = np.max(dims) / 4096.0
            features[23] = len(dims) / 20.0

        # Grid/block configuration
        config = graph.get("config", {})
        grid = config.get("grid_dim", {})
        block = config.get("block_dim", {})

        features[30] = grid.get("x", 1) / 128.0
        features[31] = grid.get("y", 1) / 128.0
        features[32] = grid.get("z", 1) / 128.0
        features[33] = block.get("x", 128) / 1024.0
        features[34] = block.get("y", 1) / 32.0

        return features


if TORCH_AVAILABLE:

    class GraphEncoder(nn.Module):
        """
        GNN-based graph encoder for kernel graphs.

        Uses message passing to learn graph representations.
        """

        def __init__(
            self,
            node_feature_dim: int = 32,
            edge_feature_dim: int = 8,
            hidden_dim: int = 128,
            output_dim: int = 128,
            num_layers: int = 3,
            aggregation: str = "mean",
        ):
            super().__init__()

            self.node_feature_dim = node_feature_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.num_layers = num_layers

            # Node embedding
            self.node_embed = nn.Linear(node_feature_dim, hidden_dim)

            # Message passing layers
            self.mp_layers = nn.ModuleList(
                [MessagePassingLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
            )

            # Readout
            self.readout = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

            self.aggregation = aggregation

        def forward(
            self,
            node_features: torch.Tensor,
            edge_index: torch.Tensor,
            batch: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Forward pass.

            Args:
                node_features: [num_nodes, node_feature_dim]
                edge_index: [2, num_edges]
                batch: [num_nodes] batch assignment

            Returns:
                Graph embeddings [batch_size, output_dim]
            """
            # Node embedding
            x = self.node_embed(node_features)

            # Message passing
            for mp_layer in self.mp_layers:
                x = mp_layer(x, edge_index)

            # Readout (graph-level pooling)
            if batch is None:
                # Single graph
                if self.aggregation == "mean":
                    graph_embedding = x.mean(dim=0, keepdim=True)
                elif self.aggregation == "sum":
                    graph_embedding = x.sum(dim=0, keepdim=True)
                elif self.aggregation == "max":
                    graph_embedding = x.max(dim=0, keepdim=True)[0]
            else:
                # Batched graphs
                from torch_scatter import scatter

                graph_embedding = scatter(x, batch, dim=0, reduce=self.aggregation)

            return self.readout(graph_embedding)

        def encode_json(self, graph_json: str) -> torch.Tensor:
            """
            Encode graph from JSON.

            Args:
                graph_json: JSON string

            Returns:
                Embedding tensor
            """
            node_features, edge_index = self._parse_graph(graph_json)
            return self.forward(node_features, edge_index)

        def _parse_graph(self, graph_json: str):
            """Parse graph JSON to tensors."""
            import json

            try:
                graph = json.loads(graph_json)
            except:
                # Empty graph fallback
                return (
                    torch.zeros(1, self.node_feature_dim),
                    torch.zeros(2, 0, dtype=torch.long),
                )

            operators = graph.get("operators", [])
            tensors = graph.get("tensors", [])

            num_nodes = len(operators) + len(tensors)
            if num_nodes == 0:
                return (
                    torch.zeros(1, self.node_feature_dim),
                    torch.zeros(2, 0, dtype=torch.long),
                )

            # Node features
            node_features = torch.zeros(num_nodes, self.node_feature_dim)

            # Operator nodes
            for i, op in enumerate(operators):
                node_features[i, 0] = 1.0  # Is operator
                # Encode operator type (one-hot simplified)
                op_type = op.get("type", "")
                node_features[i, 1] = hash(op_type) % 16 / 16.0

            # Tensor nodes
            for i, tensor in enumerate(tensors):
                idx = len(operators) + i
                node_features[idx, 0] = 0.0  # Is tensor
                dims = tensor.get("dims", [1])
                node_features[idx, 2] = len(dims) / 4.0
                if dims:
                    node_features[idx, 3] = np.prod(dims) / 1e9

            # Edges (operator -> output tensor, tensor -> operator input)
            edges = []
            for i, op in enumerate(operators):
                for inp_idx in op.get("inputs", []):
                    if inp_idx < len(tensors):
                        # Tensor -> Operator edge
                        edges.append([len(operators) + inp_idx, i])

                for out_idx in op.get("outputs", []):
                    if out_idx < len(tensors):
                        # Operator -> Tensor edge
                        edges.append([i, len(operators) + out_idx])

            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t()
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)

            return node_features, edge_index

    class MessagePassingLayer(nn.Module):
        """Simple message passing layer."""

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()

            self.linear = nn.Linear(in_dim * 2, out_dim)
            self.norm = nn.LayerNorm(out_dim)

        def forward(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
        ) -> torch.Tensor:
            """Message passing step."""
            if edge_index.shape[1] == 0:
                return x

            src, dst = edge_index

            # Aggregate messages
            messages = x[src]

            # Simple mean aggregation
            from torch_scatter import scatter

            aggregated = scatter(messages, dst, dim=0, reduce="mean", dim_size=x.shape[0])

            # Update
            combined = torch.cat([x, aggregated], dim=-1)
            out = self.linear(combined)
            out = F.relu(out)
            out = self.norm(out + x)  # Residual

            return out

else:
    # Stub when PyTorch not available
    class GraphEncoder:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required for GraphEncoder")
