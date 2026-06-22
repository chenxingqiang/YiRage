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
Feature processing for RL models.

Processes µGraph features into format suitable for neural network input.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import json
from pathlib import Path

from .mugraph_features import MuGraphFeature, OperatorFeature, TensorFeature


# Operator type to ID mapping
OPERATOR_TYPE_MAP = {
    "MATMUL": 0,
    "matmul": 0,
    "ADD": 1,
    "add": 1,
    "MUL": 2,
    "mul": 2,
    "DIV": 3,
    "div": 3,
    "EXP": 4,
    "exp": 4,
    "SILU": 5,
    "silu": 5,
    "GELU": 6,
    "gelu": 6,
    "RELU": 7,
    "relu": 7,
    "REDUCTION": 8,
    "reduction": 8,
    "RMS_NORM": 9,
    "rms_norm": 9,
    "SOFTMAX": 10,
    "softmax": 10,
    "CONCAT": 11,
    "concat": 11,
    "FORLOOP_ACCUM": 12,
    "forloop_accum": 12,
    "SQUARE": 13,
    "square": 13,
    "SQRT": 14,
    "sqrt": 14,
}


@dataclass
class FeatureNormalizer:
    """
    Normalizer for features.

    Stores running statistics for normalization.
    Supports save/load for consistent normalization during inference.
    """

    # Running statistics
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None

    # Number of samples seen
    count: int = 0

    def update(self, features: np.ndarray):
        """Update running statistics with new features."""
        if self.mean is None:
            self.mean = np.zeros(features.shape[-1], dtype=np.float64)
            self.std = np.zeros(features.shape[-1], dtype=np.float64)
            self.min_val = np.full(features.shape[-1], np.inf, dtype=np.float64)
            self.max_val = np.full(features.shape[-1], -np.inf, dtype=np.float64)

        # Welford's online algorithm for mean and variance
        batch_size = features.shape[0] if features.ndim > 1 else 1
        batch_mean = features.mean(axis=0) if features.ndim > 1 else features

        delta = batch_mean - self.mean
        self.count += batch_size
        self.mean += delta * batch_size / self.count

        # Update min/max
        if features.ndim > 1:
            self.min_val = np.minimum(self.min_val, features.min(axis=0))
            self.max_val = np.maximum(self.max_val, features.max(axis=0))
        else:
            self.min_val = np.minimum(self.min_val, features)
            self.max_val = np.maximum(self.max_val, features)

    def normalize(self, features: np.ndarray, method: str = "minmax") -> np.ndarray:
        """
        Normalize features.

        Args:
            features: Features to normalize
            method: "minmax" or "zscore"
        """
        if self.mean is None:
            return features

        if method == "zscore":
            std = np.where(self.std > 0, self.std, 1.0)
            return (features - self.mean) / std
        else:  # minmax
            range_val = self.max_val - self.min_val
            range_val = np.where(range_val > 0, range_val, 1.0)
            return (features - self.min_val) / range_val

    def save(self, path: str):
        """Save normalizer statistics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            mean=self.mean if self.mean is not None else np.array([]),
            std=self.std if self.std is not None else np.array([]),
            min_val=self.min_val if self.min_val is not None else np.array([]),
            max_val=self.max_val if self.max_val is not None else np.array([]),
            count=np.array([self.count]),
        )

    @classmethod
    def load(cls, path: str) -> "FeatureNormalizer":
        """Load normalizer statistics."""
        data = np.load(path)

        norm = cls()
        norm.mean = data["mean"] if data["mean"].size > 0 else None
        norm.std = data["std"] if data["std"].size > 0 else None
        norm.min_val = data["min_val"] if data["min_val"].size > 0 else None
        norm.max_val = data["max_val"] if data["max_val"].size > 0 else None
        norm.count = int(data["count"][0])

        return norm


class FeatureProcessor:
    """
    Process µGraph features for RL model input.

    Converts MuGraphFeature (from C++) into neural network compatible format:
    - node_features: [num_nodes, node_dim] for GNN
    - edge_index: [2, num_edges] for GNN
    - global_features: [global_dim] for MLP

    Also handles:
    - Feature normalization
    - Caching
    - Save/load normalizer state
    """

    # Feature dimensions
    NODE_DIM = 16
    GLOBAL_DIM = 48

    def __init__(
        self,
        normalize: bool = True,
        normalize_method: str = "minmax",
        cache_size: int = 1000,
    ):
        self.normalize = normalize
        self.normalize_method = normalize_method

        # Normalizers for different feature types
        self.node_normalizer = FeatureNormalizer()
        self.global_normalizer = FeatureNormalizer()

        # Cache
        from collections import OrderedDict

        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    def process(
        self,
        features: MuGraphFeature,
        update_normalizer: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Process MuGraphFeature into model input format.

        Args:
            features: µGraph features from C++
            update_normalizer: Whether to update running statistics

        Returns:
            {
                "node_features": [num_nodes, NODE_DIM],
                "edge_index": [2, num_edges],
                "global_features": [GLOBAL_DIM],
                "batch": [num_nodes] (for batching)
            }
        """
        # Check cache
        cache_key = self._get_cache_key(features)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build features
        node_features = self._build_node_features(features)
        edge_index = self._build_edge_index(features)
        global_features = self._build_global_features(features)

        # Update normalizers
        if update_normalizer:
            if node_features.size > 0:
                self.node_normalizer.update(node_features)
            self.global_normalizer.update(global_features)

        # Normalize
        if self.normalize:
            if node_features.size > 0:
                node_features = self.node_normalizer.normalize(node_features, self.normalize_method)
            global_features = self.global_normalizer.normalize(
                global_features, self.normalize_method
            )

        result = {
            "node_features": node_features.astype(np.float32),
            "edge_index": edge_index.astype(np.int64),
            "global_features": global_features.astype(np.float32),
            "batch": np.zeros(node_features.shape[0], dtype=np.int64),
        }

        # Cache result
        self._cache[cache_key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return result

    def _build_node_features(self, features: MuGraphFeature) -> np.ndarray:
        """
        Build node feature matrix.

        Nodes include both operators and tensors.
        """
        num_ops = len(features.operators)
        num_tensors = len(features.tensors)
        num_nodes = num_ops + num_tensors

        if num_nodes == 0:
            return np.zeros((1, self.NODE_DIM), dtype=np.float32)

        node_features = np.zeros((num_nodes, self.NODE_DIM), dtype=np.float32)

        # Operator nodes
        for i, op in enumerate(features.operators):
            node_features[i, 0] = 1.0  # Is operator
            node_features[i, 1] = OPERATOR_TYPE_MAP.get(op.op_type, 15) / 15.0
            node_features[i, 2] = op.num_inputs / 4.0
            node_features[i, 3] = op.num_outputs / 4.0
            node_features[i, 4] = np.log1p(op.flops) / 30.0
            node_features[i, 5] = np.log1p(op.memory_read_bytes) / 30.0
            node_features[i, 6] = np.log1p(op.memory_write_bytes) / 30.0

        # Tensor nodes
        for i, tensor in enumerate(features.tensors):
            idx = num_ops + i
            node_features[idx, 0] = 0.0  # Is tensor
            node_features[idx, 7] = len(tensor.dims) / 4.0
            node_features[idx, 8] = np.log1p(tensor.num_elements) / 30.0
            if tensor.dims:
                node_features[idx, 9] = np.log1p(max(tensor.dims)) / 15.0
                node_features[idx, 10] = np.log1p(min(tensor.dims)) / 15.0
            node_features[idx, 11] = tensor.dtype_id / 10.0
            node_features[idx, 12] = tensor.memory_level / 3.0
            node_features[idx, 13] = float(tensor.is_input)
            node_features[idx, 14] = float(tensor.is_output)
            node_features[idx, 15] = np.log1p(tensor.size_bytes) / 30.0

        return node_features

    def _build_edge_index(self, features: MuGraphFeature) -> np.ndarray:
        """
        Build edge index for GNN.

        Edges connect operators through tensors.
        """
        if not features.edges:
            # Build edges from operator connectivity
            edges = []
            num_ops = len(features.operators)

            for op in features.operators:
                # Edges from input tensors to operator
                for tensor_id in op.input_tensor_ids:
                    if 0 <= tensor_id < len(features.tensors):
                        edges.append((num_ops + tensor_id, op.op_id))

                # Edges from operator to output tensors
                for tensor_id in op.output_tensor_ids:
                    if 0 <= tensor_id < len(features.tensors):
                        edges.append((op.op_id, num_ops + tensor_id))

            if edges:
                return np.array(edges, dtype=np.int64).T
            else:
                return np.zeros((2, 0), dtype=np.int64)

        return np.array(features.edges, dtype=np.int64).T

    def _build_global_features(self, features: MuGraphFeature) -> np.ndarray:
        """
        Build global graph feature vector.
        """
        gf = np.zeros(self.GLOBAL_DIM, dtype=np.float32)

        # Structure features
        gf[0] = features.num_operators / 20.0
        gf[1] = features.num_tensors / 20.0
        gf[2] = features.graph_depth / 10.0
        gf[3] = features.graph_width / 10.0
        gf[4] = features.critical_path_length / 10.0
        gf[5] = features.parallelism_degree

        # Config features
        gf[6] = np.log1p(features.grid_dim[0]) / 7.0
        gf[7] = np.log1p(features.grid_dim[1]) / 7.0
        gf[8] = np.log1p(features.grid_dim[2]) / 7.0
        gf[9] = np.log1p(features.block_dim[0]) / 10.0
        gf[10] = np.log1p(features.block_dim[1]) / 6.0
        gf[11] = np.log1p(features.block_dim[2]) / 6.0
        gf[12] = np.log1p(features.forloop_range) / 6.0
        gf[13] = np.log1p(features.reduction_dimx) / 5.0

        # Resource usage
        gf[14] = features.occupancy
        gf[15] = features.shared_mem_usage
        gf[16] = features.register_usage

        # Performance prediction
        gf[17] = np.log1p(features.theoretical_flops) / 30.0
        gf[18] = features.memory_bandwidth_utilization
        gf[19] = np.log1p(features.arithmetic_intensity) / 10.0
        gf[20] = np.log1p(features.estimated_latency_ms) / 10.0

        # Search state
        gf[21] = features.search_level / 2.0
        gf[22] = features.search_depth / 50.0

        # Operator type histogram
        op_type_counts = np.zeros(16, dtype=np.float32)
        for op in features.operators:
            type_id = OPERATOR_TYPE_MAP.get(op.op_type, 15)
            op_type_counts[type_id] += 1
        gf[23:39] = op_type_counts / max(features.num_operators, 1)

        # Tensor shape statistics
        if features.tensors:
            all_dims = []
            for t in features.tensors:
                all_dims.extend(t.dims)
            if all_dims:
                gf[39] = np.mean(all_dims) / 4096.0
                gf[40] = np.std(all_dims) / 4096.0 if len(all_dims) > 1 else 0
                gf[41] = np.max(all_dims) / 4096.0
                gf[42] = np.min(all_dims) / 4096.0

        return gf

    def _get_cache_key(self, features: MuGraphFeature) -> str:
        """Generate cache key from features."""
        import hashlib

        content = f"{features.num_operators}_{features.num_tensors}_{features.grid_dim}_{features.block_dim}"
        return hashlib.md5(content.encode()).hexdigest()

    def save(self, path: str):
        """Save processor state (normalizers)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.node_normalizer.save(str(path / "node_normalizer.npz"))
        self.global_normalizer.save(str(path / "global_normalizer.npz"))

        # Save config
        config = {
            "normalize": self.normalize,
            "normalize_method": self.normalize_method,
            "node_dim": self.NODE_DIM,
            "global_dim": self.GLOBAL_DIM,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f)

    @classmethod
    def load(cls, path: str) -> "FeatureProcessor":
        """Load processor state."""
        path = Path(path)

        # Load config
        with open(path / "config.json") as f:
            config = json.load(f)

        processor = cls(
            normalize=config["normalize"],
            normalize_method=config["normalize_method"],
        )

        processor.node_normalizer = FeatureNormalizer.load(str(path / "node_normalizer.npz"))
        processor.global_normalizer = FeatureNormalizer.load(str(path / "global_normalizer.npz"))

        return processor
