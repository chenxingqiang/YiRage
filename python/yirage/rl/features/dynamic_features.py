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
Dynamic Feature Dictionary (Problem 4)

Replaces hardcoded 32-dim feature vectors with extensible Dict[str, Tensor]
feature representation. New features can be added without breaking saved models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import numpy as np

from .mugraph_features import MuGraphFeature


@dataclass
class FeatureSpec:
    """Specification for a single feature."""

    name: str
    dim: int
    dtype: str = "float32"
    description: str = ""
    # Whether this feature is required (vs optional)
    required: bool = False


# Registry of all known features
FEATURE_REGISTRY: Dict[str, FeatureSpec] = {
    # Graph structure
    "graph_topology": FeatureSpec("graph_topology", 8, description="Node/edge counts, depth, width"),
    "operator_histogram": FeatureSpec("operator_histogram", 16, description="Operator type distribution"),
    "tensor_stats": FeatureSpec("tensor_stats", 8, description="Tensor shape statistics"),
    # Hardware config
    "hardware_config": FeatureSpec("hardware_config", 8, description="Grid/block dims, forloop"),
    "hardware_profile": FeatureSpec("hardware_profile", 16, description="Full hardware profile"),
    # Performance
    "performance_metrics": FeatureSpec("performance_metrics", 8, description="Latency, memory, FLOPS"),
    "resource_usage": FeatureSpec("resource_usage", 4, description="Occupancy, shared mem, registers"),
    # AccelForge
    "accelforge_design": FeatureSpec("accelforge_design", 10, description="PE array, buffer, dataflow"),
    "accelforge_metrics": FeatureSpec("accelforge_metrics", 8, description="Area, energy, power"),
    # Search state
    "search_state": FeatureSpec("search_state", 4, description="Level, depth, valid found"),
    # Kernel characteristics (from Problem 1)
    "kernel_characteristics": FeatureSpec("kernel_characteristics", 12, description="Bottom-up feedback"),
    # Z3 violation (from Problem 6a)
    "z3_violation": FeatureSpec("z3_violation", 6, description="Constraint satisfaction info"),
}


class DynamicFeatureDict:
    """
    Extensible feature container using Dict[str, np.ndarray].

    Benefits over fixed-length vectors:
    - New features don't break saved models (missing → zero-padded)
    - Feature keys are human-readable (not index-based)
    - Each feature has its own normalization
    - Features can be added/removed without retraining
    """

    def __init__(
        self,
        features: Optional[Dict[str, np.ndarray]] = None,
    ):
        self._features: Dict[str, np.ndarray] = features or {}

    def set(self, key: str, value: np.ndarray):
        """Set a feature by key."""
        self._features[key] = np.asarray(value, dtype=np.float32)

    def get(self, key: str, default_dim: int = 0) -> np.ndarray:
        """Get a feature by key, returning zeros if missing."""
        if key in self._features:
            return self._features[key]
        if default_dim > 0:
            return np.zeros(default_dim, dtype=np.float32)
        spec = FEATURE_REGISTRY.get(key)
        if spec:
            return np.zeros(spec.dim, dtype=np.float32)
        return np.zeros(1, dtype=np.float32)

    def keys(self) -> Set[str]:
        return set(self._features.keys())

    def to_flat_vector(self, feature_order: Optional[List[str]] = None) -> np.ndarray:
        """
        Flatten to a single vector with deterministic ordering.

        If feature_order is provided, uses that order (zero-padding missing).
        Otherwise uses sorted keys.
        """
        if feature_order is None:
            feature_order = sorted(self._features.keys())

        parts = []
        for key in feature_order:
            if key in self._features:
                parts.append(self._features[key].flatten())
            elif key in FEATURE_REGISTRY:
                parts.append(np.zeros(FEATURE_REGISTRY[key].dim, dtype=np.float32))

        if not parts:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(parts)

    def to_dict(self) -> Dict[str, List[float]]:
        """Serialize to JSON-compatible dict."""
        return {k: v.tolist() for k, v in self._features.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DynamicFeatureDict":
        """Deserialize from dict."""
        features = {}
        for k, v in d.items():
            if isinstance(v, (list, np.ndarray)):
                features[k] = np.asarray(v, dtype=np.float32)
        return cls(features)


class DynamicFeatureProcessor:
    """
    Processes MuGraphFeature into DynamicFeatureDict format.

    Extends FeatureProcessor (processor.py) with dynamic feature support.
    """

    def __init__(self):
        from .processor import OPERATOR_TYPE_MAP
        self._op_type_map = OPERATOR_TYPE_MAP

    def process(self, features: MuGraphFeature) -> DynamicFeatureDict:
        """Convert MuGraphFeature to DynamicFeatureDict."""
        result = DynamicFeatureDict()

        # Graph topology
        result.set("graph_topology", self._build_topology(features))

        # Operator histogram
        result.set("operator_histogram", self._build_op_histogram(features))

        # Tensor stats
        result.set("tensor_stats", self._build_tensor_stats(features))

        # Hardware config
        result.set("hardware_config", self._build_hw_config(features))

        # Performance metrics
        result.set("performance_metrics", self._build_performance(features))

        # Resource usage
        result.set("resource_usage", self._build_resources(features))

        # Search state
        result.set("search_state", self._build_search_state(features))

        # AccelForge (if available)
        af = self._build_accelforge(features)
        if af is not None:
            result.set("accelforge_metrics", af)

        return result

    def _build_topology(self, f: MuGraphFeature) -> np.ndarray:
        """Build graph topology features."""
        v = np.zeros(8, dtype=np.float32)
        v[0] = f.num_operators / 20.0
        v[1] = f.num_tensors / 20.0
        v[2] = f.graph_depth / 10.0
        v[3] = f.graph_width / 10.0
        v[4] = f.critical_path_length / 10.0
        v[5] = f.parallelism_degree
        v[6] = len(f.edges) / 50.0 if f.edges else 0.0
        return v

    def _build_op_histogram(self, f: MuGraphFeature) -> np.ndarray:
        """Build operator type histogram."""
        hist = np.zeros(16, dtype=np.float32)
        for op in f.operators:
            type_id = self._op_type_map.get(op.op_type, 15)
            hist[type_id] += 1
        total = max(f.num_operators, 1)
        return hist / total

    def _build_tensor_stats(self, f: MuGraphFeature) -> np.ndarray:
        """Build tensor shape statistics."""
        v = np.zeros(8, dtype=np.float32)
        if f.tensors:
            all_dims = []
            for t in f.tensors:
                all_dims.extend(t.dims)
            if all_dims:
                v[0] = np.mean(all_dims) / 4096.0
                v[1] = np.std(all_dims) / 4096.0 if len(all_dims) > 1 else 0
                v[2] = np.max(all_dims) / 4096.0
                v[3] = np.min(all_dims) / 4096.0
            v[4] = len(f.tensors) / 20.0
            total_bytes = sum(t.size_bytes for t in f.tensors)
            v[5] = np.log1p(total_bytes) / 30.0
        return v

    def _build_hw_config(self, f: MuGraphFeature) -> np.ndarray:
        """Build hardware config features."""
        v = np.zeros(8, dtype=np.float32)
        v[0] = np.log1p(f.grid_dim[0]) / 7.0
        v[1] = np.log1p(f.grid_dim[1]) / 7.0
        v[2] = np.log1p(f.grid_dim[2]) / 7.0
        v[3] = np.log1p(f.block_dim[0]) / 10.0
        v[4] = np.log1p(f.block_dim[1]) / 6.0
        v[5] = np.log1p(f.block_dim[2]) / 6.0
        v[6] = np.log1p(f.forloop_range) / 6.0
        v[7] = np.log1p(f.reduction_dimx) / 5.0
        return v

    def _build_performance(self, f: MuGraphFeature) -> np.ndarray:
        """Build performance prediction features."""
        v = np.zeros(8, dtype=np.float32)
        v[0] = np.log1p(f.theoretical_flops) / 30.0
        v[1] = f.memory_bandwidth_utilization
        v[2] = np.log1p(f.arithmetic_intensity) / 10.0
        v[3] = np.log1p(f.estimated_latency_ms) / 10.0
        return v

    def _build_resources(self, f: MuGraphFeature) -> np.ndarray:
        """Build resource usage features."""
        v = np.zeros(4, dtype=np.float32)
        v[0] = f.occupancy
        v[1] = f.shared_mem_usage
        v[2] = f.register_usage
        return v

    def _build_search_state(self, f: MuGraphFeature) -> np.ndarray:
        """Build search state features."""
        v = np.zeros(4, dtype=np.float32)
        v[0] = f.search_level / 2.0
        v[1] = f.search_depth / 50.0
        return v

    def _build_accelforge(self, f: MuGraphFeature) -> Optional[np.ndarray]:
        """Build AccelForge features if available."""
        if f.energy_per_op_pj <= 0 and f.area_mm2 <= 0:
            return None
        v = np.zeros(8, dtype=np.float32)
        v[0] = min(f.energy_per_op_pj / 10.0, 1.0) if f.energy_per_op_pj > 0 else 0
        v[1] = min(f.area_mm2 / 100.0, 1.0) if f.area_mm2 > 0 else 0
        v[2] = min(f.total_power_mw / 10000.0, 1.0) if f.total_power_mw > 0 else 0
        v[3] = f.pe_utilization
        return v
