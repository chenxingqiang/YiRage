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
µGraph feature dataclasses.

These dataclasses represent the features extracted from C++ µGraph.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import json


@dataclass
class OperatorFeature:
    """Features for a single operator in µGraph."""

    op_id: int = 0
    op_type: str = ""
    op_type_id: int = 0
    num_inputs: int = 0
    num_outputs: int = 0

    # Performance characteristics
    flops: float = 0.0
    memory_read_bytes: float = 0.0
    memory_write_bytes: float = 0.0

    # Connectivity
    input_tensor_ids: List[int] = field(default_factory=list)
    output_tensor_ids: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OperatorFeature":
        return cls(
            op_id=d.get("op_id", 0),
            op_type=d.get("op_type", ""),
            op_type_id=d.get("op_type_id", 0),
            num_inputs=d.get("num_inputs", 0),
            num_outputs=d.get("num_outputs", 0),
            flops=d.get("flops", 0.0),
            memory_read_bytes=d.get("memory_read_bytes", 0.0),
            memory_write_bytes=d.get("memory_write_bytes", 0.0),
            input_tensor_ids=d.get("input_tensor_ids", []),
            output_tensor_ids=d.get("output_tensor_ids", []),
        )


@dataclass
class TensorFeature:
    """Features for a single tensor in µGraph."""

    tensor_id: int = 0
    dims: List[int] = field(default_factory=list)
    dtype: str = "float16"
    dtype_id: int = 0
    size_bytes: int = 0

    # Memory level: 0=register, 1=shared, 2=global
    memory_level: int = 2

    # Is input/output of the graph
    is_input: bool = False
    is_output: bool = False

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TensorFeature":
        return cls(
            tensor_id=d.get("tensor_id", 0),
            dims=d.get("dims", []),
            dtype=d.get("dtype", "float16"),
            dtype_id=d.get("dtype_id", 0),
            size_bytes=d.get("size_bytes", 0),
            memory_level=d.get("memory_level", 2),
            is_input=d.get("is_input", False),
            is_output=d.get("is_output", False),
        )

    @property
    def num_elements(self) -> int:
        if not self.dims:
            return 0
        result = 1
        for d in self.dims:
            result *= d
        return result


@dataclass
class MuGraphFeature:
    """
    Complete µGraph features extracted from C++ layer.

    This is the main dataclass that holds all features needed
    for the RL model input.
    """

    # Node features
    operators: List[OperatorFeature] = field(default_factory=list)
    tensors: List[TensorFeature] = field(default_factory=list)

    # Edge features (operator connectivity via tensors)
    edges: List[Tuple[int, int]] = field(default_factory=list)

    # Graph structure features
    num_operators: int = 0
    num_tensors: int = 0
    graph_depth: int = 0
    graph_width: int = 0
    critical_path_length: int = 0
    parallelism_degree: float = 0.0

    # Hardware configuration features
    grid_dim: Tuple[int, int, int] = (1, 1, 1)
    block_dim: Tuple[int, int, int] = (128, 1, 1)
    forloop_range: int = 1
    reduction_dimx: int = 1

    # Resource usage features
    occupancy: float = 0.0
    shared_mem_usage: float = 0.0
    register_usage: float = 0.0

    # Performance prediction features
    theoretical_flops: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    arithmetic_intensity: float = 0.0
    estimated_latency_ms: float = 0.0

    # AccelForge hardware design features
    energy_per_op_pj: float = 0.0
    area_mm2: float = 0.0
    total_power_mw: float = 0.0
    leak_power_mw: float = 0.0
    pe_utilization: float = 0.0

    # Search state features
    search_level: int = 0  # 0=kernel, 1=threadblock
    search_depth: int = 0

    @classmethod
    def from_json(cls, json_str: str) -> "MuGraphFeature":
        """Parse from C++ JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return cls()

        # Parse operators
        operators = [OperatorFeature.from_dict(op) for op in data.get("operators", [])]

        # Parse tensors
        tensors = [TensorFeature.from_dict(t) for t in data.get("tensors", [])]

        # Parse edges
        edges = [(e[0], e[1]) for e in data.get("edges", [])]

        # Parse grid/block dims
        grid = data.get("grid_dim", {})
        grid_dim = (grid.get("x", 1), grid.get("y", 1), grid.get("z", 1))

        block = data.get("block_dim", {})
        block_dim = (block.get("x", 128), block.get("y", 1), block.get("z", 1))

        return cls(
            operators=operators,
            tensors=tensors,
            edges=edges,
            num_operators=data.get("num_operators", len(operators)),
            num_tensors=data.get("num_tensors", len(tensors)),
            graph_depth=data.get("graph_depth", 0),
            graph_width=data.get("graph_width", 0),
            critical_path_length=data.get("critical_path_length", 0),
            parallelism_degree=data.get("parallelism_degree", 0.0),
            grid_dim=grid_dim,
            block_dim=block_dim,
            forloop_range=data.get("forloop_range", 1),
            reduction_dimx=data.get("reduction_dimx", 1),
            occupancy=data.get("occupancy", 0.0),
            shared_mem_usage=data.get("shared_mem_usage", 0.0),
            register_usage=data.get("register_usage", 0.0),
            theoretical_flops=data.get("theoretical_flops", 0.0),
            memory_bandwidth_utilization=data.get("memory_bandwidth_utilization", 0.0),
            arithmetic_intensity=data.get("arithmetic_intensity", 0.0),
            estimated_latency_ms=data.get("estimated_latency_ms", 0.0),
            energy_per_op_pj=data.get("energy_per_op_pj", 0.0),
            area_mm2=data.get("area_mm2", 0.0),
            total_power_mw=data.get("total_power_mw", 0.0),
            leak_power_mw=data.get("leak_power_mw", 0.0),
            pe_utilization=data.get("pe_utilization", 0.0),
            search_level=data.get("search_level", 0),
            search_depth=data.get("search_depth", 0),
        )

    def to_json(self) -> str:
        """Serialize to JSON."""
        data = {
            "operators": [
                {
                    "op_id": op.op_id,
                    "op_type": op.op_type,
                    "op_type_id": op.op_type_id,
                    "num_inputs": op.num_inputs,
                    "num_outputs": op.num_outputs,
                    "flops": op.flops,
                    "memory_read_bytes": op.memory_read_bytes,
                    "memory_write_bytes": op.memory_write_bytes,
                    "input_tensor_ids": op.input_tensor_ids,
                    "output_tensor_ids": op.output_tensor_ids,
                }
                for op in self.operators
            ],
            "tensors": [
                {
                    "tensor_id": t.tensor_id,
                    "dims": t.dims,
                    "dtype": t.dtype,
                    "dtype_id": t.dtype_id,
                    "size_bytes": t.size_bytes,
                    "memory_level": t.memory_level,
                    "is_input": t.is_input,
                    "is_output": t.is_output,
                }
                for t in self.tensors
            ],
            "edges": list(self.edges),
            "num_operators": self.num_operators,
            "num_tensors": self.num_tensors,
            "graph_depth": self.graph_depth,
            "graph_width": self.graph_width,
            "critical_path_length": self.critical_path_length,
            "parallelism_degree": self.parallelism_degree,
            "grid_dim": {"x": self.grid_dim[0], "y": self.grid_dim[1], "z": self.grid_dim[2]},
            "block_dim": {"x": self.block_dim[0], "y": self.block_dim[1], "z": self.block_dim[2]},
            "forloop_range": self.forloop_range,
            "reduction_dimx": self.reduction_dimx,
            "occupancy": self.occupancy,
            "shared_mem_usage": self.shared_mem_usage,
            "register_usage": self.register_usage,
            "theoretical_flops": self.theoretical_flops,
            "memory_bandwidth_utilization": self.memory_bandwidth_utilization,
            "arithmetic_intensity": self.arithmetic_intensity,
            "estimated_latency_ms": self.estimated_latency_ms,
            "energy_per_op_pj": self.energy_per_op_pj,
            "area_mm2": self.area_mm2,
            "total_power_mw": self.total_power_mw,
            "leak_power_mw": self.leak_power_mw,
            "pe_utilization": self.pe_utilization,
            "search_level": self.search_level,
            "search_depth": self.search_depth,
        }
        return json.dumps(data)

    @classmethod
    def from_graph_json(cls, graph_json: str) -> "MuGraphFeature":
        """
        Parse from simple graph JSON (used when C++ features unavailable).

        This is a fallback when the full C++ feature extraction is not available.
        """
        try:
            data = json.loads(graph_json)
        except json.JSONDecodeError:
            return cls()

        # Parse operators
        operators = []
        for i, op in enumerate(data.get("operators", [])):
            operators.append(
                OperatorFeature(
                    op_id=i,
                    op_type=op.get("type", ""),
                    num_inputs=len(op.get("inputs", [])),
                    num_outputs=len(op.get("outputs", [])),
                    input_tensor_ids=op.get("inputs", []),
                    output_tensor_ids=op.get("outputs", []),
                )
            )

        # Parse tensors from inputs
        tensors = []
        for i, inp in enumerate(data.get("inputs", [])):
            tensors.append(
                TensorFeature(
                    tensor_id=i,
                    dims=inp.get("dims", []),
                    dtype=inp.get("dtype", "float16"),
                    is_input=True,
                )
            )

        # Build edges
        edges = []
        num_input_tensors = len(tensors)
        for op in operators:
            for inp_id in op.input_tensor_ids:
                # Edge from tensor to operator
                edges.append((num_input_tensors + inp_id, op.op_id))

        return cls(
            operators=operators,
            tensors=tensors,
            edges=edges,
            num_operators=len(operators),
            num_tensors=len(tensors),
        )
