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
Level 0: Accelerator Design Space (AccelForge Integration)

Defines the accelerator architecture parameters that constrain Level 1 and Level 2.
This enables hardware-software co-design through RL-guided exploration of both
the accelerator architecture and kernel configuration spaces.

Hierarchy:
    Level 0 (Accelerator Design) → Level 1 (Config Policy) → Level 2 (Graph Policy)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..hardware.accelforge_bridge import (
        AccelForgeDesignPoint,
        AccelForgeMetrics,
    )

try:
    import gymnasium as gym
    from gymnasium import spaces

    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces

        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

        class StubSpace:
            """Stub space class for when gym is not available."""

            def __init__(self, *args, **kwargs):
                pass

            def sample(self):
                return None

        class StubSpaces:
            Discrete = StubSpace
            MultiDiscrete = StubSpace
            Box = StubSpace
            Dict = StubSpace

        spaces = StubSpaces()

        class StubGym:
            Env = object

        gym = StubGym()


class _NumpyMultiDiscrete:
    """Minimal MultiDiscrete-like space when gym/gymnasium is not installed."""

    __slots__ = ("nvec",)

    def __init__(self, nvec: List[int]):
        self.nvec = [int(n) for n in nvec]

    def sample(self) -> np.ndarray:
        return np.array([int(np.random.randint(0, n)) for n in self.nvec], dtype=np.int64)


# ============================================================================
# Accelerator Design Parameters
# ============================================================================

# PE array size options (rows, cols)
PE_ARRAY_SIZES = [
    (4, 4),
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
]

# Dataflow strategies
DATAFLOWS = [
    "output_stationary",
    "weight_stationary",
    "row_stationary",
]

# NoC topologies
NOC_TOPOLOGIES = [
    "mesh",
    "ring",
    "tree",
]

# Data precisions
DATA_PRECISIONS = [
    "int8",
    "fp16",
    "bf16",
    "fp32",
]

# Buffer size options (KB)
L0_BUFFER_SIZES = [0.5, 1.0, 2.0, 4.0]
L1_BUFFER_SIZES = [16.0, 32.0, 64.0, 128.0, 256.0]
L2_BUFFER_SIZES = [128.0, 256.0, 512.0, 1024.0, 2048.0]

# Clock frequency options (MHz)
CLOCK_FREQUENCIES = [500.0, 800.0, 1000.0, 1200.0, 1500.0]

# Technology node options (nm)
TECH_NODES = [3, 5, 7, 14, 28]


@dataclass
class AcceleratorDesignConstraints:
    """
    Constraints derived from Level 0 accelerator design for Level 1.

    Propagates hardware architecture choices down to kernel configuration space.
    """

    # From PE array
    max_parallelism: int = 256
    pe_array_rows: int = 16
    pe_array_cols: int = 16

    # From memory hierarchy
    max_shared_memory_kb: float = 64.0
    max_l2_cache_kb: float = 512.0
    max_tile_size: int = 128

    # From dataflow
    preferred_data_layout: str = "row_major"
    supports_weight_reuse: bool = True
    supports_output_reuse: bool = True

    # From precision
    supported_precisions: List[str] = field(default_factory=lambda: ["fp16"])
    compute_multiplier: float = 2.0  # Ops per cycle per PE

    # Peak performance
    peak_tops: float = 0.0
    peak_memory_bw_gbps: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_parallelism": self.max_parallelism,
            "pe_array_rows": self.pe_array_rows,
            "pe_array_cols": self.pe_array_cols,
            "max_shared_memory_kb": self.max_shared_memory_kb,
            "max_l2_cache_kb": self.max_l2_cache_kb,
            "max_tile_size": self.max_tile_size,
            "preferred_data_layout": self.preferred_data_layout,
            "supported_precisions": self.supported_precisions,
            "peak_tops": self.peak_tops,
        }


class AcceleratorActionSpace:
    """
    Action space for Level 0 accelerator design policy.

    Each action selects accelerator architecture parameters.
    """

    # Action dimensions and their sizes
    DIMS = {
        "pe_array_size": len(PE_ARRAY_SIZES),       # 6 options
        "dataflow": len(DATAFLOWS),                   # 3 options
        "noc_topology": len(NOC_TOPOLOGIES),          # 3 options
        "data_precision": len(DATA_PRECISIONS),       # 4 options
        "l0_buffer": len(L0_BUFFER_SIZES),            # 4 options
        "l1_buffer": len(L1_BUFFER_SIZES),            # 5 options
        "l2_buffer": len(L2_BUFFER_SIZES),            # 5 options
        "clock_freq": len(CLOCK_FREQUENCIES),         # 5 options
        "tech_node": len(TECH_NODES),                 # 5 options
    }

    def __init__(self):
        nvec = list(self.DIMS.values())
        if GYM_AVAILABLE:
            self.flat_space = spaces.MultiDiscrete(nvec)
        else:
            self.flat_space = _NumpyMultiDiscrete(nvec)

    def decode_flat(self, action: np.ndarray) -> "AccelForgeDesignPoint":
        """Decode flat action array to AccelForgeDesignPoint."""
        from ..hardware.accelforge_bridge import AccelForgeDesignPoint

        pe_idx = int(action[0]) % len(PE_ARRAY_SIZES)
        pe_rows, pe_cols = PE_ARRAY_SIZES[pe_idx]

        dataflow_idx = int(action[1]) % len(DATAFLOWS)
        dataflow = DATAFLOWS[dataflow_idx]

        noc_idx = int(action[2]) % len(NOC_TOPOLOGIES)
        noc = NOC_TOPOLOGIES[noc_idx]

        prec_idx = int(action[3]) % len(DATA_PRECISIONS)
        precision = DATA_PRECISIONS[prec_idx]

        l0_idx = int(action[4]) % len(L0_BUFFER_SIZES)
        l0 = L0_BUFFER_SIZES[l0_idx]

        l1_idx = int(action[5]) % len(L1_BUFFER_SIZES)
        l1 = L1_BUFFER_SIZES[l1_idx]

        l2_idx = int(action[6]) % len(L2_BUFFER_SIZES)
        l2 = L2_BUFFER_SIZES[l2_idx]

        clock_idx = int(action[7]) % len(CLOCK_FREQUENCIES)
        clock = CLOCK_FREQUENCIES[clock_idx]

        tech_idx = int(action[8]) % len(TECH_NODES)
        tech = TECH_NODES[tech_idx]

        return AccelForgeDesignPoint(
            pe_array_rows=pe_rows,
            pe_array_cols=pe_cols,
            l0_buffer_kb=l0,
            l1_buffer_kb=l1,
            l2_buffer_kb=l2,
            dataflow=dataflow,
            noc_topology=noc,
            data_precision=precision,
            clock_mhz=clock,
            tech_node_nm=tech,
        )

    def encode(self, design: "AccelForgeDesignPoint") -> np.ndarray:
        """Encode AccelForgeDesignPoint to flat action array."""
        action = np.zeros(len(self.DIMS), dtype=np.int64)

        # Find PE array index
        pe_tuple = (design.pe_array_rows, design.pe_array_cols)
        action[0] = PE_ARRAY_SIZES.index(pe_tuple) if pe_tuple in PE_ARRAY_SIZES else 0

        action[1] = DATAFLOWS.index(design.dataflow) if design.dataflow in DATAFLOWS else 0
        action[2] = (
            NOC_TOPOLOGIES.index(design.noc_topology)
            if design.noc_topology in NOC_TOPOLOGIES
            else 0
        )
        action[3] = (
            DATA_PRECISIONS.index(design.data_precision)
            if design.data_precision in DATA_PRECISIONS
            else 0
        )

        # Find closest buffer sizes
        action[4] = self._find_closest_idx(L0_BUFFER_SIZES, design.l0_buffer_kb)
        action[5] = self._find_closest_idx(L1_BUFFER_SIZES, design.l1_buffer_kb)
        action[6] = self._find_closest_idx(L2_BUFFER_SIZES, design.l2_buffer_kb)
        action[7] = self._find_closest_idx(CLOCK_FREQUENCIES, design.clock_mhz)
        action[8] = self._find_closest_idx(TECH_NODES, design.tech_node_nm)

        return action

    @staticmethod
    def _find_closest_idx(options: list, value: float) -> int:
        """Find index of closest value in options list."""
        return min(range(len(options)), key=lambda i: abs(options[i] - value))


class AcceleratorObservationSpace:
    """
    Observation space for Level 0 accelerator design policy.

    State includes workload characteristics, current design metrics,
    and bottom-up kernel pattern feedback from Level 2.
    """

    WORKLOAD_DIM = 16
    DESIGN_DIM = 16
    METRICS_DIM = 16
    HISTORY_DIM = 8
    KERNEL_FEEDBACK_DIM = 12  # From KernelCharacteristics.encode()

    TOTAL_DIM = WORKLOAD_DIM + DESIGN_DIM + METRICS_DIM + HISTORY_DIM + KERNEL_FEEDBACK_DIM

    def __init__(self):
        if GYM_AVAILABLE:
            self.space = spaces.Dict(
                {
                    "workload_features": spaces.Box(
                        low=-1, high=1, shape=(self.WORKLOAD_DIM,), dtype=np.float32
                    ),
                    "design_features": spaces.Box(
                        low=-1, high=1, shape=(self.DESIGN_DIM,), dtype=np.float32
                    ),
                    "metrics_features": spaces.Box(
                        low=-1, high=1, shape=(self.METRICS_DIM,), dtype=np.float32
                    ),
                    "history_features": spaces.Box(
                        low=-1, high=1, shape=(self.HISTORY_DIM,), dtype=np.float32
                    ),
                    "kernel_feedback": spaces.Box(
                        low=-1, high=1, shape=(self.KERNEL_FEEDBACK_DIM,), dtype=np.float32
                    ),
                }
            )
        else:
            self.space = None

    def encode_workload(self, workload_json: str) -> np.ndarray:
        """Encode target workload features."""
        features = np.zeros(self.WORKLOAD_DIM, dtype=np.float32)
        try:
            data = json.loads(workload_json)
            features[0] = np.log10(max(data.get("batch_size", 1), 1)) / 4.0
            features[1] = np.log10(max(data.get("sequence_length", 1024), 1)) / 5.0
            features[2] = np.log10(max(data.get("hidden_dim", 4096), 1)) / 5.0
            features[3] = data.get("num_operators", 0) / 50.0
            features[4] = data.get("total_flops", 0) / 1e12
            features[5] = data.get("total_memory_bytes", 0) / 1e9
        except (json.JSONDecodeError, TypeError):
            pass
        return features

    def encode_design(self, design: "AccelForgeDesignPoint") -> np.ndarray:
        """Encode current accelerator design."""
        from ..hardware.accelforge_bridge import (
            DATA_PRECISION_ENCODING,
            DATAFLOW_ENCODING,
            MAX_PE_ARRAY_LOG2,
            NOC_TOPOLOGY_ENCODING,
        )

        features = np.zeros(self.DESIGN_DIM, dtype=np.float32)

        features[0] = np.log2(max(design.pe_array_rows, 1)) / MAX_PE_ARRAY_LOG2
        features[1] = np.log2(max(design.pe_array_cols, 1)) / MAX_PE_ARRAY_LOG2
        features[2] = np.log2(max(design.l0_buffer_kb, 0.1) + 1) / 4.0
        features[3] = np.log2(max(design.l1_buffer_kb, 1)) / 10.0
        features[4] = np.log2(max(design.l2_buffer_kb, 1)) / 12.0

        # Encode categoricals using shared constants
        features[5] = DATAFLOW_ENCODING.get(design.dataflow, 0.0)
        features[6] = NOC_TOPOLOGY_ENCODING.get(design.noc_topology, 0.0)
        features[7] = DATA_PRECISION_ENCODING.get(design.data_precision, 0.0)

        features[8] = design.clock_mhz / 2000.0
        features[9] = design.tech_node_nm / 28.0

        return features

    def encode_metrics(self, metrics: "AccelForgeMetrics") -> np.ndarray:
        """Encode evaluation metrics."""
        features = np.zeros(self.METRICS_DIM, dtype=np.float32)

        features[0] = min(metrics.area_mm2 / 100.0, 1.0)
        features[1] = min(metrics.energy_per_op_pj / 10.0, 1.0)
        features[2] = min(metrics.latency_ms / 10.0, 1.0)
        features[3] = min(metrics.total_power_mw / 10000.0, 1.0)
        features[4] = min(metrics.leak_power_mw / 5000.0, 1.0)
        features[5] = min(metrics.peak_tops / 100.0, 1.0)
        features[6] = metrics.pe_utilization
        features[7] = metrics.buffer_utilization
        features[8] = metrics.noc_bandwidth_utilization

        return features


@dataclass
class KernelCharacteristics:
    """
    Bottom-up kernel structure feedback from Level 2 to Level 0.

    Captures structural patterns discovered during kernel search that
    should influence accelerator design decisions.
    """

    dominant_op_type: str = "unknown"
    reuse_pattern: str = "none"  # "weight_reuse", "output_reuse", "input_reuse", "none"
    memory_intensity: float = 0.0  # bytes/FLOP ratio
    compute_intensity: float = 0.0  # FLOP/byte ratio (arithmetic intensity)
    num_operators: int = 0
    num_matmuls: int = 0
    num_reductions: int = 0
    requires_large_shared_memory: bool = False
    requires_high_bandwidth: bool = False
    parallelism_degree: float = 0.0
    # Search failure analysis
    search_success_rate: float = 0.0
    common_failure_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_op_type": self.dominant_op_type,
            "reuse_pattern": self.reuse_pattern,
            "memory_intensity": self.memory_intensity,
            "compute_intensity": self.compute_intensity,
            "num_operators": self.num_operators,
            "num_matmuls": self.num_matmuls,
            "num_reductions": self.num_reductions,
            "requires_large_shared_memory": self.requires_large_shared_memory,
            "requires_high_bandwidth": self.requires_high_bandwidth,
            "parallelism_degree": self.parallelism_degree,
            "search_success_rate": self.search_success_rate,
            "common_failure_reason": self.common_failure_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KernelCharacteristics":
        valid_fields = cls.__dataclass_fields__
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def encode(self) -> np.ndarray:
        """Encode as feature vector for Level 0 observation."""
        features = np.zeros(12, dtype=np.float32)
        # Op type encoding
        op_map = {"matmul": 0.2, "reduction": 0.4, "elementwise": 0.6, "attention": 0.8}
        features[0] = op_map.get(self.dominant_op_type, 0.0)
        # Reuse pattern
        reuse_map = {"weight_reuse": 0.25, "output_reuse": 0.5, "input_reuse": 0.75, "none": 0.0}
        features[1] = reuse_map.get(self.reuse_pattern, 0.0)
        features[2] = min(self.memory_intensity / 100.0, 1.0)
        features[3] = min(self.compute_intensity / 100.0, 1.0)
        features[4] = min(self.num_operators / 20.0, 1.0)
        features[5] = min(self.num_matmuls / 10.0, 1.0)
        features[6] = min(self.num_reductions / 10.0, 1.0)
        features[7] = float(self.requires_large_shared_memory)
        features[8] = float(self.requires_high_bandwidth)
        features[9] = min(self.parallelism_degree, 1.0)
        features[10] = self.search_success_rate
        return features

    def suggest_design_adjustments(self) -> Dict[str, Any]:
        """
        Bottom-up constraint suggestion from Level 2 search patterns.

        Analyzes kernel characteristics to suggest accelerator design changes.
        """
        suggestions: Dict[str, Any] = {}

        if self.requires_large_shared_memory:
            suggestions["increase_l1_buffer"] = True
            suggestions["min_l1_buffer_kb"] = 128.0

        if self.requires_high_bandwidth:
            suggestions["prefer_noc"] = "mesh"
            suggestions["increase_l2_buffer"] = True

        if self.memory_intensity > 10.0:
            suggestions["prefer_dataflow"] = "weight_stationary"

        if self.compute_intensity > 10.0:
            suggestions["increase_pe_array"] = True

        if self.search_success_rate < 0.3:
            suggestions["relax_constraints"] = True
            if self.common_failure_reason == "pe_overflow":
                suggestions["increase_pe_array"] = True
            elif self.common_failure_reason == "buffer_overflow":
                suggestions["increase_l1_buffer"] = True

        return suggestions


@dataclass
class ParetoPoint:
    """A point on the Pareto front with kernel structure feedback."""

    design: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = float("inf")
    energy_pj: float = float("inf")
    area_mm2: float = float("inf")
    power_mw: float = float("inf")
    reward: float = 0.0
    # Bottom-up feedback from Level 2
    kernel_characteristics: Optional[Dict[str, Any]] = None

    def dominates(self, other: "ParetoPoint") -> bool:
        """Check if this point Pareto-dominates another."""
        at_least_as_good = (
            self.latency_ms <= other.latency_ms
            and self.energy_pj <= other.energy_pj
            and self.area_mm2 <= other.area_mm2
            and self.power_mw <= other.power_mw
        )
        strictly_better = (
            self.latency_ms < other.latency_ms
            or self.energy_pj < other.energy_pj
            or self.area_mm2 < other.area_mm2
            or self.power_mw < other.power_mw
        )
        return at_least_as_good and strictly_better

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "design": self.design,
            "config": self.config,
            "latency_ms": self.latency_ms,
            "energy_pj": self.energy_pj,
            "area_mm2": self.area_mm2,
            "power_mw": self.power_mw,
            "reward": self.reward,
        }
        if self.kernel_characteristics is not None:
            result["kernel_characteristics"] = self.kernel_characteristics
        return result


class ParetoFrontTracker:
    """
    Tracks the Pareto front across multi-objective evaluations.

    Maintains the set of non-dominated solutions found during search.
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.front: List[ParetoPoint] = []

    def add(self, point: ParetoPoint) -> bool:
        """
        Add a point to the Pareto front if non-dominated.

        Returns True if the point was added (non-dominated).
        """
        # Check if dominated by any existing point
        for existing in self.front:
            if existing.dominates(point):
                return False

        # Remove points dominated by new point
        self.front = [p for p in self.front if not point.dominates(p)]

        # Add new point
        self.front.append(point)

        # Trim if too large (keep most diverse)
        if len(self.front) > self.max_size:
            self._trim()

        return True

    def _trim(self):
        """Trim front to max_size, keeping diverse points."""
        # Sort by reward and keep best
        self.front.sort(key=lambda p: -p.reward)
        self.front = self.front[: self.max_size]

    def get_best(self, objective: str = "latency_ms") -> Optional[ParetoPoint]:
        """Get best point for a specific objective."""
        if not self.front:
            return None
        return min(self.front, key=lambda p: getattr(p, objective, float("inf")))

    def get_front(self) -> List[ParetoPoint]:
        """Get current Pareto front."""
        return list(self.front)

    def size(self) -> int:
        return len(self.front)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        """Export front as list of dicts."""
        return [p.to_dict() for p in self.front]


class AcceleratorEnv(gym.Env):
    """
    Level 0 Environment: Accelerator Architecture Design

    State: Workload features + current design + evaluation history
    Action: Accelerator design parameters (PE array, memory, dataflow, etc.)
    Reward: Multi-objective (latency + energy + area + power) from Level 1+2 results

    This env controls the hardware design space for Level 1 and Level 2.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()

        if config is None:
            config = {}

        self.target_workload_json = config.get("target_workload_json", "{}")

        # Reward weights
        self.reward_weight_latency = config.get("reward_weight_latency", 0.4)
        self.reward_weight_energy = config.get("reward_weight_energy", 0.2)
        self.reward_weight_area = config.get("reward_weight_area", 0.2)
        self.reward_weight_power = config.get("reward_weight_power", 0.2)

        # Budgets
        self.area_budget_mm2 = config.get("area_budget_mm2", 100.0)
        self.power_budget_mw = config.get("power_budget_mw", 5000.0)

        # Limits
        self.max_episodes = config.get("max_design_episodes", 20)

        # Action/observation spaces
        self.action_space_helper = AcceleratorActionSpace()
        self.obs_space_helper = AcceleratorObservationSpace()

        self.action_space = self.action_space_helper.flat_space
        self.observation_space = self.obs_space_helper.space

        # State
        self.current_design = None
        self.current_metrics = None
        self.episode_results: List[Dict[str, Any]] = []
        self.episode_idx = 0

        # Bottom-up feedback from Level 2
        self.kernel_feedback: Optional[KernelCharacteristics] = None
        self.kernel_feedback_history: List[KernelCharacteristics] = []

        # Expert demonstrations from C++ DFS (Problem 6b)
        self.expert_demonstrations: List[Dict[str, Any]] = []

        # Pareto tracking
        self.pareto_tracker = ParetoFrontTracker()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset for new design exploration episode."""
        if GYM_AVAILABLE:
            super().reset(seed=seed)

        if options and "target_workload_json" in options:
            self.target_workload_json = options["target_workload_json"]

        self.current_design = None
        self.current_metrics = None
        self.episode_results = []
        self.episode_idx = 0
        self.kernel_feedback = None

        obs = self._get_observation()
        return obs, {"episode_idx": 0}

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Select an accelerator design.

        This evaluates the design using AccelForge and propagates
        constraints to Level 1 and Level 2.
        """
        from ..hardware.accelforge_bridge import AccelForgeBridge

        # Decode action to design point
        self.current_design = self.action_space_helper.decode_flat(action)

        # Evaluate with AccelForge
        bridge = AccelForgeBridge()
        workload = None
        try:
            workload = json.loads(self.target_workload_json)
        except (json.JSONDecodeError, TypeError):
            pass

        self.current_metrics = bridge.evaluate(self.current_design, workload)

        # Get constraints for Level 1
        constraints = self._compute_constraints()

        self.episode_idx += 1
        done = self.episode_idx >= self.max_episodes

        # Immediate reward based on design quality
        reward = self._compute_design_reward()

        obs = self._get_observation()

        info = {
            "design": self.current_design.to_dict(),
            "metrics": self.current_metrics.to_dict(),
            "constraints": constraints.to_dict(),
            "episode_idx": self.episode_idx,
        }

        # Track Pareto front (with kernel characteristics if available)
        kc_dict = None
        if self.kernel_feedback is not None:
            kc_dict = self.kernel_feedback.to_dict()
        pareto_point = ParetoPoint(
            design=self.current_design.to_dict(),
            latency_ms=self.current_metrics.latency_ms,
            energy_pj=self.current_metrics.energy_per_op_pj,
            area_mm2=self.current_metrics.area_mm2,
            power_mw=self.current_metrics.total_power_mw,
            reward=reward,
            kernel_characteristics=kc_dict,
        )
        was_added = self.pareto_tracker.add(pareto_point)
        info["pareto_added"] = was_added
        info["pareto_size"] = self.pareto_tracker.size()

        # Include design adjustment suggestions from bottom-up feedback
        if self.kernel_feedback is not None:
            info["design_suggestions"] = self.kernel_feedback.suggest_design_adjustments()

        return obs, reward, done, False, info

    def set_level1_result(self, result: Dict[str, Any]):
        """
        Receive result from Level 1+2 search under this design.

        Now also accepts kernel_characteristics for bottom-up feedback.
        """
        self.episode_results.append(result)

        # Extract kernel characteristics for bottom-up feedback
        kc_dict = result.get("kernel_characteristics")
        if kc_dict is not None:
            kc = KernelCharacteristics.from_dict(kc_dict)
            self.kernel_feedback = kc
            self.kernel_feedback_history.append(kc)

    def get_design_reward(self) -> float:
        """Compute final reward incorporating Level 1+2 results."""
        design_reward = self._compute_design_reward()

        if not self.episode_results:
            return design_reward

        # Best kernel result under this design
        valid_results = [r for r in self.episode_results if r.get("verified", False)]
        if not valid_results:
            return design_reward - 0.5

        best = min(valid_results, key=lambda r: r.get("latency_ms", float("inf")))
        kernel_latency = best.get("latency_ms", float("inf"))

        if kernel_latency < float("inf"):
            design_reward += 0.5 * np.log(10.0 / kernel_latency + 1)

        return design_reward

    def _compute_design_reward(self) -> float:
        """Compute reward for the current design point."""
        if self.current_metrics is None:
            return -1.0

        m = self.current_metrics
        reward = 0.0

        # Latency reward
        if m.latency_ms > 0:
            reward += self.reward_weight_latency * np.log(10.0 / m.latency_ms + 1)

        # Energy efficiency reward
        if m.energy_per_op_pj > 0:
            reward += self.reward_weight_energy * np.log(10.0 / m.energy_per_op_pj + 1)

        # Area reward (prefer smaller)
        if m.area_mm2 > 0:
            reward += self.reward_weight_area * max(0.0, 1.0 - m.area_mm2 / self.area_budget_mm2)

        # Power reward (prefer lower)
        if m.total_power_mw > 0:
            reward += self.reward_weight_power * np.log(
                self.power_budget_mw / m.total_power_mw + 1
            )

        return reward

    def _compute_constraints(self) -> AcceleratorDesignConstraints:
        """Compute constraints for Level 1 from current design."""
        if self.current_design is None:
            return AcceleratorDesignConstraints()

        d = self.current_design

        # Map precision to compute multiplier
        from ..hardware.accelforge_bridge import PRECISION_OPS_PER_PE

        compute_mult = PRECISION_OPS_PER_PE.get(d.data_precision, 2.0)

        # Map dataflow to preferences
        supports_weight_reuse = d.dataflow in ("weight_stationary", "row_stationary")
        supports_output_reuse = d.dataflow in ("output_stationary", "row_stationary")

        # Max tile size from buffer (precision-aware)
        bytes_per_element = {"int8": 1, "fp16": 2, "bf16": 2, "fp32": 4}
        bpe = bytes_per_element.get(d.data_precision, 2)
        max_tile = int(np.sqrt(d.l1_buffer_kb * 1024 / bpe))

        return AcceleratorDesignConstraints(
            max_parallelism=d.total_pes,
            pe_array_rows=d.pe_array_rows,
            pe_array_cols=d.pe_array_cols,
            max_shared_memory_kb=d.l1_buffer_kb,
            max_l2_cache_kb=d.l2_buffer_kb,
            max_tile_size=max_tile,
            supports_weight_reuse=supports_weight_reuse,
            supports_output_reuse=supports_output_reuse,
            supported_precisions=[d.data_precision],
            compute_multiplier=compute_mult,
            peak_tops=self.current_metrics.peak_tops if self.current_metrics else 0.0,
        )

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get Level 0 observation."""
        workload_features = self.obs_space_helper.encode_workload(self.target_workload_json)

        if self.current_design:
            design_features = self.obs_space_helper.encode_design(self.current_design)
        else:
            design_features = np.zeros(
                self.obs_space_helper.DESIGN_DIM, dtype=np.float32
            )

        if self.current_metrics:
            metrics_features = self.obs_space_helper.encode_metrics(self.current_metrics)
        else:
            metrics_features = np.zeros(
                self.obs_space_helper.METRICS_DIM, dtype=np.float32
            )

        # History features
        history = np.zeros(self.obs_space_helper.HISTORY_DIM, dtype=np.float32)
        if self.episode_results:
            latencies = [
                r.get("latency_ms", float("inf")) for r in self.episode_results
            ]
            valid_latencies = [
                lat for lat in latencies if lat < float("inf")
            ]
            if valid_latencies:
                history[0] = np.mean(valid_latencies) / 10.0
                history[1] = np.min(valid_latencies) / 10.0
            history[2] = len(self.episode_results) / 20.0
            history[3] = self.pareto_tracker.size() / 50.0

        # Kernel feedback features (bottom-up from Level 2)
        if self.kernel_feedback is not None:
            kernel_feedback_features = self.kernel_feedback.encode()
        else:
            kernel_feedback_features = np.zeros(
                self.obs_space_helper.KERNEL_FEEDBACK_DIM, dtype=np.float32
            )

        return {
            "workload_features": workload_features,
            "design_features": design_features,
            "metrics_features": metrics_features,
            "history_features": history,
            "kernel_feedback": kernel_feedback_features,
        }

    def get_current_constraints(self) -> Optional[AcceleratorDesignConstraints]:
        """Get constraints from current design for Level 1."""
        if self.current_design is None:
            return None
        return self._compute_constraints()

    def get_pareto_front(self) -> List[Dict[str, Any]]:
        """Get the current Pareto front."""
        return self.pareto_tracker.to_dict_list()
