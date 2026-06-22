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
Level 1: Hardware Configuration Space

Defines the configuration parameters that control the µGraph search space.
Configuration choices constrain what operations are valid in Level 2.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import json

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
        # No gym available, create stub for type hints
        GYM_AVAILABLE = False

        class StubSpace:
            """Stub space class for when gym is not available."""

            def __init__(self, *args, **kwargs):
                pass

            def sample(self):
                return None

            def contains(self, x):
                return True

        class StubSpaces:
            """Stub spaces module for when gym is not available."""

            Box = StubSpace
            Discrete = StubSpace
            MultiDiscrete = StubSpace
            MultiBinary = StubSpace
            Dict = StubSpace
            Tuple = StubSpace

        class StubGym:
            """Stub gym module for when gym is not available."""

            Env = object

        spaces = StubSpaces()
        gym = StubGym()


class NumpyMultiDiscreteSpace:
    """MultiDiscrete-like space using only NumPy (when gym is not installed)."""

    __slots__ = ("nvec",)

    def __init__(self, nvec: List[int]):
        self.nvec = [int(n) for n in nvec]

    def sample(self) -> np.ndarray:
        return np.array([int(np.random.randint(0, n)) for n in self.nvec], dtype=np.int64)


# Configuration choice options
GRID_DIM_CHOICES = [1, 2, 4, 8, 16, 32, 64, 128]
BLOCK_DIM_CHOICES = [32, 64, 128, 256, 512, 1024]
FORLOOP_RANGE_CHOICES = [1, 2, 4, 8, 16, 32, 64]
REDUCTION_DIMX_CHOICES = [1, 2, 4, 8, 16, 32]
SHARED_MEM_TIERS = [16384, 32768, 49152, 65536]  # 16KB, 32KB, 48KB, 64KB
REGISTER_TIERS = [32, 64, 128, 255]  # registers per thread

# Hardware observation normalization constants for AccelForge design fields.
MAX_PE_ARRAY_LOG2 = 7.0  # log2(128), maximum PE rows/cols in the design space
MAX_L1_BUFFER_LOG2 = 12.0  # log2(4096 KB), generous upper bound for L1 buffer
MAX_L2_BUFFER_LOG2 = 16.0  # log2(65536 KB), generous upper bound for L2 buffer

# All possible imap values
ALL_IMAPS = [(ix, iy, iz) for ix in [-1, 0, 1] for iy in [-1, 0, 1] for iz in [-1, 0, 1]]


@dataclass
class HardwareConfig:
    """
    Hardware execution configuration.

    These parameters determine:
    1. How work is distributed across GPU (grid_dim)
    2. How work is parallelized within a block (block_dim)
    3. Memory and compute constraints

    This configuration CONTROLS the Level 2 search space.
    """

    # Grid dimensions (number of blocks)
    grid_dim_x: int = 1
    grid_dim_y: int = 1
    grid_dim_z: int = 1

    # Block dimensions (threads per block)
    block_dim_x: int = 128
    block_dim_y: int = 1
    block_dim_z: int = 1

    # Forloop configuration
    forloop_range: int = 1
    reduction_dimx: int = 1

    # Memory configuration
    shared_memory_size: int = 49152  # 48KB default
    num_registers: int = 64

    @property
    def grid_dim(self) -> Tuple[int, int, int]:
        return (self.grid_dim_x, self.grid_dim_y, self.grid_dim_z)

    @property
    def block_dim(self) -> Tuple[int, int, int]:
        return (self.block_dim_x, self.block_dim_y, self.block_dim_z)

    @property
    def total_threads(self) -> int:
        return self.block_dim_x * self.block_dim_y * self.block_dim_z

    @property
    def total_blocks(self) -> int:
        return self.grid_dim_x * self.grid_dim_y * self.grid_dim_z

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grid_dim": {"x": self.grid_dim_x, "y": self.grid_dim_y, "z": self.grid_dim_z},
            "block_dim": {"x": self.block_dim_x, "y": self.block_dim_y, "z": self.block_dim_z},
            "forloop_range": self.forloop_range,
            "reduction_dimx": self.reduction_dimx,
            "shared_memory_size": self.shared_memory_size,
            "num_registers": self.num_registers,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HardwareConfig":
        grid = d.get("grid_dim", {})
        block = d.get("block_dim", {})
        return cls(
            grid_dim_x=grid.get("x", 1),
            grid_dim_y=grid.get("y", 1),
            grid_dim_z=grid.get("z", 1),
            block_dim_x=block.get("x", 128),
            block_dim_y=block.get("y", 1),
            block_dim_z=block.get("z", 1),
            forloop_range=d.get("forloop_range", 1),
            reduction_dimx=d.get("reduction_dimx", 1),
            shared_memory_size=d.get("shared_memory_size", 49152),
            num_registers=d.get("num_registers", 64),
        )


class SearchSpaceConstraints:
    """
    Computes search space constraints from hardware configuration.

    The Level 1 configuration CONSTRAINS what Level 2 can do:
    - Valid imap choices depend on grid_dim
    - Valid frange choices depend on forloop_range
    - Max operators depend on memory/register budget
    - Tensor tiling depends on grid/block dims
    """

    def __init__(self, config: HardwareConfig):
        self.config = config

        # Pre-compute constraints
        self._valid_imaps = self._compute_valid_imaps()
        self._valid_franges = self._compute_valid_franges()
        self._max_operators = self._compute_max_operators()
        # Conservative default before backend precision information is applied.
        # Use 2 bytes because fp16/bf16 are the common tensor-search precisions;
        # this undercounts int8 capacity but avoids over-admitting tiles until
        # AccelForge precision metadata is applied via apply_hardware_profile().
        self.element_size_bytes = 2
        self.max_tensor_elements = config.shared_memory_size // self.element_size_bytes
        self.warp_size = config.reduction_dimx
        self.max_shared_memory = config.shared_memory_size
        self.max_tile_size: Optional[int] = None
        self.supported_precisions: List[str] = []
        self.supports_weight_reuse: Optional[bool] = None
        self.supports_output_reuse: Optional[bool] = None

    def _compute_valid_imaps(self) -> List[Tuple[int, int, int]]:
        """
        Compute valid input mappings based on grid_dim.

        imap[i] ∈ {-1, 0, 1}:
          -1: not mapped to any grid dimension
           0: mapped to blockIdx.x
           1: mapped to blockIdx.y
           2: mapped to blockIdx.z (if using iz=2)

        A mapping is only meaningful if the corresponding grid dimension > 1.
        """
        valid = []

        for imap in ALL_IMAPS:
            is_valid = True

            # Check each component
            for dim_idx, map_val in enumerate(imap):
                if map_val == 0 and self.config.grid_dim_x <= 1:
                    is_valid = False
                    break
                if map_val == 1 and self.config.grid_dim_y <= 1:
                    is_valid = False
                    break
                # Note: iz=2 would map to z, but we use iz ∈ {-1,0,1}

            if is_valid:
                valid.append(imap)

        # Always include (-1, -1, -1) as "no mapping"
        if (-1, -1, -1) not in valid:
            valid.append((-1, -1, -1))

        return valid

    def _compute_valid_franges(self) -> List[int]:
        """
        Compute valid forloop range values.

        frange must divide forloop_range evenly.
        """
        fr = self.config.forloop_range
        valid = []

        for f in FORLOOP_RANGE_CHOICES:
            if fr >= f and fr % f == 0:
                valid.append(f)

        if not valid:
            valid = [1]

        return valid

    def _compute_max_operators(self) -> int:
        """
        Compute maximum operators based on resource constraints.
        """
        # Rough estimate: each operator uses some shared memory and registers
        sm_per_op = 2048  # 2KB per operator (rough)
        reg_per_op = 16  # registers per operator (rough)

        sm_limit = self.config.shared_memory_size // sm_per_op
        reg_limit = (self.config.num_registers * self.config.total_threads) // reg_per_op

        return min(sm_limit, reg_limit, 30)  # Hard cap at 30

    @property
    def valid_imaps(self) -> List[Tuple[int, int, int]]:
        return self._valid_imaps

    @property
    def valid_franges(self) -> List[int]:
        return self._valid_franges

    @property
    def max_operators(self) -> int:
        return self._max_operators

    @max_operators.setter
    def max_operators(self, value: int):
        self._max_operators = max(1, value)

    def apply_hardware_profile(self, hardware_profile: Optional[Any]) -> "SearchSpaceConstraints":
        """Tighten constraints using a unified HardwareProfile."""
        if hardware_profile is None:
            return self

        self.warp_size = getattr(hardware_profile, "warp_size", self.warp_size)
        max_smem = int(getattr(hardware_profile, "max_shared_memory_per_block", 0) or 0)
        if max_smem > 0:
            self.max_shared_memory = min(self.max_shared_memory, max_smem)
            self.max_tensor_elements = min(self.max_tensor_elements, self.max_shared_memory // 2)

        if getattr(hardware_profile, "backend", "") != "accelforge":
            return self

        extensions = getattr(hardware_profile, "extensions", {}) or {}
        af_design = extensions.get("accelforge_design", {})
        if not af_design:
            return self

        pe_rows = int(af_design.get("pe_array_rows", 0) or 0)
        pe_cols = int(af_design.get("pe_array_cols", 0) or 0)
        if pe_rows > 0 and pe_cols > 0:
            self.max_operators = min(self.max_operators, pe_rows * pe_cols)

        l1_kb = float(af_design.get("l1_buffer_kb", 0) or 0)
        if l1_kb > 0:
            smem_bytes = int(l1_kb * 1024)
            self.max_shared_memory = min(self.max_shared_memory, smem_bytes)
            self.max_tensor_elements = min(self.max_tensor_elements, smem_bytes // 2)

        dataflow = af_design.get("dataflow", "")
        if dataflow:
            self.supports_weight_reuse = dataflow in ("weight_stationary", "row_stationary")
            self.supports_output_reuse = dataflow in ("output_stationary", "row_stationary")

        precision = af_design.get("data_precision", "")
        if precision:
            self.supported_precisions = [precision]

        return self

    def apply_accelerator_constraints(
        self,
        accelerator_constraints: Optional[Any],
    ) -> "SearchSpaceConstraints":
        """Tighten constraints using Level 0 AccelForge accelerator constraints."""
        if accelerator_constraints is None:
            return self

        if isinstance(accelerator_constraints, dict):
            get_value = accelerator_constraints.get
        else:
            def get_value(name: str, default: Any = None) -> Any:
                return getattr(accelerator_constraints, name, default)

        max_parallelism = get_value("max_parallelism")
        if max_parallelism is not None:
            self.max_operators = min(self.max_operators, int(max_parallelism))

        max_shared_memory_kb = get_value("max_shared_memory_kb")
        if max_shared_memory_kb is not None:
            smem_bytes = int(float(max_shared_memory_kb) * 1024)
            self.max_shared_memory = min(self.max_shared_memory, smem_bytes)
            self.max_tensor_elements = min(self.max_tensor_elements, smem_bytes // 2)

        max_tile_size = get_value("max_tile_size")
        if max_tile_size is not None:
            self.max_tile_size = int(max_tile_size)

        supported_precisions = get_value("supported_precisions")
        if supported_precisions is not None:
            self.supported_precisions = list(supported_precisions)

        supports_weight_reuse = get_value("supports_weight_reuse")
        if supports_weight_reuse is not None:
            self.supports_weight_reuse = bool(supports_weight_reuse)

        supports_output_reuse = get_value("supports_output_reuse")
        if supports_output_reuse is not None:
            self.supports_output_reuse = bool(supports_output_reuse)

        return self

    def get_imap_mask(self) -> np.ndarray:
        """
        Return binary mask indicating valid imaps.
        """
        mask = np.zeros(len(ALL_IMAPS), dtype=np.int8)
        for i, imap in enumerate(ALL_IMAPS):
            if imap in self._valid_imaps:
                mask[i] = 1
        return mask

    def get_frange_mask(self) -> np.ndarray:
        """
        Return binary mask indicating valid franges.
        """
        mask = np.zeros(len(FORLOOP_RANGE_CHOICES), dtype=np.int8)
        for i, fr in enumerate(FORLOOP_RANGE_CHOICES):
            if fr in self._valid_franges:
                mask[i] = 1
        return mask

    def get_tensor_tile_sizes(self, tensor_shape: List[int]) -> List[List[int]]:
        """
        Compute valid tile sizes for a tensor based on config.

        Tensor can be tiled along:
        - Grid dimensions (distributed across blocks)
        - Block dimensions (computed within a block)
        """
        if not tensor_shape:
            return [[1]]

        tiles = []

        for dim_size in tensor_shape:
            dim_tiles = [dim_size]  # No tiling

            # Tiling by grid dimensions
            for gd in [self.config.grid_dim_x, self.config.grid_dim_y, self.config.grid_dim_z]:
                if gd > 1 and dim_size % gd == 0:
                    tile = dim_size // gd
                    if tile not in dim_tiles:
                        dim_tiles.append(tile)

            # Tiling by block dimensions
            for bd in [self.config.block_dim_x, self.config.block_dim_y]:
                if bd > 1 and dim_size % bd == 0:
                    tile = dim_size // bd
                    if tile not in dim_tiles:
                        dim_tiles.append(tile)

            tiles.append(sorted(dim_tiles, reverse=True))

        return tiles

    def encode(self) -> np.ndarray:
        """
        Encode constraints as feature vector for observation.
        """
        features = np.zeros(32, dtype=np.float32)

        # Grid dim features (normalized)
        features[0] = self.config.grid_dim_x / 128.0
        features[1] = self.config.grid_dim_y / 128.0
        features[2] = self.config.grid_dim_z / 128.0

        # Block dim features
        features[3] = self.config.block_dim_x / 1024.0
        features[4] = self.config.block_dim_y / 32.0
        features[5] = self.config.block_dim_z / 32.0

        # Other config
        features[6] = self.config.forloop_range / 64.0
        features[7] = self.config.reduction_dimx / 32.0

        # Derived constraints
        features[8] = len(self._valid_imaps) / len(ALL_IMAPS)
        features[9] = len(self._valid_franges) / len(FORLOOP_RANGE_CHOICES)
        features[10] = self._max_operators / 30.0

        # Resource utilization
        features[11] = self.config.shared_memory_size / 65536.0
        features[12] = self.config.num_registers / 255.0
        features[13] = self.config.total_threads / 1024.0
        features[14] = self.config.total_blocks / (128 * 128)

        return features


class ConfigActionSpace:
    """
    Level 1 Action Space: Hardware configuration selection.
    """

    def __init__(self):
        flat_nvec = [
            len(GRID_DIM_CHOICES),
            len(GRID_DIM_CHOICES),
            len(GRID_DIM_CHOICES),
            len(BLOCK_DIM_CHOICES),
            len(BLOCK_DIM_CHOICES),
            len(FORLOOP_RANGE_CHOICES),
            len(REDUCTION_DIMX_CHOICES),
            len(SHARED_MEM_TIERS),
            len(REGISTER_TIERS),
        ]
        self.space = spaces.Dict(
            {
                # Grid dimensions
                "grid_x": spaces.Discrete(len(GRID_DIM_CHOICES)),
                "grid_y": spaces.Discrete(len(GRID_DIM_CHOICES)),
                "grid_z": spaces.Discrete(len(GRID_DIM_CHOICES)),
                # Block dimensions
                "block_x": spaces.Discrete(len(BLOCK_DIM_CHOICES)),
                "block_y": spaces.Discrete(len(BLOCK_DIM_CHOICES)),
                # Forloop
                "forloop_range": spaces.Discrete(len(FORLOOP_RANGE_CHOICES)),
                "reduction_dimx": spaces.Discrete(len(REDUCTION_DIMX_CHOICES)),
                # Memory (tiers)
                "shared_mem_tier": spaces.Discrete(len(SHARED_MEM_TIERS)),
                "register_tier": spaces.Discrete(len(REGISTER_TIERS)),
            }
        )

        if GYM_AVAILABLE:
            self.flat_space = spaces.MultiDiscrete(flat_nvec)
        else:
            self.flat_space = NumpyMultiDiscreteSpace(flat_nvec)

    def sample(self) -> Dict[str, int]:
        return self.space.sample()

    def decode(self, action: Dict[str, int]) -> HardwareConfig:
        """Decode action indices to HardwareConfig."""
        return HardwareConfig(
            grid_dim_x=GRID_DIM_CHOICES[action["grid_x"] % len(GRID_DIM_CHOICES)],
            grid_dim_y=GRID_DIM_CHOICES[action["grid_y"] % len(GRID_DIM_CHOICES)],
            grid_dim_z=GRID_DIM_CHOICES[action["grid_z"] % len(GRID_DIM_CHOICES)],
            block_dim_x=BLOCK_DIM_CHOICES[action["block_x"] % len(BLOCK_DIM_CHOICES)],
            block_dim_y=BLOCK_DIM_CHOICES[action["block_y"] % len(BLOCK_DIM_CHOICES)],
            block_dim_z=1,  # Usually 1
            forloop_range=FORLOOP_RANGE_CHOICES[
                action["forloop_range"] % len(FORLOOP_RANGE_CHOICES)
            ],
            reduction_dimx=REDUCTION_DIMX_CHOICES[
                action["reduction_dimx"] % len(REDUCTION_DIMX_CHOICES)
            ],
            shared_memory_size=SHARED_MEM_TIERS[action["shared_mem_tier"] % len(SHARED_MEM_TIERS)],
            num_registers=REGISTER_TIERS[action["register_tier"] % len(REGISTER_TIERS)],
        )

    def decode_flat(self, action: np.ndarray) -> HardwareConfig:
        """Decode flattened action array."""
        action_dict = {
            "grid_x": int(action[0]),
            "grid_y": int(action[1]),
            "grid_z": int(action[2]),
            "block_x": int(action[3]),
            "block_y": int(action[4]),
            "forloop_range": int(action[5]),
            "reduction_dimx": int(action[6]),
            "shared_mem_tier": int(action[7]),
            "register_tier": int(action[8]),
        }
        return self.decode(action_dict)


class ConfigObservationSpace:
    """
    Level 1 Observation Space: Target graph features + hardware capabilities.
    """

    GRAPH_FEATURE_DIM = 64
    HARDWARE_FEATURE_DIM = 16
    HISTORY_FEATURE_DIM = 32
    HARDWARE_BACKEND_IDX = 0
    ACCELFORGE_PE_ROWS_IDX = 10
    ACCELFORGE_PE_COLS_IDX = 11
    ACCELFORGE_L1_BUFFER_IDX = 12
    ACCELFORGE_L2_BUFFER_IDX = 13
    ACCELFORGE_DATAFLOW_IDX = 14
    ACCELFORGE_PRECISION_IDX = 15
    ACCELFORGE_DATAFLOW_ENCODING = {
        "output_stationary": 0.25,
        "weight_stationary": 0.5,
        "row_stationary": 0.75,
    }
    ACCELFORGE_PRECISION_ENCODING = {
        "int8": 0.25,
        "fp16": 0.5,
        "bf16": 0.75,
        "fp32": 1.0,
    }

    def __init__(self):
        self.space = spaces.Dict(
            {
                # Target computation graph features
                "target_graph_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.GRAPH_FEATURE_DIM,),
                    dtype=np.float32,
                ),
                # Hardware capability features
                "hardware_features": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.HARDWARE_FEATURE_DIM,),
                    dtype=np.float32,
                ),
                # History of previous configs and their results
                "history_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.HISTORY_FEATURE_DIM,),
                    dtype=np.float32,
                ),
            }
        )

    def encode_target_graph(self, graph_json: str) -> np.ndarray:
        """Extract features from target computation graph."""
        features = np.zeros(self.GRAPH_FEATURE_DIM, dtype=np.float32)

        try:
            graph = json.loads(graph_json)

            # Input tensor features
            inputs = graph.get("inputs", [])
            features[0] = len(inputs) / 10.0

            if inputs:
                # Shape statistics
                shapes = [inp.get("dims", [1]) for inp in inputs]
                all_dims = [d for s in shapes for d in s]
                if all_dims:
                    features[1] = np.mean(all_dims) / 4096.0
                    features[2] = np.max(all_dims) / 4096.0
                    features[3] = np.std(all_dims) / 4096.0 if len(all_dims) > 1 else 0

            # Operator features
            ops = graph.get("operators", [])
            features[4] = len(ops) / 20.0

            # Operator type distribution
            op_types = {}
            for op in ops:
                op_type = op.get("type", "unknown")
                op_types[op_type] = op_types.get(op_type, 0) + 1

            features[5] = op_types.get("matmul", 0) / 10.0
            features[6] = op_types.get("add", 0) / 10.0
            features[7] = op_types.get("reduction", 0) / 10.0
            features[8] = op_types.get("softmax", 0) / 10.0

        except:
            pass

        return features

    def encode_hardware(
        self,
        backend: str = "cuda",
        hardware_profile: Optional[Any] = None,
    ) -> np.ndarray:
        """Encode hardware capabilities."""
        features = np.zeros(self.HARDWARE_FEATURE_DIM, dtype=np.float32)

        if hardware_profile is not None:
            backends = ["cuda", "maca", "ascend", "cpu", "mps", "accelforge"]
            profile_backend = getattr(hardware_profile, "backend", backend)
            if profile_backend in backends:
                features[self.HARDWARE_BACKEND_IDX] = backends.index(profile_backend) / max(
                    len(backends) - 1, 1
                )

            features[1] = 1.0 if getattr(hardware_profile, "supports_tensor_cores", False) else 0.0
            features[2] = min(float(getattr(hardware_profile, "total_cores", 1)) / 4096.0, 1.0)
            features[3] = min(float(getattr(hardware_profile, "shared_memory_kb", 0)) / 256.0, 1.0)
            features[4] = min(float(getattr(hardware_profile, "warp_size", 1)) / 128.0, 1.0)
            compute_capability = getattr(hardware_profile, "compute_capability", (0, 0))
            features[5] = min(float(compute_capability[0]) / 10.0, 1.0)
            features[6] = min(
                float(getattr(hardware_profile, "memory_bandwidth_gbps", 0)) / 5000.0,
                1.0,
            )
            features[7] = min(float(getattr(hardware_profile, "peak_tflops_fp16", 0)) / 1500.0, 1.0)
            features[8] = min(float(getattr(hardware_profile, "peak_tflops_fp32", 0)) / 100.0, 1.0)
            features[9] = min(float(getattr(hardware_profile, "global_memory_gb", 0)) / 128.0, 1.0)

            extensions = getattr(hardware_profile, "extensions", {}) or {}
            af_design = extensions.get("accelforge_design", {})
            if af_design:
                features[self.ACCELFORGE_PE_ROWS_IDX] = min(
                    np.log2(max(af_design.get("pe_array_rows", 1), 1)) / MAX_PE_ARRAY_LOG2,
                    1.0,
                )
                features[self.ACCELFORGE_PE_COLS_IDX] = min(
                    np.log2(max(af_design.get("pe_array_cols", 1), 1)) / MAX_PE_ARRAY_LOG2,
                    1.0,
                )
                features[self.ACCELFORGE_L1_BUFFER_IDX] = min(
                    np.log2(max(af_design.get("l1_buffer_kb", 1), 1)) / MAX_L1_BUFFER_LOG2,
                    1.0,
                )
                features[self.ACCELFORGE_L2_BUFFER_IDX] = min(
                    np.log2(max(af_design.get("l2_buffer_kb", 1), 1)) / MAX_L2_BUFFER_LOG2,
                    1.0,
                )
                features[self.ACCELFORGE_DATAFLOW_IDX] = self.ACCELFORGE_DATAFLOW_ENCODING.get(
                    af_design.get("dataflow", ""), 0.0
                )
                features[self.ACCELFORGE_PRECISION_IDX] = self.ACCELFORGE_PRECISION_ENCODING.get(
                    af_design.get("data_precision", ""), 0.0
                )

            return features

        # Backend type
        backends = ["cuda", "maca", "ascend", "cpu"]
        if backend in backends:
            features[0] = backends.index(backend) / len(backends)

        # Hardware parameters (could be queried dynamically)
        if backend == "cuda":
            features[1] = 1.0  # Has tensor cores
            features[2] = 128 / 256  # SM count (normalized)
            features[3] = 48 / 64  # Max shared memory (KB)
            features[4] = 32 / 64  # Warp size
            features[5] = 80 / 100  # Compute capability
        elif backend == "maca":
            features[1] = 1.0
            features[2] = 64 / 256
            features[3] = 48 / 64
            features[4] = 64 / 64  # MACA warp = 64
            features[5] = 0.8

        return features
