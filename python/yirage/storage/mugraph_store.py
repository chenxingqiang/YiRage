"""
MuGraph Persistent Storage System for Training

Provides comprehensive storage for optimized muGraphs with rich metadata
to support subsequent model training (RL, supervised learning, etc.).

Directory Structure:
    ~/.yirage/mugraphs/
    ├── mps/
    │   ├── <graph_hash>_<config_hash>.json     # Full entry
    │   └── ...
    ├── cuda/
    ├── cpu/
    ├── training_data/                          # Aggregated training datasets
    │   ├── features.jsonl                      # Feature vectors
    │   ├── labels.jsonl                        # Performance labels
    │   └── trajectories.jsonl                  # Search trajectories
    └── index.json
"""

import os
import json
import hashlib
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# Default storage root
DEFAULT_STORE_ROOT = os.path.expanduser("~/.yirage/mugraphs")


def normalize_input_shapes(shapes: Optional[Union[List, Tuple]]) -> List[List[int]]:
    """Normalize runtime or stored input shape lists for equality checks."""
    if not shapes:
        return []
    out: List[List[int]] = []
    for shape in shapes:
        if shape is None:
            continue
        out.append([int(x) for x in shape])
    return out


def input_shapes_match(
    stored: Optional[Union[List, Tuple]],
    requested: Optional[Union[List, Tuple]],
) -> bool:
    """Return True when every input tensor shape matches exactly."""
    a = normalize_input_shapes(stored)
    b = normalize_input_shapes(requested)
    return bool(a) and a == b


def entry_latency_ms(entry: "MuGraphEntry") -> float:
    """Best-effort latency for ranking cached muGraph entries."""
    if entry.metadata.latency_ms > 0:
        return entry.metadata.latency_ms
    if entry.performance.latency_ms > 0:
        return entry.performance.latency_ms
    return float("inf")


def mugraph_require_shape_match() -> bool:
    """When set, persistent restore refuses shape-mismatched cache entries."""
    return os.environ.get("YIRAGE_MUGraph_REQUIRE_SHAPE_MATCH", "0").strip() in (
        "1",
        "true",
        "yes",
    )


def mugraph_shape_bucket_enabled() -> bool:
    """When enabled, fall back to bucketed shape match before global best latency."""
    val = os.environ.get("YIRAGE_MUGraph_SHAPE_BUCKET", "1").strip().lower()
    if val in ("0", "false", "no", "off"):
        return False
    return True


def bucket_dim(size: int) -> int:
    """Round a dimension up to the next power-of-two bucket (min 8 when size > 1)."""
    n = int(size)
    if n <= 1:
        return n
    if n <= 8:
        return 8
    bucket = 1
    while bucket < n:
        bucket <<= 1
    return bucket


def bucket_input_shapes(shapes: Optional[Union[List, Tuple]]) -> List[List[int]]:
    """Bucket every dimension of every input tensor (runtime dynamism)."""
    normalized = normalize_input_shapes(shapes)
    return [[bucket_dim(d) for d in shape] for shape in normalized]


def input_shapes_bucket_match(
    stored: Optional[Union[List, Tuple]],
    requested: Optional[Union[List, Tuple]],
) -> bool:
    """True when bucketed signatures match (same bucket, different exact dims)."""
    a = bucket_input_shapes(stored)
    b = bucket_input_shapes(requested)
    return bool(a) and bool(b) and a == b


class OpType(str, Enum):
    """Operator types for graph analysis."""

    MATMUL = "matmul"
    CONV2D = "conv2d"
    REDUCTION = "reduction"
    ELEMENTWISE = "elementwise"
    ACTIVATION = "activation"
    NORMALIZATION = "normalization"
    ATTENTION = "attention"
    EMBEDDING = "embedding"
    SOFTMAX = "softmax"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    CONCAT = "concat"
    SPLIT = "split"
    CUSTOM = "custom"
    UNKNOWN = "unknown"


@dataclass
class OperatorInfo:
    """Detailed information about a single operator."""

    op_id: int = 0
    op_type: str = "unknown"
    op_name: str = ""

    # Input/Output info
    num_inputs: int = 0
    num_outputs: int = 0
    input_tensor_ids: List[int] = field(default_factory=list)
    output_tensor_ids: List[int] = field(default_factory=list)

    # Compute characteristics
    flops: int = 0
    mac_ops: int = 0  # Multiply-accumulate operations

    # Memory characteristics
    memory_read_bytes: int = 0
    memory_write_bytes: int = 0

    # Compute intensity (FLOPS / memory bytes)
    arithmetic_intensity: float = 0.0

    # Operator-specific parameters
    params: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "OperatorInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TensorInfo:
    """Detailed information about a tensor."""

    tensor_id: int = 0
    name: str = ""

    # Shape and layout
    dims: List[int] = field(default_factory=list)
    strides: List[int] = field(default_factory=list)
    dtype: str = "float16"
    dtype_size_bytes: int = 2

    # Memory
    size_bytes: int = 0
    memory_level: str = "global"  # global, shared, register

    # Role
    is_input: bool = False
    is_output: bool = False
    is_weight: bool = False
    is_intermediate: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TensorInfo":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GraphStructure:
    """Complete graph structure for training."""

    # Operators and tensors
    operators: List[OperatorInfo] = field(default_factory=list)
    tensors: List[TensorInfo] = field(default_factory=list)

    # Graph topology
    edges: List[Tuple[int, int]] = field(default_factory=list)  # (src_tensor, dst_op)
    adjacency_list: Dict[int, List[int]] = field(default_factory=dict)

    # Graph metrics
    num_operators: int = 0
    num_tensors: int = 0
    num_edges: int = 0
    graph_depth: int = 0  # Longest path
    graph_width: int = 0  # Max parallel ops
    critical_path_length: int = 0
    parallelism_degree: float = 0.0

    # Compute summary
    total_flops: int = 0
    total_mac_ops: int = 0
    total_memory_read_bytes: int = 0
    total_memory_write_bytes: int = 0
    avg_arithmetic_intensity: float = 0.0

    # Operator type distribution
    op_type_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "operators": [op.to_dict() for op in self.operators],
            "tensors": [t.to_dict() for t in self.tensors],
            "edges": self.edges,
            "adjacency_list": self.adjacency_list,
            "num_operators": self.num_operators,
            "num_tensors": self.num_tensors,
            "num_edges": self.num_edges,
            "graph_depth": self.graph_depth,
            "graph_width": self.graph_width,
            "critical_path_length": self.critical_path_length,
            "parallelism_degree": self.parallelism_degree,
            "total_flops": self.total_flops,
            "total_mac_ops": self.total_mac_ops,
            "total_memory_read_bytes": self.total_memory_read_bytes,
            "total_memory_write_bytes": self.total_memory_write_bytes,
            "avg_arithmetic_intensity": self.avg_arithmetic_intensity,
            "op_type_counts": self.op_type_counts,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GraphStructure":
        gs = cls()
        gs.operators = [OperatorInfo.from_dict(op) for op in data.get("operators", [])]
        gs.tensors = [TensorInfo.from_dict(t) for t in data.get("tensors", [])]
        for key in [
            "edges",
            "adjacency_list",
            "num_operators",
            "num_tensors",
            "num_edges",
            "graph_depth",
            "graph_width",
            "critical_path_length",
            "parallelism_degree",
            "total_flops",
            "total_mac_ops",
            "total_memory_read_bytes",
            "total_memory_write_bytes",
            "avg_arithmetic_intensity",
            "op_type_counts",
        ]:
            if key in data:
                setattr(gs, key, data[key])
        return gs


@dataclass
class DeviceCapabilities:
    """Hardware device capabilities for training context."""

    # Device identification
    device_type: str = ""  # mps, cuda, cpu, ascend, maca
    device_name: str = ""
    vendor: str = ""

    # Compute capabilities
    compute_units: int = 0  # SMs for CUDA, cores for CPU
    clock_mhz: int = 0
    theoretical_tflops_fp16: float = 0.0
    theoretical_tflops_fp32: float = 0.0

    # Memory capabilities
    memory_size_gb: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    memory_type: str = ""  # HBM, GDDR6, DDR5, etc.

    # Cache hierarchy
    l1_cache_kb: int = 0
    l2_cache_mb: float = 0.0
    shared_memory_kb: int = 0

    # Thread/warp configuration
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_blocks_per_sm: int = 16

    # Special features
    tensor_cores: bool = False
    fp16_acceleration: bool = False
    bf16_support: bool = False

    # Unique device ID for tracking
    device_uuid: str = ""

    # Driver/runtime info
    driver_version: str = ""
    runtime_version: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "DeviceCapabilities":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchConfiguration:
    """Complete search configuration used during optimization."""

    # Mapping configurations
    imaps: Optional[List] = None
    omaps: Optional[List] = None
    griddims: Optional[List] = None
    blockdims: Optional[List] = None
    fmaps: Optional[List] = None
    franges: Optional[List] = None

    # Selected configuration (the one that was chosen)
    selected_grid_dim: Tuple[int, int, int] = (1, 1, 1)
    selected_block_dim: Tuple[int, int, int] = (1, 1, 1)
    selected_forloop_range: int = 1
    selected_reduction_dimx: int = 1

    # Search space size
    total_search_space_size: int = 0
    explored_candidates: int = 0
    pruned_candidates: int = 0

    # Pipeline configuration (for Hopper+)
    pipeline_stages: int = 1
    num_warp_groups: int = 1

    # Resource usage
    shared_memory_bytes: int = 0
    register_count: int = 0
    occupancy: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SearchConfiguration":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PerformanceMetrics:
    """Comprehensive performance measurements."""

    # Primary metrics
    latency_ms: float = 0.0
    throughput_tflops: float = 0.0

    # Memory metrics
    memory_bandwidth_utilization: float = 0.0  # 0.0 - 1.0
    memory_read_gb: float = 0.0
    memory_write_gb: float = 0.0
    peak_memory_mb: float = 0.0

    # Compute metrics
    compute_utilization: float = 0.0  # 0.0 - 1.0
    achieved_tflops: float = 0.0
    theoretical_tflops: float = 0.0

    # Efficiency metrics
    arithmetic_intensity_achieved: float = 0.0
    roofline_efficiency: float = 0.0  # How close to roofline

    # Profiling iterations
    warmup_iterations: int = 16
    profile_iterations: int = 1000

    # Statistical measures
    latency_std_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Comparison metrics
    speedup_vs_baseline: float = 1.0
    baseline_latency_ms: float = 0.0
    baseline_name: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "PerformanceMetrics":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchTrajectory:
    """Search trajectory for RL training."""

    # Trajectory ID
    trajectory_id: str = ""

    # States visited (encoded)
    states: List[Dict] = field(default_factory=list)

    # Actions taken
    actions: List[Dict] = field(default_factory=list)

    # Rewards received
    rewards: List[float] = field(default_factory=list)

    # State-action-reward-next_state tuples
    transitions: List[Dict] = field(default_factory=list)

    # Episode info
    episode_length: int = 0
    total_reward: float = 0.0
    final_latency_ms: float = 0.0

    # Search statistics
    unique_states_visited: int = 0
    backtrack_count: int = 0
    improvement_steps: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "SearchTrajectory":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CandidateEvaluation:
    """Evaluation result for a candidate configuration."""

    candidate_id: int = 0

    # Configuration
    grid_dim: Tuple[int, int, int] = (1, 1, 1)
    block_dim: Tuple[int, int, int] = (1, 1, 1)
    forloop_range: int = 1

    # Performance
    latency_ms: float = 0.0
    is_valid: bool = True
    error_message: str = ""

    # Ranking
    rank: int = 0
    is_best: bool = False

    # Features for prediction
    features: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "CandidateEvaluation":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingFeatures:
    """Feature vectors extracted for model training."""

    # Graph-level features
    graph_features: Dict[str, float] = field(default_factory=dict)

    # Operator-level features (aggregated)
    op_features: Dict[str, float] = field(default_factory=dict)

    # Device-level features
    device_features: Dict[str, float] = field(default_factory=dict)

    # Configuration features
    config_features: Dict[str, float] = field(default_factory=dict)

    # Combined feature vector (for direct use)
    feature_vector: List[float] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)

    # Normalized features
    normalized_features: Dict[str, float] = field(default_factory=dict)

    # Embedding vectors (if using neural encoding)
    graph_embedding: List[float] = field(default_factory=list)
    config_embedding: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingFeatures":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingLabels:
    """Labels for supervised/RL training."""

    # Primary label: optimal latency
    optimal_latency_ms: float = 0.0

    # Ranking labels
    rank_among_candidates: int = 0
    percentile: float = 0.0  # 0-100

    # Binary labels
    is_optimal: bool = False
    is_top_k: bool = False  # Top K%
    is_valid: bool = True

    # Regression labels
    speedup_ratio: float = 1.0
    efficiency_score: float = 0.0  # 0.0 - 1.0

    # Multi-class labels
    performance_tier: int = 0  # 0=poor, 1=ok, 2=good, 3=excellent

    # RL-specific labels
    reward: float = 0.0
    cumulative_reward: float = 0.0
    advantage: float = 0.0
    value_target: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingLabels":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MuGraphMetadata:
    """Complete metadata for a stored muGraph - optimized for training."""

    # === Identifiers ===
    graph_hash: str = ""
    config_hash: str = ""
    entry_id: str = ""  # Unique entry ID

    # === Timestamps ===
    created_at: str = ""
    updated_at: str = ""

    # === Version Info ===
    yirage_version: str = ""
    storage_schema_version: str = "2.0"

    # === Backend ===
    backend: str = "mps"

    # === Quick Access Fields (for compatibility) ===
    latency_ms: float = 0.0
    num_candidates_searched: int = 0
    search_time_s: float = 0.0
    device_name: str = ""
    input_shapes: List[List[int]] = field(default_factory=list)
    output_shapes: List[List[int]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "MuGraphMetadata":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MuGraphEntry:
    """
    Complete muGraph entry with all data for training.

    This is the primary storage unit containing:
    - Metadata for identification
    - Graph structure for feature extraction
    - Device capabilities for context
    - Search configuration for reproducibility
    - Performance metrics for labels
    - Training features and labels (pre-computed)
    - Search trajectory for RL
    - All candidate evaluations for learning
    """

    # === Core Metadata ===
    metadata: MuGraphMetadata = field(default_factory=MuGraphMetadata)

    # === Graph Structure ===
    graph_structure: GraphStructure = field(default_factory=GraphStructure)

    # === Device Context ===
    device_capabilities: DeviceCapabilities = field(default_factory=DeviceCapabilities)

    # === Search Configuration ===
    search_config: SearchConfiguration = field(default_factory=SearchConfiguration)

    # === Performance Results ===
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # === Training Data ===
    features: TrainingFeatures = field(default_factory=TrainingFeatures)
    labels: TrainingLabels = field(default_factory=TrainingLabels)

    # === Search Trajectory (for RL) ===
    trajectory: Optional[SearchTrajectory] = None

    # === All Candidate Evaluations ===
    candidates: List[CandidateEvaluation] = field(default_factory=list)

    # === Serialized Graph (optional) ===
    graph_json: Optional[str] = None

    # === Kernel Artifacts ===
    kernel_source_code: Optional[str] = None
    kernel_path: Optional[str] = None

    # === Raw Data ===
    extra_data: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "graph_structure": self.graph_structure.to_dict(),
            "device_capabilities": self.device_capabilities.to_dict(),
            "search_config": self.search_config.to_dict(),
            "performance": self.performance.to_dict(),
            "features": self.features.to_dict(),
            "labels": self.labels.to_dict(),
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "candidates": [c.to_dict() for c in self.candidates],
            "graph_json": self.graph_json,
            "kernel_source_code": self.kernel_source_code,
            "kernel_path": self.kernel_path,
            "extra_data": self.extra_data,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MuGraphEntry":
        """Deserialize from dictionary."""
        entry = cls()

        if "metadata" in data:
            entry.metadata = MuGraphMetadata.from_dict(data["metadata"])
        if "graph_structure" in data:
            entry.graph_structure = GraphStructure.from_dict(data["graph_structure"])
        if "device_capabilities" in data:
            entry.device_capabilities = DeviceCapabilities.from_dict(data["device_capabilities"])
        if "search_config" in data:
            entry.search_config = SearchConfiguration.from_dict(data["search_config"])
        if "performance" in data:
            entry.performance = PerformanceMetrics.from_dict(data["performance"])
        if "features" in data:
            entry.features = TrainingFeatures.from_dict(data["features"])
        if "labels" in data:
            entry.labels = TrainingLabels.from_dict(data["labels"])
        if data.get("trajectory"):
            entry.trajectory = SearchTrajectory.from_dict(data["trajectory"])
        if "candidates" in data:
            entry.candidates = [CandidateEvaluation.from_dict(c) for c in data["candidates"]]

        entry.graph_json = data.get("graph_json")
        entry.kernel_source_code = data.get("kernel_source_code")
        entry.kernel_path = data.get("kernel_path")
        entry.extra_data = data.get("extra_data", {})

        return entry

    def compute_training_features(self):
        """Compute training features from graph and config."""
        features = {}

        # Graph features
        gs = self.graph_structure
        features["num_operators"] = gs.num_operators
        features["num_tensors"] = gs.num_tensors
        features["num_edges"] = gs.num_edges
        features["graph_depth"] = gs.graph_depth
        features["graph_width"] = gs.graph_width
        features["parallelism_degree"] = gs.parallelism_degree
        features["total_flops"] = gs.total_flops
        features["total_memory_bytes"] = gs.total_memory_read_bytes + gs.total_memory_write_bytes
        features["avg_arithmetic_intensity"] = gs.avg_arithmetic_intensity

        # Op type distribution
        for op_type, count in gs.op_type_counts.items():
            features[f"op_count_{op_type}"] = count

        # Device features
        dc = self.device_capabilities
        features["compute_units"] = dc.compute_units
        features["memory_bandwidth_gbps"] = dc.memory_bandwidth_gbps
        features["warp_size"] = dc.warp_size
        features["shared_memory_kb"] = dc.shared_memory_kb

        # Config features
        sc = self.search_config
        features["grid_dim_x"] = sc.selected_grid_dim[0]
        features["grid_dim_y"] = sc.selected_grid_dim[1]
        features["grid_dim_z"] = sc.selected_grid_dim[2]
        features["block_dim_x"] = sc.selected_block_dim[0]
        features["block_dim_y"] = sc.selected_block_dim[1]
        features["block_dim_z"] = sc.selected_block_dim[2]
        features["forloop_range"] = sc.selected_forloop_range
        features["occupancy"] = sc.occupancy
        features["shared_memory_bytes"] = sc.shared_memory_bytes

        self.features.graph_features = {
            k: v
            for k, v in features.items()
            if k.startswith(("num_", "graph_", "total_", "avg_", "op_count_"))
        }
        self.features.device_features = {
            k: v
            for k, v in features.items()
            if k in ["compute_units", "memory_bandwidth_gbps", "warp_size", "shared_memory_kb"]
        }
        self.features.config_features = {
            k: v
            for k, v in features.items()
            if k.startswith(("grid_", "block_", "forloop_", "occupancy", "shared_memory_bytes"))
        }

        # Build feature vector
        self.features.feature_names = sorted(features.keys())
        self.features.feature_vector = [
            float(features.get(k, 0)) for k in self.features.feature_names
        ]

    def compute_training_labels(self, all_latencies: List[float] = None):
        """Compute training labels from performance data."""
        perf = self.performance

        self.labels.optimal_latency_ms = perf.latency_ms
        self.labels.is_valid = perf.latency_ms > 0

        if all_latencies and perf.latency_ms > 0:
            sorted_latencies = sorted([l for l in all_latencies if l > 0])
            if sorted_latencies:
                rank = (
                    sorted_latencies.index(perf.latency_ms) + 1
                    if perf.latency_ms in sorted_latencies
                    else len(sorted_latencies)
                )
                self.labels.rank_among_candidates = rank
                self.labels.percentile = (1 - rank / len(sorted_latencies)) * 100
                self.labels.is_optimal = rank == 1
                self.labels.is_top_k = rank <= max(1, len(sorted_latencies) // 10)

                # Speedup vs worst
                worst = sorted_latencies[-1]
                self.labels.speedup_ratio = worst / perf.latency_ms if perf.latency_ms > 0 else 1.0

                # Efficiency score (normalized to 0-1)
                best = sorted_latencies[0]
                self.labels.efficiency_score = (
                    best / perf.latency_ms if perf.latency_ms > 0 else 0.0
                )

                # Performance tier
                if self.labels.percentile >= 90:
                    self.labels.performance_tier = 3  # Excellent
                elif self.labels.percentile >= 70:
                    self.labels.performance_tier = 2  # Good
                elif self.labels.percentile >= 40:
                    self.labels.performance_tier = 1  # OK
                else:
                    self.labels.performance_tier = 0  # Poor

                # RL reward (normalized)
                self.labels.reward = -perf.latency_ms / worst if worst > 0 else 0


class MuGraphStore:
    """
    Persistent storage manager for optimized muGraphs.

    Organizes muGraphs by device type (backend) in separate directories.
    Provides rich data storage for model training.
    """

    def __init__(self, root_path: Optional[str] = None):
        """Initialize the MuGraph store."""
        self.root_path = Path(root_path or DEFAULT_STORE_ROOT)
        self._ensure_directories()
        self._index: Dict[str, List[str]] = {}
        self._load_index()

    def _ensure_directories(self):
        """Create directory structure."""
        backends = ["mps", "cuda", "cpu", "ascend", "maca", "triton"]

        for backend in backends:
            (self.root_path / backend).mkdir(parents=True, exist_ok=True)

        # Training data directory
        (self.root_path / "training_data").mkdir(parents=True, exist_ok=True)

        logger.info(f"MuGraph store initialized at: {self.root_path}")

    def _get_backend_dir(self, backend: str) -> Path:
        return self.root_path / backend

    def _index_path(self) -> Path:
        return self.root_path / "index.json"

    def _load_index(self):
        """Load the global index."""
        index_path = self._index_path()
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self):
        """Save the global index."""
        try:
            with open(self._index_path(), "w") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save index: {e}")

    @staticmethod
    def _compute_config_hash(
        imaps: Optional[List] = None,
        omaps: Optional[List] = None,
        griddims: Optional[List] = None,
        blockdims: Optional[List] = None,
        fmaps: Optional[List] = None,
        franges: Optional[List] = None,
    ) -> str:
        """Compute hash for search configuration."""
        config_str = json.dumps(
            {
                "imaps": imaps,
                "omaps": omaps,
                "griddims": griddims,
                "blockdims": blockdims,
                "fmaps": fmaps,
                "franges": franges,
            },
            sort_keys=True,
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _get_entry_path(self, graph_hash: str, config_hash: str, backend: str) -> Path:
        return self._get_backend_dir(backend) / f"{graph_hash}_{config_hash}.json"

    def save(
        self,
        graph_hash: str,
        optimized_graph: Any,
        backend: str,
        # Search config
        imaps: Optional[List] = None,
        omaps: Optional[List] = None,
        griddims: Optional[List] = None,
        blockdims: Optional[List] = None,
        fmaps: Optional[List] = None,
        franges: Optional[List] = None,
        # Selected config
        selected_grid_dim: Tuple[int, int, int] = None,
        selected_block_dim: Tuple[int, int, int] = None,
        selected_forloop_range: int = 1,
        # Performance
        latency_ms: float = 0.0,
        latency_stats: Dict = None,  # min, max, std, p50, p95, p99
        memory_bytes: int = 0,
        # Search stats
        num_candidates_searched: int = 0,
        search_time_s: float = 0.0,
        # Graph info
        input_shapes: Optional[List] = None,
        output_shapes: Optional[List] = None,
        graph_structure: GraphStructure = None,
        operators: List[Dict] = None,
        tensors: List[Dict] = None,
        # Device info
        device_name: str = "",
        device_info: Optional[Dict] = None,
        device_capabilities: DeviceCapabilities = None,
        # Candidates
        candidate_evaluations: List[Dict] = None,
        all_latencies: List[float] = None,
        # Trajectory
        search_trajectory: Dict = None,
        # Kernel
        kernel_source_code: str = None,
        # Extra
        extra_data: Optional[Dict] = None,
    ) -> str:
        """
        Save an optimized muGraph with comprehensive training data.
        """
        config_hash = self._compute_config_hash(imaps, omaps, griddims, blockdims, fmaps, franges)

        now = datetime.now().isoformat()
        entry_id = f"{graph_hash}_{config_hash}_{int(time.time())}"

        # Build metadata
        metadata = MuGraphMetadata(
            graph_hash=graph_hash,
            config_hash=config_hash,
            entry_id=entry_id,
            created_at=now,
            updated_at=now,
            backend=backend,
            latency_ms=latency_ms,
            num_candidates_searched=num_candidates_searched,
            search_time_s=search_time_s,
            device_name=device_name,
            input_shapes=input_shapes or [],
            output_shapes=output_shapes or [],
        )

        # Get version
        try:
            import yirage

            metadata.yirage_version = getattr(yirage, "__version__", "")
        except:
            pass

        # Build search config
        search_config = SearchConfiguration(
            imaps=imaps,
            omaps=omaps,
            griddims=griddims,
            blockdims=blockdims,
            fmaps=fmaps,
            franges=franges,
            selected_grid_dim=selected_grid_dim or (1, 1, 1),
            selected_block_dim=selected_block_dim or (1, 1, 1),
            selected_forloop_range=selected_forloop_range,
            total_search_space_size=len(griddims or [])
            * len(blockdims or [])
            * len(franges or [1]),
            explored_candidates=num_candidates_searched,
        )

        # Build performance metrics
        performance = PerformanceMetrics(
            latency_ms=latency_ms,
            peak_memory_mb=memory_bytes / (1024 * 1024) if memory_bytes else 0,
        )
        if latency_stats:
            performance.latency_std_ms = latency_stats.get("std", 0)
            performance.latency_min_ms = latency_stats.get("min", 0)
            performance.latency_max_ms = latency_stats.get("max", 0)
            performance.latency_p50_ms = latency_stats.get("p50", 0)
            performance.latency_p95_ms = latency_stats.get("p95", 0)
            performance.latency_p99_ms = latency_stats.get("p99", 0)

        # Build device capabilities
        if device_capabilities is None:
            device_capabilities = DeviceCapabilities(
                device_type=backend,
                device_name=device_name,
            )
            if device_info:
                for k, v in device_info.items():
                    if hasattr(device_capabilities, k):
                        setattr(device_capabilities, k, v)

        # Build graph structure
        if graph_structure is None:
            graph_structure = GraphStructure(
                num_operators=len(operators) if operators else 0,
                num_tensors=len(tensors) if tensors else 0,
            )
            if operators:
                graph_structure.operators = [OperatorInfo.from_dict(op) for op in operators]
            if tensors:
                graph_structure.tensors = [TensorInfo.from_dict(t) for t in tensors]

        # Build candidates
        candidates = []
        if candidate_evaluations:
            for i, c in enumerate(candidate_evaluations):
                candidates.append(
                    CandidateEvaluation(
                        candidate_id=i,
                        latency_ms=c.get("latency_ms", 0),
                        is_valid=c.get("is_valid", True),
                        grid_dim=tuple(c.get("grid_dim", (1, 1, 1))),
                        block_dim=tuple(c.get("block_dim", (1, 1, 1))),
                        is_best=c.get("is_best", False),
                    )
                )

        # Build trajectory
        trajectory = None
        if search_trajectory:
            trajectory = SearchTrajectory.from_dict(search_trajectory)

        # Serialize graph (CyKNGraph -> JSON string for cross-session restore)
        from .graph_serde import serialize_optimized_graph

        graph_json = serialize_optimized_graph(optimized_graph)

        # Create entry
        entry = MuGraphEntry(
            metadata=metadata,
            graph_structure=graph_structure,
            device_capabilities=device_capabilities,
            search_config=search_config,
            performance=performance,
            trajectory=trajectory,
            candidates=candidates,
            graph_json=graph_json,
            kernel_source_code=kernel_source_code,
            extra_data=extra_data or {},
        )

        # Compute training features and labels
        entry.compute_training_features()
        entry.compute_training_labels(all_latencies)

        # Save to file
        entry_path = self._get_entry_path(graph_hash, config_hash, backend)
        with open(entry_path, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

        # Update index
        index_key = f"{backend}:{graph_hash}"
        if index_key not in self._index:
            self._index[index_key] = []
        if str(entry_path) not in self._index[index_key]:
            self._index[index_key].append(str(entry_path))
        self._save_index()

        logger.info(f"Saved muGraph to: {entry_path}")
        return str(entry_path)

    def find(
        self,
        graph_hash: str,
        backend: str,
        imaps: Optional[List] = None,
        omaps: Optional[List] = None,
        griddims: Optional[List] = None,
        blockdims: Optional[List] = None,
        fmaps: Optional[List] = None,
        franges: Optional[List] = None,
    ) -> Optional[MuGraphEntry]:
        """Find a cached muGraph by hash and configuration."""
        config_hash = self._compute_config_hash(imaps, omaps, griddims, blockdims, fmaps, franges)

        entry_path = self._get_entry_path(graph_hash, config_hash, backend)

        if entry_path.exists():
            try:
                with open(entry_path, "r") as f:
                    data = json.load(f)
                return MuGraphEntry.from_dict(data)
            except Exception as e:
                logger.warning(f"Failed to load entry: {e}")

        return None

    def find_all_for_graph(
        self,
        graph_hash: str,
        backend: Optional[str] = None,
    ) -> List[MuGraphEntry]:
        """Find all cached muGraphs for a given input graph."""
        entries = []
        backends = [backend] if backend else ["mps", "cuda", "cpu", "ascend", "maca", "triton"]

        for be in backends:
            backend_dir = self._get_backend_dir(be)
            if not backend_dir.exists():
                continue

            for entry_file in backend_dir.glob(f"{graph_hash}_*.json"):
                try:
                    with open(entry_file, "r") as f:
                        data = json.load(f)
                    entries.append(MuGraphEntry.from_dict(data))
                except Exception as e:
                    logger.warning(f"Failed to load {entry_file}: {e}")

        return entries

    def find_best(
        self,
        graph_hash: str,
        backend: str,
        input_shapes: Optional[List] = None,
        *,
        require_shape_match: Optional[bool] = None,
    ) -> Optional[MuGraphEntry]:
        """Find the best (lowest latency) muGraph.

        When ``input_shapes`` is provided, prefer entries profiled at the same
        input shapes (runtime dynamism). Then try power-of-two **shape buckets**
        when ``YIRAGE_MUGraph_SHAPE_BUCKET`` is enabled (default on). Falls back
        to global best latency unless ``require_shape_match`` is True
        (or ``YIRAGE_MUGraph_REQUIRE_SHAPE_MATCH=1``).
        """
        entries = self.find_all_for_graph(graph_hash, backend)

        if not entries:
            return None

        if require_shape_match is None:
            require_shape_match = mugraph_require_shape_match()

        if input_shapes is not None:
            normalized = normalize_input_shapes(input_shapes)
            if normalized:
                matched = [
                    e
                    for e in entries
                    if input_shapes_match(e.metadata.input_shapes, normalized)
                ]
                if matched:
                    matched.sort(key=entry_latency_ms)
                    return matched[0]

                if mugraph_shape_bucket_enabled() and not require_shape_match:
                    bucketed = [
                        e
                        for e in entries
                        if input_shapes_bucket_match(
                            e.metadata.input_shapes, normalized
                        )
                    ]
                    if bucketed:
                        bucketed.sort(key=entry_latency_ms)
                        return bucketed[0]

                if require_shape_match:
                    return None

        entries.sort(key=entry_latency_ms)
        return entries[0]

    def list_all(
        self,
        backend: Optional[str] = None,
        limit: int = 100,
    ) -> List[MuGraphEntry]:
        """List all stored muGraphs."""
        entries = []
        backends = [backend] if backend else ["mps", "cuda", "cpu", "ascend", "maca", "triton"]

        for be in backends:
            backend_dir = self._get_backend_dir(be)
            if not backend_dir.exists():
                continue

            for entry_file in backend_dir.glob("*.json"):
                if len(entries) >= limit:
                    break
                try:
                    with open(entry_file, "r") as f:
                        data = json.load(f)
                    entries.append(MuGraphEntry.from_dict(data))
                except Exception as e:
                    logger.warning(f"Failed to load {entry_file}: {e}")

        return entries[:limit]

    def delete(self, graph_hash: str, config_hash: str, backend: str) -> bool:
        """Delete a specific muGraph entry."""
        entry_path = self._get_entry_path(graph_hash, config_hash, backend)

        if entry_path.exists():
            entry_path.unlink()
            index_key = f"{backend}:{graph_hash}"
            if index_key in self._index:
                self._index[index_key] = [p for p in self._index[index_key] if p != str(entry_path)]
                if not self._index[index_key]:
                    del self._index[index_key]
                self._save_index()
            return True
        return False

    def clear(self, backend: Optional[str] = None):
        """Clear all stored muGraphs."""
        backends = [backend] if backend else ["mps", "cuda", "cpu", "ascend", "maca", "triton"]

        for be in backends:
            backend_dir = self._get_backend_dir(be)
            if backend_dir.exists():
                for entry_file in backend_dir.glob("*.json"):
                    entry_file.unlink()

        if backend:
            self._index = {k: v for k, v in self._index.items() if not k.startswith(f"{backend}:")}
        else:
            self._index = {}
        self._save_index()

    def get_stats(self, backend: Optional[str] = None) -> Dict:
        """Get statistics about stored muGraphs."""
        backends = [backend] if backend else ["mps", "cuda", "cpu", "ascend", "maca", "triton"]

        stats = {
            "total_entries": 0,
            "by_backend": {},
            "total_size_bytes": 0,
            "avg_latency_ms": 0.0,
            "best_latency_ms": float("inf"),
            "worst_latency_ms": 0.0,
            "total_candidates_explored": 0,
            "total_search_time_s": 0.0,
        }

        latencies = []

        for be in backends:
            backend_dir = self._get_backend_dir(be)
            if not backend_dir.exists():
                continue

            count = 0
            size = 0

            for entry_file in backend_dir.glob("*.json"):
                count += 1
                size += entry_file.stat().st_size

                try:
                    with open(entry_file, "r") as f:
                        data = json.load(f)
                    meta = data.get("metadata", {})
                    lat = meta.get("latency_ms", 0)
                    if lat > 0:
                        latencies.append(lat)
                    stats["total_candidates_explored"] += meta.get("num_candidates_searched", 0)
                    stats["total_search_time_s"] += meta.get("search_time_s", 0)
                except:
                    pass

            stats["by_backend"][be] = {"count": count, "size_bytes": size}
            stats["total_entries"] += count
            stats["total_size_bytes"] += size

        if latencies:
            stats["avg_latency_ms"] = sum(latencies) / len(latencies)
            stats["best_latency_ms"] = min(latencies)
            stats["worst_latency_ms"] = max(latencies)

        return stats

    def export_training_data(
        self,
        output_dir: Optional[str] = None,
        backend: Optional[str] = None,
        format: str = "jsonl",
    ) -> Dict[str, str]:
        """
        Export training data for ML model training.

        Returns:
            Dictionary with paths to exported files.
        """
        output_path = Path(output_dir or (self.root_path / "training_data"))
        output_path.mkdir(parents=True, exist_ok=True)

        entries = self.list_all(backend=backend, limit=100000)

        # Export features
        features_path = output_path / f"features_{backend or 'all'}.{format}"
        labels_path = output_path / f"labels_{backend or 'all'}.{format}"
        trajectories_path = output_path / f"trajectories_{backend or 'all'}.{format}"

        with open(features_path, "w") as f_feat, open(labels_path, "w") as f_labels, open(
            trajectories_path, "w"
        ) as f_traj:

            for entry in entries:
                # Features
                feat_record = {
                    "entry_id": entry.metadata.entry_id,
                    "graph_hash": entry.metadata.graph_hash,
                    "backend": entry.metadata.backend,
                    **entry.features.to_dict(),
                }
                f_feat.write(json.dumps(feat_record) + "\n")

                # Labels
                label_record = {
                    "entry_id": entry.metadata.entry_id,
                    "graph_hash": entry.metadata.graph_hash,
                    **entry.labels.to_dict(),
                }
                f_labels.write(json.dumps(label_record) + "\n")

                # Trajectories
                if entry.trajectory:
                    traj_record = {
                        "entry_id": entry.metadata.entry_id,
                        "graph_hash": entry.metadata.graph_hash,
                        **entry.trajectory.to_dict(),
                    }
                    f_traj.write(json.dumps(traj_record) + "\n")

        return {
            "features": str(features_path),
            "labels": str(labels_path),
            "trajectories": str(trajectories_path),
        }

    def export(self, output_path: str, backend: Optional[str] = None):
        """Export all muGraphs to a single archive."""
        import tarfile

        with tarfile.open(output_path, "w:gz") as tar:
            backends = [backend] if backend else ["mps", "cuda", "cpu", "ascend", "maca", "triton"]

            for be in backends:
                backend_dir = self._get_backend_dir(be)
                if backend_dir.exists():
                    for entry_file in backend_dir.glob("*.json"):
                        tar.add(entry_file, arcname=f"{be}/{entry_file.name}")

            if self._index_path().exists():
                tar.add(self._index_path(), arcname="index.json")

    def import_archive(self, archive_path: str):
        """Import muGraphs from an archive."""
        import tarfile

        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(self.root_path)

        self._load_index()


# Global singleton
_default_store: Optional[MuGraphStore] = None


def get_mugraph_store(root_path: Optional[str] = None) -> MuGraphStore:
    """Get the global MuGraphStore instance."""
    global _default_store
    if _default_store is None:
        _default_store = MuGraphStore(root_path)
    return _default_store


def save_mugraph(graph_hash: str, optimized_graph: Any, backend: str, **kwargs) -> str:
    """Convenience function to save a muGraph."""
    return get_mugraph_store().save(graph_hash, optimized_graph, backend, **kwargs)


def find_mugraph(graph_hash: str, backend: str, **kwargs) -> Optional[MuGraphEntry]:
    """Convenience function to find a cached muGraph."""
    return get_mugraph_store().find(graph_hash, backend, **kwargs)


def find_best_mugraph(
    graph_hash: str,
    backend: str,
    input_shapes: Optional[List] = None,
    **kwargs,
) -> Optional[MuGraphEntry]:
    """Convenience function to find the best muGraph."""
    return get_mugraph_store().find_best(
        graph_hash, backend, input_shapes=input_shapes, **kwargs
    )
