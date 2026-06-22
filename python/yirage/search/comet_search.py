# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
COMET Search Strategy for Compound Operations.

This module implements the search strategy for COMET's compound operations,
exploring the design space of:
- Fusion levels (which ops to fuse)
- Tile sizes (M_tile, N_tile, K_tile)
- Collective placement (where to insert collectives)
- Scheduling strategy (sequential, pipelined, parallel)

Reference: COMET paper (Negi et al.) - Section IV-C
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math
import time
import itertools


# =============================================================================
# COMET Enums (mirroring C++ type.h)
# =============================================================================

class CompoundOpType(Enum):
    """Types of compound operations supported by COMET."""
    NONE = 0
    GEMM_SOFTMAX = 1
    GEMM_LAYERNORM = 2
    SELF_ATTENTION = 3
    GATED_MLP = 4
    RMS_NORM_LINEAR = 5


class SchedulingStrategy(Enum):
    """Scheduling strategies for compound operations."""
    SEQUENTIAL = 0
    PIPELINED = 1
    PARALLEL = 2


class CollectiveOpType(Enum):
    """Types of collective operations."""
    NONE = 0
    ALLREDUCE = 1
    ALLGATHER = 2
    REDUCESCATTER = 3
    BROADCAST = 4
    P2P_SEND = 5
    P2P_RECV = 6


class MemoryLevel(Enum):
    """Memory hierarchy levels."""
    REGISTER = 0
    L1_CACHE = 1
    L2_CACHE = 2
    SHARED_MEM = 3
    DRAM = 4
    HBM = 5


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class COMETSearchConfig:
    """Configuration for COMET search strategy."""
    
    # Search parameters
    max_iterations: int = 1000
    timeout_seconds: float = 600.0
    random_seed: int = 42
    
    # Fusion search
    max_fusion_depth: int = 5
    enable_fusion_search: bool = True
    
    # Tile size search space
    tile_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    auto_tune_tiles: bool = True
    
    # Collective optimization
    optimize_collectives: bool = True
    num_devices: int = 1
    noc_bandwidth_gbps: float = 600.0
    
    # Scheduling search
    search_scheduling: bool = True
    scheduling_options: List[SchedulingStrategy] = field(
        default_factory=lambda: [
            SchedulingStrategy.SEQUENTIAL,
            SchedulingStrategy.PIPELINED,
            SchedulingStrategy.PARALLEL
        ]
    )
    
    # Hardware parameters
    dram_bandwidth_gbps: float = 900.0
    onchip_bandwidth_gbps: float = 3000.0
    peak_tflops: float = 312.0
    
    # Search objective
    objective: str = "minimize_latency"  # "minimize_latency", "minimize_energy", "balance"
    energy_weight: float = 0.3


@dataclass
class CompoundPattern:
    """Detected compound operation pattern."""
    
    op_type: CompoundOpType
    op_indices: List[int]
    input_dims: List[int] = field(default_factory=list)
    output_dims: List[int] = field(default_factory=list)
    
    # Estimated fusion benefits
    memory_reduction_ratio: float = 0.0
    latency_reduction_ratio: float = 0.0
    
    def get_fusion_benefit(self) -> float:
        """Compute overall fusion benefit score."""
        return 0.6 * self.memory_reduction_ratio + 0.4 * self.latency_reduction_ratio


@dataclass
class TileConfig:
    """Tile configuration for compound operations."""
    
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 64
    
    def __iter__(self):
        return iter([self.tile_m, self.tile_n, self.tile_k])
    
    def __hash__(self):
        return hash((self.tile_m, self.tile_n, self.tile_k))


@dataclass
class COMETCandidate:
    """Search candidate for COMET optimization."""
    
    # Configuration
    pattern: Optional[CompoundPattern] = None
    tile_config: TileConfig = field(default_factory=TileConfig)
    scheduling: SchedulingStrategy = SchedulingStrategy.PIPELINED
    
    # Collective placement
    collective_ops: List[Tuple[CollectiveOpType, int]] = field(default_factory=list)
    
    # Cost estimates
    latency_ns: float = 0.0
    energy_pj: float = 0.0
    memory_bytes: int = 0
    
    # Search score
    score: float = 0.0


# =============================================================================
# COMET Cost Model
# =============================================================================

class COMETCostModel:
    """
    COMET Cost Model for compound operation evaluation.
    
    Implements latency and energy estimation based on COMET equations.
    """
    
    def __init__(self, config: Optional[COMETSearchConfig] = None):
        if config is None:
            config = COMETSearchConfig()
        
        self.dram_bandwidth_gbps = config.dram_bandwidth_gbps
        self.onchip_bandwidth_gbps = config.onchip_bandwidth_gbps
        self.noc_bandwidth_gbps = config.noc_bandwidth_gbps
        self.peak_tflops = config.peak_tflops
        self.noc_latency_ns = 100.0
    
    def estimate_compute_latency_ns(
        self,
        flops: int,
        utilization: float = 0.7
    ) -> float:
        """Estimate compute latency given FLOPs."""
        tflops_effective = self.peak_tflops * utilization
        return flops / (tflops_effective * 1e3)  # ns
    
    def estimate_memory_latency_ns(
        self,
        data_bytes: int,
        src: MemoryLevel,
        dst: MemoryLevel,
        num_tiles: int = 1
    ) -> float:
        """
        Estimate memory transfer latency.
        
        COMET Eq. 4: L_mem = bytes / bandwidth
        """
        if src == MemoryLevel.DRAM or dst == MemoryLevel.DRAM:
            bandwidth = self.dram_bandwidth_gbps
        else:
            bandwidth = self.onchip_bandwidth_gbps
        
        bytes_per_ns = bandwidth * 1e9 / 1e9
        latency = data_bytes / bytes_per_ns
        
        # Per-tile overhead
        latency += num_tiles * 10.0
        
        return latency
    
    def estimate_collective_latency_ns(
        self,
        collective_type: CollectiveOpType,
        data_bytes: int,
        num_participants: int
    ) -> float:
        """
        Estimate collective communication latency.
        
        COMET Eq. 5: L_coll = alpha + f(n, data) * beta
        """
        if num_participants <= 1:
            return 0.0
        
        alpha = self.noc_latency_ns
        bytes_per_ns = self.noc_bandwidth_gbps * 1e9 / 1e9
        beta = 1.0 / bytes_per_ns
        
        n = num_participants
        data = float(data_bytes)
        
        if collective_type == CollectiveOpType.ALLREDUCE:
            # Ring all-reduce: 2 * (n-1)/n * data * beta
            return alpha + 2.0 * (n - 1.0) / n * data * beta
        elif collective_type == CollectiveOpType.ALLGATHER:
            # All-gather: (n-1) * data * beta
            return alpha + (n - 1.0) / n * data * n * beta
        elif collective_type == CollectiveOpType.REDUCESCATTER:
            # Reduce-scatter: (n-1)/n * data * beta
            return alpha + (n - 1.0) / n * data * beta
        elif collective_type == CollectiveOpType.BROADCAST:
            # Broadcast: log2(n) * alpha + (n-1)/n * data * beta
            return math.log2(n) * alpha + (n - 1.0) / n * data * beta
        elif collective_type in (CollectiveOpType.P2P_SEND, CollectiveOpType.P2P_RECV):
            # Point-to-point: alpha + data * beta
            return alpha + data * beta
        else:
            return 0.0
    
    def estimate_scheduling_overhead_ns(
        self,
        strategy: SchedulingStrategy,
        num_ops: int
    ) -> float:
        """Estimate scheduling overhead."""
        if strategy == SchedulingStrategy.SEQUENTIAL:
            base = 50.0
        elif strategy == SchedulingStrategy.PIPELINED:
            base = 200.0
        else:  # PARALLEL
            base = 500.0
        
        return base + num_ops * 5.0
    
    def estimate_gemm_softmax_latency_ns(
        self,
        M: int, K: int, N: int,
        tile_config: TileConfig,
        fused: bool = True
    ) -> float:
        """Estimate latency for GEMM-Softmax compound operation."""
        # Compute FLOPs
        gemm_flops = 2 * M * K * N
        softmax_flops = 4 * M * N  # exp, sum, div for each element
        total_flops = gemm_flops + softmax_flops
        
        compute_latency = self.estimate_compute_latency_ns(total_flops)
        
        # Memory traffic
        if fused:
            # Input A, B, output result only
            memory_bytes = (M * K + K * N + M * N) * 4  # FP32
        else:
            # Unfused: intermediate materialized
            memory_bytes = (M * K + K * N + M * N + M * N) * 4
        
        memory_latency = self.estimate_memory_latency_ns(
            memory_bytes, MemoryLevel.DRAM, MemoryLevel.REGISTER
        )
        
        return max(compute_latency, memory_latency)
    
    def estimate_self_attention_latency_ns(
        self,
        batch: int, heads: int, seq_len: int, head_dim: int,
        tile_config: TileConfig,
        fused: bool = True
    ) -> float:
        """Estimate latency for self-attention compound operation."""
        # Q*K^T FLOPs
        qk_flops = 2 * batch * heads * seq_len * seq_len * head_dim
        # Softmax FLOPs
        softmax_flops = 4 * batch * heads * seq_len * seq_len
        # Attn * V FLOPs
        av_flops = 2 * batch * heads * seq_len * head_dim * seq_len
        
        total_flops = qk_flops + softmax_flops + av_flops
        compute_latency = self.estimate_compute_latency_ns(total_flops)
        
        # Memory traffic
        elem_size = 4  # FP32
        qkv_size = 3 * batch * heads * seq_len * head_dim * elem_size
        
        output_size = batch * heads * seq_len * head_dim * elem_size
        
        if fused:
            # Fused: only QKV input and output
            memory_bytes = qkv_size + output_size
        else:
            # Unfused: intermediates materialized
            scores_size = batch * heads * seq_len * seq_len * elem_size
            probs_size = scores_size
            memory_bytes = qkv_size + scores_size + probs_size + output_size
        
        memory_latency = self.estimate_memory_latency_ns(
            memory_bytes, MemoryLevel.DRAM, MemoryLevel.REGISTER
        )
        
        return max(compute_latency, memory_latency)
    
    def compute_score(
        self,
        latency_ns: float,
        energy_pj: float,
        objective: str = "minimize_latency",
        energy_weight: float = 0.3
    ) -> float:
        """Compute optimization score (higher is better)."""
        latency_score = 1.0 / (1.0 + latency_ns / 1e6)
        energy_score = 1.0 / (1.0 + energy_pj / 1e9)
        
        if objective == "minimize_latency":
            return latency_score
        elif objective == "minimize_energy":
            return energy_score
        else:  # balance
            return (1.0 - energy_weight) * latency_score + energy_weight * energy_score


# =============================================================================
# Pattern Detection
# =============================================================================

def detect_compound_patterns(
    op_types: List[str],
    op_connections: Optional[Dict[int, List[int]]] = None
) -> List[CompoundPattern]:
    """
    Detect compound operation patterns in a graph.
    
    Args:
        op_types: List of operation type strings
        op_connections: Optional graph connectivity
        
    Returns:
        List of detected compound patterns
    """
    patterns = []
    n_ops = len(op_types)
    used = set()
    
    for i in range(n_ops):
        if i in used:
            continue
        
        # Try to match Self-Attention FIRST (most complex pattern)
        # Self-attention includes GEMM-Softmax, so check it first
        if _is_self_attention(op_types, i):
            pattern = CompoundPattern(
                op_type=CompoundOpType.SELF_ATTENTION,
                op_indices=list(range(i, min(i + 6, n_ops))),
                memory_reduction_ratio=0.7,
                latency_reduction_ratio=0.4
            )
            patterns.append(pattern)
            used.update(pattern.op_indices)
            continue
        
        # Try to match GEMM-Softmax: matmul -> exp -> reduction -> div
        if _is_gemm_softmax(op_types, i):
            pattern = CompoundPattern(
                op_type=CompoundOpType.GEMM_SOFTMAX,
                op_indices=list(range(i, min(i + 4, n_ops))),
                memory_reduction_ratio=0.5,
                latency_reduction_ratio=0.3
            )
            patterns.append(pattern)
            used.update(pattern.op_indices)
            continue
        
        # Try to match GEMM-LayerNorm
        if _is_gemm_layernorm(op_types, i):
            pattern = CompoundPattern(
                op_type=CompoundOpType.GEMM_LAYERNORM,
                op_indices=list(range(i, min(i + 5, n_ops))),
                memory_reduction_ratio=0.5,
                latency_reduction_ratio=0.25
            )
            patterns.append(pattern)
            used.update(pattern.op_indices)
            continue
        
        # Try to match Gated MLP
        if _is_gated_mlp(op_types, i):
            pattern = CompoundPattern(
                op_type=CompoundOpType.GATED_MLP,
                op_indices=list(range(i, min(i + 4, n_ops))),
                memory_reduction_ratio=0.4,
                latency_reduction_ratio=0.2
            )
            patterns.append(pattern)
            used.update(pattern.op_indices)
    
    return patterns


def _is_gemm_softmax(op_types: List[str], start: int) -> bool:
    """Check for GEMM-Softmax pattern."""
    n = len(op_types)
    if start >= n:
        return False
    
    if op_types[start].lower() not in ("matmul", "gemm"):
        return False
    
    # Look for softmax-like ops following
    for i in range(start + 1, min(start + 4, n)):
        if op_types[i].lower() in ("exp", "softmax"):
            return True
    
    return False


def _is_gemm_layernorm(op_types: List[str], start: int) -> bool:
    """Check for GEMM-LayerNorm pattern."""
    n = len(op_types)
    if start >= n:
        return False
    
    if op_types[start].lower() not in ("matmul", "gemm"):
        return False
    
    # Look for layernorm-like ops (reduction followed by scale/shift)
    has_reduction = False
    has_scale = False
    
    for i in range(start + 1, min(start + 5, n)):
        op_lower = op_types[i].lower()
        if "reduction" in op_lower or "mean" in op_lower or "var" in op_lower:
            has_reduction = True
        if op_lower in ("mul", "multiply", "scale", "add"):
            has_scale = True
    
    # LayerNorm requires both reduction (for mean/var) and scale
    return has_reduction and has_scale


def _is_self_attention(op_types: List[str], start: int) -> bool:
    """Check for Self-Attention pattern."""
    n = len(op_types)
    if start >= n:
        return False
    
    if op_types[start].lower() not in ("matmul", "gemm"):
        return False
    
    # Look for two matmuls with softmax between
    matmul_count = 1
    has_softmax = False
    
    for i in range(start + 1, min(start + 6, n)):
        if op_types[i].lower() in ("matmul", "gemm"):
            matmul_count += 1
        if op_types[i].lower() in ("exp", "softmax"):
            has_softmax = True
    
    return matmul_count >= 2 and has_softmax


def _is_gated_mlp(op_types: List[str], start: int) -> bool:
    """Check for Gated MLP pattern."""
    n = len(op_types)
    if start >= n:
        return False
    
    if op_types[start].lower() not in ("matmul", "gemm"):
        return False
    
    # Look for activation and element-wise mul
    has_activation = False
    has_mul = False
    
    for i in range(start + 1, min(start + 4, n)):
        if op_types[i].lower() in ("silu", "gelu", "swish"):
            has_activation = True
        if op_types[i].lower() in ("mul", "multiply"):
            has_mul = True
    
    return has_activation or has_mul


# =============================================================================
# COMET Search Strategy
# =============================================================================

class COMETSearchStrategy:
    """
    COMET Search Strategy for compound operation optimization.
    
    Implements the search algorithm from the COMET paper, exploring:
    1. Fusion decisions
    2. Tile size optimization
    3. Collective placement
    4. Scheduling strategy selection
    """
    
    def __init__(self, config: Optional[COMETSearchConfig] = None):
        if config is None:
            config = COMETSearchConfig()
        
        # Validate config
        self._validate_config(config)
        
        self.config = config
        self.cost_model = COMETCostModel(config)
        
        # Search state
        self.candidates: List[COMETCandidate] = []
        self.best_candidate: Optional[COMETCandidate] = None
        
        # Statistics
        self.patterns_detected = 0
        self.candidates_generated = 0
        self.candidates_evaluated = 0
    
    def _validate_config(self, config: COMETSearchConfig) -> None:
        """Validate search configuration."""
        if not config.tile_sizes:
            raise ValueError("tile_sizes cannot be empty")
        if config.num_devices < 1:
            raise ValueError("num_devices must be >= 1")
        if config.dram_bandwidth_gbps <= 0:
            raise ValueError("dram_bandwidth_gbps must be positive")
        if config.onchip_bandwidth_gbps <= 0:
            raise ValueError("onchip_bandwidth_gbps must be positive")
        if config.peak_tflops <= 0:
            raise ValueError("peak_tflops must be positive")
        if config.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if not (0.0 <= config.energy_weight <= 1.0):
            raise ValueError("energy_weight must be in [0, 1]")
    
    def search(
        self,
        op_types: List[str],
        problem_dims: Dict[str, int],
        op_connections: Optional[Dict[int, List[int]]] = None
    ) -> COMETCandidate:
        """
        Run COMET search to find optimal configuration.
        
        Args:
            op_types: List of operation type strings
            problem_dims: Problem dimensions (M, K, N, etc.)
            op_connections: Optional graph connectivity
            
        Returns:
            Best COMETCandidate found
        """
        start_time = time.time()
        timeout = self.config.timeout_seconds
        
        # Step 1: Detect patterns
        patterns = detect_compound_patterns(op_types, op_connections)
        self.patterns_detected = len(patterns)
        
        if not patterns:
            # No compound patterns, return default
            return COMETCandidate()
        
        # Step 2: Generate candidates
        self.candidates = self._generate_candidates(patterns, problem_dims)
        self.candidates_generated = len(self.candidates)
        
        # Step 3: Evaluate candidates with timeout
        for candidate in self.candidates:
            # Check timeout
            if time.time() - start_time > timeout:
                break
            
            self._evaluate_candidate(candidate, problem_dims)
            self.candidates_evaluated += 1
        
        # Step 4: Select best from evaluated candidates
        evaluated = [c for c in self.candidates if c.score > 0]
        if evaluated:
            self.best_candidate = max(evaluated, key=lambda c: c.score)
        elif self.candidates:
            # If no candidates were evaluated (immediate timeout), evaluate first
            self._evaluate_candidate(self.candidates[0], problem_dims)
            self.best_candidate = self.candidates[0]
        
        return self.best_candidate or COMETCandidate()
    
    def _generate_candidates(
        self,
        patterns: List[CompoundPattern],
        problem_dims: Dict[str, int]
    ) -> List[COMETCandidate]:
        """Generate search candidates."""
        candidates = []
        
        # Generate tile configurations
        tile_configs = self._generate_tile_configs(problem_dims)
        
        for pattern in patterns:
            for tile_config in tile_configs:
                for scheduling in self.config.scheduling_options:
                    candidate = COMETCandidate(
                        pattern=pattern,
                        tile_config=tile_config,
                        scheduling=scheduling
                    )
                    candidates.append(candidate)
        
        # Limit candidates if too many
        max_candidates = self.config.max_iterations
        if len(candidates) > max_candidates:
            # Prioritize by pattern benefit
            candidates.sort(key=lambda c: c.pattern.get_fusion_benefit() if c.pattern else 0, reverse=True)
            candidates = candidates[:max_candidates]
        
        return candidates
    
    def _generate_tile_configs(
        self,
        problem_dims: Dict[str, int]
    ) -> List[TileConfig]:
        """Generate valid tile configurations."""
        configs = []
        
        M = problem_dims.get("M", 1024)
        K = problem_dims.get("K", 1024)
        N = problem_dims.get("N", 1024)
        
        for tm in self.config.tile_sizes:
            if tm > M:
                continue
            for tn in self.config.tile_sizes:
                if tn > N:
                    continue
                for tk in self.config.tile_sizes:
                    if tk > K:
                        continue
                    configs.append(TileConfig(tm, tn, tk))
        
        return configs
    
    def _evaluate_candidate(
        self,
        candidate: COMETCandidate,
        problem_dims: Dict[str, int]
    ) -> None:
        """Evaluate a candidate using the cost model."""
        if not candidate.pattern:
            candidate.score = 0.0
            return
        
        M = problem_dims.get("M", 1024)
        K = problem_dims.get("K", 1024)
        N = problem_dims.get("N", 1024)
        
        # Estimate latency based on pattern type
        if candidate.pattern.op_type == CompoundOpType.GEMM_SOFTMAX:
            candidate.latency_ns = self.cost_model.estimate_gemm_softmax_latency_ns(
                M, K, N, candidate.tile_config, fused=True
            )
        elif candidate.pattern.op_type == CompoundOpType.SELF_ATTENTION:
            batch = problem_dims.get("batch", 1)
            heads = problem_dims.get("heads", 8)
            seq_len = problem_dims.get("seq_len", M)
            head_dim = problem_dims.get("head_dim", K // heads)
            candidate.latency_ns = self.cost_model.estimate_self_attention_latency_ns(
                batch, heads, seq_len, head_dim, candidate.tile_config, fused=True
            )
        else:
            # Generic estimate
            flops = 2 * M * K * N
            candidate.latency_ns = self.cost_model.estimate_compute_latency_ns(flops)
        
        # Add scheduling overhead
        num_ops = len(candidate.pattern.op_indices) if candidate.pattern else 1
        candidate.latency_ns += self.cost_model.estimate_scheduling_overhead_ns(
            candidate.scheduling, num_ops
        )
        
        # Add collective overhead if distributed
        if self.config.num_devices > 1:
            for coll_type, data_bytes in candidate.collective_ops:
                candidate.latency_ns += self.cost_model.estimate_collective_latency_ns(
                    coll_type, data_bytes, self.config.num_devices
                )
        
        # Estimate energy (simplified)
        candidate.energy_pj = candidate.latency_ns * 100  # Approximate
        
        # Compute score
        candidate.score = self.cost_model.compute_score(
            candidate.latency_ns,
            candidate.energy_pj,
            self.config.objective,
            self.config.energy_weight
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        return {
            "patterns_detected": self.patterns_detected,
            "candidates_generated": self.candidates_generated,
            "candidates_evaluated": self.candidates_evaluated,
            "best_score": self.best_candidate.score if self.best_candidate else 0.0,
            "best_latency_ns": self.best_candidate.latency_ns if self.best_candidate else 0.0,
        }


# =============================================================================
# High-Level API
# =============================================================================

def optimize_compound_graph(
    op_types: List[str],
    problem_dims: Dict[str, int],
    config: Optional[COMETSearchConfig] = None
) -> Dict[str, Any]:
    """
    Optimize a compound operation graph using COMET search.
    
    Args:
        op_types: List of operation type strings
        problem_dims: Problem dimensions (M, K, N, etc.)
        config: Optional COMET search configuration
        
    Returns:
        Dictionary with optimization results
    """
    strategy = COMETSearchStrategy(config)
    best = strategy.search(op_types, problem_dims)
    
    return {
        "success": best.score > 0,
        "best_candidate": best,
        "tile_config": {
            "tile_m": best.tile_config.tile_m,
            "tile_n": best.tile_config.tile_n,
            "tile_k": best.tile_config.tile_k,
        },
        "scheduling": best.scheduling.name,
        "latency_ns": best.latency_ns,
        "statistics": strategy.get_statistics(),
    }
