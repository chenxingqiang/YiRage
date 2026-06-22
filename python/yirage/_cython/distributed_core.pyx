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
Cython wrapper for distributed search C++ API.

Provides Python bindings for:
- SearchPartition: Search space partitioning
- SearchFeedback: RL training data collection
- Partitioned search execution with C++ core
- RL context for GPU verification

This module provides two modes:
1. Native C++ mode: When Cython bindings are compiled
2. Pure Python mode: Fallback for testing/development
"""

from libc.stdlib cimport malloc, free
from libc.string cimport strlen
from cpython.bytes cimport PyBytes_FromString
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool as cpp_bool

# Try to import C++ bindings
cdef bint CPP_AVAILABLE = False
try:
    cimport distributed_core
    from CCore cimport CppKNGraph, dim3, int3
    CPP_AVAILABLE = True
except ImportError:
    pass

import json
import ctypes
import time


cdef class PySearchPartition:
    """
    Python wrapper for C++ SearchPartition.
    
    Represents a partition of the search space for distributed search.
    """
    
    cdef int _partition_id
    cdef int _total_partitions
    cdef list _grid_dim_range
    cdef list _block_dim_range
    cdef list _imap_range
    cdef list _omap_range
    cdef list _fmap_range
    cdef list _frange_range
    cdef size_t _estimated_candidates
    
    def __cinit__(
        self,
        int partition_id=0,
        int total_partitions=1,
        list grid_dim_range=None,
        list block_dim_range=None,
        list imap_range=None,
        list omap_range=None,
        list fmap_range=None,
        list frange_range=None,
        size_t estimated_candidates=0,
    ):
        self._partition_id = partition_id
        self._total_partitions = total_partitions
        self._grid_dim_range = grid_dim_range or []
        self._block_dim_range = block_dim_range or []
        self._imap_range = imap_range or []
        self._omap_range = omap_range or []
        self._fmap_range = fmap_range or []
        self._frange_range = frange_range or []
        self._estimated_candidates = estimated_candidates
    
    @property
    def partition_id(self) -> int:
        return self._partition_id
    
    @property
    def total_partitions(self) -> int:
        return self._total_partitions
    
    @property
    def grid_dim_range(self) -> list:
        return self._grid_dim_range
    
    @property
    def block_dim_range(self) -> list:
        return self._block_dim_range
    
    @property
    def estimated_candidates(self) -> int:
        return self._estimated_candidates
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {
            "partition_id": self._partition_id,
            "total_partitions": self._total_partitions,
            "grid_dim_range": [
                {"x": g[0], "y": g[1], "z": g[2]}
                for g in self._grid_dim_range
            ],
            "block_dim_range": [
                {"x": b[0], "y": b[1], "z": b[2]}
                for b in self._block_dim_range
            ],
            "imap_range": [
                {"x": m[0], "y": m[1], "z": m[2]}
                for m in self._imap_range
            ],
            "omap_range": [
                {"x": m[0], "y": m[1], "z": m[2]}
                for m in self._omap_range
            ],
            "fmap_range": self._fmap_range,
            "frange_range": self._frange_range,
            "estimated_candidates": self._estimated_candidates,
        }
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, str json_str) -> "PySearchPartition":
        """Create from JSON string."""
        data = json.loads(json_str)
        
        grid_range = [
            (g["x"], g["y"], g["z"])
            for g in data.get("grid_dim_range", [])
        ]
        block_range = [
            (b["x"], b["y"], b["z"])
            for b in data.get("block_dim_range", [])
        ]
        imap_range = [
            (m["x"], m["y"], m["z"])
            for m in data.get("imap_range", [])
        ]
        omap_range = [
            (m["x"], m["y"], m["z"])
            for m in data.get("omap_range", [])
        ]
        
        return cls(
            partition_id=data.get("partition_id", 0),
            total_partitions=data.get("total_partitions", 1),
            grid_dim_range=grid_range,
            block_dim_range=block_range,
            imap_range=imap_range,
            omap_range=omap_range,
            fmap_range=data.get("fmap_range", []),
            frange_range=data.get("frange_range", []),
            estimated_candidates=data.get("estimated_candidates", 0),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return json.loads(self.to_json())
    
    @classmethod
    def from_dict(cls, dict data) -> "PySearchPartition":
        """Create from dictionary."""
        return cls.from_json(json.dumps(data))


cdef class PyCandidateInfo:
    """
    Python wrapper for C++ CandidateInfo.
    
    Contains information about a candidate configuration explored during search.
    """
    
    cdef dict _data
    
    def __cinit__(self, dict data=None):
        self._data = data or {
            "candidate_id": 0,
            "grid_dim": (1, 1, 1),
            "block_dim": (128, 1, 1),
            "imaps": [],
            "omap": (0, 0, 0),
            "frange": 1,
            "search_depth": 0,
            "operator_count": 0,
            "verified": False,
            "fingerprint_time_ms": 0.0,
            "estimated_performance_ms": 0.0,
            "rejection_reason": "",
        }
    
    @property
    def candidate_id(self) -> int:
        return self._data.get("candidate_id", 0)
    
    @property
    def grid_dim(self) -> tuple:
        return tuple(self._data.get("grid_dim", (1, 1, 1)))
    
    @property
    def block_dim(self) -> tuple:
        return tuple(self._data.get("block_dim", (128, 1, 1)))
    
    @property
    def verified(self) -> bool:
        return self._data.get("verified", False)
    
    @property
    def estimated_performance_ms(self) -> float:
        return self._data.get("estimated_performance_ms", 0.0)
    
    def to_dict(self) -> dict:
        return self._data.copy()
    
    @classmethod
    def from_dict(cls, dict data) -> "PyCandidateInfo":
        return cls(data)


cdef class PySearchFeedback:
    """
    Python wrapper for C++ SearchFeedback.
    
    Aggregated feedback from a search run for RL training.
    """
    
    cdef dict _data
    
    def __cinit__(self, dict data=None):
        self._data = data or {
            "partition_id": 0,
            "total_partitions": 1,
            "candidates": [],
            "valid_candidate_ids": [],
            "total_states_explored": 0,
            "valid_graphs_found": 0,
            "candidates_generated": 0,
            "search_time_seconds": 0.0,
            "best_performance_ms": float("inf"),
            "best_candidate_id": -1,
        }
    
    @property
    def partition_id(self) -> int:
        return self._data.get("partition_id", 0)
    
    @property
    def total_states_explored(self) -> int:
        return self._data.get("total_states_explored", 0)
    
    @property
    def valid_graphs_found(self) -> int:
        return self._data.get("valid_graphs_found", 0)
    
    @property
    def best_performance_ms(self) -> float:
        return self._data.get("best_performance_ms", float("inf"))
    
    @property
    def candidates(self) -> list:
        return [
            PyCandidateInfo.from_dict(c)
            for c in self._data.get("candidates", [])
        ]
    
    def to_dict(self) -> dict:
        return self._data.copy()
    
    def to_json(self) -> str:
        return json.dumps(self._data)
    
    @classmethod
    def from_dict(cls, dict data) -> "PySearchFeedback":
        return cls(data)
    
    @classmethod
    def from_json(cls, str json_str) -> "PySearchFeedback":
        return cls(json.loads(json_str))
    
    def get_summary(self) -> str:
        """Get summary statistics."""
        return (
            f"Partition {self.partition_id}: "
            f"explored={self.total_states_explored}, "
            f"valid={self.valid_graphs_found}, "
            f"best={self.best_performance_ms:.3f}ms"
        )


def create_partitions_py(dict config, int num_partitions) -> list:
    """
    Create search partitions from configuration.
    
    This is a Python implementation that works without C++ bindings.
    When C++ is available, it will use the native implementation.
    
    Args:
        config: Search configuration with grid_dims, block_dims, etc.
        num_partitions: Number of partitions to create
        
    Returns:
        List of PySearchPartition objects
    """
    grid_dims = config.get("grid_dims", [(1, 1, 1)])
    block_dims = config.get("block_dims", [(128, 1, 1)])
    imaps = config.get("imaps", [])
    omaps = config.get("omaps", [])
    fmaps = config.get("fmaps", [])
    franges = config.get("franges", [])
    
    # Partition by grid dimensions (primary)
    grids_per_partition = max(1, len(grid_dims) // num_partitions)
    
    partitions = []
    for i in range(num_partitions):
        start_idx = i * grids_per_partition
        end_idx = start_idx + grids_per_partition if i < num_partitions - 1 else len(grid_dims)
        
        my_grids = grid_dims[start_idx:end_idx] if start_idx < len(grid_dims) else []
        
        # Estimate candidates
        estimated = len(my_grids) * len(block_dims) * max(1, len(franges))
        
        partition = PySearchPartition(
            partition_id=i,
            total_partitions=num_partitions,
            grid_dim_range=my_grids,
            block_dim_range=block_dims,
            imap_range=imaps,
            omap_range=omaps,
            fmap_range=fmaps,
            frange_range=franges,
            estimated_candidates=estimated,
        )
        partitions.append(partition)
    
    return partitions


def search_partition_py(
    graph_json: str,
    partition: PySearchPartition,
    config: dict,
    bint collect_feedback=True,
) -> dict:
    """
    Execute search on a partition.
    
    This is a Python fallback implementation.
    When C++ is available, it will use the native implementation.
    
    Args:
        graph_json: Computation graph as JSON
        partition: Search partition
        config: Search configuration
        collect_feedback: Whether to collect feedback data
        
    Returns:
        Search results with graphs and feedback
    """
    # Try to use C++ implementation
    try:
        from yirage.core import search, CyKNGraph, cy_from_json
        
        # Load graph from JSON
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(graph_json)
            temp_path = f.name
        
        try:
            input_graph = cy_from_json(temp_path)
            
            # Convert partition to search parameters
            griddims = [tuple(g) for g in partition.grid_dim_range]
            blockdims = [tuple(b) for b in partition.block_dim_range]
            
            # Execute search
            new_graphs = search(
                input_graph,
                backend=config.get("backend", "cuda"),
                griddims=griddims or None,
                blockdims=blockdims or None,
                verbose=config.get("verbose", False),
                is_formal_verified=config.get("formal_verify", False),
            )
            
            return {
                "partition_id": partition.partition_id,
                "num_graphs": len(new_graphs),
                "graphs": new_graphs,
                "feedback": None,  # Feedback from C++ would be collected separately
            }
            
        finally:
            import os
            os.unlink(temp_path)
            
    except ImportError:
        # Fall back to simulated search
        pass
    
    # Simulated search (for testing without C++)
    candidates = []
    
    for grid in partition.grid_dim_range:
        for block in partition.block_dim_range:
            candidates.append({
                "grid_dim": grid,
                "block_dim": block,
                "latency_ms": 1.0 / (grid[0] * block[0]),  # Fake metric
                "verified": True,
            })
    
    best = min(candidates, key=lambda x: x["latency_ms"]) if candidates else None
    
    feedback = None
    if collect_feedback:
        feedback = PySearchFeedback({
            "partition_id": partition.partition_id,
            "total_partitions": partition.total_partitions,
            "candidates": candidates,
            "total_states_explored": len(candidates),
            "valid_graphs_found": len(candidates),
            "best_performance_ms": best["latency_ms"] if best else float("inf"),
        })
    
    return {
        "partition_id": partition.partition_id,
        "num_candidates": len(candidates),
        "candidates": candidates,
        "best": best,
        "feedback": feedback.to_dict() if feedback else None,
    }


# =============================================================================
# C++ Binding Functions (when compiled with Cython)
# =============================================================================

def is_cpp_available() -> bool:
    """Check if C++ bindings are available."""
    return CPP_AVAILABLE


def create_partitions_cpp(str config_json, int num_partitions) -> str:
    """
    Create search partitions using C++ implementation.
    
    Args:
        config_json: Search configuration as JSON
        num_partitions: Number of partitions to create
        
    Returns:
        JSON string containing partition array
    """
    if not CPP_AVAILABLE:
        raise RuntimeError("C++ bindings not available")
    
    cdef bytes config_bytes = config_json.encode('utf-8')
    cdef const char* config_ptr = config_bytes
    cdef char* result
    
    with nogil:
        result = distributed_core.create_partitions(
            num_partitions,
            config_ptr
        )
    
    if result == NULL:
        return "[]"
    
    try:
        return result.decode('utf-8')
    finally:
        with nogil:
            distributed_core.free_json_string(result)


def search_partition_cpp(
    graph,  # CyKNGraph
    str partition_json,
    str config_json,
    bint collect_feedback = True,
    int max_num_graphs = 1024,
) -> dict:
    """
    Execute search on a partition using C++ implementation.
    
    Args:
        graph: CyKNGraph input computation graph
        partition_json: Partition configuration as JSON
        config_json: Search configuration as JSON
        collect_feedback: Whether to collect feedback data
        max_num_graphs: Maximum number of graphs to return
        
    Returns:
        Dictionary with:
        - num_graphs: int
        - graphs: list of CyKNGraph
        - feedback_json: str or None
    """
    if not CPP_AVAILABLE:
        raise RuntimeError("C++ bindings not available")
    
    # Import CyKNGraph from core
    from yirage.core import CyKNGraph
    
    cdef bytes partition_bytes = partition_json.encode('utf-8')
    cdef bytes config_bytes = config_json.encode('utf-8')
    cdef const char* partition_ptr = partition_bytes
    cdef const char* config_ptr = config_bytes
    cdef CppKNGraph* input_graph = <CppKNGraph*><unsigned long long>ctypes.cast(
        graph.p_kgraph if hasattr(graph, 'p_kgraph') else graph, 
        ctypes.c_void_p
    ).value
    
    # Allocate output arrays
    cdef CppKNGraph** new_graphs = <CppKNGraph**>malloc(max_num_graphs * sizeof(CppKNGraph*))
    cdef char* feedback_json_ptr = NULL
    cdef int num_graphs
    
    try:
        with nogil:
            num_graphs = distributed_core.search_partition(
                input_graph,
                partition_ptr,
                config_ptr,
                collect_feedback,
                max_num_graphs,
                new_graphs,
                &feedback_json_ptr if collect_feedback else NULL
            )
        
        # Convert results to Python
        graphs = []
        for i in range(num_graphs):
            ptr = ctypes.cast(<unsigned long long>new_graphs[i], ctypes.c_void_p)
            graphs.append(CyKNGraph(ptr))
        
        feedback = None
        if feedback_json_ptr != NULL:
            feedback = feedback_json_ptr.decode('utf-8')
            with nogil:
                distributed_core.free_json_string(feedback_json_ptr)
        
        return {
            "num_graphs": num_graphs,
            "graphs": graphs,
            "feedback_json": feedback,
        }
        
    finally:
        free(new_graphs)


# =============================================================================
# RL Context for GPU Verification
# =============================================================================

cdef class RLSearchContextCpp:
    """
    Cython wrapper for C++ RLSearchContext.
    
    Provides GPU-based verification for the RL closed loop.
    """
    
    cdef void* _ctx
    cdef str _backend
    cdef int _gpu_id
    cdef bint _initialized
    
    def __cinit__(
        self,
        str target_graph_json,
        str backend = "cuda",
        int gpu_id = 0,
    ):
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ bindings not available")
        
        cdef bytes target_bytes = target_graph_json.encode('utf-8')
        cdef bytes backend_bytes = backend.encode('utf-8')
        cdef const char* target_ptr = target_bytes
        cdef const char* backend_ptr = backend_bytes
        
        with nogil:
            self._ctx = distributed_core.rl_context_create(
                target_ptr,
                backend_ptr,
                gpu_id
            )
        
        if self._ctx == NULL:
            raise RuntimeError("Failed to create RLSearchContext")
        
        self._backend = backend
        self._gpu_id = gpu_id
        self._initialized = True
    
    def __dealloc__(self):
        if self._initialized and self._ctx != NULL:
            with nogil:
                distributed_core.rl_context_destroy(self._ctx)
            self._ctx = NULL
    
    def reset(self, str new_target_json = ""):
        """Reset context for new episode."""
        cdef bytes target_bytes = new_target_json.encode('utf-8') if new_target_json else b""
        cdef const char* target_ptr = <const char*>target_bytes if len(target_bytes) > 0 else NULL
        
        with nogil:
            distributed_core.rl_context_reset(
                self._ctx, 
                target_ptr
            )
    
    def apply_action(self, int action_type, dict config) -> bool:
        """Apply RL action to search."""
        cdef str config_json = json.dumps(config)
        cdef bytes config_bytes = config_json.encode('utf-8')
        cdef const char* config_ptr = config_bytes
        cdef int result
        
        with nogil:
            result = distributed_core.rl_context_apply_action(
                self._ctx,
                action_type,
                config_ptr
            )
        
        return result != 0
    
    def verify(self) -> dict:
        """
        Verify current kernel on GPU.
        
        This is the critical GPU operation for the RL closed loop.
        """
        cdef char* result_json
        
        with nogil:
            result_json = distributed_core.rl_context_verify(self._ctx)
        
        if result_json == NULL:
            return {"verified": False, "rejection_reason": "null_result"}
        
        try:
            return json.loads(result_json.decode('utf-8'))
        finally:
            with nogil:
                distributed_core.rl_free_string(result_json)
    
    def profile(self, int warmup_iters = 10, int profile_iters = 100) -> dict:
        """Profile kernel performance on GPU."""
        cdef char* result_json
        
        with nogil:
            result_json = distributed_core.rl_context_profile(
                self._ctx,
                warmup_iters,
                profile_iters
            )
        
        if result_json == NULL:
            return {"latency_ms": float("inf")}
        
        try:
            return json.loads(result_json.decode('utf-8'))
        finally:
            with nogil:
                distributed_core.rl_free_string(result_json)
    
    def get_state(self) -> dict:
        """Get current search state for RL observation."""
        cdef char* state_json
        
        with nogil:
            state_json = distributed_core.rl_context_get_state(self._ctx)
        
        if state_json == NULL:
            return {}
        
        try:
            return json.loads(state_json.decode('utf-8'))
        finally:
            with nogil:
                distributed_core.rl_free_string(state_json)
    
    def extract_features(self) -> str:
        """Extract features from current µGraph for RL model."""
        cdef char* features_json
        
        with nogil:
            features_json = distributed_core.rl_context_extract_features(self._ctx)
        
        if features_json == NULL:
            return "{}"
        
        try:
            return features_json.decode('utf-8')
        finally:
            with nogil:
                distributed_core.rl_free_string(features_json)
    
    def is_done(self) -> bool:
        """Check if search is complete."""
        cdef int result
        
        with nogil:
            result = distributed_core.rl_context_is_done(self._ctx)
        
        return result != 0
    
    @property
    def backend(self) -> str:
        return self._backend
    
    @property
    def gpu_id(self) -> int:
        return self._gpu_id


# =============================================================================
# Training Sample Extraction
# =============================================================================

# Export training sample extraction
def extract_training_samples(feedback: PySearchFeedback) -> list:
    """
    Extract RL training samples from search feedback.
    
    Args:
        feedback: Search feedback data
        
    Returns:
        List of training samples (state, action, reward, next_state, done)
    """
    samples = []
    candidates = feedback._data.get("candidates", [])
    num_valid_found = 0
    
    for i, cand in enumerate(candidates):
        sample = {
            "state": {
                "search_depth": cand.get("search_depth", 0),
                "operator_count": cand.get("operator_count", 0),
                "grid_dim": list(cand.get("grid_dim", (1, 1, 1))),
                "block_dim": list(cand.get("block_dim", (128, 1, 1))),
                "num_valid_found_so_far": num_valid_found,
            },
            "action": {
                "imaps": cand.get("imaps", []),
                "omap": list(cand.get("omap", (0, 0, 0))),
                "frange": cand.get("frange", 1),
            },
            "reward": 0.0,
            "done": i == len(candidates) - 1,
            "next_state": None,
        }
        
        # Calculate reward
        if cand.get("verified", False):
            sample["reward"] = 1.0
            perf = cand.get("estimated_performance_ms", 0)
            if perf > 0:
                sample["reward"] += 1.0 / perf
            num_valid_found += 1
        else:
            sample["reward"] = -0.5
        
        # Add depth penalty
        sample["reward"] -= 0.01 * cand.get("search_depth", 0)
        
        # Next state
        if i + 1 < len(candidates):
            next_cand = candidates[i + 1]
            sample["next_state"] = {
                "search_depth": next_cand.get("search_depth", 0),
                "operator_count": next_cand.get("operator_count", 0),
                "grid_dim": list(next_cand.get("grid_dim", (1, 1, 1))),
                "block_dim": list(next_cand.get("block_dim", (128, 1, 1))),
            }
        
        samples.append(sample)
    
    return samples
