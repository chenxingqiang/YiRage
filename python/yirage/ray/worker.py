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
Search worker for distributed execution.

Workers run on CPU and call the C++ search core for actual computation.
Supports both Cython bindings (when compiled) and pure Python fallback.
"""

from typing import Dict, Any, List, Optional
import json
import time
import tempfile
import os


# Check for available search backends
_CYTHON_AVAILABLE = False
_CORE_AVAILABLE = False

try:
    from yirage._cython.distributed_core import (
        PySearchPartition,
        PySearchFeedback,
        search_partition_py,
    )

    _CYTHON_AVAILABLE = True
except ImportError:
    pass

try:
    from yirage.core import search, CyKNGraph, cy_from_json, cy_to_json

    _CORE_AVAILABLE = True
except ImportError:
    pass


class SearchWorker:
    """
    Search worker that executes partition search.

    This is the non-Ray version for single-process usage.
    For distributed usage, use create_ray_worker().

    The worker attempts to use backends in this order:
    1. C++ Cython bindings (if compiled)
    2. Python search with C++ core (if available)
    3. Pure Python simulation (fallback)
    """

    def __init__(self, worker_id: int = 0):
        self.worker_id = worker_id
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Detect available search backend."""
        if _CYTHON_AVAILABLE:
            return "cython"
        elif _CORE_AVAILABLE:
            return "core"
        else:
            return "simulation"

    def search(
        self,
        graph_json: str,
        partition_json: str,
        config_json: str,
        collect_feedback: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute search on assigned partition.

        Args:
            graph_json: Computation graph as JSON
            partition_json: Partition configuration as JSON
            config_json: Search configuration as JSON
            collect_feedback: Whether to collect feedback data

        Returns:
            {
                "worker_id": int,
                "graphs": [...],
                "feedback": {...} or None,
                "elapsed_seconds": float,
                "backend": str,
            }
        """
        start_time = time.time()

        # Parse inputs
        partition = json.loads(partition_json)
        config = json.loads(config_json)

        # Execute search based on available backend
        if self._backend == "cython":
            result = self._search_cython(graph_json, partition, config, collect_feedback)
        elif self._backend == "core":
            result = self._search_core(graph_json, partition, config, collect_feedback)
        else:
            result = self._search_simulation(graph_json, partition, config, collect_feedback)

        result["worker_id"] = self.worker_id
        result["elapsed_seconds"] = time.time() - start_time
        result["backend"] = self._backend

        return result

    def _search_cython(
        self,
        graph_json: str,
        partition: Dict,
        config: Dict,
        collect_feedback: bool,
    ) -> Dict[str, Any]:
        """Search using Cython bindings."""
        from yirage._cython.distributed_core import (
            PySearchPartition,
            search_partition_py,
        )

        # Convert partition dict to PySearchPartition
        py_partition = PySearchPartition.from_dict(partition)

        # Execute search
        result = search_partition_py(
            graph_json,
            py_partition,
            config,
            collect_feedback,
        )

        return {
            "graphs": result.get("graphs", []),
            "candidates": result.get("candidates", []),
            "best": result.get("best"),
            "feedback": result.get("feedback"),
        }

    def _search_core(
        self,
        graph_json: str,
        partition: Dict,
        config: Dict,
        collect_feedback: bool,
    ) -> Dict[str, Any]:
        """Search using C++ core (without partition-aware bindings)."""
        from yirage.core import search, cy_from_json

        # Write graph to temp file for loading
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(graph_json)
            temp_path = f.name

        try:
            input_graph = cy_from_json(temp_path)

            # Extract partition constraints
            grid_dims = partition.get("grid_dim_range", [])
            block_dims = partition.get("block_dim_range", [])

            # Convert to tuples
            griddims = (
                [
                    (
                        tuple(g)
                        if isinstance(g, (list, tuple))
                        else (g.get("x", 1), g.get("y", 1), g.get("z", 1))
                    )
                    for g in grid_dims
                ]
                if grid_dims
                else None
            )
            blockdims = (
                [
                    (
                        tuple(b)
                        if isinstance(b, (list, tuple))
                        else (b.get("x", 128), b.get("y", 1), b.get("z", 1))
                    )
                    for b in block_dims
                ]
                if block_dims
                else None
            )

            # Execute search with constraints
            new_graphs = search(
                input_graph,
                backend=config.get("backend", "cuda"),
                griddims=griddims,
                blockdims=blockdims,
                verbose=config.get("verbose", False),
                is_formal_verified=config.get("formal_verify", False),
            )

            # Build serializable result (CyKNGraph cannot be pickled across Ray workers)
            candidates = []
            for i, _g in enumerate(new_graphs):
                candidates.append(
                    {
                        "graph_id": i,
                        "verified": True,
                    }
                )

            return {
                "graphs": candidates,
                "candidates": candidates,
                "best": candidates[0] if candidates else None,
                "feedback": (
                    {
                        "partition_id": partition.get("partition_id", 0),
                        "total_partitions": partition.get("total_partitions", 1),
                        "candidates": candidates,
                        "total_states_explored": len(candidates),
                        "valid_graphs_found": len(candidates),
                    }
                    if collect_feedback
                    else None
                ),
            }

        finally:
            os.unlink(temp_path)

    def _search_simulation(
        self,
        graph_json: str,
        partition: Dict,
        config: Dict,
        collect_feedback: bool,
    ) -> Dict[str, Any]:
        """Simulated search for testing without C++ core."""
        grid_range = partition.get("grid_dim_range", [(1, 1, 1)])
        block_range = partition.get("block_dim_range", [(128, 1, 1)])

        # Parse grid/block dims
        def parse_dim(d):
            if isinstance(d, (list, tuple)):
                return tuple(d)
            elif isinstance(d, dict):
                return (d.get("x", 1), d.get("y", 1), d.get("z", 1))
            return (1, 1, 1)

        candidates = []
        for grid in grid_range:
            grid = parse_dim(grid)
            for block in block_range:
                block = parse_dim(block)

                # Simple performance model
                parallelism = grid[0] * grid[1] * grid[2]
                latency = 1.0 / max(parallelism, 1)

                candidates.append(
                    {
                        "grid_dim": grid,
                        "block_dim": block,
                        "latency_ms": latency,
                        "verified": True,
                    }
                )

        best = min(candidates, key=lambda x: x["latency_ms"]) if candidates else None

        return {
            "graphs": [],
            "candidates": candidates,
            "best": best,
            "feedback": (
                {
                    "partition_id": partition.get("partition_id", 0),
                    "total_partitions": partition.get("total_partitions", 1),
                    "candidates": candidates,
                    "total_states_explored": len(candidates),
                    "valid_graphs_found": len(candidates),
                    "best_performance_ms": best["latency_ms"] if best else float("inf"),
                }
                if collect_feedback
                else None
            ),
        }

    @property
    def backend(self) -> str:
        """Get the active search backend."""
        return self._backend


def create_ray_worker(num_cpus: float = 1.0):
    """
    Create a Ray remote worker class.

    Args:
        num_cpus: Number of CPUs per worker

    Returns:
        Ray remote actor class
    """
    import ray

    @ray.remote(num_cpus=num_cpus)
    class RaySearchWorker:
        """
        Ray remote actor for distributed search.

        Runs on CPU, no GPU required.
        """

        def __init__(self, worker_id: int):
            self.worker_id = worker_id
            self._local_worker = SearchWorker(worker_id)

        def search(
            self,
            graph_json: str,
            partition_json: str,
            config_json: str,
            collect_feedback: bool = True,
        ) -> Dict[str, Any]:
            """Execute search on assigned partition."""
            return self._local_worker.search(
                graph_json,
                partition_json,
                config_json,
                collect_feedback,
            )

        def get_worker_id(self) -> int:
            """Get worker ID."""
            return self.worker_id

        def is_ready(self) -> bool:
            """Check if worker is ready."""
            return True

    return RaySearchWorker


def create_workers(
    num_workers: int,
    use_ray: bool = True,
    num_cpus_per_worker: float = 1.0,
) -> List[Any]:
    """
    Create search workers.

    Args:
        num_workers: Number of workers to create
        use_ray: Whether to use Ray for distributed execution
        num_cpus_per_worker: CPUs allocated per worker (Ray only)

    Returns:
        List of workers (local or Ray actors)
    """
    if use_ray:
        try:
            import ray

            if not ray.is_initialized():
                ray.init()

            RayWorker = create_ray_worker(num_cpus_per_worker)
            workers = [RayWorker.remote(i) for i in range(num_workers)]

            # Wait for workers to be ready
            ray.get([w.is_ready.remote() for w in workers])

            return workers

        except ImportError:
            print("Warning: Ray not available, falling back to local workers")

    # Local workers
    return [SearchWorker(i) for i in range(num_workers)]
