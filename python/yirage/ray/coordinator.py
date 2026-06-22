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
Distributed search coordinator using Ray.

Coordinates multiple CPU workers to search the configuration space
in parallel. No GPU required for search - GPUs only needed for
final profiling.
"""

from typing import List, Dict, Any, Optional
import json
import time
import os

from .partition import SearchPartition, create_partitions, partition_config_from_search_config
from .feedback import SearchFeedback, save_training_data
from .worker import create_workers, SearchWorker


class DistributedSearchCoordinator:
    """
    Coordinates distributed kernel search using Ray.

    Features:
    - Partitions search space across workers
    - Each worker runs on CPU, calling C++ search core
    - Aggregates results from all workers
    - Collects feedback data for RL training
    """

    def __init__(
        self,
        num_workers: Optional[int] = None,
        ray_address: Optional[str] = None,
        use_ray: bool = True,
    ):
        """
        Initialize coordinator.

        Args:
            num_workers: Number of workers (default: CPU count)
            ray_address: Ray cluster address (default: local)
            use_ray: Whether to use Ray (set False for single-process)
        """
        self.num_workers = num_workers or os.cpu_count() or 4
        self.ray_address = ray_address
        self.use_ray = use_ray

        # Initialize Ray if needed
        if use_ray:
            self._init_ray()

        # Create workers
        self.workers = create_workers(
            num_workers=self.num_workers,
            use_ray=use_ray,
        )

        # Results
        self.all_graphs: List[Dict] = []
        self.all_feedback: List[SearchFeedback] = []
        self.search_stats: Dict[str, Any] = {}

    def _init_ray(self):
        """Initialize Ray."""
        try:
            import ray

            if not ray.is_initialized():
                if self.ray_address:
                    ray.init(address=self.ray_address)
                else:
                    ray.init()

            print(f"Ray initialized: {ray.cluster_resources()}")

        except ImportError:
            print("Warning: Ray not available, using local execution")
            self.use_ray = False

    def parallel_search(
        self,
        computation_graph: Any,
        config: Optional[Dict] = None,
        backend: str = "cuda",
        collect_feedback: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute parallel distributed search.

        Args:
            computation_graph: Target computation graph
            config: Search configuration
            backend: Target backend (cuda, maca, ascend, etc.)
            collect_feedback: Whether to collect RL training data
            verbose: Print progress

        Returns:
            {
                "graphs": [...],  # All valid graphs found
                "best_graph": ...,  # Best graph
                "feedback": [...],  # Feedback from each worker
                "statistics": {...},  # Search statistics
            }
        """
        start_time = time.time()
        config = config or {}

        # Serialize computation graph
        graph_json = self._serialize_graph(computation_graph)

        # Create partitions
        partitions = partition_config_from_search_config(config, self.num_workers)

        if verbose:
            print(f"Created {len(partitions)} partitions for {self.num_workers} workers")
            total_estimated = sum(p.estimated_candidates for p in partitions)
            print(f"Estimated total candidates: {total_estimated}")

        # Prepare config with backend
        full_config = {
            **config,
            "backend": backend,
        }
        config_json = json.dumps(full_config)

        # Dispatch to workers
        if self.use_ray:
            results = self._parallel_search_ray(
                graph_json, partitions, config_json, collect_feedback, verbose
            )
        else:
            results = self._parallel_search_local(
                graph_json, partitions, config_json, collect_feedback, verbose
            )

        # Aggregate results
        elapsed = time.time() - start_time

        all_graphs = []
        all_feedback = []
        total_states = 0
        total_valid = 0

        for result in results:
            all_graphs.extend(result.get("graphs", []))

            if result.get("feedback"):
                fb = SearchFeedback.from_dict(result["feedback"])
                all_feedback.append(fb)
                total_states += fb.total_states_explored
                total_valid += fb.valid_graphs_found

        # Store results
        self.all_graphs = all_graphs
        self.all_feedback = all_feedback
        self.search_stats = {
            "num_workers": self.num_workers,
            "num_partitions": len(partitions),
            "total_states_explored": total_states,
            "valid_graphs_found": total_valid,
            "elapsed_seconds": elapsed,
            "states_per_second": total_states / max(elapsed, 0.001),
        }

        if verbose:
            print(f"\nSearch completed in {elapsed:.2f}s")
            print(f"States explored: {total_states}")
            print(f"Valid graphs: {total_valid}")
            print(f"Rate: {self.search_stats['states_per_second']:.0f} states/s")

        # Select best
        best_graph = self._select_best(all_graphs)

        return {
            "graphs": all_graphs,
            "best_graph": best_graph,
            "feedback": [fb.to_dict() for fb in all_feedback],
            "statistics": self.search_stats,
        }

    def _parallel_search_ray(
        self,
        graph_json: str,
        partitions: List[SearchPartition],
        config_json: str,
        collect_feedback: bool,
        verbose: bool,
    ) -> List[Dict]:
        """Execute parallel search using Ray."""
        import ray

        # Dispatch tasks
        futures = []
        for worker, partition in zip(self.workers, partitions):
            future = worker.search.remote(
                graph_json,
                partition.to_json(),
                config_json,
                collect_feedback,
            )
            futures.append(future)

        # Collect results with progress
        results = []
        if verbose:
            print(f"Waiting for {len(futures)} workers...")

            for i, future in enumerate(futures):
                result = ray.get(future)
                results.append(result)
                print(
                    f"  Worker {i}: {result.get('elapsed_seconds', 0):.2f}s, "
                    f"graphs: {len(result.get('graphs', []))}"
                )
        else:
            results = ray.get(futures)

        return results

    def _parallel_search_local(
        self,
        graph_json: str,
        partitions: List[SearchPartition],
        config_json: str,
        collect_feedback: bool,
        verbose: bool,
    ) -> List[Dict]:
        """Execute parallel search locally (single process)."""
        results = []

        for i, (worker, partition) in enumerate(zip(self.workers, partitions)):
            if verbose:
                print(f"  Worker {i} starting...")

            result = worker.search(
                graph_json,
                partition.to_json(),
                config_json,
                collect_feedback,
            )
            results.append(result)

            if verbose:
                print(f"  Worker {i}: {result.get('elapsed_seconds', 0):.2f}s")

        return results

    def _serialize_graph(self, graph: Any) -> str:
        """Serialize computation graph to JSON."""
        import tempfile

        # Try C++ binding first (cy_to_json writes to a file path)
        try:
            from yirage.core import cy_to_json

            cygraph = None
            if hasattr(graph, "cygraph"):
                cygraph = graph.cygraph
            elif hasattr(graph, "p_kgraph"):
                cygraph = graph

            if cygraph is not None:
                fd, path = tempfile.mkstemp(suffix=".json", prefix="yirage_coord_graph_")
                os.close(fd)
                try:
                    cy_to_json(cygraph, path)
                    with open(path, "r", encoding="utf-8") as f:
                        return f.read()
                finally:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        except ImportError:
            pass

        # Fallback to Python methods
        if hasattr(graph, "to_json"):
            return graph.to_json()
        elif isinstance(graph, dict):
            import json

            return json.dumps(graph)

        return "{}"

    def _select_best(self, graphs: List[Dict]) -> Optional[Dict]:
        """Select best graph from candidates based on estimated performance."""
        if not graphs:
            return None

        # Filter verified graphs and select by latency
        verified = [g for g in graphs if g.get("verified", False)]
        if not verified:
            verified = graphs

        # Sort by latency (lower is better)
        sorted_graphs = sorted(
            verified, key=lambda g: g.get("latency_ms", g.get("estimated_latency_ms", float("inf")))
        )

        return sorted_graphs[0] if sorted_graphs else None

    def get_training_data(self) -> Dict[str, Any]:
        """
        Get RL training data from collected feedback.

        Returns:
            {
                "num_samples": int,
                "samples": [...],
            }
        """
        from .feedback import extract_training_samples

        all_samples = []
        for fb in self.all_feedback:
            samples = extract_training_samples(fb)
            all_samples.extend([s.to_dict() for s in samples])

        return {
            "num_samples": len(all_samples),
            "samples": all_samples,
        }

    def save_training_data(self, filepath: str):
        """Save training data to file."""
        save_training_data(self.all_feedback, filepath)

    def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        return self.search_stats

    def shutdown(self):
        """Shutdown workers."""
        if self.use_ray:
            import ray

            for worker in self.workers:
                ray.kill(worker)
        self.workers = []
