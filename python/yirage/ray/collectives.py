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
Ray Collective Operations for Distributed Training.

Provides efficient collective patterns for:
1. Scatter-gather for distributed search
2. All-reduce for gradient aggregation
3. Broadcast for configuration sharing
4. Ring-reduce for efficient communication
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, TypeVar, Generic
import json

# Check Ray availability
try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

T = TypeVar("T")


@dataclass
class CollectiveConfig:
    """Configuration for collective operations."""

    # Communication settings
    timeout_s: float = 300.0
    max_retries: int = 3

    # Batching
    batch_size: int = 100

    # Compression
    use_compression: bool = False


class CollectiveOperations:
    """
    Collective operations for distributed computation.

    Provides patterns commonly used in distributed ML training
    and kernel optimization search.
    """

    def __init__(self, config: Optional[CollectiveConfig] = None):
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray not installed")

        self.config = config or CollectiveConfig()

    def scatter(
        self,
        data: List[T],
        workers: List[Any],
        process_fn: Callable[[T, int], Any],
    ) -> List[Any]:
        """
        Scatter data to workers and gather results.

        Args:
            data: Data items to distribute
            workers: Ray actor handles
            process_fn: Function to apply on each worker

        Returns:
            Results from all workers
        """
        if not ray.is_initialized():
            raise RuntimeError("Ray not initialized")

        # Distribute data round-robin
        num_workers = len(workers)
        futures = []

        for i, item in enumerate(data):
            worker = workers[i % num_workers]
            # Assuming workers have a process method
            future = process_fn(worker, item, i)
            futures.append(future)

        # Gather results
        return ray.get(futures, timeout=self.config.timeout_s)

    def gather(
        self,
        workers: List[Any],
        get_fn: Callable[[Any], Any],
    ) -> List[Any]:
        """
        Gather data from all workers.

        Args:
            workers: Ray actor handles
            get_fn: Function to get data from each worker

        Returns:
            Data from all workers
        """
        futures = [get_fn(w) for w in workers]
        return ray.get(futures, timeout=self.config.timeout_s)

    def broadcast(
        self,
        data: T,
        workers: List[Any],
        set_fn: Callable[[Any, T], Any],
    ) -> None:
        """
        Broadcast data to all workers.

        Uses Ray object store for efficient single-copy broadcast.

        Args:
            data: Data to broadcast
            workers: Ray actor handles
            set_fn: Function to set data on each worker
        """
        # Put data in object store (single copy)
        data_ref = ray.put(data)

        # All workers reference same object
        futures = [set_fn(w, data_ref) for w in workers]
        ray.get(futures, timeout=self.config.timeout_s)

    def reduce(
        self,
        workers: List[Any],
        get_fn: Callable[[Any], Dict],
        reduce_fn: Callable[[List[Dict]], Dict],
    ) -> Dict:
        """
        Reduce data from all workers.

        Args:
            workers: Ray actor handles
            get_fn: Function to get data from each worker
            reduce_fn: Function to aggregate results

        Returns:
            Aggregated result
        """
        # Gather from all workers
        results = self.gather(workers, get_fn)

        # Apply reduction
        return reduce_fn(results)

    def all_reduce(
        self,
        workers: List[Any],
        get_fn: Callable[[Any], Dict],
        reduce_fn: Callable[[List[Dict]], Dict],
        set_fn: Callable[[Any, Dict], Any],
    ) -> Dict:
        """
        All-reduce: reduce data and broadcast result to all workers.

        Args:
            workers: Ray actor handles
            get_fn: Function to get data from each worker
            reduce_fn: Function to aggregate results
            set_fn: Function to set result on each worker

        Returns:
            Aggregated result
        """
        # Reduce
        result = self.reduce(workers, get_fn, reduce_fn)

        # Broadcast result back
        self.broadcast(result, workers, set_fn)

        return result


# Utility functions for common reductions


def sum_reduce(results: List[Dict]) -> Dict:
    """Sum reduction for numeric dictionaries."""
    if not results:
        return {}

    aggregated = {}
    for key in results[0].keys():
        values = [r.get(key, 0) for r in results]
        if all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = sum(values)
        else:
            # For non-numeric, take first
            aggregated[key] = values[0]

    return aggregated


def mean_reduce(results: List[Dict]) -> Dict:
    """Mean reduction for numeric dictionaries."""
    if not results:
        return {}

    aggregated = {}
    for key in results[0].keys():
        values = [r.get(key, 0) for r in results]
        if all(isinstance(v, (int, float)) for v in values):
            aggregated[key] = sum(values) / len(values)
        else:
            aggregated[key] = values[0]

    return aggregated


def min_reduce(results: List[Dict], key: str = "latency_ms") -> Dict:
    """Find minimum by specified key."""
    if not results:
        return {}

    return min(results, key=lambda x: x.get(key, float("inf")))


def max_reduce(results: List[Dict], key: str = "throughput") -> Dict:
    """Find maximum by specified key."""
    if not results:
        return {}

    return max(results, key=lambda x: x.get(key, 0))


def concat_reduce(results: List[Dict], list_key: str = "candidates") -> Dict:
    """Concatenate list values."""
    if not results:
        return {list_key: []}

    all_items = []
    for r in results:
        items = r.get(list_key, [])
        all_items.extend(items)

    return {list_key: all_items}


# Higher-level distributed patterns


class DistributedSearchPattern:
    """
    Pattern for distributed kernel search.

    Implements efficient parallel search with result aggregation.
    """

    def __init__(
        self,
        num_workers: int = 4,
        config: Optional[CollectiveConfig] = None,
    ):
        self.num_workers = num_workers
        self.collectives = CollectiveOperations(config)

    def parallel_search(
        self,
        graph: Dict,
        search_space: Dict,
        workers: List[Any],
    ) -> Dict:
        """
        Execute parallel search across workers.

        Args:
            graph: Computation graph
            search_space: Search configuration space
            workers: Worker actors

        Returns:
            Aggregated search results with best kernel
        """
        # Broadcast graph to all workers
        graph_ref = ray.put(graph)

        # Partition search space
        partitions = self._partition_search_space(search_space, len(workers))

        # Scatter partitions to workers
        @ray.remote
        def search_worker(worker, graph, partition, worker_id):
            return worker.search.remote(graph, partition, {"worker_id": worker_id})

        futures = []
        for i, (worker, partition) in enumerate(zip(workers, partitions)):
            future = worker.search.remote(graph_ref, partition, {"worker_id": i})
            futures.append(future)

        # Gather results
        results = ray.get(futures)

        # Reduce to find best
        all_candidates = []
        for r in results:
            all_candidates.extend(r.get("candidates", []))

        best = min(
            [c for c in all_candidates if c.get("verified", False)],
            key=lambda x: x.get("latency_ms", float("inf")),
            default=None,
        )

        return {
            "best": best,
            "all_candidates": all_candidates,
            "num_workers": len(workers),
            "total_candidates": len(all_candidates),
        }

    def _partition_search_space(
        self,
        search_space: Dict,
        num_partitions: int,
    ) -> List[Dict]:
        """Partition search space for workers."""
        grid_dims = search_space.get("grid_dims", [(1, 1, 1)])
        block_dims = search_space.get("block_dims", [(128, 1, 1)])

        grids_per_partition = max(1, len(grid_dims) // num_partitions)

        partitions = []
        for i in range(num_partitions):
            start = i * grids_per_partition
            end = start + grids_per_partition if i < num_partitions - 1 else len(grid_dims)

            partitions.append(
                {
                    "partition_id": i,
                    "grid_dim_range": grid_dims[start:end],
                    "block_dim_range": block_dims,
                }
            )

        return partitions


class DistributedTrainingPattern:
    """
    Pattern for distributed RL training.

    Supports gradient synchronization and experience sharing.
    """

    def __init__(
        self,
        num_workers: int = 4,
        config: Optional[CollectiveConfig] = None,
    ):
        self.num_workers = num_workers
        self.collectives = CollectiveOperations(config)

    def sync_gradients(
        self,
        workers: List[Any],
    ) -> Dict:
        """
        Synchronize gradients across workers.

        Uses all-reduce to aggregate and distribute gradients.

        Args:
            workers: Worker actors with get_gradients/set_gradients methods

        Returns:
            Aggregated gradients
        """
        return self.collectives.all_reduce(
            workers,
            get_fn=lambda w: ray.get(w.get_gradients.remote()),
            reduce_fn=mean_reduce,
            set_fn=lambda w, g: w.set_gradients.remote(g),
        )

    def share_experiences(
        self,
        workers: List[Any],
        sample_size: int = 100,
    ) -> List[Dict]:
        """
        Share experiences across workers.

        Each worker contributes experiences, all receive full set.

        Args:
            workers: Worker actors with get_experiences method
            sample_size: Number of experiences to sample per worker

        Returns:
            All shared experiences
        """
        # Gather experiences
        all_experiences = self.collectives.reduce(
            workers,
            get_fn=lambda w: ray.get(w.get_experiences.remote(sample_size)),
            reduce_fn=lambda results: concat_reduce(results, "experiences"),
        )

        experiences = all_experiences.get("experiences", [])

        # Broadcast back
        exp_ref = ray.put(experiences)
        futures = [w.set_experiences.remote(exp_ref) for w in workers]
        ray.get(futures)

        return experiences


# Export
__all__ = [
    "CollectiveConfig",
    "CollectiveOperations",
    "DistributedSearchPattern",
    "DistributedTrainingPattern",
    "sum_reduce",
    "mean_reduce",
    "min_reduce",
    "max_reduce",
    "concat_reduce",
]
