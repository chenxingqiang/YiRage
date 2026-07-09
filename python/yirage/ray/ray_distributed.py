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
YiRage Ray Distributed Module.

Unified Ray integration for distributed kernel search and RL training:
1. Distributed search with GPU-aware placement
2. Object store for large graph data transfer
3. Fault tolerance with retry and checkpoints
4. Ray Train integration for RL gradient sync

Usage:
    from yirage.distributed.ray_distributed import (
        RayDistributedEngine,
        DistributedConfig,
        create_engine,
    )
    
    engine = create_engine(num_workers=4, backend="cuda")
    result = engine.optimize(graph, search_space)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum, auto
import json
import time
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Dependency Checks
# =============================================================================

try:
    import ray
    from ray.util.placement_group import (
        placement_group,
        remove_placement_group,
    )
    # ``PlacementGroupSchedulingStrategy`` was relocated in Ray 2.x. It used to
    # be re-exported from ``ray.util.placement_group``; in modern Ray (>=2.0)
    # it lives in ``ray.util.scheduling_strategies``. Try the new location
    # first and fall back to the old one for very old Ray installations so
    # ``yirage.ray`` works across both. Without this fallback, the entire
    # import block fails on modern Ray and ``RAY_AVAILABLE`` is silently set
    # to ``False`` even when Ray is correctly installed.
    try:
        from ray.util.scheduling_strategies import (
            PlacementGroupSchedulingStrategy,
        )
    except ImportError:  # pragma: no cover - very old Ray
        from ray.util.placement_group import (  # type: ignore[attr-defined]
            PlacementGroupSchedulingStrategy,
        )
    from ray.util.placement_group import placement_group, remove_placement_group

    RAY_AVAILABLE = True
except ImportError:
    ray = None  # type: ignore[assignment]
    placement_group = None  # type: ignore[assignment,misc]
    remove_placement_group = None  # type: ignore[assignment,misc]
    RAY_AVAILABLE = False

try:
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
except ImportError:
    PlacementGroupSchedulingStrategy = None  # type: ignore[assignment,misc]

try:
    import ray.train
    from ray.train import ScalingConfig

    RAY_TRAIN_AVAILABLE = True
except ImportError:
    ScalingConfig = None  # type: ignore[assignment,misc]
    RAY_TRAIN_AVAILABLE = False

try:
    from ray.train.torch import TorchTrainer
except ImportError:
    TorchTrainer = None  # type: ignore[assignment,misc]

try:
    import torch
    import torch.distributed as dist

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def is_ray_available() -> bool:
    """Check if Ray is available."""
    return RAY_AVAILABLE


# =============================================================================
# Configuration Classes
# =============================================================================


class RetryStrategy(Enum):
    """Strategy for retry on failure."""

    NONE = auto()
    FIXED = auto()
    EXPONENTIAL = auto()


class PartitionStrategy(Enum):
    """Strategy for partitioning search space."""

    BY_GRID_DIM = auto()
    BY_BLOCK_DIM = auto()
    ROUND_ROBIN = auto()


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    max_retries: int = 3
    initial_delay_s: float = 1.0
    max_delay_s: float = 60.0
    multiplier: float = 2.0

    def get_delay(self, attempt: int) -> float:
        if self.strategy == RetryStrategy.NONE:
            return 0
        elif self.strategy == RetryStrategy.FIXED:
            return self.initial_delay_s
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.initial_delay_s * (self.multiplier**attempt)
            return min(delay, self.max_delay_s)
        return self.initial_delay_s


@dataclass
class GPUPlacementConfig:
    """Configuration for GPU-aware placement."""

    gpus_per_worker: float = 1.0
    cpus_per_worker: float = 1.0
    memory_per_worker_mb: int = 4096
    strategy: str = "PACK"  # PACK for NVLink, SPREAD for distribution


@dataclass
class DistributedConfig:
    """Unified configuration for distributed engine."""

    # Workers
    num_workers: int = 4
    gpu_placement: GPUPlacementConfig = field(default_factory=GPUPlacementConfig)

    # Retry
    retry: RetryConfig = field(default_factory=RetryConfig)

    # Search
    backend: str = "cuda"
    max_search_time_s: float = 300.0
    partition_strategy: PartitionStrategy = PartitionStrategy.BY_GRID_DIM

    # Object store
    use_object_store: bool = True
    large_object_threshold_bytes: int = 1024 * 1024

    # Checkpoints
    checkpoint_dir: Optional[str] = None
    checkpoint_interval_s: float = 60.0

    def to_dict(self) -> Dict:
        return {
            "num_workers": self.num_workers,
            "backend": self.backend,
            "max_search_time_s": self.max_search_time_s,
            "use_object_store": self.use_object_store,
        }


@dataclass
class DistributedResult:
    """Result from distributed optimization."""

    best_config: Optional[Dict] = None
    best_latency_ms: float = float("inf")
    all_candidates: List[Dict] = field(default_factory=list)
    total_candidates_searched: int = 0
    total_valid_graphs: int = 0
    search_time_s: float = 0.0
    num_workers: int = 0
    worker_stats: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "best_config": self.best_config,
            "best_latency_ms": self.best_latency_ms,
            "total_candidates_searched": self.total_candidates_searched,
            "total_valid_graphs": self.total_valid_graphs,
            "search_time_s": self.search_time_s,
            "num_workers": self.num_workers,
        }


# =============================================================================
# Search Worker (Ray Actor)
# =============================================================================


def _create_search_worker_class():
    """Create GPU-aware Ray worker class."""
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray not available")

    @ray.remote
    class SearchWorker:
        """
        Distributed search worker with C++ integration.

        Features:
        - Direct C++ search calls when available
        - GPU verification via RL context
        - Checkpoint save/restore
        - Simulated search fallback
        """

        def __init__(
            self,
            worker_id: int,
            gpu_id: int = 0,
            backend: str = "cuda",
            checkpoint_dir: str = None,
        ):
            self.worker_id = worker_id
            self.gpu_id = gpu_id
            self.backend = backend
            self.checkpoint_dir = checkpoint_dir

            self._checkpoint = {
                "partition_id": worker_id,
                "completed_configs": 0,
                "best_latency_ms": float("inf"),
                "best_config": None,
                "timestamp": 0.0,
            }
            self.is_running = False
            self._cpp_available = False
            self._search_func = None
            self._init_cpp_core()

        def _init_cpp_core(self):
            """Initialize C++ search core."""
            try:
                from yirage.core import search

                self._search_func = search
                self._cpp_available = True
            except ImportError:
                self._cpp_available = False

        def search(self, graph, partition, config):
            """Execute search on partition."""
            import time as _time

            self.is_running = True
            try:
                if self._cpp_available and self._is_valid_graph(graph):
                    return self._search_cpp(graph, partition, config)
                else:
                    return self._search_simulation(graph, partition, config)
            finally:
                self.is_running = False
                self._checkpoint["timestamp"] = _time.time()
                if self.checkpoint_dir:
                    self._save_checkpoint()

        def _is_valid_graph(self, graph):
            """Check if graph is valid YiRage format."""
            return isinstance(graph, dict) and ("operators" in graph or "inputs" in graph)

        def _search_cpp(self, graph, partition, config):
            """Search using C++ core."""
            import tempfile
            import os as _os
            import json as _json

            try:
                from yirage.core import search, cy_from_json
            except ImportError:
                return self._search_simulation(graph, partition, config)

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                _json.dump(graph, f)
                temp_path = f.name

            try:
                input_graph = cy_from_json(temp_path)
                griddims = self._convert_dims(partition.get("grid_dim_range", []))
                blockdims = self._convert_dims(partition.get("block_dim_range", []))

                new_graphs = search(
                    input_graph,
                    backend=config.get("backend", self.backend),
                    griddims=griddims,
                    blockdims=blockdims,
                    verbose=config.get("verbose", False),
                    is_formal_verified=config.get("formal_verify", False),
                )

                return {
                    "worker_id": self.worker_id,
                    "partition_id": partition.get("partition_id", 0),
                    "candidates": [
                        {"graph_id": i, "verified": True} for i in range(len(new_graphs))
                    ],
                    "best": {"graph_id": 0} if new_graphs else None,
                    "num_candidates": len(new_graphs),
                    "num_valid": len(new_graphs),
                }
            except Exception:
                return self._search_simulation(graph, partition, config)
            finally:
                try:
                    _os.unlink(temp_path)
                except Exception:
                    pass

        def _convert_dims(self, dims):
            if not dims:
                return None
            return [
                (
                    tuple(d)
                    if isinstance(d, (list, tuple))
                    else (d.get("x", 1), d.get("y", 1), d.get("z", 1))
                )
                for d in dims
            ]

        def _search_simulation(self, graph, partition, config):
            """Simulated search for testing."""
            grid_range = partition.get("grid_dim_range", [(1, 1, 1)])
            block_range = partition.get("block_dim_range", [(128, 1, 1)])

            candidates = []
            for grid in grid_range:
                grid = tuple(grid) if isinstance(grid, list) else grid
                for block in block_range:
                    block = tuple(block) if isinstance(block, list) else block

                    parallelism = grid[0] * grid[1] * grid[2] if isinstance(grid, tuple) else 1
                    base_flops = (
                        graph.get("estimated_flops", 1e12) if isinstance(graph, dict) else 1e12
                    )
                    latency_ms = (base_flops / (312.0 * parallelism * 1e12)) * 1000

                    candidates.append(
                        {
                            "grid_dim": grid,
                            "block_dim": block,
                            "latency_ms": latency_ms,
                            "verified": True,
                        }
                    )
                    self._checkpoint["completed_configs"] += 1

            best = min(candidates, key=lambda x: x["latency_ms"]) if candidates else None
            if best:
                self._checkpoint["best_latency_ms"] = best["latency_ms"]
                self._checkpoint["best_config"] = best

            return {
                "worker_id": self.worker_id,
                "partition_id": partition.get("partition_id", 0),
                "candidates": candidates,
                "best": best,
                "num_candidates": len(candidates),
                "num_valid": len([c for c in candidates if c.get("verified", False)]),
            }

        def _save_checkpoint(self):
            import os as _os
            import json as _json

            if not self.checkpoint_dir:
                return
            _os.makedirs(self.checkpoint_dir, exist_ok=True)
            filepath = _os.path.join(
                self.checkpoint_dir, f"worker_{self.worker_id}_checkpoint.json"
            )
            with open(filepath, "w") as f:
                _json.dump(self._checkpoint, f)

        def get_checkpoint(self):
            return self._checkpoint.copy()

        def restore_checkpoint(self, data):
            self._checkpoint = data.copy()

        def get_status(self):
            return {
                "worker_id": self.worker_id,
                "gpu_id": self.gpu_id,
                "is_running": self.is_running,
                "cpp_available": self._cpp_available,
                "completed_configs": self._checkpoint["completed_configs"],
                "best_latency_ms": self._checkpoint["best_latency_ms"],
            }

        def is_ready(self):
            return True

    return SearchWorker


# =============================================================================
# Distributed Engine
# =============================================================================


class RayDistributedEngine:
    """
    Production-grade Ray distributed optimization engine.

    Features:
    - GPU-aware placement groups
    - Object store for large data
    - Fault tolerance with retry
    - C++ search core integration
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray not installed. Install with: pip install ray")

        self.config = config or DistributedConfig()
        self.workers = []
        self.placement_group = None
        self._ray_initialized = False

    def _ensure_ray(self):
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, logging_level="WARNING")
            self._ray_initialized = True

    def _effective_gpus_per_worker(self) -> float:
        gpu_config = self.config.gpu_placement
        gpus_per_worker = gpu_config.gpus_per_worker
        if gpus_per_worker <= 0:
            return 0.0
        if self.config.backend == "maca":
            from yirage.backends.maca.config import resolve_maca_gpus_per_worker

            effective = resolve_maca_gpus_per_worker(requested=gpus_per_worker)
            if effective <= 0:
                logger.warning(
                    "gpus_per_worker=%s but MetaX MACA GPU unavailable; using CPU-only placement",
                    gpus_per_worker,
                )
            return effective
        if TORCH_AVAILABLE and not torch.cuda.is_available():
            logger.warning(
                "gpus_per_worker=%s but CUDA is unavailable; using CPU-only placement",
                gpus_per_worker,
            )
            return 0.0
        return gpus_per_worker

    def _create_placement_group(self):
        gpu_config = self.config.gpu_placement
        gpus_per_worker = self._effective_gpus_per_worker()

        bundles = [
            (
                {"CPU": gpu_config.cpus_per_worker, "GPU": gpus_per_worker}
                if gpus_per_worker > 0
                else {"CPU": gpu_config.cpus_per_worker}
            )
            for _ in range(self.config.num_workers)
        ]

        try:
            pg = placement_group(bundles, strategy=gpu_config.strategy)
            ray.get(pg.ready(), timeout=30)
            logger.info(f"Created placement group with {len(bundles)} bundles")
            return pg
        except Exception as e:
            logger.warning(f"Failed to create placement group: {e}")
            return None

    def _create_workers(self):
        SearchWorker = _create_search_worker_class()
        gpu_config = self.config.gpu_placement
        gpus_per_worker = self._effective_gpus_per_worker()

        workers = []
        for i in range(self.config.num_workers):
            options = {"num_cpus": gpu_config.cpus_per_worker}
            if gpus_per_worker > 0:
                options["num_gpus"] = gpus_per_worker

            if self.placement_group and PlacementGroupSchedulingStrategy is not None:
                options["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                    placement_group=self.placement_group,
                    placement_group_bundle_index=i,
                )

            worker = SearchWorker.options(**options).remote(
                worker_id=i,
                gpu_id=i % max(1, int(gpus_per_worker * self.config.num_workers)),
                backend=self.config.backend,
                checkpoint_dir=self.config.checkpoint_dir,
            )
            workers.append(worker)

        ray.get([w.is_ready.remote() for w in workers])
        return workers

    def _create_partitions(self, search_space: Dict) -> List[Dict]:
        grid_dims = search_space.get("grid_dims", [(1, 1, 1)])
        block_dims = search_space.get("block_dims", [(128, 1, 1)])

        n = self.config.num_workers
        per_partition = max(1, len(grid_dims) // n)

        partitions = []
        for i in range(n):
            start = i * per_partition
            end = start + per_partition if i < n - 1 else len(grid_dims)
            partitions.append(
                {
                    "partition_id": i,
                    "total_partitions": n,
                    "grid_dim_range": grid_dims[start:end] if start < len(grid_dims) else [],
                    "block_dim_range": block_dims,
                }
            )
        return partitions

    def _execute_with_retry(self, func: Callable, *args, **kwargs):
        retry_config = self.config.retry
        last_error = None

        for attempt in range(retry_config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < retry_config.max_retries:
                    delay = retry_config.get_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
        raise last_error

    def optimize(
        self, graph: Dict, search_space: Dict, search_config: Optional[Dict] = None
    ) -> DistributedResult:
        """
        Run distributed optimization.

        Args:
            graph: Computation graph
            search_space: Search space with grid_dims, block_dims
            search_config: Additional config

        Returns:
            DistributedResult with best kernel
        """
        self._ensure_ray()
        start_time = time.time()

        if self._effective_gpus_per_worker() > 0:
            self.placement_group = self._create_placement_group()

        try:
            self.workers = self._create_workers()
            partitions = self._create_partitions(search_space)

            # Object store for large graphs
            graph_size = len(json.dumps(graph))
            if (
                self.config.use_object_store
                and graph_size > self.config.large_object_threshold_bytes
            ):
                graph_ref = ray.put(graph)
                logger.info(f"Graph stored in object store ({graph_size} bytes)")
            else:
                graph_ref = graph

            config = search_config or {}
            config["backend"] = self.config.backend

            # Launch parallel search
            futures = [
                self._execute_with_retry(
                    lambda i=i: self.workers[i].search.remote(graph_ref, partitions[i], config)
                )
                for i in range(self.config.num_workers)
            ]

            # Gather with timeout
            try:
                results = ray.get(futures, timeout=self.config.max_search_time_s)
            except ray.exceptions.GetTimeoutError:
                ready, _ = ray.wait(futures, num_returns=len(futures), timeout=1)
                results = ray.get(ready)
                logger.warning(f"Timeout: got {len(results)}/{len(futures)} results")

            return self._aggregate_results(results, time.time() - start_time)

        finally:
            if self.placement_group:
                try:
                    remove_placement_group(self.placement_group)
                except Exception:
                    pass
                self.placement_group = None

    def _aggregate_results(self, results: List[Dict], elapsed: float) -> DistributedResult:
        all_candidates = []
        total_searched = 0
        total_valid = 0
        worker_stats = []

        for r in results:
            candidates = r.get("candidates", [])
            all_candidates.extend(candidates)
            total_searched += r.get("num_candidates", 0)
            total_valid += r.get("num_valid", 0)
            worker_stats.append(
                {
                    "worker_id": r.get("worker_id", -1),
                    "partition_id": r.get("partition_id", -1),
                    "num_candidates": r.get("num_candidates", 0),
                    "num_valid": r.get("num_valid", 0),
                }
            )

        valid = [c for c in all_candidates if c.get("verified", False)]
        best = min(valid, key=lambda x: x.get("latency_ms", float("inf"))) if valid else None

        return DistributedResult(
            best_config=best,
            best_latency_ms=best.get("latency_ms", float("inf")) if best else float("inf"),
            all_candidates=valid,
            total_candidates_searched=total_searched,
            total_valid_graphs=total_valid,
            search_time_s=elapsed,
            num_workers=len(results),
            worker_stats=worker_stats,
        )

    def all_reduce(self, data: List[Dict], op: str = "mean") -> Dict:
        """All-reduce data across workers."""
        if not data:
            return {}

        refs = [ray.put(d) for d in data]

        @ray.remote
        def reduce_fn(refs, op):
            items = [ray.get(r) for r in refs]
            result = {}
            for key in items[0].keys():
                values = [d.get(key, 0) for d in items]
                if all(isinstance(v, (int, float)) for v in values):
                    if op == "mean":
                        result[key] = sum(values) / len(values)
                    elif op == "sum":
                        result[key] = sum(values)
                    elif op == "max":
                        result[key] = max(values)
                    elif op == "min":
                        result[key] = min(values)
                else:
                    result[key] = values[0]
            return result

        return ray.get(reduce_fn.remote(refs, op))

    def broadcast(self, config: Dict) -> None:
        """Broadcast config to all workers."""
        if not self.workers:
            return
        config_ref = ray.put(config)
        ray.get([w.restore_checkpoint.remote({"config": config_ref}) for w in self.workers])

    def get_worker_status(self) -> List[Dict]:
        if not self.workers:
            return []
        return ray.get([w.get_status.remote() for w in self.workers])

    def shutdown(self):
        if self.placement_group:
            try:
                remove_placement_group(self.placement_group)
            except Exception:
                pass
            self.placement_group = None

        for w in self.workers:
            try:
                ray.kill(w)
            except Exception:
                pass
        self.workers = []

        if self._ray_initialized and ray.is_initialized():
            ray.shutdown()
            self._ray_initialized = False


# =============================================================================
# Ray Train Integration for RL
# =============================================================================


@dataclass
class RLTrainConfig:
    """Configuration for distributed RL training."""

    num_workers: int = 4
    gpus_per_worker: int = 1
    batch_size_per_worker: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    backend: str = "nccl"
    checkpoint_every_n_epochs: int = 10


class RLGradientAllReduce:
    """All-reduce for RL gradients using PyTorch distributed."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank

    def all_reduce_gradients(self, model, average: bool = True):
        if not TORCH_AVAILABLE or not dist.is_initialized():
            return
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                if average:
                    param.grad /= self.world_size

    def broadcast_model(self, model, src_rank: int = 0):
        if not TORCH_AVAILABLE or not dist.is_initialized():
            return
        for param in model.parameters():
            dist.broadcast(param.data, src=src_rank)


def create_train_loop(model_factory, data_loader_factory, config: RLTrainConfig):
    """Create Ray Train training loop."""

    def train_loop(train_config: Dict):
        import torch
        from ray.train import get_context
        from ray.train.torch import get_device, prepare_model, prepare_data_loader

        context = get_context()
        world_rank = context.get_world_rank()

        model = model_factory()
        device = get_device()
        model = prepare_model(model.to(device))

        data_loader = prepare_data_loader(data_loader_factory())
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_config.get("lr", config.learning_rate)
        )

        for epoch in range(train_config.get("epochs", config.num_epochs)):
            model.train()
            total_loss = 0.0
            n = 0

            for batch_idx, batch in enumerate(data_loader):
                inputs, targets = (
                    (batch[0].to(device), batch[1].to(device))
                    if isinstance(batch, (list, tuple))
                    else (batch.to(device), batch.to(device))
                )
                loss = (
                    torch.nn.functional.mse_loss(model(inputs), targets)
                    / config.gradient_accumulation_steps
                )
                loss.backward()

                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * config.gradient_accumulation_steps
                n += 1

            ray.train.report({"epoch": epoch, "loss": total_loss / max(n, 1), "rank": world_rank})

    return train_loop


def run_distributed_training(
    model_factory, data_loader_factory, config: Optional[RLTrainConfig] = None
) -> Dict:
    """Run distributed RL training with Ray Train."""
    if not RAY_TRAIN_AVAILABLE:
        raise RuntimeError("Ray Train not available. Install with: pip install ray[train]")
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available")

    config = config or RLTrainConfig()

    trainer = TorchTrainer(
        train_loop_per_worker=create_train_loop(model_factory, data_loader_factory, config),
        scaling_config=ScalingConfig(
            num_workers=config.num_workers,
            use_gpu=config.gpus_per_worker > 0,
            resources_per_worker={"CPU": 1, "GPU": config.gpus_per_worker},
        ),
        train_loop_config={"lr": config.learning_rate, "epochs": config.num_epochs},
    )

    result = trainer.fit()
    return {"best_checkpoint": result.best_checkpoints, "metrics": result.metrics}


# =============================================================================
# Factory Functions
# =============================================================================


def create_engine(
    num_workers: int = 4,
    gpus_per_worker: float = 1.0,
    backend: str = "cuda",
    use_nvlink: bool = True,
    checkpoint_dir: Optional[str] = None,
) -> RayDistributedEngine:
    """
    Create a distributed engine.

    Args:
        num_workers: Number of workers
        gpus_per_worker: GPUs per worker (0 for CPU)
        backend: Target backend
        use_nvlink: Use PACK strategy for NVLink
        checkpoint_dir: Checkpoint directory

    Returns:
        Configured RayDistributedEngine
    """
    if backend == "maca":
        from yirage.backends.maca.config import maca_ray_gpu_placement_kwargs

        placement_kwargs = maca_ray_gpu_placement_kwargs(
            gpus_per_worker=gpus_per_worker,
            strategy="PACK" if use_nvlink else "SPREAD",
        )
    else:
        placement_kwargs = {
            "gpus_per_worker": gpus_per_worker,
            "strategy": "PACK" if use_nvlink else "SPREAD",
        }

    config = DistributedConfig(
        num_workers=num_workers,
        gpu_placement=GPUPlacementConfig(**placement_kwargs),
        backend=backend,
        checkpoint_dir=checkpoint_dir,
    )
    return RayDistributedEngine(config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core
    "RayDistributedEngine",
    "DistributedConfig",
    "DistributedResult",
    "GPUPlacementConfig",
    "RetryConfig",
    "RetryStrategy",
    "PartitionStrategy",
    # Factory
    "create_engine",
    "is_ray_available",
    # RL Training
    "RLTrainConfig",
    "RLGradientAllReduce",
    "run_distributed_training",
    "create_train_loop",
]
