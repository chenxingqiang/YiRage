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
Benchmarking utilities for Ray distributed optimization.

Provides performance measurement for:
- Object store throughput
- Worker scaling efficiency
- All-reduce latency
- End-to-end optimization time
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
import json
import statistics

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time_s: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_s": self.total_time_s,
            "mean_time_ms": self.mean_time_ms,
            "std_time_ms": self.std_time_ms,
            "min_time_ms": self.min_time_ms,
            "max_time_ms": self.max_time_ms,
            "throughput": self.throughput,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean_time_ms:.2f} ± {self.std_time_ms:.2f} ms "
            f"(min={self.min_time_ms:.2f}, max={self.max_time_ms:.2f}, n={self.iterations})"
        )


def _compute_stats(times_ms: List[float]) -> Dict[str, float]:
    """Compute statistics from timing data."""
    if not times_ms:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}

    return {
        "mean": statistics.mean(times_ms),
        "std": statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
        "min": min(times_ms),
        "max": max(times_ms),
    }


def benchmark_object_store(
    data_sizes_kb: List[int] = [1, 10, 100, 1000, 10000],
    iterations: int = 10,
) -> List[BenchmarkResult]:
    """
    Benchmark Ray object store put/get performance.

    Args:
        data_sizes_kb: Data sizes to test in KB
        iterations: Number of iterations per size

    Returns:
        List of benchmark results
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray not available")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    results = []

    for size_kb in data_sizes_kb:
        # Create data
        data = {"payload": "x" * (size_kb * 1024)}

        # Benchmark put
        put_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            ref = ray.put(data)
            put_times.append((time.perf_counter() - start) * 1000)

        put_stats = _compute_stats(put_times)
        results.append(
            BenchmarkResult(
                name=f"object_store_put_{size_kb}kb",
                iterations=iterations,
                total_time_s=sum(put_times) / 1000,
                mean_time_ms=put_stats["mean"],
                std_time_ms=put_stats["std"],
                min_time_ms=put_stats["min"],
                max_time_ms=put_stats["max"],
                throughput=size_kb / put_stats["mean"] * 1000 if put_stats["mean"] > 0 else 0,
                metadata={"size_kb": size_kb, "operation": "put"},
            )
        )

        # Benchmark get
        ref = ray.put(data)
        get_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = ray.get(ref)
            get_times.append((time.perf_counter() - start) * 1000)

        get_stats = _compute_stats(get_times)
        results.append(
            BenchmarkResult(
                name=f"object_store_get_{size_kb}kb",
                iterations=iterations,
                total_time_s=sum(get_times) / 1000,
                mean_time_ms=get_stats["mean"],
                std_time_ms=get_stats["std"],
                min_time_ms=get_stats["min"],
                max_time_ms=get_stats["max"],
                throughput=size_kb / get_stats["mean"] * 1000 if get_stats["mean"] > 0 else 0,
                metadata={"size_kb": size_kb, "operation": "get"},
            )
        )

    return results


def benchmark_worker_scaling(
    worker_counts: List[int] = [1, 2, 4, 8],
    task_count: int = 100,
    task_duration_ms: float = 10.0,
) -> List[BenchmarkResult]:
    """
    Benchmark worker scaling efficiency.

    Args:
        worker_counts: Number of workers to test
        task_count: Total tasks to execute
        task_duration_ms: Simulated task duration

    Returns:
        List of benchmark results
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray not available")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    @ray.remote
    def simulated_task(duration_ms: float) -> float:
        import time

        time.sleep(duration_ms / 1000)
        return duration_ms

    results = []

    for num_workers in worker_counts:
        # Limit concurrency to num_workers
        times = []

        for _ in range(3):  # 3 iterations
            start = time.perf_counter()

            # Submit tasks in batches
            futures = []
            for i in range(task_count):
                futures.append(simulated_task.remote(task_duration_ms))

                # Control concurrency
                if len(futures) >= num_workers:
                    ray.get(futures)
                    futures = []

            # Get remaining
            if futures:
                ray.get(futures)

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        stats = _compute_stats(times)

        # Calculate efficiency
        ideal_time = (task_count * task_duration_ms) / num_workers
        efficiency = ideal_time / stats["mean"] * 100 if stats["mean"] > 0 else 0

        results.append(
            BenchmarkResult(
                name=f"scaling_{num_workers}_workers",
                iterations=3,
                total_time_s=sum(times) / 1000,
                mean_time_ms=stats["mean"],
                std_time_ms=stats["std"],
                min_time_ms=stats["min"],
                max_time_ms=stats["max"],
                throughput=task_count / (stats["mean"] / 1000) if stats["mean"] > 0 else 0,
                metadata={
                    "num_workers": num_workers,
                    "task_count": task_count,
                    "task_duration_ms": task_duration_ms,
                    "efficiency_pct": efficiency,
                },
            )
        )

    return results


def benchmark_all_reduce(
    gradient_sizes: List[int] = [100, 1000, 10000],
    num_workers: int = 4,
    iterations: int = 10,
) -> List[BenchmarkResult]:
    """
    Benchmark all-reduce gradient aggregation.

    Args:
        gradient_sizes: Number of gradient elements
        num_workers: Number of simulated workers
        iterations: Number of iterations

    Returns:
        List of benchmark results
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray not available")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    results = []

    for size in gradient_sizes:
        times = []

        for _ in range(iterations):
            # Create gradient data
            gradients = [
                {f"param_{j}": float(i + j) for j in range(size)} for i in range(num_workers)
            ]

            start = time.perf_counter()

            # Put in object store
            refs = [ray.put(g) for g in gradients]

            # Reduce - pass refs individually so Ray auto-resolves them
            @ray.remote
            def reduce_grads(*grads):
                result = {}
                for key in grads[0].keys():
                    values = [g[key] for g in grads]
                    result[key] = sum(values) / len(values)
                return result

            reduced = ray.get(reduce_grads.remote(*refs))

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        stats = _compute_stats(times)

        results.append(
            BenchmarkResult(
                name=f"all_reduce_{size}_params",
                iterations=iterations,
                total_time_s=sum(times) / 1000,
                mean_time_ms=stats["mean"],
                std_time_ms=stats["std"],
                min_time_ms=stats["min"],
                max_time_ms=stats["max"],
                throughput=size / stats["mean"] * 1000 if stats["mean"] > 0 else 0,
                metadata={
                    "gradient_size": size,
                    "num_workers": num_workers,
                },
            )
        )

    return results


def run_all_benchmarks(verbose: bool = True) -> Dict[str, List[BenchmarkResult]]:
    """
    Run all benchmarks and return results.

    Args:
        verbose: Print results as they complete

    Returns:
        Dictionary of benchmark results by category
    """
    if not RAY_AVAILABLE:
        raise RuntimeError("Ray not available")

    results = {}

    if verbose:
        print("=" * 60)
        print("  Ray Integration Benchmarks")
        print("=" * 60)

    # Object store
    if verbose:
        print("\n[Object Store]")
    results["object_store"] = benchmark_object_store()
    if verbose:
        for r in results["object_store"]:
            print(f"  {r}")

    # Worker scaling
    if verbose:
        print("\n[Worker Scaling]")
    results["scaling"] = benchmark_worker_scaling(
        worker_counts=[1, 2, 4],
        task_count=20,
        task_duration_ms=5.0,
    )
    if verbose:
        for r in results["scaling"]:
            eff = r.metadata.get("efficiency_pct", 0)
            print(f"  {r} (efficiency: {eff:.1f}%)")

    # All-reduce
    if verbose:
        print("\n[All-Reduce]")
    results["all_reduce"] = benchmark_all_reduce(
        gradient_sizes=[100, 1000],
        iterations=5,
    )
    if verbose:
        for r in results["all_reduce"]:
            print(f"  {r}")

    if verbose:
        print("\n" + "=" * 60)
        print("  Benchmarks Complete")
        print("=" * 60)

    return results


def save_benchmark_results(
    results: Dict[str, List[BenchmarkResult]],
    filepath: str,
) -> None:
    """Save benchmark results to JSON file."""
    data = {
        category: [r.to_dict() for r in result_list] for category, result_list in results.items()
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


# Export
__all__ = [
    "BenchmarkResult",
    "benchmark_object_store",
    "benchmark_worker_scaling",
    "benchmark_all_reduce",
    "run_all_benchmarks",
    "save_benchmark_results",
]


if __name__ == "__main__":
    results = run_all_benchmarks(verbose=True)
