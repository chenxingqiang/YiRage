# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Core Benchmark Module

Provides precise timing and profiling for kernel performance measurement.
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum, auto


class TimingMethod(Enum):
    """Method for timing kernel execution."""

    PYTHON_PERF = auto()  # time.perf_counter
    TORCH_CUDA = auto()  # torch.cuda.Event
    TORCH_MPS = auto()  # torch.mps synchronization
    HARDWARE_COUNTER = auto()  # Native hardware counters


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarks."""

    warmup_iterations: int = 10
    benchmark_iterations: int = 100
    min_duration_seconds: float = 0.5
    timing_method: TimingMethod = TimingMethod.PYTHON_PERF
    sync_before: bool = True
    sync_after: bool = True
    collect_memory: bool = True
    collect_variance: bool = True
    percentiles: List[float] = field(default_factory=lambda: [50, 90, 95, 99])


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    backend: str

    # Timing statistics (in milliseconds)
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    median_ms: float

    # Percentiles
    percentiles: Dict[str, float] = field(default_factory=dict)

    # Raw data
    all_times_ms: List[float] = field(default_factory=list)

    # Memory (if collected)
    peak_memory_bytes: Optional[int] = None
    allocated_memory_bytes: Optional[int] = None

    # Throughput
    iterations: int = 0
    total_time_seconds: float = 0.0
    throughput_per_second: float = 0.0

    # Performance metrics
    gflops: Optional[float] = None
    memory_bandwidth_gbps: Optional[float] = None

    # Metadata
    device_name: Optional[str] = None
    config: Optional[BenchmarkConfig] = None

    def speedup_over(self, baseline: "BenchmarkResult") -> float:
        """Calculate speedup over baseline."""
        if self.mean_ms <= 0:
            return float("inf")
        return baseline.mean_ms / self.mean_ms

    def __str__(self) -> str:
        return (
            f"{self.name} ({self.backend}): "
            f"{self.mean_ms:.3f} ± {self.std_ms:.3f} ms "
            f"(min: {self.min_ms:.3f}, max: {self.max_ms:.3f})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "backend": self.backend,
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "median_ms": self.median_ms,
            "percentiles": self.percentiles,
            "iterations": self.iterations,
            "throughput_per_second": self.throughput_per_second,
            "gflops": self.gflops,
            "memory_bandwidth_gbps": self.memory_bandwidth_gbps,
            "device_name": self.device_name,
        }


class Benchmark:
    """
    Kernel benchmark runner.

    Example:
        bench = Benchmark("matmul", backend="cuda")

        # Benchmark a function
        result = bench.run(kernel_fn, input_a, input_b)

        # Benchmark a compiled graph
        result = bench.run_graph(compiled_graph, inputs)
    """

    def __init__(
        self,
        name: str,
        backend: str = "cpu",
        config: Optional[BenchmarkConfig] = None,
    ):
        self.name = name
        self.backend = backend
        self.config = config or BenchmarkConfig()

        # Set timing method based on backend
        if backend == "cuda" and self.config.timing_method == TimingMethod.PYTHON_PERF:
            self.config.timing_method = TimingMethod.TORCH_CUDA
        elif backend == "mps" and self.config.timing_method == TimingMethod.PYTHON_PERF:
            self.config.timing_method = TimingMethod.TORCH_MPS

    def run(self, fn: Callable, *args, **kwargs) -> BenchmarkResult:
        """
        Run benchmark on a callable.

        Args:
            fn: Function to benchmark
            *args: Arguments to pass
            **kwargs: Keyword arguments to pass

        Returns:
            BenchmarkResult with timing statistics
        """
        # Warmup
        self._warmup(fn, args, kwargs)

        # Collect memory before
        memory_before = self._get_memory()

        # Run benchmark
        times = self._benchmark(fn, args, kwargs)

        # Collect memory after
        memory_after = self._get_memory()

        # Compute statistics
        return self._compute_result(times, memory_before, memory_after)

    def run_graph(
        self,
        graph: Any,
        inputs: List[Any],
    ) -> BenchmarkResult:
        """
        Run benchmark on a compiled graph.

        Args:
            graph: Compiled kernel graph
            inputs: Input tensors

        Returns:
            BenchmarkResult with timing statistics
        """
        if hasattr(graph, "__call__"):
            return self.run(graph, *inputs)
        elif hasattr(graph, "execute"):
            return self.run(graph.execute, *inputs)
        else:
            raise TypeError("Graph must be callable or have execute method")

    def _warmup(self, fn: Callable, args: tuple, kwargs: dict):
        """Run warmup iterations."""
        self._sync()
        for _ in range(self.config.warmup_iterations):
            fn(*args, **kwargs)
        self._sync()

    def _benchmark(self, fn: Callable, args: tuple, kwargs: dict) -> List[float]:
        """Run benchmark iterations and collect times."""
        times = []

        if self.config.timing_method == TimingMethod.TORCH_CUDA:
            times = self._benchmark_cuda(fn, args, kwargs)
        elif self.config.timing_method == TimingMethod.TORCH_MPS:
            times = self._benchmark_mps(fn, args, kwargs)
        else:
            times = self._benchmark_python(fn, args, kwargs)

        return times

    def _benchmark_python(self, fn: Callable, args: tuple, kwargs: dict) -> List[float]:
        """Benchmark using Python time.perf_counter."""
        times = []
        total_time = 0.0
        iterations = 0

        while (
            iterations < self.config.benchmark_iterations
            or total_time < self.config.min_duration_seconds
        ):

            if self.config.sync_before:
                self._sync()

            start = time.perf_counter()
            fn(*args, **kwargs)

            if self.config.sync_after:
                self._sync()

            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
            total_time += elapsed
            iterations += 1

            if iterations >= self.config.benchmark_iterations * 10:
                break  # Prevent infinite loop

        return times

    def _benchmark_cuda(self, fn: Callable, args: tuple, kwargs: dict) -> List[float]:
        """Benchmark using CUDA events."""
        try:
            import torch

            if not torch.cuda.is_available():
                return self._benchmark_python(fn, args, kwargs)

            times = []

            for _ in range(self.config.benchmark_iterations):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                fn(*args, **kwargs)
                end_event.record()

                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))

            return times
        except ImportError:
            return self._benchmark_python(fn, args, kwargs)

    def _benchmark_mps(self, fn: Callable, args: tuple, kwargs: dict) -> List[float]:
        """Benchmark on MPS backend."""
        try:
            import torch

            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                return self._benchmark_python(fn, args, kwargs)

            times = []

            for _ in range(self.config.benchmark_iterations):
                torch.mps.synchronize()
                start = time.perf_counter()
                fn(*args, **kwargs)
                torch.mps.synchronize()
                elapsed = time.perf_counter() - start
                times.append(elapsed * 1000)

            return times
        except ImportError:
            return self._benchmark_python(fn, args, kwargs)

    def _sync(self):
        """Synchronize device."""
        try:
            import torch

            if self.backend == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elif self.backend == "mps" and hasattr(torch.backends, "mps"):
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
        except ImportError:
            pass

    def _get_memory(self) -> Optional[Dict[str, int]]:
        """Get current memory usage."""
        if not self.config.collect_memory:
            return None

        try:
            import torch

            if self.backend == "cuda" and torch.cuda.is_available():
                return {
                    "allocated": torch.cuda.memory_allocated(),
                    "reserved": torch.cuda.memory_reserved(),
                    "max_allocated": torch.cuda.max_memory_allocated(),
                }
            elif self.backend == "mps" and hasattr(torch.backends, "mps"):
                if torch.backends.mps.is_available():
                    return {
                        "allocated": torch.mps.current_allocated_memory(),
                    }
        except (ImportError, AttributeError):
            pass

        return None

    def _compute_result(
        self,
        times: List[float],
        memory_before: Optional[Dict],
        memory_after: Optional[Dict],
    ) -> BenchmarkResult:
        """Compute benchmark result from times."""
        if not times:
            return BenchmarkResult(
                name=self.name,
                backend=self.backend,
                mean_ms=0,
                std_ms=0,
                min_ms=0,
                max_ms=0,
                median_ms=0,
            )

        mean_ms = statistics.mean(times)
        std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
        min_ms = min(times)
        max_ms = max(times)
        median_ms = statistics.median(times)

        # Percentiles
        percentiles = {}
        sorted_times = sorted(times)
        for p in self.config.percentiles:
            idx = int(len(sorted_times) * p / 100)
            idx = min(idx, len(sorted_times) - 1)
            percentiles[f"p{int(p)}"] = sorted_times[idx]

        # Memory
        peak_memory = None
        if memory_after and "max_allocated" in memory_after:
            peak_memory = memory_after["max_allocated"]

        total_time = sum(times) / 1000  # seconds
        throughput = len(times) / total_time if total_time > 0 else 0

        return BenchmarkResult(
            name=self.name,
            backend=self.backend,
            mean_ms=mean_ms,
            std_ms=std_ms,
            min_ms=min_ms,
            max_ms=max_ms,
            median_ms=median_ms,
            percentiles=percentiles,
            all_times_ms=times if self.config.collect_variance else [],
            peak_memory_bytes=peak_memory,
            iterations=len(times),
            total_time_seconds=total_time,
            throughput_per_second=throughput,
            config=self.config,
        )


def run_benchmark(
    fn: Callable,
    *args,
    name: str = "benchmark",
    backend: str = "cpu",
    warmup: int = 10,
    iterations: int = 100,
    **kwargs,
) -> BenchmarkResult:
    """
    Quick benchmark function.

    Args:
        fn: Function to benchmark
        *args: Arguments to pass
        name: Benchmark name
        backend: Target backend
        warmup: Warmup iterations
        iterations: Benchmark iterations
        **kwargs: Keyword arguments to pass

    Returns:
        BenchmarkResult
    """
    config = BenchmarkConfig(
        warmup_iterations=warmup,
        benchmark_iterations=iterations,
    )
    bench = Benchmark(name, backend, config)
    return bench.run(fn, *args, **kwargs)
