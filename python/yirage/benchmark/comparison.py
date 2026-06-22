# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark Comparison Module

Compare performance across backends and against baselines.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import json

from .benchmark import Benchmark, BenchmarkResult, BenchmarkConfig


@dataclass
class ComparisonResult:
    """Result from comparing benchmarks."""

    name: str
    baseline: BenchmarkResult
    compared: BenchmarkResult
    speedup: float
    improvement_percent: float

    def __str__(self) -> str:
        direction = "faster" if self.speedup > 1 else "slower"
        return (
            f"{self.name}: {self.compared.backend} is {abs(self.speedup):.2f}x "
            f"{direction} than {self.baseline.backend} "
            f"({self.improvement_percent:+.1f}%)"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "baseline_backend": self.baseline.backend,
            "baseline_ms": self.baseline.mean_ms,
            "compared_backend": self.compared.backend,
            "compared_ms": self.compared.mean_ms,
            "speedup": self.speedup,
            "improvement_percent": self.improvement_percent,
        }


def compare_backends(
    fn: Callable,
    args: List[Any],
    backends: List[str],
    name: str = "comparison",
    config: Optional[BenchmarkConfig] = None,
    baseline_backend: Optional[str] = None,
) -> List[ComparisonResult]:
    """
    Compare performance across multiple backends.

    Args:
        fn: Function to benchmark
        args: Arguments to pass
        backends: List of backends to compare
        name: Benchmark name
        config: Benchmark configuration
        baseline_backend: Backend to use as baseline (first if None)

    Returns:
        List of ComparisonResult

    Example:
        results = compare_backends(
            matmul_fn, [a, b],
            backends=["cpu", "mps", "cuda"],
            name="matmul"
        )
        for r in results:
            print(r)
    """
    if not backends:
        return []

    config = config or BenchmarkConfig()
    results: Dict[str, BenchmarkResult] = {}

    # Run benchmarks on each backend
    for backend in backends:
        try:
            bench = Benchmark(name, backend, config)
            results[backend] = bench.run(fn, *args)
        except Exception as e:
            print(f"Failed to benchmark on {backend}: {e}")
            results[backend] = BenchmarkResult(
                name=name,
                backend=backend,
                mean_ms=float("inf"),
                std_ms=0,
                min_ms=float("inf"),
                max_ms=float("inf"),
                median_ms=float("inf"),
            )

    # Determine baseline
    baseline_backend = baseline_backend or backends[0]
    baseline = results.get(baseline_backend)

    if baseline is None:
        return []

    # Compare each to baseline
    comparisons = []
    for backend in backends:
        if backend == baseline_backend:
            continue

        compared = results.get(backend)
        if compared is None:
            continue

        speedup = baseline.mean_ms / compared.mean_ms if compared.mean_ms > 0 else 0
        improvement = (
            (baseline.mean_ms - compared.mean_ms) / baseline.mean_ms * 100
            if baseline.mean_ms > 0
            else 0
        )

        comparisons.append(
            ComparisonResult(
                name=name,
                baseline=baseline,
                compared=compared,
                speedup=speedup,
                improvement_percent=improvement,
            )
        )

    return comparisons


def compare_with_baseline(
    yirage_fn: Callable,
    baseline_fn: Callable,
    args: List[Any],
    name: str = "yirage_vs_baseline",
    backend: str = "cpu",
    config: Optional[BenchmarkConfig] = None,
) -> ComparisonResult:
    """
    Compare YiRage kernel against a baseline (e.g., PyTorch).

    Args:
        yirage_fn: YiRage compiled function
        baseline_fn: Baseline function (e.g., torch.mm)
        args: Arguments to pass
        name: Comparison name
        backend: Target backend
        config: Benchmark configuration

    Returns:
        ComparisonResult

    Example:
        result = compare_with_baseline(
            yirage_matmul, torch.mm,
            [a, b],
            name="matmul_1024",
            backend="cuda"
        )
        print(f"YiRage is {result.speedup:.2f}x faster!")
    """
    config = config or BenchmarkConfig()

    # Benchmark baseline
    baseline_bench = Benchmark(f"{name}_baseline", backend, config)
    baseline_result = baseline_bench.run(baseline_fn, *args)
    baseline_result.backend = f"{backend}_baseline"

    # Benchmark YiRage
    yirage_bench = Benchmark(f"{name}_yirage", backend, config)
    yirage_result = yirage_bench.run(yirage_fn, *args)
    yirage_result.backend = f"{backend}_yirage"

    # Compute speedup
    speedup = baseline_result.mean_ms / yirage_result.mean_ms if yirage_result.mean_ms > 0 else 0
    improvement = (
        (baseline_result.mean_ms - yirage_result.mean_ms) / baseline_result.mean_ms * 100
        if baseline_result.mean_ms > 0
        else 0
    )

    return ComparisonResult(
        name=name,
        baseline=baseline_result,
        compared=yirage_result,
        speedup=speedup,
        improvement_percent=improvement,
    )


class BackendComparer:
    """
    Comprehensive backend comparison tool.

    Example:
        comparer = BackendComparer()
        comparer.add("matmul", matmul_fn, [a, b])
        comparer.add("attention", attention_fn, [q, k, v])

        report = comparer.run(backends=["cpu", "mps", "cuda"])
        print(report)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self._benchmarks: List[Tuple[str, Callable, List]] = []

    def add(self, name: str, fn: Callable, args: List[Any]):
        """Add a benchmark."""
        self._benchmarks.append((name, fn, args))

    def run(
        self,
        backends: List[str],
        baseline_backend: Optional[str] = None,
    ) -> str:
        """
        Run all benchmarks across backends and generate report.

        Args:
            backends: Backends to compare
            baseline_backend: Baseline backend

        Returns:
            Formatted comparison report
        """
        all_results: Dict[str, List[ComparisonResult]] = {}

        for name, fn, args in self._benchmarks:
            results = compare_backends(
                fn,
                args,
                backends,
                name=name,
                config=self.config,
                baseline_backend=baseline_backend,
            )
            all_results[name] = results

        return self._format_report(all_results, backends, baseline_backend or backends[0])

    def _format_report(
        self,
        all_results: Dict[str, List[ComparisonResult]],
        backends: List[str],
        baseline: str,
    ) -> str:
        """Format comparison report."""
        lines = [
            "=" * 80,
            "Backend Comparison Report",
            f"Baseline: {baseline}",
            "=" * 80,
            "",
        ]

        # Header
        header = f"{'Benchmark':<20}"
        for backend in backends:
            if backend != baseline:
                header += f"{backend:>15}"
        lines.append(header)
        lines.append("-" * 80)

        # Results
        for name, results in all_results.items():
            row = f"{name:<20}"
            for r in results:
                if r.speedup >= 1:
                    row += f"{r.speedup:>14.2f}x"
                else:
                    row += f"{1/r.speedup:>13.2f}x⬇"
            lines.append(row)

        lines.append("-" * 80)

        report = "\n".join(lines)
        print(report)
        return report
