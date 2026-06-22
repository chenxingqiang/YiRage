# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark Suite for running multiple benchmarks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import json
import os

from .benchmark import Benchmark, BenchmarkResult, BenchmarkConfig


@dataclass
class SuiteResult:
    """Results from a benchmark suite run."""

    suite_name: str
    timestamp: datetime
    results: List[BenchmarkResult]
    total_time_seconds: float

    def get_result(self, name: str) -> Optional[BenchmarkResult]:
        """Get result by benchmark name."""
        for r in self.results:
            if r.name == name:
                return r
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suite_name": self.suite_name,
            "timestamp": self.timestamp.isoformat(),
            "total_time_seconds": self.total_time_seconds,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str):
        """Save results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class BenchmarkSuite:
    """
    Suite for running multiple benchmarks.

    Example:
        suite = BenchmarkSuite("llm_kernels")

        # Add benchmarks
        suite.add("matmul", matmul_fn, [a, b])
        suite.add("attention", attention_fn, [q, k, v])

        # Run all
        results = suite.run_all()

        # Generate report
        suite.report()
    """

    def __init__(
        self,
        name: str = "benchmark_suite",
        backend: str = "cpu",
        config: Optional[BenchmarkConfig] = None,
    ):
        self.name = name
        self.backend = backend
        self.config = config or BenchmarkConfig()

        self._benchmarks: List[Dict[str, Any]] = []
        self._results: Optional[SuiteResult] = None

    def add(
        self,
        name: str,
        fn: Callable,
        args: List[Any],
        kwargs: Optional[Dict] = None,
    ):
        """Add a benchmark to the suite."""
        self._benchmarks.append(
            {
                "name": name,
                "fn": fn,
                "args": args,
                "kwargs": kwargs or {},
            }
        )

    def add_graph(
        self,
        name: str,
        graph: Any,
        inputs: List[Any],
    ):
        """Add a graph benchmark."""
        if hasattr(graph, "__call__"):
            fn = graph
        elif hasattr(graph, "execute"):
            fn = graph.execute
        else:
            raise TypeError("Graph must be callable")

        self.add(name, fn, inputs)

    def run_all(self, verbose: bool = True) -> SuiteResult:
        """
        Run all benchmarks in the suite.

        Args:
            verbose: Print progress

        Returns:
            SuiteResult with all results
        """
        import time

        start_time = time.time()
        results = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"Running Benchmark Suite: {self.name}")
            print(f"Backend: {self.backend}")
            print(f"Benchmarks: {len(self._benchmarks)}")
            print(f"{'='*60}\n")

        for i, bench_info in enumerate(self._benchmarks):
            name = bench_info["name"]
            fn = bench_info["fn"]
            args = bench_info["args"]
            kwargs = bench_info["kwargs"]

            if verbose:
                print(f"[{i+1}/{len(self._benchmarks)}] {name}...", end=" ", flush=True)

            try:
                bench = Benchmark(name, self.backend, self.config)
                result = bench.run(fn, *args, **kwargs)
                results.append(result)

                if verbose:
                    print(f"{result.mean_ms:.3f} ms (±{result.std_ms:.3f})")
            except Exception as e:
                if verbose:
                    print(f"FAILED: {e}")
                # Add failed result
                results.append(
                    BenchmarkResult(
                        name=name,
                        backend=self.backend,
                        mean_ms=float("inf"),
                        std_ms=0,
                        min_ms=float("inf"),
                        max_ms=float("inf"),
                        median_ms=float("inf"),
                    )
                )

        total_time = time.time() - start_time

        self._results = SuiteResult(
            suite_name=self.name,
            timestamp=datetime.now(),
            results=results,
            total_time_seconds=total_time,
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"Suite completed in {total_time:.2f} seconds")
            print(f"{'='*60}\n")

        return self._results

    def report(self, format: str = "text") -> str:
        """
        Generate benchmark report.

        Args:
            format: Output format ('text', 'markdown', 'json')

        Returns:
            Formatted report string
        """
        if self._results is None:
            return "No results. Run the suite first."

        if format == "json":
            return json.dumps(self._results.to_dict(), indent=2)
        elif format == "markdown":
            return self._report_markdown()
        else:
            return self._report_text()

    def _report_text(self) -> str:
        """Generate text report."""
        lines = [
            f"\nBenchmark Suite: {self._results.suite_name}",
            f"Date: {self._results.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Backend: {self.backend}",
            "-" * 70,
            f"{'Name':<25} {'Mean (ms)':<12} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10}",
            "-" * 70,
        ]

        for r in self._results.results:
            lines.append(
                f"{r.name:<25} {r.mean_ms:<12.3f} {r.std_ms:<10.3f} "
                f"{r.min_ms:<10.3f} {r.max_ms:<10.3f}"
            )

        lines.extend(
            [
                "-" * 70,
                f"Total time: {self._results.total_time_seconds:.2f} seconds",
            ]
        )

        report = "\n".join(lines)
        print(report)
        return report

    def _report_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Benchmark Suite: {self._results.suite_name}",
            f"",
            f"**Date:** {self._results.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Backend:** {self.backend}",
            f"",
            "| Name | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |",
            "|------|-----------|----------|----------|----------|",
        ]

        for r in self._results.results:
            lines.append(
                f"| {r.name} | {r.mean_ms:.3f} | {r.std_ms:.3f} | "
                f"{r.min_ms:.3f} | {r.max_ms:.3f} |"
            )

        lines.extend(
            [
                "",
                f"**Total time:** {self._results.total_time_seconds:.2f} seconds",
            ]
        )

        return "\n".join(lines)


def create_standard_suite(backend: str = "cpu") -> BenchmarkSuite:
    """
    Create a standard benchmark suite for LLM kernels.

    Args:
        backend: Target backend

    Returns:
        BenchmarkSuite with standard LLM benchmarks
    """
    suite = BenchmarkSuite("standard_llm_suite", backend)

    # Add standard benchmarks when inputs are provided
    # This is a template that needs actual tensors

    return suite
