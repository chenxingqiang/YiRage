# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Benchmark Framework

Provides comprehensive benchmarking for kernel optimization across backends.

Usage:
    from yirage.benchmark import Benchmark, BenchmarkSuite
    
    # Single benchmark
    bench = Benchmark(name="matmul_1024", backend="cuda")
    result = bench.run(graph, inputs)
    print(f"Latency: {result.mean_ms:.3f} ms")
    
    # Benchmark suite
    suite = BenchmarkSuite()
    suite.add_graph("matmul", graph, inputs)
    results = suite.run_all()
    suite.report()
"""

from .benchmark import (
    Benchmark,
    BenchmarkResult,
    BenchmarkConfig,
    run_benchmark,
)

from .suite import (
    BenchmarkSuite,
    SuiteResult,
    create_standard_suite,
)

from .comparison import (
    ComparisonResult,
    compare_backends,
    compare_with_baseline,
)

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkConfig",
    "run_benchmark",
    "BenchmarkSuite",
    "SuiteResult",
    "create_standard_suite",
    "ComparisonResult",
    "compare_backends",
    "compare_with_baseline",
]
