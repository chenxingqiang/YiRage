# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark Module Tests

Tests for the benchmark framework.
"""

import pytest
from typing import List, Dict

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def benchmark_module():
    """Import the benchmark module."""
    try:
        from yirage import benchmark
        return benchmark
    except ImportError:
        pytest.skip("yirage.benchmark module not available")


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports."""

    def test_benchmark_class_import(self, benchmark_module):
        """Test Benchmark class can be imported."""
        assert hasattr(benchmark_module, "Benchmark")

    def test_benchmark_result_import(self, benchmark_module):
        """Test BenchmarkResult class can be imported."""
        assert hasattr(benchmark_module, "BenchmarkResult")

    def test_benchmark_config_import(self, benchmark_module):
        """Test BenchmarkConfig class can be imported."""
        assert hasattr(benchmark_module, "BenchmarkConfig")

    def test_benchmark_suite_import(self, benchmark_module):
        """Test BenchmarkSuite class can be imported."""
        assert hasattr(benchmark_module, "BenchmarkSuite")

    def test_suite_result_import(self, benchmark_module):
        """Test SuiteResult class can be imported."""
        assert hasattr(benchmark_module, "SuiteResult")


# =============================================================================
# Benchmark Class Tests
# =============================================================================


class TestBenchmarkClass:
    """Tests for Benchmark class."""

    def test_benchmark_exists(self, benchmark_module):
        """Test Benchmark class exists."""
        Benchmark = benchmark_module.Benchmark
        assert Benchmark is not None


# =============================================================================
# BenchmarkResult Tests
# =============================================================================


class TestBenchmarkResult:
    """Tests for BenchmarkResult class."""

    def test_result_exists(self, benchmark_module):
        """Test BenchmarkResult class exists."""
        BenchmarkResult = benchmark_module.BenchmarkResult
        assert BenchmarkResult is not None


# =============================================================================
# BenchmarkConfig Tests
# =============================================================================


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig class."""

    def test_config_exists(self, benchmark_module):
        """Test BenchmarkConfig class exists."""
        BenchmarkConfig = benchmark_module.BenchmarkConfig
        assert BenchmarkConfig is not None


# =============================================================================
# BenchmarkSuite Tests
# =============================================================================


class TestBenchmarkSuite:
    """Tests for BenchmarkSuite class."""

    def test_suite_exists(self, benchmark_module):
        """Test BenchmarkSuite class exists."""
        BenchmarkSuite = benchmark_module.BenchmarkSuite
        assert BenchmarkSuite is not None


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_run_benchmark_exists(self, benchmark_module):
        """Test run_benchmark function exists."""
        assert hasattr(benchmark_module, "run_benchmark")
        assert callable(benchmark_module.run_benchmark)

    def test_create_standard_suite_exists(self, benchmark_module):
        """Test create_standard_suite function exists."""
        assert hasattr(benchmark_module, "create_standard_suite")
        assert callable(benchmark_module.create_standard_suite)


# =============================================================================
# Comparison Functions Tests
# =============================================================================


class TestComparisonFunctions:
    """Tests for comparison functions."""

    def test_comparison_result_exists(self, benchmark_module):
        """Test ComparisonResult class exists."""
        assert hasattr(benchmark_module, "ComparisonResult")

    def test_compare_backends_exists(self, benchmark_module):
        """Test compare_backends function exists."""
        assert hasattr(benchmark_module, "compare_backends")
        assert callable(benchmark_module.compare_backends)

    def test_compare_with_baseline_exists(self, benchmark_module):
        """Test compare_with_baseline function exists."""
        assert hasattr(benchmark_module, "compare_with_baseline")
        assert callable(benchmark_module.compare_with_baseline)


# =============================================================================
# All Exports Test
# =============================================================================


class TestAllExports:
    """Tests for __all__ exports."""

    def test_all_exports_defined(self, benchmark_module):
        """Test __all__ is defined."""
        assert hasattr(benchmark_module, "__all__")

    def test_all_exports_accessible(self, benchmark_module):
        """Test all items in __all__ are accessible."""
        for name in benchmark_module.__all__:
            assert hasattr(benchmark_module, name), f"Export '{name}' not accessible"


# =============================================================================
# Parameterized Tests
# =============================================================================


@pytest.mark.parametrize("class_name", [
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkSuite",
    "SuiteResult",
    "ComparisonResult",
])
def test_class_exists(benchmark_module, class_name):
    """Test expected classes exist."""
    assert hasattr(benchmark_module, class_name), f"Class '{class_name}' missing"


@pytest.mark.parametrize("function_name", [
    "run_benchmark",
    "create_standard_suite",
    "compare_backends",
    "compare_with_baseline",
])
def test_function_callable(benchmark_module, function_name):
    """Test expected functions are callable."""
    func = getattr(benchmark_module, function_name, None)
    assert callable(func), f"{function_name} is not callable"
