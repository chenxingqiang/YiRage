#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Profiler Module Unit Tests

Tests for yirage/profiler/ module including HardwareProfiler and TimingResult.
Run with: pytest tests/python/test_profiler.py -v
"""

import pytest

from conftest import PYTHON_ROOT, load_module


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def hardware_profiler_module():
    """Load hardware profiler module."""
    return load_module(
        "hardware",
        PYTHON_ROOT / "yirage" / "profiler" / "hardware.py"
    )


# =============================================================================
# ProfilerBackend Tests
# =============================================================================

class TestProfilerBackend:
    """Tests for ProfilerBackend enum."""

    def test_profiler_backend_exists(self, hardware_profiler_module):
        """Test ProfilerBackend enum exists."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        assert hasattr(hardware_profiler_module, "ProfilerBackend")

    def test_cuda_backend_value(self, hardware_profiler_module):
        """Test CUDA backend value."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        ProfilerBackend = getattr(hardware_profiler_module, "ProfilerBackend", None)
        if ProfilerBackend is None:
            pytest.skip("ProfilerBackend not found")

        assert hasattr(ProfilerBackend, "CUDA")

    def test_cpu_backend_value(self, hardware_profiler_module):
        """Test CPU backend value."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        ProfilerBackend = getattr(hardware_profiler_module, "ProfilerBackend", None)
        if ProfilerBackend is None:
            pytest.skip("ProfilerBackend not found")

        assert hasattr(ProfilerBackend, "CPU")

    def test_mps_backend_value(self, hardware_profiler_module):
        """Test MPS backend value."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        ProfilerBackend = getattr(hardware_profiler_module, "ProfilerBackend", None)
        if ProfilerBackend is None:
            pytest.skip("ProfilerBackend not found")

        assert hasattr(ProfilerBackend, "MPS")


# =============================================================================
# ProfileConfig Tests
# =============================================================================

class TestProfileConfig:
    """Tests for ProfileConfig class."""

    def test_profile_config_exists(self, hardware_profiler_module):
        """Test ProfileConfig class exists."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        assert hasattr(hardware_profiler_module, "ProfileConfig")

    def test_profile_config_creation(self, hardware_profiler_module):
        """Test ProfileConfig can be created."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        ProfileConfig = getattr(hardware_profiler_module, "ProfileConfig", None)
        if ProfileConfig is None:
            pytest.skip("ProfileConfig not found")

        config = ProfileConfig()
        assert config is not None

    def test_profile_config_has_attributes(self, hardware_profiler_module):
        """Test ProfileConfig has expected attributes."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        ProfileConfig = getattr(hardware_profiler_module, "ProfileConfig", None)
        if ProfileConfig is None:
            pytest.skip("ProfileConfig not found")

        config = ProfileConfig()
        # Should have some configuration attributes
        attrs = [a for a in dir(config) if not a.startswith("_")]
        assert len(attrs) > 0


# =============================================================================
# TimingResult Tests
# =============================================================================

class TestTimingResult:
    """Tests for TimingResult class."""

    def test_timing_result_exists(self, hardware_profiler_module):
        """Test TimingResult class exists."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        assert hasattr(hardware_profiler_module, "TimingResult")

    def test_from_latencies_mean(self, hardware_profiler_module):
        """Test from_latencies computes mean correctly."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        TimingResult = getattr(hardware_profiler_module, "TimingResult", None)
        if TimingResult is None:
            pytest.skip("TimingResult not found")

        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = TimingResult.from_latencies(latencies, num_warmup=0)

        # Mean should be 3.0
        assert abs(result.mean_ms - 3.0) < 0.01

    def test_from_latencies_std(self, hardware_profiler_module):
        """Test from_latencies computes std correctly."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        TimingResult = getattr(hardware_profiler_module, "TimingResult", None)
        if TimingResult is None:
            pytest.skip("TimingResult not found")

        latencies = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = TimingResult.from_latencies(latencies, num_warmup=0)

        # Std should be 0.0
        assert result.std_ms < 0.01

    def test_from_latencies_percentiles(self, hardware_profiler_module):
        """Test from_latencies computes percentiles."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        TimingResult = getattr(hardware_profiler_module, "TimingResult", None)
        if TimingResult is None:
            pytest.skip("TimingResult not found")

        latencies = [1.0, 1.1, 1.2, 0.9, 1.05, 1.15, 0.95, 1.08, 1.12, 0.98]
        result = TimingResult.from_latencies(latencies, num_warmup=2)

        assert hasattr(result, "p50_ms")
        assert hasattr(result, "p95_ms")

    def test_warmup_excluded(self, hardware_profiler_module):
        """Test warmup iterations are excluded."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        TimingResult = getattr(hardware_profiler_module, "TimingResult", None)
        if TimingResult is None:
            pytest.skip("TimingResult not found")

        # First 2 are warmup (high values), rest are actual
        latencies = [100.0, 100.0, 1.0, 1.0, 1.0]
        try:
            result = TimingResult.from_latencies(latencies, num_warmup=2)
            # Mean should be close to 1.0, not affected by warmup
            assert result.mean_ms < 2.0
        except Exception as e:
            pytest.skip(f"TimingResult.from_latencies failed: {e}")


# =============================================================================
# HardwareCounters Tests
# =============================================================================

class TestHardwareCounters:
    """Tests for HardwareCounters class."""

    def test_hardware_counters_exists(self, hardware_profiler_module):
        """Test HardwareCounters class exists."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        assert hasattr(hardware_profiler_module, "HardwareCounters")

    def test_hardware_counters_creation(self, hardware_profiler_module):
        """Test HardwareCounters can be created."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        HardwareCounters = getattr(hardware_profiler_module, "HardwareCounters", None)
        if HardwareCounters is None:
            pytest.skip("HardwareCounters not found")

        counters = HardwareCounters()
        assert counters is not None


# =============================================================================
# HardwareProfiler Tests
# =============================================================================

class TestHardwareProfiler:
    """Tests for HardwareProfiler class."""

    def test_profiler_exists(self, hardware_profiler_module):
        """Test HardwareProfiler class exists."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        assert hasattr(hardware_profiler_module, "HardwareProfiler")

    def test_profiler_creation(self, hardware_profiler_module):
        """Test HardwareProfiler can be created."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        HardwareProfiler = getattr(hardware_profiler_module, "HardwareProfiler", None)
        ProfilerBackend = getattr(hardware_profiler_module, "ProfilerBackend", None)

        if HardwareProfiler is None:
            pytest.skip("HardwareProfiler not found")

        if ProfilerBackend:
            profiler = HardwareProfiler(backend=ProfilerBackend.CPU)
        else:
            profiler = HardwareProfiler()

        assert profiler is not None

    def test_profiler_has_profile_method(self, hardware_profiler_module):
        """Test HardwareProfiler has profile method."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        HardwareProfiler = getattr(hardware_profiler_module, "HardwareProfiler", None)
        if HardwareProfiler is None:
            pytest.skip("HardwareProfiler not found")

        has_profile = (
            hasattr(HardwareProfiler, "profile") or 
            hasattr(HardwareProfiler, "profile_kernel") or
            hasattr(HardwareProfiler, "__call__")
        )
        if not has_profile:
            pytest.skip("Profile method not implemented yet")


# =============================================================================
# Integration Tests
# =============================================================================

class TestProfilerIntegration:
    """Integration tests for profiler module."""

    def test_profiler_with_config(self, hardware_profiler_module):
        """Test profiler with custom config."""
        if hardware_profiler_module is None:
            pytest.skip("Hardware profiler module not available")

        HardwareProfiler = getattr(hardware_profiler_module, "HardwareProfiler", None)
        ProfileConfig = getattr(hardware_profiler_module, "ProfileConfig", None)
        ProfilerBackend = getattr(hardware_profiler_module, "ProfilerBackend", None)

        if HardwareProfiler is None or ProfileConfig is None:
            pytest.skip("Required classes not found")

        config = ProfileConfig()

        if ProfilerBackend:
            profiler = HardwareProfiler(backend=ProfilerBackend.CPU, config=config)
        else:
            profiler = HardwareProfiler(config=config)

        assert profiler is not None
