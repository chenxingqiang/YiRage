#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
AccelForge Integration Tests

Tests for the YiRage × AccelForge hardware-software co-design integration.
Run with: pytest tests/python/test_rl/test_accelforge.py -v
"""

import pytest
import sys
import json
import types
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))


# =============================================================================
# AccelForge Bridge Tests
# =============================================================================


class TestAccelForgeDesignPoint:
    """Tests for AccelForgeDesignPoint dataclass."""

    def test_default_creation(self):
        """Test default design point creation."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        design = AccelForgeDesignPoint()
        assert design.pe_array_rows == 16
        assert design.pe_array_cols == 16
        assert design.dataflow == "output_stationary"
        assert design.data_precision == "fp16"
        assert design.noc_topology == "mesh"

    def test_custom_creation(self):
        """Test custom design point creation."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        design = AccelForgeDesignPoint(
            pe_array_rows=32,
            pe_array_cols=32,
            l1_buffer_kb=128.0,
            dataflow="weight_stationary",
            data_precision="int8",
        )
        assert design.pe_array_rows == 32
        assert design.total_pes == 1024
        assert design.dataflow == "weight_stationary"

    def test_total_pes(self):
        """Test total PE count."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        design = AccelForgeDesignPoint(pe_array_rows=8, pe_array_cols=8)
        assert design.total_pes == 64

    def test_total_buffer(self):
        """Test total buffer calculation."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        design = AccelForgeDesignPoint(
            pe_array_rows=4,
            pe_array_cols=4,
            l0_buffer_kb=1.0,
            l1_buffer_kb=64.0,
            l2_buffer_kb=512.0,
        )
        # l0 * total_pes + l1 + l2 = 1.0 * 16 + 64 + 512 = 592
        assert design.total_buffer_kb == 592.0

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        design = AccelForgeDesignPoint(
            pe_array_rows=32,
            pe_array_cols=16,
            dataflow="row_stationary",
        )
        d = design.to_dict()
        restored = AccelForgeDesignPoint.from_dict(d)
        assert restored.pe_array_rows == 32
        assert restored.pe_array_cols == 16
        assert restored.dataflow == "row_stationary"


class TestAccelForgeMetrics:
    """Tests for AccelForgeMetrics dataclass."""

    def test_default_creation(self):
        """Test default metrics."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeMetrics

        metrics = AccelForgeMetrics()
        assert metrics.area_mm2 == 0.0
        assert metrics.confidence == 0.85

    def test_to_dict(self):
        """Test serialization."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeMetrics

        metrics = AccelForgeMetrics(
            area_mm2=5.0,
            energy_per_op_pj=1.5,
            latency_ms=0.5,
            total_power_mw=100.0,
        )
        d = metrics.to_dict()
        assert d["area_mm2"] == 5.0
        assert d["energy_per_op_pj"] == 1.5


class TestAccelForgeBridge:
    """Tests for AccelForgeBridge main class."""

    def test_creation(self):
        """Test bridge creation."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeBridge

        bridge = AccelForgeBridge()
        assert bridge is not None

    def test_evaluate_default(self):
        """Test evaluating default design point."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )

        bridge = AccelForgeBridge()
        design = AccelForgeDesignPoint()
        metrics = bridge.evaluate(design)

        assert metrics.area_mm2 > 0
        assert metrics.energy_per_op_pj > 0
        assert metrics.peak_tops > 0
        assert metrics.total_power_mw > 0

    def test_evaluate_with_workload(self):
        """Test evaluating with workload specification."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )

        bridge = AccelForgeBridge()
        design = AccelForgeDesignPoint(pe_array_rows=32, pe_array_cols=32)
        workload = {"estimated_flops": 1e9, "memory_bytes": 1e6}
        metrics = bridge.evaluate(design, workload)

        assert metrics.latency_ms > 0
        assert metrics.achieved_tops > 0

    def test_evaluate_cache(self):
        """Test that caching works."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )

        bridge = AccelForgeBridge()
        design = AccelForgeDesignPoint()

        metrics1 = bridge.evaluate(design)
        metrics2 = bridge.evaluate(design)

        # Should return same results from cache
        assert metrics1.area_mm2 == metrics2.area_mm2

    def test_to_hardware_profile(self):
        """Test conversion to HardwareProfile."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )

        bridge = AccelForgeBridge()
        design = AccelForgeDesignPoint(pe_array_rows=16, pe_array_cols=16)
        profile = bridge.to_hardware_profile(design)

        assert profile.backend == "accelforge"
        assert profile.total_cores == 256
        assert profile.is_accelforge
        assert "accelforge_design" in profile.extensions
        assert "accelforge_metrics" in profile.extensions

    def test_get_design_space(self):
        """Test design space enumeration."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeBridge

        bridge = AccelForgeBridge()
        designs = bridge.get_design_space()

        assert len(designs) > 0
        # 4 PE sizes * 3 dataflows * 4 precisions = 48
        assert len(designs) == 48

    def test_clear_cache(self):
        """Test cache clearing."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )

        bridge = AccelForgeBridge()
        design = AccelForgeDesignPoint()
        bridge.evaluate(design)
        assert len(bridge._cache) > 0

        bridge.clear_cache()
        assert len(bridge._cache) == 0


class TestIsAccelForgeAvailable:
    """Test availability check."""

    def test_availability_function(self):
        """Test is_accelforge_available function."""
        from yirage.rl.hardware.accelforge_bridge import is_accelforge_available

        # Returns bool regardless
        result = is_accelforge_available()
        assert isinstance(result, bool)


# =============================================================================
# HardwareProfile AccelForge Extension Tests
# =============================================================================


class TestHardwareProfileAccelForge:
    """Tests for AccelForge extensions in HardwareProfile."""

    def test_from_accelforge_dict(self):
        """Test creating profile from dict."""
        from yirage.rl.hardware.profile import HardwareProfile

        design_dict = {
            "pe_array_rows": 32,
            "pe_array_cols": 32,
            "dataflow": "weight_stationary",
        }
        profile = HardwareProfile.from_accelforge(design_dict)
        assert profile.backend == "accelforge"
        assert profile.total_cores == 1024

    def test_from_accelforge_design_point(self):
        """Test creating profile from AccelForgeDesignPoint."""
        from yirage.rl.hardware.profile import HardwareProfile
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        design = AccelForgeDesignPoint(pe_array_rows=64, pe_array_cols=64)
        profile = HardwareProfile.from_accelforge(design)
        assert profile.total_cores == 4096
        assert profile.is_accelforge


class TestAccelForgeConfigFusion:
    """Tests for unified YiRage HardwareConfig coupling with AccelForge profiles."""

    def test_config_coupling_reuses_level1_hardware_config(self):
        """config_coupling should export the single Level 1 HardwareConfig type."""
        from yirage.rl.hardware.config_coupling import HardwareConfig as CoupledHardwareConfig
        from yirage.rl.search.config_space import HardwareConfig as Level1HardwareConfig

        assert CoupledHardwareConfig is Level1HardwareConfig

    def test_accelforge_profile_generates_level1_config_and_constraints(self):
        """AccelForge design should drive YiRage config and constraints."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
            AccelForgeMetrics,
        )
        from yirage.rl.hardware.config_coupling import ConfigGenerator
        from yirage.rl.search.config_space import HardwareConfig, SearchSpaceConstraints

        design = AccelForgeDesignPoint(
            pe_array_rows=8,
            pe_array_cols=16,
            l1_buffer_kb=32.0,
            dataflow="weight_stationary",
            data_precision="int8",
        )
        profile = AccelForgeBridge().to_hardware_profile(
            design,
            AccelForgeMetrics(peak_tops=128.0, pe_utilization=0.5),
        )

        config = ConfigGenerator(profile).generate()
        assert isinstance(config, HardwareConfig)
        assert config.block_dim_x == 16
        assert config.shared_memory_size == 32 * 1024
        assert config.num_registers == 64

        constraints = SearchSpaceConstraints(config).apply_hardware_profile(profile)
        assert constraints.max_shared_memory == 32 * 1024
        assert constraints.supported_precisions == ["int8"]
        assert constraints.supports_weight_reuse is True
        assert constraints.max_operators <= design.total_pes

    def test_config_env_uses_accelforge_generated_config(self):
        """ConfigEnv should use HardwareProfile-derived config for AccelForge."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
            AccelForgeMetrics,
        )
        from yirage.rl.search.hierarchical_env import ConfigEnv

        design = AccelForgeDesignPoint(pe_array_rows=4, pe_array_cols=8, l1_buffer_kb=16.0)
        profile = AccelForgeBridge().to_hardware_profile(
            design,
            AccelForgeMetrics(peak_tops=16.0),
        )
        env = ConfigEnv(
            {
                "backend": "accelforge",
                "hardware_profile": profile,
                "accelerator_constraints": {
                    "max_parallelism": 16,
                    "supported_precisions": ["fp16"],
                },
            }
        )

        action = np.zeros(len(env.action_space.nvec), dtype=np.int64)
        _, _, _, _, info = env.step(action)

        assert env.current_config.block_dim_x == 8
        assert env.current_config.shared_memory_size == 16 * 1024
        assert info["constraints"]["max_operators"] <= 16
        assert info["constraints"]["supported_precisions"] == ["fp16"]

    def test_accelforge_hardware_profile_is_encoded_in_observation(self):
        """Observation encoding should include AccelForge-specific hardware fields."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
            AccelForgeMetrics,
        )
        from yirage.rl.search.config_space import ConfigObservationSpace

        design = AccelForgeDesignPoint(
            pe_array_rows=32,
            pe_array_cols=16,
            l1_buffer_kb=64.0,
            data_precision="bf16",
        )
        profile = AccelForgeBridge().to_hardware_profile(
            design,
            AccelForgeMetrics(peak_tops=256.0),
        )

        obs_space = ConfigObservationSpace()
        features = obs_space.encode_hardware("accelforge", profile)
        bf16_precision_encoding = obs_space.ACCELFORGE_PRECISION_ENCODING["bf16"]

        assert features.shape == (ConfigObservationSpace.HARDWARE_FEATURE_DIM,)
        assert features[obs_space.HARDWARE_BACKEND_IDX] == 1.0
        assert features[obs_space.ACCELFORGE_PE_ROWS_IDX] > 0.0
        assert features[obs_space.ACCELFORGE_PE_COLS_IDX] > 0.0
        assert features[obs_space.ACCELFORGE_PRECISION_IDX] == bf16_precision_encoding

    def test_is_accelforge_property(self):
        """Test is_accelforge property."""
        from yirage.rl.hardware.profile import HardwareProfile

        gpu = HardwareProfile(backend="cuda")
        assert not gpu.is_accelforge

        af = HardwareProfile(backend="accelforge")
        assert af.is_accelforge

    def test_accelforge_metrics_property(self):
        """Test accelforge_metrics property."""
        from yirage.rl.hardware.profile import HardwareProfile

        # Non-AccelForge profile
        gpu = HardwareProfile(backend="cuda")
        assert gpu.accelforge_metrics is None

        # AccelForge profile
        af = HardwareProfile(
            backend="accelforge",
            extensions={"accelforge_metrics": {"area_mm2": 5.0}},
        )
        assert af.accelforge_metrics is not None
        assert af.accelforge_metrics["area_mm2"] == 5.0

    def test_feature_vector_with_accelforge(self):
        """Test feature vector includes AccelForge features."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )

        bridge = AccelForgeBridge()
        design = AccelForgeDesignPoint(pe_array_rows=32, pe_array_cols=32)
        profile = bridge.to_hardware_profile(design)

        features = profile.to_feature_vector()
        assert features.shape == (32,)

        # AccelForge backend should be set (index 5)
        assert features[5] == 1.0

        # AccelForge-specific features should be non-zero
        assert features[21] > 0  # PE rows
        assert features[22] > 0  # PE cols
        assert features[26] > 0  # Area

    def test_feature_vector_without_accelforge(self):
        """Test feature vector for non-AccelForge backend."""
        from yirage.rl.hardware.profile import HardwareProfile

        profile = HardwareProfile(backend="cuda", total_cores=5120)
        features = profile.to_feature_vector()
        assert features.shape == (32,)
        assert features[0] == 1.0  # CUDA backend
        assert features[5] == 0.0  # Not AccelForge


# =============================================================================
# PerformanceEstimate Extension Tests
# =============================================================================


class TestPerformanceEstimateExtension:
    """Tests for energy/area/power fields in PerformanceEstimate."""

    def test_energy_fields(self):
        """Test energy/area/power fields exist."""
        from yirage.rl.hardware.profile import PerformanceEstimate

        est = PerformanceEstimate(
            estimated_latency_ms=1.0,
            energy_pj=2.5,
            area_mm2=10.0,
            power_mw=500.0,
            leak_power_mw=50.0,
        )
        assert est.energy_pj == 2.5
        assert est.area_mm2 == 10.0
        assert est.power_mw == 500.0
        assert est.leak_power_mw == 50.0

    def test_to_dict_includes_energy(self):
        """Test to_dict includes new fields."""
        from yirage.rl.hardware.profile import PerformanceEstimate

        est = PerformanceEstimate(energy_pj=3.0, area_mm2=5.0, power_mw=200.0)
        d = est.to_dict()
        assert "energy_pj" in d
        assert "area_mm2" in d
        assert "power_mw" in d
        assert d["energy_pj"] == 3.0


# =============================================================================
# AccelForge Detector Tests
# =============================================================================


class TestAccelForgeDetector:
    """Tests for AccelForgeDetector."""

    def test_is_available(self):
        """Test detector is always available."""
        from yirage.rl.hardware.detector import AccelForgeDetector

        detector = AccelForgeDetector()
        assert detector.is_available()

    def test_detect_default(self):
        """Test detecting default design."""
        from yirage.rl.hardware.detector import AccelForgeDetector

        detector = AccelForgeDetector()
        profile = detector.detect()
        assert profile is not None
        assert profile.backend == "accelforge"

    def test_detect_custom(self):
        """Test detecting custom design."""
        from yirage.rl.hardware.detector import AccelForgeDetector

        detector = AccelForgeDetector(
            design_point={"pe_array_rows": 64, "pe_array_cols": 64}
        )
        profile = detector.detect()
        assert profile.total_cores == 4096

    def test_get_detector_accelforge(self):
        """Test factory function returns AccelForge detector."""
        from yirage.rl.hardware.detector import get_detector

        detector = get_detector("accelforge")
        assert detector is not None
        assert detector.is_available()


# =============================================================================
# AccelForge Performance Estimation Tests
# =============================================================================


class TestAccelForgeEstimation:
    """Tests for AccelForge-enhanced performance estimation."""

    def test_accelforge_estimate(self):
        """Test performance estimation with AccelForge backend."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )
        from yirage.rl.hardware.config_coupling import HardwareConfig, HardwareSearchCoupling

        bridge = AccelForgeBridge()
        design = AccelForgeDesignPoint()
        profile = bridge.to_hardware_profile(design)

        coupling = HardwareSearchCoupling(profile)
        config = HardwareConfig(grid_dim_x=4, block_dim_x=16)

        estimate = coupling.estimate_performance(
            config,
            {"theoretical_flops": 1e9, "memory_bytes": 1e6},
        )

        assert estimate.estimated_latency_ms > 0
        assert estimate.energy_pj > 0
        assert estimate.area_mm2 > 0
        assert estimate.power_mw > 0
        assert estimate.confidence >= 0.6


# =============================================================================
# Accelerator Space Tests
# =============================================================================


class TestAcceleratorActionSpace:
    """Tests for Level 0 action space."""

    def test_creation(self):
        """Test action space creation."""
        from yirage.rl.search.accelerator_space import AcceleratorActionSpace

        space = AcceleratorActionSpace()
        assert space is not None

    def test_decode_encode_roundtrip(self):
        """Test action encode-decode round-trip."""
        from yirage.rl.search.accelerator_space import AcceleratorActionSpace

        space = AcceleratorActionSpace()

        if space.flat_space is not None:
            action = space.flat_space.sample()
            design = space.decode_flat(action)
            re_encoded = space.encode(design)

            # Decode again and check consistency
            design2 = space.decode_flat(re_encoded)
            assert design.pe_array_rows == design2.pe_array_rows
            assert design.dataflow == design2.dataflow


class TestAcceleratorObservationSpace:
    """Tests for Level 0 observation space."""

    def test_creation(self):
        """Test observation space creation."""
        from yirage.rl.search.accelerator_space import AcceleratorObservationSpace

        space = AcceleratorObservationSpace()
        assert space is not None

    def test_encode_workload(self):
        """Test workload encoding."""
        from yirage.rl.search.accelerator_space import AcceleratorObservationSpace

        space = AcceleratorObservationSpace()
        features = space.encode_workload('{"batch_size": 32, "sequence_length": 1024}')
        assert features.shape == (space.WORKLOAD_DIM,)

    def test_encode_design(self):
        """Test design encoding."""
        from yirage.rl.search.accelerator_space import AcceleratorObservationSpace
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        space = AcceleratorObservationSpace()
        design = AccelForgeDesignPoint(pe_array_rows=32, pe_array_cols=32)
        features = space.encode_design(design)
        assert features.shape == (space.DESIGN_DIM,)
        assert features[0] > 0  # PE rows


class TestAcceleratorEnv:
    """Tests for Level 0 AcceleratorEnv."""

    @pytest.fixture
    def env(self):
        from yirage.rl.search.accelerator_space import AcceleratorEnv

        return AcceleratorEnv({"max_design_episodes": 5})

    def test_creation(self, env):
        """Test env creation."""
        assert env is not None

    def test_reset(self, env):
        """Test reset returns observation."""
        obs, info = env.reset()
        assert "workload_features" in obs
        assert "design_features" in obs
        assert "metrics_features" in obs

    def test_step(self, env):
        """Test step returns valid results."""
        env.reset()

        if env.action_space is not None:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            assert isinstance(reward, float)
            assert "design" in info
            assert "metrics" in info
            assert "constraints" in info

    def test_multi_step(self, env):
        """Test multiple steps."""
        env.reset()

        if env.action_space is not None:
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, done, truncated, info = env.step(action)
                if done:
                    break

    def test_pareto_tracking(self, env):
        """Test Pareto front is tracked."""
        env.reset()

        if env.action_space is not None:
            for _ in range(5):
                action = env.action_space.sample()
                env.step(action)

            front = env.get_pareto_front()
            assert isinstance(front, list)


# =============================================================================
# Pareto Front Tracker Tests
# =============================================================================


class TestParetoFrontTracker:
    """Tests for ParetoFrontTracker."""

    def test_creation(self):
        """Test tracker creation."""
        from yirage.rl.search.accelerator_space import ParetoFrontTracker

        tracker = ParetoFrontTracker()
        assert tracker.size() == 0

    def test_add_single(self):
        """Test adding single point."""
        from yirage.rl.search.accelerator_space import ParetoFrontTracker, ParetoPoint

        tracker = ParetoFrontTracker()
        point = ParetoPoint(latency_ms=1.0, energy_pj=2.0, area_mm2=5.0, power_mw=100.0)
        added = tracker.add(point)
        assert added
        assert tracker.size() == 1

    def test_dominance(self):
        """Test Pareto dominance."""
        from yirage.rl.search.accelerator_space import ParetoPoint

        p1 = ParetoPoint(latency_ms=1.0, energy_pj=1.0, area_mm2=1.0, power_mw=1.0)
        p2 = ParetoPoint(latency_ms=2.0, energy_pj=2.0, area_mm2=2.0, power_mw=2.0)

        assert p1.dominates(p2)
        assert not p2.dominates(p1)

    def test_non_dominated_front(self):
        """Test Pareto front with non-dominated points."""
        from yirage.rl.search.accelerator_space import ParetoFrontTracker, ParetoPoint

        tracker = ParetoFrontTracker()

        # Two non-dominated points (trade-off between latency and energy)
        p1 = ParetoPoint(latency_ms=1.0, energy_pj=5.0, area_mm2=5.0, power_mw=100.0)
        p2 = ParetoPoint(latency_ms=5.0, energy_pj=1.0, area_mm2=3.0, power_mw=50.0)

        tracker.add(p1)
        tracker.add(p2)

        assert tracker.size() == 2

    def test_dominated_point_rejected(self):
        """Test that dominated points are not added."""
        from yirage.rl.search.accelerator_space import ParetoFrontTracker, ParetoPoint

        tracker = ParetoFrontTracker()

        p1 = ParetoPoint(latency_ms=1.0, energy_pj=1.0, area_mm2=1.0, power_mw=1.0)
        p2 = ParetoPoint(latency_ms=2.0, energy_pj=2.0, area_mm2=2.0, power_mw=2.0)

        tracker.add(p1)
        added = tracker.add(p2)

        assert not added
        assert tracker.size() == 1

    def test_get_best(self):
        """Test getting best point for objective."""
        from yirage.rl.search.accelerator_space import ParetoFrontTracker, ParetoPoint

        tracker = ParetoFrontTracker()
        tracker.add(ParetoPoint(latency_ms=1.0, energy_pj=5.0, area_mm2=5.0, power_mw=100.0))
        tracker.add(ParetoPoint(latency_ms=5.0, energy_pj=1.0, area_mm2=3.0, power_mw=50.0))

        best_latency = tracker.get_best("latency_ms")
        assert best_latency.latency_ms == 1.0

        best_energy = tracker.get_best("energy_pj")
        assert best_energy.energy_pj == 1.0


# =============================================================================
# AccelForge Verifier Tests
# =============================================================================


class TestAccelForgeVerifier:
    """Tests for AccelForgeVerifier."""

    def test_creation(self):
        """Test verifier creation."""
        from yirage.rl.verifier.accelforge_verifier import AccelForgeVerifier

        verifier = AccelForgeVerifier()
        assert verifier is not None

    def test_verify_empty(self):
        """Test verification with empty graphs."""
        from yirage.rl.verifier.accelforge_verifier import AccelForgeVerifier

        verifier = AccelForgeVerifier()
        result = verifier.verify_fingerprint("{}", "{}")
        assert result.verified
        assert result.fingerprint_time_ms >= 0

    def test_verify_invalid_json(self):
        """Test verification with invalid JSON."""
        from yirage.rl.verifier.accelforge_verifier import AccelForgeVerifier

        verifier = AccelForgeVerifier()
        result = verifier.verify_fingerprint("not json", "{}")
        assert not result.verified
        assert result.rejection_reason == "invalid_json"

    def test_profile_kernel(self):
        """Test kernel profiling."""
        from yirage.rl.verifier.accelforge_verifier import AccelForgeVerifier

        verifier = AccelForgeVerifier()
        result = verifier.profile_kernel(
            '{"operators": [{"flops": 1000000}]}',
            input_shapes=[[32, 1024]],
        )
        assert result.latency_ms > 0

    def test_get_full_metrics(self):
        """Test getting full AccelForge metrics."""
        from yirage.rl.verifier.accelforge_verifier import AccelForgeVerifier

        verifier = AccelForgeVerifier()
        metrics = verifier.get_full_metrics(
            '{"operators": [{"flops": 1000000}]}',
            input_shapes=[[32, 1024]],
        )
        assert isinstance(metrics, dict)
        assert "area_mm2" in metrics
        assert "energy_per_op_pj" in metrics

    def test_custom_design(self):
        """Test verifier with custom design point."""
        from yirage.rl.verifier.accelforge_verifier import AccelForgeVerifier

        verifier = AccelForgeVerifier(
            design_point={"pe_array_rows": 64, "pe_array_cols": 64}
        )
        assert verifier.design.pe_array_rows == 64


# =============================================================================
# Multi-Objective Reward Tests
# =============================================================================


class TestMultiObjectiveReward:
    """Tests for multi-objective reward function in ConfigEnv."""

    def test_latency_only_reward(self):
        """Test reward with latency only (legacy behavior)."""
        from yirage.rl.search.hierarchical_env import ConfigEnv

        env = ConfigEnv({"reward_weight_latency": 1.0})
        env.episode_results = [{"verified": True, "latency_ms": 1.0}]
        reward = env.get_config_reward()
        assert reward > 0

    def test_multi_objective_reward(self):
        """Test reward with latency + energy + area."""
        from yirage.rl.search.hierarchical_env import ConfigEnv

        env = ConfigEnv({
            "reward_weight_latency": 0.5,
            "reward_weight_energy": 0.2,
            "reward_weight_area": 0.15,
            "reward_weight_power": 0.15,
        })
        env.episode_results = [
            {
                "verified": True,
                "latency_ms": 1.0,
                "energy_pj": 2.0,
                "area_mm2": 10.0,
                "power_mw": 500.0,
            }
        ]
        reward = env.get_config_reward()
        assert reward > 0

    def test_no_results_penalty(self):
        """Test penalty when no results."""
        from yirage.rl.search.hierarchical_env import ConfigEnv

        env = ConfigEnv()
        reward = env.get_config_reward()
        assert reward == -1.0

    def test_no_valid_results_penalty(self):
        """Test penalty when no valid results."""
        from yirage.rl.search.hierarchical_env import ConfigEnv

        env = ConfigEnv()
        env.episode_results = [{"verified": False}]
        reward = env.get_config_reward()
        assert reward == -0.5


# =============================================================================
# MuGraphFeature Extension Tests
# =============================================================================


class TestMuGraphFeatureExtension:
    """Tests for AccelForge fields in MuGraphFeature."""

    def test_energy_fields_exist(self):
        """Test energy/area/power fields exist in MuGraphFeature."""
        from yirage.rl.features.mugraph_features import MuGraphFeature

        features = MuGraphFeature(
            energy_per_op_pj=2.5,
            area_mm2=10.0,
            total_power_mw=500.0,
            leak_power_mw=50.0,
            pe_utilization=0.8,
        )
        assert features.energy_per_op_pj == 2.5
        assert features.area_mm2 == 10.0
        assert features.total_power_mw == 500.0

    def test_json_roundtrip_with_energy(self):
        """Test JSON serialization includes new fields."""
        from yirage.rl.features.mugraph_features import MuGraphFeature

        features = MuGraphFeature(
            energy_per_op_pj=2.5,
            area_mm2=10.0,
            total_power_mw=500.0,
        )
        json_str = features.to_json()
        data = json.loads(json_str)
        assert data["energy_per_op_pj"] == 2.5
        assert data["area_mm2"] == 10.0

        # Round-trip
        restored = MuGraphFeature.from_json(json_str)
        assert restored.energy_per_op_pj == 2.5
        assert restored.area_mm2 == 10.0


# =============================================================================
# Feature Extractor Extension Tests
# =============================================================================


class TestFeatureExtractorWithAccelForge:
    """Tests for AccelForge-enhanced feature extraction."""

    def test_extract_with_accelforge(self):
        """Test extract_with_accelforge method."""
        from yirage.rl.features.extractor import GraphFeatureExtractor
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )

        bridge = AccelForgeBridge()
        profile = bridge.to_hardware_profile(AccelForgeDesignPoint())

        extractor = GraphFeatureExtractor(use_cpp=False)
        features = extractor.extract_with_accelforge(
            graph_json='{"operators": [], "inputs": []}',
            hardware_profile=profile,
        )

        assert features.energy_per_op_pj > 0
        assert features.area_mm2 > 0

    def test_extract_without_accelforge(self):
        """Test standard extraction still works."""
        from yirage.rl.features.extractor import GraphFeatureExtractor
        from yirage.rl.hardware.profile import HardwareProfile

        extractor = GraphFeatureExtractor(use_cpp=False)
        features = extractor.extract_with_accelforge(
            graph_json='{"operators": [], "inputs": []}',
            hardware_profile=HardwareProfile(backend="cuda"),
        )

        # Non-AccelForge profile: energy fields should be 0
        assert features.energy_per_op_pj == 0.0


# =============================================================================
# Co-Design Trainer Tests
# =============================================================================


class TestCoDesignTrainer:
    """Tests for co-design training mode."""

    def test_codesign_mode(self):
        """Test trainer supports codesign mode."""
        from yirage.rl.search.hierarchical_trainer import (
            HierarchicalTrainer,
            HierarchicalTrainingConfig,
        )

        config = HierarchicalTrainingConfig(
            mode="codesign",
            max_iterations=2,
            config_episodes_per_iter=2,
            accelerator_episodes_per_iter=2,
        )
        trainer = HierarchicalTrainer(config)
        result = trainer.train(
            target_graphs=['{"operators": [], "inputs": [], "outputs": []}'],
        )
        assert result["mode"] == "codesign"
        assert "pareto_front" in result

    def test_training_config_accelforge_fields(self):
        """Test training config has AccelForge-related fields."""
        from yirage.rl.search.hierarchical_trainer import HierarchicalTrainingConfig

        config = HierarchicalTrainingConfig(
            accelerator_enabled=True,
            area_budget_mm2=50.0,
            power_budget_mw=3000.0,
        )
        assert config.accelerator_enabled
        assert config.area_budget_mm2 == 50.0
        assert config.power_budget_mw == 3000.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline(self):
        """Test complete AccelForge → HardwareProfile → PerformanceEstimate pipeline."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )
        from yirage.rl.hardware.config_coupling import HardwareConfig, HardwareSearchCoupling

        # 1. Create design and evaluate
        bridge = AccelForgeBridge()
        design = AccelForgeDesignPoint(pe_array_rows=16, pe_array_cols=16)
        metrics = bridge.evaluate(design)

        # 2. Convert to HardwareProfile
        profile = bridge.to_hardware_profile(design, metrics)
        assert profile.backend == "accelforge"

        # 3. Get feature vector
        features = profile.to_feature_vector()
        assert features.shape == (32,)

        # 4. Estimate performance
        coupling = HardwareSearchCoupling(profile)
        config = HardwareConfig()
        estimate = coupling.estimate_performance(
            config, {"theoretical_flops": 1e9}
        )
        assert estimate.energy_pj > 0
        assert estimate.area_mm2 > 0
        assert estimate.confidence >= 0.6

    def test_accelforge_verifier_pipeline(self):
        """Test AccelForge verifier pipeline."""
        from yirage.rl.verifier.accelforge_verifier import AccelForgeVerifier

        verifier = AccelForgeVerifier()

        # Verify
        verify_result = verifier.verify_fingerprint('{"operators": []}', '{"operators": []}')
        assert verify_result.verified

        # Profile
        profile_result = verifier.profile_kernel(
            '{"operators": [{"flops": 1e6}]}',
            input_shapes=[[32, 1024]],
        )
        assert profile_result.latency_ms > 0

    def test_imports_from_rl_module(self):
        """Test that all new symbols are importable from yirage.rl."""
        from yirage.rl.hardware import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
            AccelForgeMetrics,
            AccelForgeDetector,
            is_accelforge_available,
        )
        from yirage.rl.search import (
            AcceleratorEnv,
            AcceleratorActionSpace,
            AcceleratorObservationSpace,
            AcceleratorDesignConstraints,
            ParetoFrontTracker,
            ParetoPoint,
        )
        from yirage.rl.verifier import AccelForgeVerifier

        # All imports should succeed
        assert AccelForgeBridge is not None
        assert AcceleratorEnv is not None
        assert AccelForgeVerifier is not None


# =============================================================================
# Hardware-Config Coupling Tests
# =============================================================================


class TestAccelForgeHardwareCoupling:
    """Tests that AccelForge is properly coupled with hardware config flow."""

    def test_config_generator_accelforge_backend(self):
        """ConfigGenerator produces AccelForge-derived configs, not GPU defaults."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )
        from yirage.rl.hardware.config_coupling import ConfigGenerator

        design = AccelForgeDesignPoint(
            pe_array_rows=32, pe_array_cols=32, l1_buffer_kb=128.0,
            dataflow="weight_stationary",
        )
        bridge = AccelForgeBridge()
        profile = bridge.to_hardware_profile(design)

        gen = ConfigGenerator(profile)
        config = gen.generate()

        # block_dim_x should be derived from pe_cols (32), not GPU defaults (256)
        assert config.block_dim_x == 32
        # grid_dim_x from pe_rows
        assert config.grid_dim_x == 32
        # shared memory from L1 buffer
        assert config.shared_memory_size == int(128.0 * 1024)

    def test_config_generator_different_designs_give_different_configs(self):
        """Different AccelForge designs produce different kernel configs."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )
        from yirage.rl.hardware.config_coupling import ConfigGenerator

        bridge = AccelForgeBridge()

        d1 = AccelForgeDesignPoint(pe_array_rows=8, pe_array_cols=8, l1_buffer_kb=32.0)
        d2 = AccelForgeDesignPoint(pe_array_rows=64, pe_array_cols=64, l1_buffer_kb=256.0)

        p1 = bridge.to_hardware_profile(d1)
        p2 = bridge.to_hardware_profile(d2)

        c1 = ConfigGenerator(p1).generate()
        c2 = ConfigGenerator(p2).generate()

        assert c1.block_dim_x != c2.block_dim_x or c1.grid_dim_x != c2.grid_dim_x
        assert c1.shared_memory_size != c2.shared_memory_size

    def test_constraints_incorporate_accelforge_profile(self):
        """HardwareSearchCoupling auto-derives constraints from AccelForge profile."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )
        from yirage.rl.hardware.config_coupling import HardwareConfig, HardwareSearchCoupling

        design = AccelForgeDesignPoint(
            pe_array_rows=8, pe_array_cols=8,
            l1_buffer_kb=16.0,
            dataflow="weight_stationary",
            data_precision="fp16",
        )
        bridge = AccelForgeBridge()
        profile = bridge.to_hardware_profile(design)

        coupling = HardwareSearchCoupling(profile)
        constraints = coupling.get_constraints(HardwareConfig())

        # max_operators should be limited by PE count (64)
        assert constraints["max_operators"] <= 64
        # max_shared_memory should be limited by L1 buffer (16KB)
        assert constraints["max_shared_memory"] <= 16 * 1024
        # Precision should be propagated
        assert constraints["supported_precisions"] == ["fp16"]
        # Dataflow reuse info
        assert constraints["supports_weight_reuse"] is True
        assert constraints["supports_output_reuse"] is False

    def test_constraints_with_accelerator_constraints(self):
        """Explicit AcceleratorDesignConstraints further restrict search space."""
        from yirage.rl.hardware.config_coupling import HardwareConfig, HardwareSearchCoupling
        from yirage.rl.hardware.profile import HardwareProfile
        from yirage.rl.search.accelerator_space import AcceleratorDesignConstraints

        profile = HardwareProfile(backend="cuda", total_cores=1024)
        coupling = HardwareSearchCoupling(profile)

        accel_c = AcceleratorDesignConstraints(
            max_parallelism=32,
            max_shared_memory_kb=16.0,
            max_tile_size=64,
            supported_precisions=["int8"],
        )

        constraints = coupling.get_constraints(
            HardwareConfig(), accelerator_constraints=accel_c
        )

        assert constraints["max_operators"] <= 32
        assert constraints["max_shared_memory"] <= 16 * 1024
        assert constraints["max_tile_size"] == 64
        assert constraints["supported_precisions"] == ["int8"]

    def test_graph_env_uses_coupled_design(self):
        """ConstrainedGraphEnv uses the design from env_config, not standalone default."""
        from yirage.rl.search.hierarchical_env import (
            ConstrainedGraphEnv,
            HierarchicalEnvConfig,
        )
        from yirage.rl.search.config_space import HardwareConfig, SearchSpaceConstraints

        config = HierarchicalEnvConfig(
            backend="accelforge",
            accelforge_design={
                "pe_array_rows": 64,
                "pe_array_cols": 64,
                "l1_buffer_kb": 256.0,
                "dataflow": "row_stationary",
                "data_precision": "int8",
            },
        )
        hw_config = HardwareConfig()
        constraints = SearchSpaceConstraints(hw_config)
        env = ConstrainedGraphEnv(constraints, vars(config))

        # The env should have the coupled design, not standalone
        assert env.env_config.accelforge_design is not None
        assert env.env_config.accelforge_design["pe_array_rows"] == 64
        assert env.env_config.accelforge_design["dataflow"] == "row_stationary"

    def test_finish_step_includes_accelforge_metrics_from_target_graph(self):
        """FINISH should surface AccelForge metrics when target_graph_json is set."""
        import json

        import numpy as np

        from yirage.rl.search.config_space import HardwareConfig, SearchSpaceConstraints
        from yirage.rl.search.graph_space import GraphAction
        from yirage.rl.search.hierarchical_env import (
            ConstrainedGraphEnv,
            HierarchicalEnvConfig,
        )

        cy_graph = [
            {
                "op_type": "kn_matmul_op",
                "input_tensors": [
                    {"guid": 1, "num_dims": 2, "dim": [8, 32, 0, 0]},
                    {"guid": 2, "num_dims": 2, "dim": [32, 64, 0, 0]},
                ],
                "output_tensors": [
                    {"guid": 3, "num_dims": 2, "dim": [8, 64, 0, 0]},
                ],
            }
        ]
        config = HierarchicalEnvConfig(
            backend="accelforge",
            target_graph_json=json.dumps(cy_graph),
            accelforge_design={"pe_array_rows": 8, "pe_array_cols": 8, "data_precision": "fp16"},
        )
        env = ConstrainedGraphEnv(
            SearchSpaceConstraints(HardwareConfig()),
            vars(config),
        )
        env.reset()
        assert env.kernel_graph_json != "{}"

        finish_info = {}
        for action_type in (
            GraphAction.ADD_KN_OP,
            GraphAction.CREATE_TB,
            GraphAction.ADD_TB_OP,
            GraphAction.FINISH,
        ):
            action = np.zeros(8, dtype=int)
            action[0] = action_type
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                finish_info = info
                break

        assert finish_info.get("verified") is True
        assert "accelforge_metrics" in finish_info
        assert finish_info["accelforge_metrics"].get("latency_ms", 0) > 0

    def test_level0_design_propagates_to_level2(self):
        """In co-design mode, Level 0 design should propagate to graph env."""
        from yirage.rl.search.hierarchical_env import (
            HierarchicalSearchEnv,
            HierarchicalEnvConfig,
        )

        config = HierarchicalEnvConfig(
            backend="accelforge",
            accelforge_design={
                "pe_array_rows": 32,
                "pe_array_cols": 32,
            },
        )
        env = HierarchicalSearchEnv(vars(config))
        env.reset()

        # Step through Level 1 (which triggers Level 2)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        # Graph env should have inherited the AccelForge design
        assert env.graph_env.env_config.backend == "accelforge"
        assert env.graph_env.env_config.accelforge_design is not None

    def test_env_config_new_fields(self):
        """HierarchicalEnvConfig supports hardware_profile and accelforge_design."""
        from yirage.rl.search.hierarchical_env import HierarchicalEnvConfig
        from yirage.rl.hardware.profile import HardwareProfile

        profile = HardwareProfile(backend="accelforge", device_name="test")
        config = HierarchicalEnvConfig(
            backend="accelforge",
            hardware_profile=profile,
            accelforge_design={"pe_array_rows": 16},
            accelerator_constraints={"max_parallelism": 256},
        )

        assert config.hardware_profile is not None
        assert config.hardware_profile.backend == "accelforge"
        assert config.accelforge_design["pe_array_rows"] == 16
        assert config.accelerator_constraints["max_parallelism"] == 256

        # Roundtrip through vars()
        d = vars(config)
        config2 = HierarchicalEnvConfig(**d)
        assert config2.hardware_profile is not None
        assert config2.accelforge_design is not None

    def test_search_space_constraints_setter(self):
        """SearchSpaceConstraints.max_operators is settable for AccelForge coupling."""
        from yirage.rl.search.config_space import HardwareConfig, SearchSpaceConstraints

        constraints = SearchSpaceConstraints(HardwareConfig())
        original = constraints.max_operators

        constraints.max_operators = 10
        assert constraints.max_operators == 10

        # Ensure minimum is 1
        constraints.max_operators = 0
        assert constraints.max_operators == 1

    def test_codesign_propagates_design_to_inner_env(self):
        """Co-design training should propagate design info to inner envs."""
        from yirage.rl.search.hierarchical_trainer import (
            HierarchicalTrainer,
            HierarchicalTrainingConfig,
        )

        config = HierarchicalTrainingConfig(
            mode="codesign",
            max_iterations=1,
            config_episodes_per_iter=1,
            accelerator_episodes_per_iter=1,
        )
        trainer = HierarchicalTrainer(config)
        result = trainer.train(
            target_graphs=['{"operators": [{"flops": 100}], "inputs": [], "outputs": []}'],
        )
        assert result["mode"] == "codesign"
        # The inner env should have received AccelForge design info
        assert "stats" in result

    def test_performance_estimate_accelforge_uses_coupled_design(self):
        """estimate_performance for AccelForge uses design from profile, not standalone."""
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
        )
        from yirage.rl.hardware.config_coupling import HardwareConfig, HardwareSearchCoupling

        # Create specific design
        design = AccelForgeDesignPoint(
            pe_array_rows=64, pe_array_cols=64,
            dataflow="row_stationary", data_precision="fp32",
        )
        bridge = AccelForgeBridge()
        profile = bridge.to_hardware_profile(design)

        coupling = HardwareSearchCoupling(profile)
        config = HardwareConfig()
        estimate = coupling.estimate_performance(
            config, {"theoretical_flops": 1e9, "memory_bytes": 1e6}
        )

        # Should have AccelForge metrics
        assert estimate.energy_pj > 0
        assert estimate.area_mm2 > 0
        assert estimate.power_mw > 0
        assert estimate.confidence >= 0.6

    def test_gpu_backend_unaffected_by_accelforge_coupling(self):
        """GPU backend constraints should not include AccelForge-specific fields."""
        from yirage.rl.hardware.config_coupling import (
            ConfigGenerator,
            HardwareConfig,
            HardwareSearchCoupling,
        )
        from yirage.rl.hardware.profile import HardwareProfile

        gpu_profile = HardwareProfile(
            backend="cuda", total_cores=4096, max_threads_per_block=1024,
        )
        gen = ConfigGenerator(gpu_profile)
        config = gen.generate()

        # GPU config should use GPU-style defaults, not AccelForge path
        assert config.block_dim_x in [64, 128, 256, 512, 1024]

        coupling = HardwareSearchCoupling(gpu_profile)
        constraints = coupling.get_constraints(config)

        # Should not have AccelForge-specific keys
        assert "supported_precisions" not in constraints
        assert "supports_weight_reuse" not in constraints


# =============================================================================
# Real AccelForge API Path Tests
# =============================================================================


class TestAccelForgeBridgeRealAPI:
    """
    Tests that exercise the real AccelForge Spec.from_yaml path.

    Each test calls bridge.evaluate() which (when accelforge is installed)
    runs the real AccelForge mapper and returns physically-grounded metrics.
    When accelforge is not installed, use a local Spec test double so these
    integration-path tests still exercise YiRage's AccelForge bridge plumbing.
    """

    @staticmethod
    def _install_fake_accelforge(monkeypatch):
        """Install a small AccelForge Spec-compatible test double."""

        class FakeResults:
            def __init__(self, spec):
                self.spec = spec

            def energy(self, per_component=False):
                mac = self.spec.mac_energy * self.n_computes()
                l1 = self.spec.l1_energy * self.spec.bits_per_value * self.spec.m_dim
                l2 = self.spec.l2_energy * self.spec.bits_per_value * self.spec.n_dim
                main = self.spec.main_energy * self.spec.bits_per_value * self.spec.k_dim
                if per_component:
                    return {
                        "MAC": mac,
                        "L1Buffer": l1,
                        "L2Buffer": l2,
                        "MainMemory": main,
                    }
                return mac + l1 + l2 + main

            def latency(self):
                return max(1, int(self.n_computes() / max(self.spec.total_pes, 1)))

            def n_computes(self):
                return max(1, self.spec.m_dim * self.spec.k_dim * self.spec.n_dim)

            def resource_usage(self):
                return {"L1Buffer": 0.5}

        class FakeSpec:
            def __init__(self, arch_yaml, workload_yaml):
                self.arch_yaml = arch_yaml
                self.workload_yaml = workload_yaml
                self.total_pes = self._extract_int(r"n_parallel_instances:\s*(\d+)", 256)
                self.mac_area = self._extract_compute_float("area", 0.0012)
                self.mac_leak = self._extract_compute_float("leak_power", 0.06)
                memory_blocks = arch_yaml.split("- !Memory")
                self.main_energy = self._extract_block_energy(memory_blocks, "MainMemory", 10.0)
                self.l2_energy = self._extract_block_energy(memory_blocks, "L2Buffer", 0.5)
                self.l1_energy = self._extract_block_energy(memory_blocks, "L1Buffer", 0.1)
                self.mac_energy = self._extract_compute_energy(0.2)
                self.m_dim = self._extract_int(r"M:\s*(\d+)", 128, workload_yaml)
                self.k_dim = self._extract_int(r"K:\s*(\d+)", 256, workload_yaml)
                self.n_dim = self._extract_int(r"N:\s*(\d+)", 256, workload_yaml)
                self.bits_per_value = self._extract_int(r"All:\s*(\d+)", 16, workload_yaml)

            @classmethod
            def from_yaml(cls, arch_path, workload_path):
                return cls(Path(arch_path).read_text(), Path(workload_path).read_text())

            def map_workload_to_arch(self, print_progress=False):
                return FakeResults(self)

            def calculate_component_area_energy_latency_leak(self):
                l2_area = self._extract_memory_float("L2Buffer", "area", 1.024)
                l1_area = self._extract_memory_float("L1Buffer", "area", 0.128)
                l2_leak = self._extract_memory_float("L2Buffer", "leak_power", 51.2)
                l1_leak = self._extract_memory_float("L1Buffer", "leak_power", 6.4)
                mac_total_area = self.mac_area * self.total_pes
                mac_total_leak = self.mac_leak * self.total_pes
                return types.SimpleNamespace(
                    arch=types.SimpleNamespace(
                        nodes=[
                            types.SimpleNamespace(
                                name="MAC",
                                total_area=mac_total_area,
                                area=self.mac_area,
                                total_leak_power=mac_total_leak,
                                leak_power=self.mac_leak,
                            ),
                            types.SimpleNamespace(
                                name="L2Buffer",
                                total_area=l2_area,
                                area=l2_area,
                                total_leak_power=l2_leak,
                                leak_power=l2_leak,
                            ),
                            types.SimpleNamespace(
                                name="L1Buffer",
                                total_area=l1_area,
                                area=l1_area,
                                total_leak_power=l1_leak,
                                leak_power=l1_leak,
                            ),
                        ]
                    )
                )

            def _extract_int(self, pattern, default, text=None):
                import re

                match = re.search(pattern, self.arch_yaml if text is None else text)
                return int(match.group(1)) if match else default

            def _extract_compute_float(self, field, default):
                import re

                match = re.search(r"- !Compute[\s\S]*?" + field + r":\s*([\d.]+)", self.arch_yaml)
                return float(match.group(1)) if match else default

            def _extract_compute_energy(self, default):
                import re

                match = re.search(r"- !Compute[\s\S]*?energy:\s*([\d.]+)", self.arch_yaml)
                return float(match.group(1)) if match else default

            @staticmethod
            def _extract_block_energy(blocks, name, default):
                import re

                for block in blocks:
                    if f"name: {name}" in block:
                        match = re.search(r"energy:\s*([\d.]+)", block)
                        return float(match.group(1)) if match else default
                return default

            def _extract_memory_float(self, name, field, default):
                import re

                pattern = r"name:\s*" + name + r"[\s\S]*?" + field + r":\s*([\d.]+)"
                match = re.search(pattern, self.arch_yaml)
                return float(match.group(1)) if match else default

        fake_module = types.ModuleType("accelforge")
        fake_module.Spec = FakeSpec
        monkeypatch.setitem(sys.modules, "accelforge", fake_module)
        return FakeSpec

    @pytest.fixture(autouse=True)
    def skip_if_unavailable(self, monkeypatch):
        import yirage.rl.hardware.accelforge_bridge as bridge_mod

        if not bridge_mod.ACCELFORGE_AVAILABLE:
            self._install_fake_accelforge(monkeypatch)
            monkeypatch.setattr(bridge_mod, "ACCELFORGE_AVAILABLE", True)
            monkeypatch.setattr(bridge_mod, "ACCELFORGE_VERSION", None)
            monkeypatch.setattr(bridge_mod, "ACCELFORGE_IMPORT_ERROR", "")

    @pytest.fixture
    def bridge(self):
        from yirage.rl.hardware.accelforge_bridge import AccelForgeBridge

        # Use a short time limit so CI does not hang
        return AccelForgeBridge(config={"mapper_time_limit": 30.0})

    def test_accelforge_spec_class_stored(self, bridge):
        """_init_accelforge must store the Spec class, not the whole module."""
        from accelforge import Spec

        assert bridge._af_model is Spec

    def test_design_to_arch_yaml_structure(self, bridge):
        """_design_to_arch_yaml should produce valid YAML with required keys
        and physics-based numeric values within expected ranges."""
        import re
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        design = AccelForgeDesignPoint()  # 16×16, fp16, 7 nm, 1 GHz
        yaml_str = bridge._design_to_arch_yaml(design)

        assert "arch:" in yaml_str
        assert "MainMemory" in yaml_str
        assert "L2Buffer" in yaml_str
        assert "L1Buffer" in yaml_str
        assert "MAC" in yaml_str
        assert "mapper:" in yaml_str
        assert "max_loops_minus_ranks:" in yaml_str
        assert "time_limit:" in yaml_str
        # AccelForge multiplies MAC area by n_parallel_instances
        assert f"n_parallel_instances: {design.total_pes}" in yaml_str

        # ---- Physics model numeric checks (7 nm, fp16) ----
        # DRAM energy: 10.0 pJ/bit (node-independent)
        assert "energy: 10.000000" in yaml_str

        # L2 energy: 0.5 × (7/7) = 0.5 pJ/bit
        assert "energy: 0.500000" in yaml_str

        # L1 energy (mesh, noc_scale=1.0): 0.1 × 1.0 = 0.1 pJ/bit
        assert "energy: 0.100000" in yaml_str

        # MAC compute energy: 0.2 × (7/7) = 0.2 pJ/MAC for fp16
        assert "energy: 0.200000" in yaml_str

        # MAC area per PE: l0_area_per_pe + pe_compute_area = (1*0.0002 + 0.001)*1 = 0.0012 mm²
        # Extract the area line for MAC block
        mac_block = yaml_str.split("- !Compute")[-1]
        mac_area_match = re.search(r"area:\s*([\d.]+)", mac_block)
        assert mac_area_match, "MAC area not found"
        mac_area_per_pe = float(mac_area_match.group(1))
        assert 0.0 < mac_area_per_pe < 0.01, (
            f"MAC area per PE should be per-instance, got {mac_area_per_pe}"
        )

        # Test at 28 nm: tech_scale = 4, L2 energy = 0.5 × 4 = 2.0 pJ/bit
        design_28 = AccelForgeDesignPoint(tech_node_nm=28)
        yaml_28 = bridge._design_to_arch_yaml(design_28)
        assert "energy: 2.000000" in yaml_28  # L2 at 28 nm

        # Ring NoC: L1 effective energy = 0.1 × 1.5 = 0.15 pJ/bit
        design_ring = AccelForgeDesignPoint(noc_topology="ring")
        yaml_ring = bridge._design_to_arch_yaml(design_ring)
        assert "energy: 0.150000" in yaml_ring

    def test_workload_to_yaml_structure(self, bridge):
        """_workload_to_yaml should produce valid Einsum YAML."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        design = AccelForgeDesignPoint()

        # Default workload
        yaml_default = bridge._workload_to_yaml(design, None)
        assert "workload:" in yaml_default
        assert "MatMul" in yaml_default
        assert "Input" in yaml_default
        assert "Weight" in yaml_default
        assert "Output" in yaml_default
        assert "bits_per_value" in yaml_default

        # Shape-based workload
        yaml_shaped = bridge._workload_to_yaml(
            design, {"batch_size": 2, "sequence_length": 32,
                     "hidden_dim": 128, "output_dim": 128}
        )
        assert "M: 64" in yaml_shaped   # 2 × 32
        assert "K: 128" in yaml_shaped
        assert "N: 128" in yaml_shaped

        # estimated_flops-only workload (cube-root approximation)
        yaml_flops = bridge._workload_to_yaml(design, {"estimated_flops": 2e6})
        assert "workload:" in yaml_flops

    def test_evaluate_returns_physical_metrics(self, bridge):
        """Real AccelForge path should return physically meaningful metrics."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        design = AccelForgeDesignPoint(
            pe_array_rows=16,
            pe_array_cols=16,
            data_precision="fp16",
            l1_buffer_kb=64.0,
            l2_buffer_kb=512.0,
            tech_node_nm=7,
        )
        workload = {"batch_size": 1, "sequence_length": 32,
                    "hidden_dim": 64, "output_dim": 64}
        metrics = bridge.evaluate(design, workload)

        # Confidence should reflect real AccelForge (not analytical fallback)
        assert metrics.confidence == 0.90

        # Latency must be positive
        assert metrics.latency_ms > 0

        # Area: 16×16 PE array at 7nm — expect 0.5–10 mm²
        assert 0.1 < metrics.area_mm2 < 20.0

        # PE area < total area (buffers also contribute)
        assert metrics.pe_area_mm2 < metrics.area_mm2

        # Energy per op must be positive
        assert metrics.energy_per_op_pj > 0

        # Peak TOPS: 256 PEs × 2 ops × 1 GHz = 0.512 TOPS
        assert abs(metrics.peak_tops - 0.512) < 0.01

    def test_larger_pe_array_lower_latency(self, bridge):
        """Doubling PE count should approximately halve latency."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        workload = {"batch_size": 1, "sequence_length": 32,
                    "hidden_dim": 128, "output_dim": 128}

        small = AccelForgeDesignPoint(pe_array_rows=8, pe_array_cols=8)
        large = AccelForgeDesignPoint(pe_array_rows=16, pe_array_cols=16)

        m_small = bridge.evaluate(small, workload)
        m_large = bridge.evaluate(large, workload)

        # 4× more PEs → latency should be ~4× lower
        assert m_large.latency_ms < m_small.latency_ms

    def test_larger_pe_array_larger_area(self, bridge):
        """Larger PE array should occupy more chip area."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        small = AccelForgeDesignPoint(pe_array_rows=8, pe_array_cols=8)
        large = AccelForgeDesignPoint(pe_array_rows=16, pe_array_cols=16)

        m_small = bridge.evaluate(small)
        m_large = bridge.evaluate(large)

        assert m_large.area_mm2 > m_small.area_mm2

    def test_advanced_node_lower_area(self, bridge):
        """7 nm design should occupy less area than 28 nm."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        d7 = AccelForgeDesignPoint(tech_node_nm=7)
        d28 = AccelForgeDesignPoint(tech_node_nm=28)

        m7 = bridge.evaluate(d7)
        m28 = bridge.evaluate(d28)

        assert m7.area_mm2 < m28.area_mm2

    def test_noc_topology_affects_energy(self, bridge):
        """Ring topology (1.5× L1 energy) should yield higher total energy than mesh."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        workload = {"batch_size": 1, "sequence_length": 16,
                    "hidden_dim": 64, "output_dim": 64}
        mesh = AccelForgeDesignPoint(noc_topology="mesh")
        ring = AccelForgeDesignPoint(noc_topology="ring")

        m_mesh = bridge.evaluate(mesh, workload)
        m_ring = bridge.evaluate(ring, workload)

        assert m_ring.energy_per_op_pj >= m_mesh.energy_per_op_pj

    def test_int8_lower_energy_than_fp32(self, bridge):
        """INT8 compute energy should be lower than FP32 at the same tech node."""
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        workload = {"batch_size": 1, "sequence_length": 16,
                    "hidden_dim": 64, "output_dim": 64}
        int8 = AccelForgeDesignPoint(data_precision="int8")
        fp32 = AccelForgeDesignPoint(data_precision="fp32")

        m_int8 = bridge.evaluate(int8, workload)
        m_fp32 = bridge.evaluate(fp32, workload)

        assert m_int8.energy_per_op_pj < m_fp32.energy_per_op_pj

    def test_yaml_generation_roundtrip(self, bridge):
        """The generated YAMLs should be parseable by AccelForge's Spec."""
        import tempfile, os
        from accelforge import Spec
        from yirage.rl.hardware.accelforge_bridge import AccelForgeDesignPoint

        design = AccelForgeDesignPoint()
        arch_yaml = bridge._design_to_arch_yaml(design)
        workload_yaml = bridge._workload_to_yaml(design, None)

        arch_path = workload_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                             delete=False) as f:
                f.write(arch_yaml)
                arch_path = f.name
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                             delete=False) as f:
                f.write(workload_yaml)
                workload_path = f.name

            spec = Spec.from_yaml(arch_path, workload_path)
            # If we get here without an exception the YAML is valid
            assert spec is not None
        finally:
            for p in (arch_path, workload_path):
                if p and os.path.exists(p):
                    os.unlink(p)


# =============================================================================
# mugraph_to_workload Tests
# =============================================================================


class TestMugraphToWorkload:
    """
    Tests for the YiRage µGraph → AccelForge workload translator.

    Verifies that actual YiRage operator types and tensor shapes are correctly
    translated into AccelForge Einsum dimensions instead of just passing a
    scalar estimated_flops proxy.

    All tests are pure-Python (no AccelForge install required).
    """

    def _make_graph(self, op_type, input_shapes, output_shapes=None,
                    flops=1_000_000, extra_graph_fields=None):
        import json
        tensors = []
        tid = 0
        input_ids = []
        for shape in input_shapes:
            tensors.append({"tensor_id": tid, "dims": shape})
            input_ids.append(tid)
            tid += 1
        output_ids = []
        for shape in (output_shapes or []):
            tensors.append({"tensor_id": tid, "dims": shape})
            output_ids.append(tid)
            tid += 1
        graph = {
            "operators": [{"op_id": 0, "op_type": op_type, "flops": flops,
                           "input_tensor_ids": input_ids,
                           "output_tensor_ids": output_ids}],
            "tensors": tensors,
        }
        if extra_graph_fields:
            graph.update(extra_graph_fields)
        return json.dumps(graph)

    # ------ matmul family ------

    def test_matmul_2d(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = self._make_graph("matmul", [[64, 128], [128, 256]])
        w = mugraph_to_workload(g)
        assert w["m_dim"] == 64 and w["k_dim"] == 128 and w["n_dim"] == 256
        assert w["op_type"] == "matmul"

    def test_batch_matmul(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = self._make_graph("batch_matmul", [[8, 32, 64], [8, 64, 128]])
        w = mugraph_to_workload(g)
        assert w["m_dim"] == 32 and w["k_dim"] == 64 and w["n_dim"] == 128

    def test_bmm_alias(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = self._make_graph("bmm", [[4, 16, 32], [4, 32, 64]])
        w = mugraph_to_workload(g)
        assert w["m_dim"] == 16 and w["k_dim"] == 32 and w["n_dim"] == 64

    def test_matmul_no_tensor_shapes_uses_flops(self):
        import json
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = json.dumps({"operators": [{"op_id": 0, "op_type": "matmul",
                                        "flops": 2 * 64**3}], "tensors": []})
        w = mugraph_to_workload(g)
        assert w["m_dim"] == w["k_dim"] == w["n_dim"] and w["m_dim"] > 0

    # ------ attention ------

    def test_attention_seq_head_dims(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = self._make_graph("attention", [[2, 8, 512, 64], [2, 8, 512, 64],
                                             [2, 8, 512, 64]])
        w = mugraph_to_workload(g)
        assert w["m_dim"] == 512 and w["k_dim"] == 64 and w["n_dim"] == 512

    def test_softmax_attention_alias(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = self._make_graph("softmax_attention", [[1, 4, 128, 32]])
        w = mugraph_to_workload(g)
        assert w["m_dim"] == 128 and w["k_dim"] == 32

    # ------ convolution ------

    def test_conv_linearised_matmul(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = self._make_graph("conv", [[1, 32, 28, 28], [64, 32, 3, 3]])
        w = mugraph_to_workload(g)
        assert w["m_dim"] == 676 and w["k_dim"] == 288 and w["n_dim"] == 64

    def test_convolution_alias(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = self._make_graph("convolution", [[1, 16, 8, 8], [32, 16, 1, 1]])
        w = mugraph_to_workload(g)
        assert w["m_dim"] == 64 and w["k_dim"] == 16 and w["n_dim"] == 32

    # ------ reduction ------

    def test_reduction_with_reduction_dimx(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = self._make_graph("reduction", [[32, 256]],
                             extra_graph_fields={"reduction_dimx": 256})
        w = mugraph_to_workload(g)
        assert w["k_dim"] == 256 and w["n_dim"] == 1

    def test_reduction_without_reduction_dimx(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = self._make_graph("reduction", [[8, 512]])
        w = mugraph_to_workload(g)
        assert w["k_dim"] == 512 and w["n_dim"] == 1

    # ------ mixed graphs ------

    def test_matmul_dominates_elementwise(self):
        import json
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = json.dumps({
            "operators": [
                {"op_id": 0, "op_type": "relu", "flops": 1000,
                 "input_tensor_ids": [0], "output_tensor_ids": [1]},
                {"op_id": 1, "op_type": "matmul", "flops": 5e5,
                 "input_tensor_ids": [2, 3], "output_tensor_ids": [4]},
            ],
            "tensors": [
                {"tensor_id": 2, "dims": [32, 128]},
                {"tensor_id": 3, "dims": [128, 64]},
            ],
        })
        w = mugraph_to_workload(g)
        assert w["op_type"] == "matmul"
        assert w["m_dim"] == 32 and w["n_dim"] == 64

    def test_attention_dominates_reduction(self):
        import json
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        g = json.dumps({
            "operators": [
                {"op_id": 0, "op_type": "reduction", "flops": 1e5,
                 "input_tensor_ids": [0]},
                {"op_id": 1, "op_type": "attention", "flops": 5e4,
                 "input_tensor_ids": [1], "output_tensor_ids": [2]},
            ],
            "tensors": [{"tensor_id": 1, "dims": [1, 4, 64, 32]}],
        })
        w = mugraph_to_workload(g)
        assert w["op_type"] == "attention"

    # ------ fallback paths ------

    def test_empty_graph_fallback(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        w = mugraph_to_workload("{}")
        assert "estimated_flops" in w and w["estimated_flops"] >= 1.0

    def test_invalid_json_fallback(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        w = mugraph_to_workload("not valid json")
        assert "estimated_flops" in w

    def test_empty_string_fallback(self):
        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload
        w = mugraph_to_workload("")
        assert "estimated_flops" in w

    # ------ integration: priority-0 path in _workload_to_yaml ------

    def test_m_k_n_direct_path_in_workload_yaml(self):
        from yirage.rl.hardware.accelforge_bridge import (
            AccelForgeBridge,
            AccelForgeDesignPoint,
            mugraph_to_workload,
        )
        bridge = AccelForgeBridge()
        design = AccelForgeDesignPoint()
        graph_json = self._make_graph("matmul", [[48, 96], [96, 192]])
        workload = mugraph_to_workload(graph_json)
        yaml_str = bridge._workload_to_yaml(design, workload)
        assert "M: 48" in yaml_str
        assert "K: 96" in yaml_str
        assert "N: 192" in yaml_str

    def test_cy_to_json_kn_matmul_list(self):
        import json

        from yirage.rl.hardware.accelforge_bridge import mugraph_to_workload

        cy_graph = [
            {
                "op_type": "kn_matmul_op",
                "input_tensors": [
                    {"guid": 1, "num_dims": 2, "dim": [32, 128, 0, 0]},
                    {"guid": 2, "num_dims": 2, "dim": [128, 256, 0, 0]},
                ],
                "output_tensors": [
                    {"guid": 3, "num_dims": 2, "dim": [32, 256, 0, 0]},
                ],
            }
        ]
        w = mugraph_to_workload(json.dumps(cy_graph))
        assert w["m_dim"] == 32 and w["k_dim"] == 128 and w["n_dim"] == 256
        assert w["op_type"] == "matmul"
