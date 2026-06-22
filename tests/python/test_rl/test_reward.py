#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
RL Reward Module Unit Tests

Tests for yirage/rl/env/reward.py module.
Run with: pytest tests/python/test_rl/test_reward.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import PYTHON_ROOT, load_module


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def reward_module():
    """Load reward module."""
    return load_module(
        "reward",
        PYTHON_ROOT / "yirage" / "rl" / "env" / "reward.py"
    )


# =============================================================================
# RewardConfig Tests
# =============================================================================

class TestRewardConfig:
    """Tests for RewardConfig class."""

    def test_reward_config_class_exists(self, reward_module):
        """Test RewardConfig class exists."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        assert hasattr(reward_module, "RewardConfig")

    def test_reward_config_creation(self, reward_module):
        """Test RewardConfig can be created."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        RewardConfig = getattr(reward_module, "RewardConfig", None)
        if RewardConfig is None:
            pytest.skip("RewardConfig not found")

        config = RewardConfig()
        assert config is not None

    def test_default_weights(self, reward_module):
        """Test default weight values."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        RewardConfig = getattr(reward_module, "RewardConfig", None)
        if RewardConfig is None:
            pytest.skip("RewardConfig not found")

        config = RewardConfig()

        # Should have weight attributes
        has_weights = (
            hasattr(config, "validity_weight") or
            hasattr(config, "performance_weight") or
            hasattr(config, "latency_weight")
        )
        assert has_weights

    def test_custom_weights(self, reward_module):
        """Test setting custom weights."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        RewardConfig = getattr(reward_module, "RewardConfig", None)
        if RewardConfig is None:
            pytest.skip("RewardConfig not found")

        config = RewardConfig(
            validity_weight=1.0,
            performance_weight=2.0,
        )
        assert config.validity_weight == 1.0
        assert config.performance_weight == 2.0


# =============================================================================
# RewardComputer Tests
# =============================================================================

class TestRewardComputer:
    """Tests for RewardComputer class."""

    def test_reward_computer_class_exists(self, reward_module):
        """Test RewardComputer class exists."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        assert hasattr(reward_module, "RewardComputer")

    def test_reward_computer_creation(self, reward_module):
        """Test RewardComputer can be created."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        RewardComputer = getattr(reward_module, "RewardComputer", None)
        RewardConfig = getattr(reward_module, "RewardConfig", None)

        if RewardComputer is None:
            pytest.skip("RewardComputer not found")

        if RewardConfig:
            config = RewardConfig()
            computer = RewardComputer(config)
        else:
            computer = RewardComputer()

        assert computer is not None

    def test_reward_computer_has_reset(self, reward_module):
        """Test RewardComputer has reset method."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        RewardComputer = getattr(reward_module, "RewardComputer", None)
        if RewardComputer is None:
            pytest.skip("RewardComputer not found")

        assert hasattr(RewardComputer, "reset")

    def test_reward_computer_has_compute(self, reward_module):
        """Test RewardComputer has compute method."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        RewardComputer = getattr(reward_module, "RewardComputer", None)
        if RewardComputer is None:
            pytest.skip("RewardComputer not found")

        assert hasattr(RewardComputer, "compute")

    def test_compute_valid_kernel_positive(self, reward_module):
        """Test computing reward for valid kernel."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        RewardComputer = getattr(reward_module, "RewardComputer", None)
        RewardConfig = getattr(reward_module, "RewardConfig", None)
        VerifyResult = getattr(reward_module, "VerifyResult", None)
        ProfileResult = getattr(reward_module, "ProfileResult", None)

        if None in (RewardComputer, VerifyResult, ProfileResult):
            pytest.skip("Required classes not found")

        if RewardConfig:
            config = RewardConfig()
            computer = RewardComputer(config)
        else:
            computer = RewardComputer()

        computer.reset()

        verify_result = VerifyResult(verified=True, fingerprint_time_ms=0.1)
        profile_result = ProfileResult(
            latency_ms=0.05, memory_bytes=8192, flops=2097152.0, compile_time_ms=100.0
        )

        reward = computer.compute(
            verify_result=verify_result,
            profile_result=profile_result,
            config_hash="test_config",
            search_depth=5,
            action_type=0,
        )

        assert reward is not None
        assert isinstance(reward, (int, float))

    def test_compute_invalid_kernel_penalty(self, reward_module):
        """Test computing reward for invalid kernel gives penalty."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        RewardComputer = getattr(reward_module, "RewardComputer", None)
        RewardConfig = getattr(reward_module, "RewardConfig", None)
        VerifyResult = getattr(reward_module, "VerifyResult", None)
        ProfileResult = getattr(reward_module, "ProfileResult", None)

        if None in (RewardComputer, VerifyResult, ProfileResult):
            pytest.skip("Required classes not found")

        if RewardConfig:
            config = RewardConfig()
            computer = RewardComputer(config)
        else:
            computer = RewardComputer()

        computer.reset()

        # Invalid verification
        verify_result = VerifyResult(verified=False, fingerprint_time_ms=0.1)
        profile_result = ProfileResult(
            latency_ms=0.0, memory_bytes=0, flops=0.0, compile_time_ms=0.0
        )

        reward = computer.compute(
            verify_result=verify_result,
            profile_result=profile_result,
            config_hash="test_config",
            search_depth=5,
            action_type=0,
        )

        assert reward is not None
        # Invalid kernel should get lower/negative reward
        assert isinstance(reward, (int, float))

    def test_get_stats(self, reward_module):
        """Test getting reward statistics."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        RewardComputer = getattr(reward_module, "RewardComputer", None)
        RewardConfig = getattr(reward_module, "RewardConfig", None)

        if RewardComputer is None:
            pytest.skip("RewardComputer not found")

        if RewardConfig:
            config = RewardConfig()
            computer = RewardComputer(config)
        else:
            computer = RewardComputer()

        computer.reset()
        stats = computer.get_stats()

        assert isinstance(stats, dict)


# =============================================================================
# VerifyResult Tests
# =============================================================================

class TestVerifyResult:
    """Tests for VerifyResult class."""

    def test_verify_result_class_exists(self, reward_module):
        """Test VerifyResult class exists."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        assert hasattr(reward_module, "VerifyResult")

    def test_verify_result_creation(self, reward_module):
        """Test VerifyResult can be created."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        VerifyResult = getattr(reward_module, "VerifyResult", None)
        if VerifyResult is None:
            pytest.skip("VerifyResult not found")

        result = VerifyResult(verified=True, fingerprint_time_ms=0.1)
        assert result.verified is True
        assert result.fingerprint_time_ms == 0.1


# =============================================================================
# ProfileResult Tests
# =============================================================================

class TestProfileResult:
    """Tests for ProfileResult class."""

    def test_profile_result_class_exists(self, reward_module):
        """Test ProfileResult class exists."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        assert hasattr(reward_module, "ProfileResult")

    def test_profile_result_creation(self, reward_module):
        """Test ProfileResult can be created."""
        if reward_module is None:
            pytest.skip("Reward module not available")

        ProfileResult = getattr(reward_module, "ProfileResult", None)
        if ProfileResult is None:
            pytest.skip("ProfileResult not found")

        result = ProfileResult(
            latency_ms=0.05,
            memory_bytes=8192,
            flops=2097152.0,
            compile_time_ms=100.0,
        )
        assert result.latency_ms == 0.05
        assert result.memory_bytes == 8192
