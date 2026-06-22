#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
RL Environment Unit Tests

Tests for yirage/rl/env/ module including YiRageSearchEnv.
Run with: pytest tests/python/test_rl/test_env.py -v
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import safe_import


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def env_module():
    """Load environment module."""
    return safe_import("yirage.rl.env.yirage_env")


@pytest.fixture(scope="module")
def observation_module():
    """Load observation module."""
    return safe_import("yirage.rl.env.observation")


# =============================================================================
# YiRageSearchEnv Tests
# =============================================================================

class TestYiRageSearchEnv:
    """Tests for YiRageSearchEnv class."""

    def test_env_class_exists(self, env_module):
        """Test YiRageSearchEnv class exists."""
        if env_module is None:
            pytest.skip("Environment module not available")

        assert hasattr(env_module, "YiRageSearchEnv")

    def test_env_config_class_exists(self, env_module):
        """Test EnvConfig class exists."""
        if env_module is None:
            pytest.skip("Environment module not available")

        assert hasattr(env_module, "EnvConfig")

    def test_env_config_creation(self, env_module):
        """Test EnvConfig can be created."""
        if env_module is None:
            pytest.skip("Environment module not available")

        EnvConfig = getattr(env_module, "EnvConfig", None)
        if EnvConfig is None:
            pytest.skip("EnvConfig not found")

        config = EnvConfig(
            target_graph_json={"operators": [], "inputs": [], "outputs": []},
            backend="cpu",
        )
        assert config is not None

    def test_env_has_reset_method(self, env_module):
        """Test environment has reset method."""
        if env_module is None:
            pytest.skip("Environment module not available")

        YiRageSearchEnv = getattr(env_module, "YiRageSearchEnv", None)
        if YiRageSearchEnv is None:
            pytest.skip("YiRageSearchEnv not found")

        assert hasattr(YiRageSearchEnv, "reset")

    def test_env_has_step_method(self, env_module):
        """Test environment has step method."""
        if env_module is None:
            pytest.skip("Environment module not available")

        YiRageSearchEnv = getattr(env_module, "YiRageSearchEnv", None)
        if YiRageSearchEnv is None:
            pytest.skip("YiRageSearchEnv not found")

        assert hasattr(YiRageSearchEnv, "step")


# =============================================================================
# Observation Tests
# =============================================================================

class TestSearchState:
    """Tests for SearchState class."""

    def test_search_state_class_exists(self, observation_module):
        """Test SearchState class exists."""
        if observation_module is None:
            pytest.skip("Observation module not available")

        assert hasattr(observation_module, "SearchState")

    def test_search_state_creation(self, observation_module):
        """Test SearchState can be created."""
        if observation_module is None:
            pytest.skip("Observation module not available")

        SearchState = getattr(observation_module, "SearchState", None)
        if SearchState is None:
            pytest.skip("SearchState not found")

        state = SearchState(
            search_level=1,
            search_depth=5,
            num_kn_operators=3,
            num_tb_operators=2,
            num_tensors=4,
            num_valid_found=1,
            best_latency_ms=0.5,
            current_grid_dim=(4, 2, 1),
            current_block_dim=(128, 1, 1),
            backend="cuda",
            compute_capability=80,
        )
        assert state is not None
        assert state.search_level == 1

    def test_search_state_attributes(self, observation_module):
        """Test SearchState has expected attributes."""
        if observation_module is None:
            pytest.skip("Observation module not available")

        SearchState = getattr(observation_module, "SearchState", None)
        if SearchState is None:
            pytest.skip("SearchState not found")

        state = SearchState(
            search_level=1, search_depth=5, num_kn_operators=3,
            num_tb_operators=2, num_tensors=4, num_valid_found=1,
            best_latency_ms=0.5, current_grid_dim=(4, 2, 1),
            current_block_dim=(128, 1, 1), backend="cuda", compute_capability=80,
        )

        assert hasattr(state, "search_level")
        assert hasattr(state, "search_depth")
        assert hasattr(state, "best_latency_ms")


class TestObservationSpace:
    """Tests for ObservationSpace class."""

    def test_observation_space_class_exists(self, observation_module):
        """Test ObservationSpace class exists."""
        if observation_module is None:
            pytest.skip("Observation module not available")

        assert hasattr(observation_module, "ObservationSpace")

    def test_observation_space_creation(self, observation_module):
        """Test ObservationSpace can be created."""
        if observation_module is None:
            pytest.skip("Observation module not available")

        ObservationSpace = getattr(observation_module, "ObservationSpace", None)
        if ObservationSpace is None:
            pytest.skip("ObservationSpace not found")

        space = ObservationSpace()
        assert space is not None


class TestObservationEncoder:
    """Tests for ObservationEncoder class."""

    def test_observation_encoder_class_exists(self, observation_module):
        """Test ObservationEncoder class exists."""
        if observation_module is None:
            pytest.skip("Observation module not available")

        assert hasattr(observation_module, "ObservationEncoder")

    def test_encoder_creation(self, observation_module):
        """Test ObservationEncoder can be created."""
        if observation_module is None:
            pytest.skip("Observation module not available")

        ObservationEncoder = getattr(observation_module, "ObservationEncoder", None)
        ObservationSpace = getattr(observation_module, "ObservationSpace", None)

        if ObservationEncoder is None or ObservationSpace is None:
            pytest.skip("Required classes not found")

        space = ObservationSpace()
        encoder = ObservationEncoder(space)
        assert encoder is not None

    def test_encode_returns_dict(self, observation_module):
        """Test encode method returns dictionary."""
        if observation_module is None:
            pytest.skip("Observation module not available")

        ObservationEncoder = getattr(observation_module, "ObservationEncoder", None)
        ObservationSpace = getattr(observation_module, "ObservationSpace", None)
        SearchState = getattr(observation_module, "SearchState", None)

        if None in (ObservationEncoder, ObservationSpace, SearchState):
            pytest.skip("Required classes not found")

        state = SearchState(
            search_level=1, search_depth=5, num_kn_operators=3,
            num_tb_operators=2, num_tensors=4, num_valid_found=1,
            best_latency_ms=0.5, current_grid_dim=(4, 2, 1),
            current_block_dim=(128, 1, 1), backend="cuda", compute_capability=80,
        )

        space = ObservationSpace()
        encoder = ObservationEncoder(space)
        obs = encoder.encode(state)

        assert isinstance(obs, dict)

    def test_encode_has_graph_embedding(self, observation_module):
        """Test encoded observation has graph_embedding."""
        if observation_module is None:
            pytest.skip("Observation module not available")

        ObservationEncoder = getattr(observation_module, "ObservationEncoder", None)
        ObservationSpace = getattr(observation_module, "ObservationSpace", None)
        SearchState = getattr(observation_module, "SearchState", None)

        if None in (ObservationEncoder, ObservationSpace, SearchState):
            pytest.skip("Required classes not found")

        state = SearchState(
            search_level=1, search_depth=5, num_kn_operators=3,
            num_tb_operators=2, num_tensors=4, num_valid_found=1,
            best_latency_ms=0.5, current_grid_dim=(4, 2, 1),
            current_block_dim=(128, 1, 1), backend="cuda", compute_capability=80,
        )

        space = ObservationSpace()
        encoder = ObservationEncoder(space)
        obs = encoder.encode(state)

        assert "graph_embedding" in obs

    def test_encode_has_action_mask(self, observation_module):
        """Test encoded observation has action_mask."""
        if observation_module is None:
            pytest.skip("Observation module not available")

        ObservationEncoder = getattr(observation_module, "ObservationEncoder", None)
        ObservationSpace = getattr(observation_module, "ObservationSpace", None)
        SearchState = getattr(observation_module, "SearchState", None)

        if None in (ObservationEncoder, ObservationSpace, SearchState):
            pytest.skip("Required classes not found")

        state = SearchState(
            search_level=1, search_depth=5, num_kn_operators=3,
            num_tb_operators=2, num_tensors=4, num_valid_found=1,
            best_latency_ms=0.5, current_grid_dim=(4, 2, 1),
            current_block_dim=(128, 1, 1), backend="cuda", compute_capability=80,
        )

        space = ObservationSpace()
        encoder = ObservationEncoder(space)
        obs = encoder.encode(state)

        assert "action_mask" in obs
