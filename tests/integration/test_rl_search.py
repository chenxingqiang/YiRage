#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
RL-Search Integration Tests

Tests for RL environment integration with C++ search engine.
Run with: pytest tests/integration/test_rl_search.py -v
"""

import pytest
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from conftest import load_module


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def env_module():
    """Load RL environment module."""
    return load_module(
        "search_env",
        PYTHON_ROOT / "yirage" / "rl" / "env" / "search_env.py"
    )


@pytest.fixture(scope="module")
def features_module():
    """Load features module."""
    return load_module(
        "mugraph_features",
        PYTHON_ROOT / "yirage" / "rl" / "features" / "mugraph_features.py"
    )


@pytest.fixture
def sample_graph_json():
    """Sample graph JSON for testing."""
    return {
        "operators": [
            {"op_id": 0, "op_type": "matmul", "input_tensor_ids": [0, 1], "output_tensor_ids": [2]},
        ],
        "inputs": [
            {"tensor_id": 0, "dims": [32, 64]},
            {"tensor_id": 1, "dims": [64, 128]},
        ],
        "outputs": [
            {"tensor_id": 2, "dims": [32, 128]},
        ],
    }


# =============================================================================
# RL-Search Integration Tests
# =============================================================================

class TestRLSearchIntegration:
    """Tests for RL-Search integration."""

    def test_env_config_accepts_graph_json(self, env_module, sample_graph_json):
        """Test environment config accepts graph JSON."""
        if env_module is None:
            pytest.skip("Environment module not available")
        
        EnvConfig = getattr(env_module, "EnvConfig", None)
        if EnvConfig is None:
            pytest.skip("EnvConfig not found")
        
        config = EnvConfig(
            target_graph_json=sample_graph_json,
            backend="cpu",
        )
        assert config is not None

    def test_observation_from_mugraph(self, features_module):
        """Test observation can be generated from µGraph features."""
        if features_module is None:
            pytest.skip("Features module not available")
        
        MuGraphFeature = getattr(features_module, "MuGraphFeature", None)
        if MuGraphFeature is None:
            pytest.skip("MuGraphFeature not found")
        
        sample_json = json.dumps({
            "operators": [{"op_id": 0, "op_type": "matmul"}],
            "tensors": [{"tensor_id": 0, "dims": [32, 64]}],
            "num_operators": 1,
            "num_tensors": 1,
            "graph_depth": 1,
            "graph_width": 1,
            "grid_dim": {"x": 1, "y": 1, "z": 1},
            "block_dim": {"x": 128, "y": 1, "z": 1},
        })
        
        features = MuGraphFeature.from_json(sample_json)
        assert features is not None
        assert len(features.operators) == 1

    def test_action_space_matches_search(self):
        """Test action space matches search engine capabilities."""
        config_module = load_module(
            "config_space",
            PYTHON_ROOT / "yirage" / "rl" / "search" / "config_space.py"
        )
        
        if config_module is None:
            pytest.skip("Config space module not available")
        
        HardwareConfig = getattr(config_module, "HardwareConfig", None)
        SearchSpaceConstraints = getattr(config_module, "SearchSpaceConstraints", None)
        
        if HardwareConfig is None or SearchSpaceConstraints is None:
            pytest.skip("Required classes not found")
        
        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )
        
        constraints = SearchSpaceConstraints(config)
        
        # Action space should be derived from constraints
        assert len(constraints.valid_imaps) > 0
        assert constraints.max_operators > 0

    def test_reward_from_verification(self):
        """Test reward computation from verification results."""
        reward_module = load_module(
            "reward",
            PYTHON_ROOT / "yirage" / "rl" / "env" / "reward.py"
        )
        
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
        
        # Simulate verification result
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


# =============================================================================
# Search Flow Tests
# =============================================================================

class TestSearchFlow:
    """Tests for complete search flow."""

    def test_hierarchical_search_flow(self):
        """Test hierarchical search flow (config -> graph)."""
        config_module = load_module(
            "config_space",
            PYTHON_ROOT / "yirage" / "rl" / "search" / "config_space.py"
        )
        graph_module = load_module(
            "graph_space",
            PYTHON_ROOT / "yirage" / "rl" / "search" / "graph_space.py"
        )
        
        if config_module is None or graph_module is None:
            pytest.skip("Required modules not available")
        
        HardwareConfig = getattr(config_module, "HardwareConfig", None)
        SearchSpaceConstraints = getattr(config_module, "SearchSpaceConstraints", None)
        ConstrainedGraphActionSpace = getattr(graph_module, "ConstrainedGraphActionSpace", None)
        
        if None in (HardwareConfig, SearchSpaceConstraints, ConstrainedGraphActionSpace):
            pytest.skip("Required classes not found")
        
        # Level 1: Hardware config
        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )
        
        # Level 2: Constraints from config
        constraints = SearchSpaceConstraints(config)
        
        # Level 3: Constrained graph action space
        action_space = ConstrainedGraphActionSpace(constraints)
        
        # All should be valid
        assert config is not None
        assert constraints is not None
        assert action_space is not None
        assert len(action_space.valid_imaps) > 0
