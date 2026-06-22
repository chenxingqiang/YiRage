#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
RL Search Module Unit Tests

Tests for yirage/rl/search/ module including HardwareConfig and SearchSpaceConstraints.
Run with: pytest tests/python/test_rl/test_search.py -v
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
def config_space_module():
    """Load config space module."""
    return safe_import("yirage.rl.search.config_space")


@pytest.fixture(scope="module")
def graph_space_module():
    """Load graph space module."""
    return safe_import("yirage.rl.search.graph_space")


# =============================================================================
# HardwareConfig Tests
# =============================================================================

class TestHardwareConfig:
    """Tests for HardwareConfig class."""

    def test_hardware_config_class_exists(self, config_space_module):
        """Test HardwareConfig class exists."""
        if config_space_module is None:
            pytest.skip("Config space module not available")

        assert hasattr(config_space_module, "HardwareConfig")

    def test_hardware_config_creation(self, config_space_module):
        """Test HardwareConfig can be created."""
        if config_space_module is None:
            pytest.skip("Config space module not available")

        HardwareConfig = getattr(config_space_module, "HardwareConfig", None)
        if HardwareConfig is None:
            pytest.skip("HardwareConfig not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )
        assert config is not None

    def test_grid_dim_tuple(self, config_space_module):
        """Test grid_dim returns tuple."""
        if config_space_module is None:
            pytest.skip("Config space module not available")

        HardwareConfig = getattr(config_space_module, "HardwareConfig", None)
        if HardwareConfig is None:
            pytest.skip("HardwareConfig not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        assert config.grid_dim == (4, 2, 1)

    def test_block_dim_tuple(self, config_space_module):
        """Test block_dim returns tuple."""
        if config_space_module is None:
            pytest.skip("Config space module not available")

        HardwareConfig = getattr(config_space_module, "HardwareConfig", None)
        if HardwareConfig is None:
            pytest.skip("HardwareConfig not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        assert config.block_dim == (128, 1, 1)

    def test_shared_memory_positive(self, config_space_module):
        """Test shared_memory_size is positive."""
        if config_space_module is None:
            pytest.skip("Config space module not available")

        HardwareConfig = getattr(config_space_module, "HardwareConfig", None)
        if HardwareConfig is None:
            pytest.skip("HardwareConfig not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        assert config.shared_memory_size > 0


# =============================================================================
# SearchSpaceConstraints Tests
# =============================================================================

class TestSearchSpaceConstraints:
    """Tests for SearchSpaceConstraints class."""

    def test_constraints_class_exists(self, config_space_module):
        """Test SearchSpaceConstraints class exists."""
        if config_space_module is None:
            pytest.skip("Config space module not available")

        assert hasattr(config_space_module, "SearchSpaceConstraints")

    def test_constraints_creation(self, config_space_module):
        """Test SearchSpaceConstraints can be created."""
        if config_space_module is None:
            pytest.skip("Config space module not available")

        HardwareConfig = getattr(config_space_module, "HardwareConfig", None)
        SearchSpaceConstraints = getattr(config_space_module, "SearchSpaceConstraints", None)

        if HardwareConfig is None or SearchSpaceConstraints is None:
            pytest.skip("Required classes not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        constraints = SearchSpaceConstraints(config)
        assert constraints is not None

    def test_valid_imaps_generation(self, config_space_module):
        """Test valid_imaps is generated."""
        if config_space_module is None:
            pytest.skip("Config space module not available")

        HardwareConfig = getattr(config_space_module, "HardwareConfig", None)
        SearchSpaceConstraints = getattr(config_space_module, "SearchSpaceConstraints", None)

        if HardwareConfig is None or SearchSpaceConstraints is None:
            pytest.skip("Required classes not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        constraints = SearchSpaceConstraints(config)
        assert len(constraints.valid_imaps) > 0

    def test_valid_franges_generation(self, config_space_module):
        """Test valid_franges is generated."""
        if config_space_module is None:
            pytest.skip("Config space module not available")

        HardwareConfig = getattr(config_space_module, "HardwareConfig", None)
        SearchSpaceConstraints = getattr(config_space_module, "SearchSpaceConstraints", None)

        if HardwareConfig is None or SearchSpaceConstraints is None:
            pytest.skip("Required classes not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        constraints = SearchSpaceConstraints(config)
        assert len(constraints.valid_franges) > 0

    def test_max_operators_limit(self, config_space_module):
        """Test max_operators is set."""
        if config_space_module is None:
            pytest.skip("Config space module not available")

        HardwareConfig = getattr(config_space_module, "HardwareConfig", None)
        SearchSpaceConstraints = getattr(config_space_module, "SearchSpaceConstraints", None)

        if HardwareConfig is None or SearchSpaceConstraints is None:
            pytest.skip("Required classes not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        constraints = SearchSpaceConstraints(config)
        assert constraints.max_operators > 0


# =============================================================================
# ConstrainedGraphActionSpace Tests
# =============================================================================

class TestConstrainedGraphActionSpace:
    """Tests for ConstrainedGraphActionSpace class."""

    def test_action_space_class_exists(self, graph_space_module):
        """Test ConstrainedGraphActionSpace class exists."""
        if graph_space_module is None:
            pytest.skip("Graph space module not available")

        assert hasattr(graph_space_module, "ConstrainedGraphActionSpace")

    def test_action_space_creation(self, graph_space_module, config_space_module):
        """Test ConstrainedGraphActionSpace can be created."""
        if graph_space_module is None or config_space_module is None:
            pytest.skip("Required modules not available")

        HardwareConfig = getattr(config_space_module, "HardwareConfig", None)
        SearchSpaceConstraints = getattr(config_space_module, "SearchSpaceConstraints", None)
        ConstrainedGraphActionSpace = getattr(graph_space_module, "ConstrainedGraphActionSpace", None)

        if None in (HardwareConfig, SearchSpaceConstraints, ConstrainedGraphActionSpace):
            pytest.skip("Required classes not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        constraints = SearchSpaceConstraints(config)
        action_space = ConstrainedGraphActionSpace(constraints)
        assert action_space is not None

    def test_action_space_has_valid_imaps(self, graph_space_module, config_space_module):
        """Test action space has valid_imaps."""
        if graph_space_module is None or config_space_module is None:
            pytest.skip("Required modules not available")

        HardwareConfig = getattr(config_space_module, "HardwareConfig", None)
        SearchSpaceConstraints = getattr(config_space_module, "SearchSpaceConstraints", None)
        ConstrainedGraphActionSpace = getattr(graph_space_module, "ConstrainedGraphActionSpace", None)

        if None in (HardwareConfig, SearchSpaceConstraints, ConstrainedGraphActionSpace):
            pytest.skip("Required classes not found")

        config = HardwareConfig(
            grid_dim_x=4, grid_dim_y=2, grid_dim_z=1,
            block_dim_x=128, block_dim_y=1, block_dim_z=1,
            forloop_range=16, reduction_dimx=16,
            shared_memory_size=49152, num_registers=64,
        )

        constraints = SearchSpaceConstraints(config)
        action_space = ConstrainedGraphActionSpace(constraints)

        assert len(action_space.valid_imaps) > 0
