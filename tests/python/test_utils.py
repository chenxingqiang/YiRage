# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Utilities Module Tests

Tests for utility functions.
"""

import pytest

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def utils_module():
    """Import the utils module."""
    try:
        from yirage import utils
        return utils
    except ImportError:
        pytest.skip("yirage.utils module not available")


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Tests for module imports."""

    def test_get_shared_memory_capacity_import(self, utils_module):
        """Test get_shared_memory_capacity can be imported."""
        assert hasattr(utils_module, "get_shared_memory_capacity")

    def test_get_nvcc_compiler_import(self, utils_module):
        """Test get_nvcc_compiler can be imported."""
        assert hasattr(utils_module, "get_nvcc_compiler")

    def test_visualizer_available_flag(self, utils_module):
        """Test VISUALIZER_AVAILABLE flag exists."""
        assert hasattr(utils_module, "VISUALIZER_AVAILABLE")
        assert isinstance(utils_module.VISUALIZER_AVAILABLE, bool)


# =============================================================================
# get_shared_memory_capacity Tests
# =============================================================================


class TestGetSharedMemoryCapacity:
    """Tests for get_shared_memory_capacity function."""

    def test_function_exists(self, utils_module):
        """Test function exists."""
        assert hasattr(utils_module, "get_shared_memory_capacity")
        assert callable(utils_module.get_shared_memory_capacity)

    def test_returns_int(self, utils_module):
        """Table lookup by compute capability (no live GPU required)."""
        result = utils_module.get_shared_memory_capacity(80)
        assert isinstance(result, int)

    def test_returns_positive_or_zero(self, utils_module):
        """Known CCs return positive shared memory limit (bytes)."""
        result = utils_module.get_shared_memory_capacity(90)
        assert result > 0


# =============================================================================
# get_nvcc_compiler Tests
# =============================================================================


class TestGetNvccCompiler:
    """Tests for get_nvcc_compiler function."""

    def test_function_exists(self, utils_module):
        """Test function exists."""
        assert hasattr(utils_module, "get_nvcc_compiler")
        assert callable(utils_module.get_nvcc_compiler)

    def test_returns_string_or_none(self, utils_module):
        """Test function returns string or None."""
        result = utils_module.get_nvcc_compiler()
        assert result is None or isinstance(result, str)


# =============================================================================
# Visualizer Tests (conditional)
# =============================================================================


class TestVisualizer:
    """Tests for visualizer functions (if available)."""

    def test_visualizer_flag_exists(self, utils_module):
        """Test VISUALIZER_AVAILABLE flag exists."""
        assert hasattr(utils_module, "VISUALIZER_AVAILABLE")

    def test_visualizer_imports_when_available(self, utils_module):
        """Test visualizer functions imported when available."""
        if utils_module.VISUALIZER_AVAILABLE:
            assert hasattr(utils_module, "visualizer")
            assert hasattr(utils_module, "handle_graph_data")
            assert hasattr(utils_module, "kernel_graph")
            assert hasattr(utils_module, "block_graph")


# =============================================================================
# All Exports Test
# =============================================================================


class TestAllExports:
    """Tests for __all__ exports."""

    def test_all_exports_defined(self, utils_module):
        """Test __all__ is defined."""
        assert hasattr(utils_module, "__all__")

    def test_all_exports_accessible(self, utils_module):
        """Test all items in __all__ are accessible."""
        for name in utils_module.__all__:
            assert hasattr(utils_module, name), f"Export '{name}' not accessible"

    def test_core_exports_present(self, utils_module):
        """Test core exports are present."""
        core_exports = [
            "get_shared_memory_capacity",
            "get_nvcc_compiler",
            "VISUALIZER_AVAILABLE",
        ]
        for name in core_exports:
            assert name in utils_module.__all__, f"'{name}' not in __all__"
