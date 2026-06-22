# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for yirage.global_config module.

Tests the GlobalConfig class and the module-level singleton.
"""

import importlib.util
from pathlib import Path

import pytest

# Load the module directly to bypass yirage.__init__ (which requires native core)
_PYTHON_ROOT = Path(__file__).parent.parent.parent / "python"


def _load_global_config():
    """Load global_config module directly."""
    path = _PYTHON_ROOT / "yirage" / "global_config.py"
    spec = importlib.util.spec_from_file_location("yirage.global_config", path)
    if spec is None or spec.loader is None:
        pytest.skip("global_config module not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def global_config_module():
    """Load and return the global_config module."""
    return _load_global_config()


@pytest.fixture
def fresh_config(global_config_module):
    """Return a fresh GlobalConfig instance for each test."""
    return global_config_module.GlobalConfig()


# =============================================================================
# GlobalConfig Class Tests
# =============================================================================


class TestGlobalConfigDefaults:
    """Test default values of GlobalConfig."""

    def test_verbose_default(self, fresh_config):
        """Test verbose defaults to False."""
        assert fresh_config.verbose is False

    def test_gpu_device_id_default(self, fresh_config):
        """Test gpu_device_id defaults to 0."""
        assert fresh_config.gpu_device_id == 0

    def test_bypass_compile_errors_default(self, fresh_config):
        """Test bypass_compile_errors defaults to False."""
        assert fresh_config.bypass_compile_errors is False


class TestGlobalConfigMutation:
    """Test that GlobalConfig properties can be set."""

    def test_set_verbose(self, fresh_config):
        """Test setting verbose flag."""
        fresh_config.verbose = True
        assert fresh_config.verbose is True

    def test_set_gpu_device_id(self, fresh_config):
        """Test setting gpu_device_id."""
        fresh_config.gpu_device_id = 3
        assert fresh_config.gpu_device_id == 3

    def test_set_bypass_compile_errors(self, fresh_config):
        """Test setting bypass_compile_errors."""
        fresh_config.bypass_compile_errors = True
        assert fresh_config.bypass_compile_errors is True


class TestGlobalConfigSingleton:
    """Test module-level singleton behaviour."""

    def test_singleton_exists(self, global_config_module):
        """Test that a module-level global_config instance exists."""
        assert hasattr(global_config_module, "global_config")
        assert isinstance(
            global_config_module.global_config,
            global_config_module.GlobalConfig,
        )

    def test_singleton_is_instance(self, global_config_module):
        """Test that global_config is a GlobalConfig instance."""
        gc = global_config_module.global_config
        assert gc.verbose is False or gc.verbose is True  # bool
        assert isinstance(gc.gpu_device_id, int)
        assert gc.bypass_compile_errors is False or gc.bypass_compile_errors is True


class TestGlobalConfigIsolation:
    """Test that different instances are independent."""

    def test_instances_are_independent(self, global_config_module):
        """Test two separate instances do not share state."""
        a = global_config_module.GlobalConfig()
        b = global_config_module.GlobalConfig()
        a.verbose = True
        assert b.verbose is False

    def test_mutation_does_not_affect_other(self, global_config_module):
        """Test modifying one instance doesn't affect another."""
        a = global_config_module.GlobalConfig()
        b = global_config_module.GlobalConfig()
        a.gpu_device_id = 7
        assert b.gpu_device_id == 0
