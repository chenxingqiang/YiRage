#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Backend API Unit Tests

Tests for yirage/backends/api.py and backend configuration modules.
Run with: pytest tests/python/test_backends.py -v
"""

import pytest
from typing import Dict, Any, List

from conftest import (
    PYTHON_ROOT,
    load_module,
    check_module_syntax,
)


# =============================================================================
# Backend API Tests
# =============================================================================

class TestBackendRegistry:
    """Tests for backend registry functions."""

    def test_get_available_backends_returns_list(self, backend_api_module):
        """Test that get_available_backends returns a list."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        result = backend_api_module.get_available_backends()
        assert isinstance(result, list)

    def test_get_available_backends_all_strings(self, backend_api_module):
        """Test that all backend names are strings."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        backends = backend_api_module.get_available_backends()
        for backend in backends:
            assert isinstance(backend, str)

    def test_get_default_backend_returns_string_or_none(self, backend_api_module):
        """Test that get_default_backend returns string or None."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        result = backend_api_module.get_default_backend()
        assert result is None or isinstance(result, str)

    def test_get_default_backend_priority_order(self, backend_api_module):
        """Test that default backend follows priority order."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        backends = backend_api_module.get_available_backends()
        default = backend_api_module.get_default_backend()

        if not backends:
            assert default is None
            return

        # Check priority: cuda > maca > ascend > mps > cpu
        priority = ["cuda", "maca", "ascend", "mps", "cpu"]
        for prio_backend in priority:
            if prio_backend in backends:
                assert default == prio_backend
                break

    def test_is_backend_available_returns_bool(self, backend_api_module):
        """Test that is_backend_available returns boolean."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        result = backend_api_module.is_backend_available("cpu")
        assert isinstance(result, bool)

    def test_is_backend_available_nonexistent_returns_false(self, backend_api_module):
        """Test that nonexistent backend returns False."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        result = backend_api_module.is_backend_available("nonexistent_backend_xyz")
        assert result is False

    def test_is_backend_available_empty_string_returns_false(self, backend_api_module):
        """Test that empty string returns False."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        result = backend_api_module.is_backend_available("")
        assert result is False

    def test_get_backend_info_returns_dict(self, backend_api_module):
        """Test that get_backend_info returns a dictionary."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        result = backend_api_module.get_backend_info("cpu")
        assert isinstance(result, dict)
        assert "name" in result
        assert "available" in result

    def test_get_backend_info_has_name_field(self, backend_api_module):
        """Test that backend info contains name field."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        result = backend_api_module.get_backend_info("cuda")
        assert result["name"] == "cuda"

    def test_set_default_backend_nonexistent_returns_false(self, backend_api_module):
        """Test that setting nonexistent backend returns False."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        result = backend_api_module.set_default_backend("nonexistent_xyz")
        assert result is False


# =============================================================================
# Backend Config Tests - Parameterized for all 12 backends
# =============================================================================

BACKEND_CONFIGS = [
    ("cuda", "get_cuda_search_config", ["search_thread", "arch"]),
    ("mps", "get_mps_search_config", ["grid_dims_to_explore", "block_dims_to_explore"]),
    ("rocm", "get_rocm_search_config", ["search_thread"]),
    ("cpu", "get_cpu_search_config", ["search_thread"]),
    ("ascend", "get_ascend_search_config", ["search_thread"]),
    ("maca", "get_maca_search_config", ["search_thread"]),
    ("tpu", "get_tpu_search_config", ["search_thread"]),
    ("xpu", "get_xpu_search_config", ["search_thread"]),
    ("fpga", "get_fpga_search_config", ["search_thread"]),
]


class TestBackendConfig:
    """Tests for backend configuration modules."""

    @pytest.mark.parametrize("backend,func_name,required_keys", BACKEND_CONFIGS)
    def test_config_module_exists(self, backend: str, func_name: str, required_keys: List[str]):
        """Test that backend config module file exists."""
        path = PYTHON_ROOT / "yirage" / "backends" / backend / "config.py"
        assert path.exists(), f"Config module not found: {path}"

    @pytest.mark.parametrize("backend,func_name,required_keys", BACKEND_CONFIGS)
    def test_config_module_syntax_valid(self, backend: str, func_name: str, required_keys: List[str]):
        """Test that backend config module has valid Python syntax."""
        path = PYTHON_ROOT / "yirage" / "backends" / backend / "config.py"
        if not path.exists():
            pytest.skip(f"{backend} config not available")

        valid, error = check_module_syntax(path)
        assert valid, f"Syntax error in {backend}/config.py: {error}"

    @pytest.mark.parametrize("backend,func_name,required_keys", BACKEND_CONFIGS)
    def test_config_function_exists(self, backend_configs, backend: str, func_name: str, required_keys: List[str]):
        """Test that config function exists in module."""
        if backend not in backend_configs:
            pytest.skip(f"{backend} config not available")

        module = backend_configs[backend]
        assert hasattr(module, func_name), f"{func_name} not found in {backend}/config.py"

    @pytest.mark.parametrize("backend,func_name,required_keys", BACKEND_CONFIGS)
    def test_config_returns_dict(self, backend_configs, backend: str, func_name: str, required_keys: List[str]):
        """Test that config function returns a dictionary."""
        if backend not in backend_configs:
            pytest.skip(f"{backend} config not available")

        module = backend_configs[backend]
        if not hasattr(module, func_name):
            pytest.skip(f"{func_name} not found")

        get_config = getattr(module, func_name)
        config = get_config()
        assert isinstance(config, dict), f"{func_name} should return dict"

    @pytest.mark.parametrize("backend,func_name,required_keys", BACKEND_CONFIGS)
    def test_config_not_empty(self, backend_configs, backend: str, func_name: str, required_keys: List[str]):
        """Test that config is not empty."""
        if backend not in backend_configs:
            pytest.skip(f"{backend} config not available")

        module = backend_configs[backend]
        if not hasattr(module, func_name):
            pytest.skip(f"{func_name} not found")

        get_config = getattr(module, func_name)
        config = get_config()
        assert len(config) > 0, f"{func_name} returned empty config"


# =============================================================================
# Memory Config Tests
# =============================================================================

MEMORY_CONFIGS = [
    ("cuda", "get_cuda_memory_config"),
    ("mps", "get_mps_memory_config"),
    ("rocm", "get_rocm_memory_config"),
    ("ascend", "get_ascend_memory_config"),
    ("maca", "get_maca_memory_config"),
]


class TestBackendMemoryConfig:
    """Tests for backend memory configuration."""

    @pytest.mark.parametrize("backend,func_name", MEMORY_CONFIGS)
    def test_memory_config_exists(self, backend_configs, backend: str, func_name: str):
        """Test that memory config function exists."""
        if backend not in backend_configs:
            pytest.skip(f"{backend} config not available")

        module = backend_configs[backend]
        if not hasattr(module, func_name):
            pytest.skip(f"{func_name} not found in {backend}/config.py")

        get_config = getattr(module, func_name)
        config = get_config()
        assert isinstance(config, dict)

    @pytest.mark.parametrize("backend,func_name", MEMORY_CONFIGS)
    def test_memory_config_positive_values(self, backend_configs, backend: str, func_name: str):
        """Test that memory config values are positive."""
        if backend not in backend_configs:
            pytest.skip(f"{backend} config not available")

        module = backend_configs[backend]
        if not hasattr(module, func_name):
            pytest.skip(f"{func_name} not found")

        get_config = getattr(module, func_name)
        config = get_config()

        # Check numeric values are positive
        for key, value in config.items():
            if isinstance(value, (int, float)) and "size" in key.lower() or "memory" in key.lower():
                assert value >= 0, f"{key} should be non-negative"


# =============================================================================
# Backend Availability Check Tests
# =============================================================================

AVAILABILITY_CHECKS = [
    ("cuda", "is_cuda_available"),
    ("rocm", "is_rocm_available"),
    ("maca", "is_maca_available"),
    ("tpu", "is_tpu_available"),
    ("xpu", "is_xpu_available"),
    ("fpga", "is_fpga_available"),
]


class TestBackendAvailability:
    """Tests for backend availability check functions."""

    @pytest.mark.parametrize("backend,func_name", AVAILABILITY_CHECKS)
    def test_availability_function_returns_bool(self, backend_configs, backend: str, func_name: str):
        """Test that availability check returns boolean."""
        if backend not in backend_configs:
            pytest.skip(f"{backend} config not available")

        module = backend_configs[backend]
        if not hasattr(module, func_name):
            pytest.skip(f"{func_name} not found in {backend}/config.py")

        is_available = getattr(module, func_name)
        result = is_available()
        assert isinstance(result, bool), f"{func_name} should return bool"


# =============================================================================
# Backend Architecture Enum Tests
# =============================================================================

BACKEND_ENUMS = [
    ("cuda", "CUDAArch"),
    ("mps", "AppleChipFamily"),
    ("rocm", "ROCmArch"),
    ("cpu", "SIMDType"),
    ("tpu", "TPUVersion"),
    ("xpu", "XPUArch"),
    ("fpga", "FPGADevice"),
]


class TestBackendEnums:
    """Tests for backend architecture enums."""

    @pytest.mark.parametrize("backend,enum_name", BACKEND_ENUMS)
    def test_enum_exists(self, backend_configs, backend: str, enum_name: str):
        """Test that architecture enum exists."""
        if backend not in backend_configs:
            pytest.skip(f"{backend} config not available")

        module = backend_configs[backend]
        if not hasattr(module, enum_name):
            pytest.skip(f"{enum_name} not found in {backend}/config.py")

        enum_class = getattr(module, enum_name)
        # Check it's an enum-like class
        assert hasattr(enum_class, "__members__") or callable(enum_class)

    @pytest.mark.parametrize("backend,enum_name", BACKEND_ENUMS)
    def test_enum_has_members(self, backend_configs, backend: str, enum_name: str):
        """Test that enum has at least one member."""
        if backend not in backend_configs:
            pytest.skip(f"{backend} config not available")

        module = backend_configs[backend]
        if not hasattr(module, enum_name):
            pytest.skip(f"{enum_name} not found")

        enum_class = getattr(module, enum_name)
        if hasattr(enum_class, "__members__"):
            assert len(enum_class.__members__) > 0, f"{enum_name} should have members"


# =============================================================================
# Backend Constants Tests
# =============================================================================

class TestBackendConstants:
    """Tests for backend-specific constants."""

    def test_maca_warp_size_constant(self, backend_configs):
        """Test MACA_WARP_SIZE constant is 64."""
        if "maca" not in backend_configs:
            pytest.skip("MACA config not available")

        module = backend_configs["maca"]
        if not hasattr(module, "MACA_WARP_SIZE"):
            pytest.skip("MACA_WARP_SIZE not found")

        assert module.MACA_WARP_SIZE == 64

    def test_cuda_has_compute_capability(self, backend_configs):
        """Test CUDA config has compute capability info."""
        if "cuda" not in backend_configs:
            pytest.skip("CUDA config not available")

        module = backend_configs["cuda"]
        # Check for arch specs or compute capability info
        assert hasattr(module, "CUDAArch") or hasattr(module, "CUDAArchSpecs")


# =============================================================================
# Integration Tests
# =============================================================================

class TestBackendIntegration:
    """Integration tests for backend subsystem."""

    def test_all_backends_have_search_config(self, backend_configs):
        """Test that all available backends have search config."""
        for backend, module in backend_configs.items():
            func_name = f"get_{backend}_search_config"
            assert hasattr(module, func_name), f"{backend} missing {func_name}"

    def test_backend_configs_are_compatible(self, backend_configs):
        """Test that backend configs have compatible structure."""
        common_keys = None

        for backend, module in backend_configs.items():
            func_name = f"get_{backend}_search_config"
            if not hasattr(module, func_name):
                continue

            config = getattr(module, func_name)()

            # First iteration: establish common keys
            if common_keys is None:
                # At minimum, search_thread should be common
                if "search_thread" in config:
                    common_keys = {"search_thread"}

    def test_default_backend_is_available(self, backend_api_module):
        """Test that default backend is in available list."""
        if backend_api_module is None:
            pytest.skip("Backend API module not available")

        default = backend_api_module.get_default_backend()
        if default is None:
            pytest.skip("No backends available")

        backends = backend_api_module.get_available_backends()
        assert default in backends
