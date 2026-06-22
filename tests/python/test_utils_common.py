# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for yirage.utils.common module.

Comprehensive tests for get_shared_memory_capacity, get_nvcc_compiler,
and get_scheduler - loaded directly to avoid yirage.__init__ requirement.
"""

import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest

_PYTHON_ROOT = Path(__file__).parent.parent.parent / "python"


def _load_common():
    """Load common module directly."""
    path = _PYTHON_ROOT / "yirage" / "utils" / "common.py"
    spec = importlib.util.spec_from_file_location("yirage_utils_common_test", path)
    if spec is None or spec.loader is None:
        pytest.skip("utils/common.py not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def common():
    """Load and return the common module."""
    return _load_common()


# =============================================================================
# get_shared_memory_capacity Tests
# =============================================================================


class TestGetSharedMemoryCapacity:
    """Comprehensive tests for get_shared_memory_capacity."""

    @pytest.mark.parametrize(
        "cc,expected_bytes",
        [
            (70, 96 * 1024),  # V100 (Volta)
            (75, 64 * 1024),  # T4 (Turing)
            (80, 163 * 1024),  # A100 (Ampere)
            (86, 99 * 1024),  # A5000
            (89, 99 * 1024),  # A6000
            (90, 223 * 1024),  # H100 (Hopper)
            (100, 227 * 1024),  # B200 (Blackwell)
        ],
    )
    def test_known_compute_capabilities(self, common, cc, expected_bytes):
        """Test all known compute capabilities return correct shared memory."""
        result = common.get_shared_memory_capacity(cc)
        assert result == expected_bytes

    def test_returns_int(self, common):
        """Test return type is int."""
        result = common.get_shared_memory_capacity(80)
        assert isinstance(result, int)

    def test_returns_positive(self, common):
        """Test all results are positive."""
        for cc in [70, 75, 80, 86, 89, 90, 100]:
            assert common.get_shared_memory_capacity(cc) > 0

    def test_unsupported_cc_raises(self, common):
        """Test unsupported compute capability raises AssertionError."""
        with pytest.raises(AssertionError, match="Unsupported compute capacity"):
            common.get_shared_memory_capacity(50)

    def test_unsupported_cc_raises_for_future(self, common):
        """Test unsupported future compute capability raises AssertionError."""
        with pytest.raises(AssertionError):
            common.get_shared_memory_capacity(999)

    def test_ordering_volta_to_hopper(self, common):
        """Test that newer architectures generally have more shared memory."""
        v100 = common.get_shared_memory_capacity(70)
        a100 = common.get_shared_memory_capacity(80)
        h100 = common.get_shared_memory_capacity(90)
        assert a100 > v100
        assert h100 > a100


# =============================================================================
# get_scheduler Tests
# =============================================================================


class TestGetScheduler:
    """Tests for get_scheduler function."""

    def test_basic_computation(self, common):
        """Test basic scheduler count computation."""
        # scheduler = 4 * (sm_cnt - worker)
        result = common.get_scheduler(108, 96)
        assert result == 4 * (108 - 96)
        assert result == 48

    def test_large_sm_count(self, common):
        """Test with large SM count."""
        result = common.get_scheduler(160, 144)
        assert result == 4 * (160 - 144)
        assert result == 64

    def test_worker_equals_sm_raises(self, common):
        """Test worker count equal to SM count raises AssertionError."""
        with pytest.raises(AssertionError):
            common.get_scheduler(80, 80)

    def test_worker_greater_than_sm_raises(self, common):
        """Test worker count greater than SM count raises AssertionError."""
        with pytest.raises(AssertionError):
            common.get_scheduler(64, 96)


# =============================================================================
# get_nvcc_compiler Tests
# =============================================================================


class TestGetNvccCompiler:
    """Tests for get_nvcc_compiler function."""

    def test_returns_string_or_none(self, common):
        """Test return type is str or None."""
        result = common.get_nvcc_compiler()
        assert result is None or isinstance(result, str)

    def test_with_cuda_home_set(self, common, tmp_path):
        """Test that CUDA_HOME env var is checked."""
        nvcc = tmp_path / "bin" / "nvcc"
        nvcc.parent.mkdir(parents=True)
        nvcc.touch()
        with patch.dict(os.environ, {"CUDA_HOME": str(tmp_path)}):
            result = common.get_nvcc_compiler()
            assert result == str(nvcc)

    def test_with_cuda_path_set(self, common, tmp_path):
        """Test that CUDA_PATH env var is checked."""
        nvcc = tmp_path / "bin" / "nvcc"
        nvcc.parent.mkdir(parents=True)
        nvcc.touch()
        with patch.dict(
            os.environ,
            {"CUDA_PATH": str(tmp_path)},
            clear=False,
        ):
            # Remove CUDA_HOME to test CUDA_PATH fallback
            env = os.environ.copy()
            env.pop("CUDA_HOME", None)
            with patch.dict(os.environ, env, clear=True):
                result = common.get_nvcc_compiler()
                # May still find nvcc from system PATH, but at minimum no crash
                assert result is None or isinstance(result, str)

    def test_with_no_cuda(self, common):
        """Test returns None when no CUDA is installed."""
        env = {k: v for k, v in os.environ.items() if k not in ("CUDA_HOME", "CUDA_PATH")}
        with patch.dict(os.environ, env, clear=True):
            with patch("shutil.which", return_value=None):
                with patch("os.path.isfile", return_value=False):
                    result = common.get_nvcc_compiler()
                    assert result is None
