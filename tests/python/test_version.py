# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
Tests for yirage.version module.

Tests version string format and accessibility.
"""

import importlib.util
import re
from pathlib import Path

import pytest

_PYTHON_ROOT = Path(__file__).parent.parent.parent / "python"


def _load_version():
    """Load version module directly."""
    path = _PYTHON_ROOT / "yirage" / "version.py"
    spec = importlib.util.spec_from_file_location("yirage_version_test", path)
    if spec is None or spec.loader is None:
        pytest.skip("version module not found")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def version_module():
    """Load and return the version module."""
    return _load_version()


class TestVersion:
    """Test version information."""

    def test_version_exists(self, version_module):
        """Test __version__ attribute exists."""
        assert hasattr(version_module, "__version__")

    def test_version_is_string(self, version_module):
        """Test __version__ is a string."""
        assert isinstance(version_module.__version__, str)

    def test_version_not_empty(self, version_module):
        """Test __version__ is not empty."""
        assert len(version_module.__version__) > 0

    def test_version_semver_format(self, version_module):
        """Test __version__ follows semantic versioning (MAJOR.MINOR.PATCH)."""
        pattern = r"^\d+\.\d+\.\d+([a-zA-Z0-9\.\-]*)?$"
        assert re.match(
            pattern, version_module.__version__
        ), f"Version '{version_module.__version__}' does not match semver format"

    def test_version_components(self, version_module):
        """Test version can be split into numeric components."""
        parts = version_module.__version__.split(".")
        assert len(parts) >= 3
        # First three parts should be numeric
        for part in parts[:3]:
            # Remove any pre-release suffix from the last part
            numeric = re.match(r"^(\d+)", part)
            assert numeric is not None, f"Non-numeric version part: {part}"
            assert int(numeric.group(1)) >= 0
