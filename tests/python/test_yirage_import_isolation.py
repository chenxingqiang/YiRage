# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Tests for yirage import isolation between RL shim and native runtime."""

import sys
import types

import pytest

from tests.python._yirage_test_support import (
    native_core_available,
    purge_yirage_modules,
    restore_real_yirage_if_shimmed,
)


def _install_yirage_shim() -> None:
    from pathlib import Path

    python_root = Path(__file__).resolve().parents[2] / "python"
    module = types.ModuleType("yirage")
    module.__path__ = [str(python_root / "yirage")]
    module.__package__ = "yirage"
    module._is_test_shim = True  # type: ignore[attr-defined]
    sys.modules["yirage"] = module


@pytest.mark.skipif(not native_core_available(), reason="Native yirage.core not built")
def test_restore_real_yirage_after_rl_shim():
    _install_yirage_shim()
    assert restore_real_yirage_if_shimmed() is True
    import yirage

    assert hasattr(yirage, "__version__")
    assert not getattr(yirage, "_is_test_shim", False)


def test_restore_noop_without_shim():
  purge_yirage_modules()
  try:
    import yirage  # noqa: F401
  except ImportError:
    pytest.skip("yirage not installed")
  assert restore_real_yirage_if_shimmed() is False
