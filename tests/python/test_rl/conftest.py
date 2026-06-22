#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Test shims for RL modules that do not require the native YiRage core runtime.
"""

import sys
import types
from pathlib import Path

from tests.python._yirage_test_support import native_core_available, purge_yirage_modules

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_PYTHON_ROOT = _PROJECT_ROOT / "python"

_SHIM_INSTALLED_BY_US = False


def _install_namespace_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    module._is_test_shim = True  # type: ignore[attr-defined]
    sys.modules[name] = module


# Only install the shim when the native runtime is not built. When ``yirage.core``
# is available, a normal ``import yirage`` must win so integration tests are not
# polluted by this bare namespace stub.
if not native_core_available():
    _install_namespace_package("yirage", _PYTHON_ROOT / "yirage")
    _SHIM_INSTALLED_BY_US = True


def pytest_sessionfinish(session, exitstatus):
    if _SHIM_INSTALLED_BY_US:
        purge_yirage_modules()
