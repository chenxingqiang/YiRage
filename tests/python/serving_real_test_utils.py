# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers: real torch serving tests only (no numpy stub engine).

Policy: Serving verification lives in pytest + ``real_torch_e2e`` /
``segment_torch_bench`` only — do NOT add ``demo/serving/*smoke*.py``.
See AGENTS.md § Serving 验证禁令.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


def import_serving():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "yirage" not in sys.modules or not hasattr(sys.modules.get("yirage"), "core"):
        try:
            import yirage as yr  # noqa: F401
        except ImportError:
            stub = types.ModuleType("yirage")
            stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
            sys.modules["yirage"] = stub
    for key in list(sys.modules):
        if key == "yirage.serving" or key.startswith("yirage.serving."):
            del sys.modules[key]
    mod = importlib.import_module("yirage.serving")
    mod.require_torch()
    return mod


@pytest.fixture(scope="module")
def serving():
    return import_serving()


@pytest.fixture(scope="module")
def torch(serving):
    import torch

    return torch
