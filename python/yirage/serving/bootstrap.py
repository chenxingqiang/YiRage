# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Import ``yirage.serving`` on CPU without ``yirage.core`` / GPU build.

Cloud Agent and CI use this bootstrap so RuntimeFusion contracts run reliably
with only NumPy installed.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any, Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def serving_pkg_dir() -> Path:
    return repo_root() / "python" / "yirage"


def bootstrap_yirage_stub(*, force_reload: bool = False) -> None:
    """Register ``yirage`` for serving imports; prefer a real built package when present."""
    root = repo_root()
    pkg_root = str(root / "python")
    yirage_dir = str(serving_pkg_dir())
    if pkg_root not in sys.path:
        sys.path.insert(0, pkg_root)
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)

    existing = sys.modules.get("yirage")
    if not force_reload and existing is not None and hasattr(existing, "float32"):
        return

    if (
        not force_reload
        and existing is not None
        and hasattr(existing, "__path__")
        and yirage_dir in list(existing.__path__)  # type: ignore[attr-defined]
    ):
        return

    try:
        import yirage as yr  # noqa: F401

        if hasattr(yr, "float32"):
            return
    except ImportError:
        pass

    stub = types.ModuleType("yirage")
    stub.__path__ = [yirage_dir]  # type: ignore[attr-defined]
    sys.modules["yirage"] = stub

    if force_reload:
        for key in list(sys.modules):
            if key == "yirage.serving" or key.startswith("yirage.serving."):
                del sys.modules[key]


def import_serving(*, force_reload: bool = False) -> Any:
    """Return the ``yirage.serving`` module (CPU-safe)."""
    bootstrap_yirage_stub(force_reload=force_reload)
    return importlib.import_module("yirage.serving")


def require_numpy():
    try:
        import numpy as np  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "yirage.serving CPU contracts require NumPy; install with: pip install numpy"
        ) from e
