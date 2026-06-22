# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for YiRage pytest isolation (RL shim vs native runtime)."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def is_yirage_test_shim(module: Optional[Any]) -> bool:
    return getattr(module, "_is_test_shim", False) is True


def native_core_available() -> bool:
    try:
        spec = importlib.util.find_spec("yirage.core")
        if spec is None:
            return False
        import yirage.core  # noqa: F401

        return True
    except (ImportError, OSError):
        return False


def purge_yirage_modules() -> None:
    for name in list(sys.modules):
        if name == "yirage" or name.startswith("yirage."):
            del sys.modules[name]


def restore_real_yirage_if_shimmed() -> bool:
    """Replace RL test shim with the real package when the native core is built."""
    mod = sys.modules.get("yirage")
    if mod is None or not is_yirage_test_shim(mod):
        return False
    if not native_core_available():
        return False
    purge_yirage_modules()
    import yirage  # noqa: F401

    return True


def ensure_native_library_path() -> None:
    """Prepend Rust helper libs to LD_LIBRARY_PATH when present under build/."""
    build = PROJECT_ROOT / "build"
    parts = [
        build / "abstract_subexpr" / "release",
        build / "formal_verifier" / "release",
    ]
    extra = os.pathsep.join(str(p) for p in parts if p.exists())
    if not extra:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if extra not in current.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = f"{extra}{os.pathsep}{current}" if current else extra
