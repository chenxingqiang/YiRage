# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for Serving Loop pytest (torch execution; no numpy stub engine).

Policy: Serving verification lives in pytest + ``torch_e2e`` /
``segment_torch_bench`` only — do NOT add ``demo/serving/*smoke*.py``.
See AGENTS.md § Serving 验证禁令.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest


def maca_integration_enabled() -> bool:
    """MetaX / MACA tier tests opt-in only (not CPU Serving cert)."""
    return os.environ.get("YIRAGE_MACA_INTEGRATION") == "1"


def vllm_available() -> bool:
    """True when ``vllm`` is importable (does not load ``yirage.core``)."""
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def sglang_available() -> bool:
    """True when ``sglang`` is importable (optional tier; does not load ``yirage.core``)."""
    try:
        import sglang  # noqa: F401

        return True
    except ImportError:
        return False


def require_vllm_installed() -> None:
    """Fail pytest when vLLM is missing — CPU Serving cert gate."""
    if not vllm_available():
        pytest.fail(
            "CPU Serving verification requires `pip install vllm transformers`. "
            "vLLM Qwen2 fork e2e must not be skipped on CPU."
        )


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
