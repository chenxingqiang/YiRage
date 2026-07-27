#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CLI entry for Serving Loop CPU cert (avoids importing yirage/__init__.py).

Usage::

    PYTHONPATH=python python3 scripts/serving_cpu_cert.py --quick
    make test-serving-cpu-cert
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _install_yirage_stub() -> None:
    root = Path(__file__).resolve().parents[1]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    stub = types.ModuleType("yirage")
    stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
    sys.modules["yirage"] = stub
    for key in list(sys.modules):
        if key == "yirage.serving" or key.startswith("yirage.serving."):
            del sys.modules[key]


def _bootstrap_and_run(argv: list[str]) -> int:
    _install_yirage_stub()
    cpu_cert = importlib.import_module("yirage.serving.cpu_cert")
    return int(cpu_cert.main(argv))


if __name__ == "__main__":
    raise SystemExit(_bootstrap_and_run(sys.argv[1:]))
