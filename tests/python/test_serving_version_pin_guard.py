#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Live RF version pin guard (cert gate repair, 2026-11).

Scans test sources for *live* RuntimeFusion version assertions -- pins that
compare the CURRENT runtime version against an ``"sXX"`` literal -- and fails
when any pin drifts from ``RuntimeFusion([]).inspect()["version"]``.

Covered live shapes:
  - ``...inspect()["version"] == "sXX"``       (any receiver, same line)
  - ``info["version"] == "sXX"``                (info = rf.inspect())
  - ``report.serving_version == "sXX"``         (integration cert report)

Archive-style assertions that validate an explicitly-passed ``version=``
parameter (``payload["version"]``, ``report.version``, ``archive.version``,
``baseline.version``) are intentionally out of scope: those pins describe the
feature round that produced the archive, not the live runtime.

Background: S22-S50 bumped the runtime version without updating older live
pins, leaving the CPU cert gate red for 28 rounds (20 stale pins found
2026-11-16 while re-verifying main). This guard makes that drift class fail
loudly at test time.
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path

import pytest


def _import_serving():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if "yirage" not in sys.modules or not hasattr(sys.modules["yirage"], "__path__"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    for key in list(sys.modules):
        if key == "yirage.serving" or key.startswith("yirage.serving."):
            del sys.modules[key]
    return importlib.import_module("yirage.serving")


@pytest.fixture(scope="module")
def serving():
    return _import_serving()


_LIVE_PATTERNS = [
    re.compile(r'inspect\(\)\["version"\]\s*==\s*"(s\d+)"'),
    re.compile(r'info\["version"\]\s*==\s*"(s\d+)"'),
    re.compile(r'serving_version\s*==\s*"(s\d+)"'),
]


def _live_pins():
    """Return [(relative_path, lineno, pinned_version)] for live pins."""
    repo_root = Path(__file__).resolve().parents[2]
    pins = []
    for sub in ("tests/python", "tests/integration"):
        for path in sorted((repo_root / sub).glob("*.py")):
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                for pattern in _LIVE_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        pins.append(
                            (str(path.relative_to(repo_root)), lineno, match.group(1))
                        )
                        break
    return pins


def test_live_version_pins_found(serving):
    """The scan itself must keep finding live pins; otherwise the guard roots."""
    pins = _live_pins()
    assert len(pins) >= 20, (
        f"expected >=20 live RF version pins across test sources, found {len(pins)}; "
        "if the live-pin assertion shape changed, update _LIVE_PATTERNS"
    )


def test_live_version_pins_match_runtime(serving):
    current = serving.RuntimeFusion([]).inspect()["version"]
    stale = [
        f"{path}:{lineno} pins {pinned} (runtime is {current})"
        for path, lineno, pinned in _live_pins()
        if pinned != current
    ]
    assert not stale, (
        "stale live RF version pins found -- bump them together with "
        "RuntimeFusion.inspect()['version'] (archive-style version= pins are "
        "exempt):\n" + "\n".join(stale)
    )
