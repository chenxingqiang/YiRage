# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU demo manifest contract for the infinite optimization loop (Loop R60)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.cpu_cert_utils import cpu_demo_loop_manifest  # noqa: E402

# Map manifest pytest names to functions in test_cpu_demos.py
_DEMO_TEST_MODULE = "tests.integration.test_cpu_demos"


def test_cpu_demo_loop_manifest_scripts_exist():
    for entry in cpu_demo_loop_manifest():
        script = _REPO / entry["script"]
        assert script.is_file(), f"missing demo script: {entry['script']}"


def test_cpu_demo_loop_manifest_covers_test_cpu_demos():
    import importlib

    demo_tests = importlib.import_module(_DEMO_TEST_MODULE)
    manifest_tests = {e["pytest"] for e in cpu_demo_loop_manifest()}
    module_tests = {
        name
        for name in dir(demo_tests)
        if name.startswith("test_") and callable(getattr(demo_tests, name))
    }
    assert manifest_tests <= module_tests, (
        f"manifest tests not in {_DEMO_TEST_MODULE}: {manifest_tests - module_tests}"
    )


def test_cpu_demo_loop_manifest_layers_are_known():
    allowed = {"perceive", "verify", "evolve"}
    for entry in cpu_demo_loop_manifest():
        assert entry["layer"] in allowed
