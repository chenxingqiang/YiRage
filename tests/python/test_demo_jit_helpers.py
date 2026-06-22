# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEMO_JIT = _PROJECT_ROOT / "demo" / "demo_jit.py"


def _load_demo_jit():
    spec = importlib.util.spec_from_file_location("demo_jit", _DEMO_JIT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


demo_jit = _load_demo_jit()


def test_backend_for_device_cpu_and_mps():
    assert demo_jit._backend_for_device("cpu") == "cpu"
    assert demo_jit._backend_for_device("mps") == "mps"
    assert demo_jit._backend_for_device("cuda:0") == "cuda"


def test_resolve_device_cpu():
    assert demo_jit._resolve_device("cpu") == "cpu"


def test_main_cpu_quiet_exits_zero():
    code = demo_jit.main(["--device", "cpu", "--quiet"])
    assert code == 0
