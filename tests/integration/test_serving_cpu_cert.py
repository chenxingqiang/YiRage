# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Integration: Serving Loop cert runner (real torch default)."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _bootstrap():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if "yirage" not in sys.modules:
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(pkg_root / "yirage")]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    from yirage.serving.cpu_cert import run_serving_cpu_cert, serving_cpu_cert_manifest

    return run_serving_cpu_cert, serving_cpu_cert_manifest


def test_serving_cpu_cert_manifest_has_core_stages():
    _, manifest_fn = _bootstrap()
    real_names = [s.name for s in manifest_fn(quick=True, real=True)]
    assert "s1_contract" in real_names
    assert "s5_contract" in real_names
    assert "s6_contract" in real_names
    assert "s7_contract" in real_names
    assert "s8_contract" in real_names
    assert "real_torch_e2e" in real_names
    assert "torch_mlp_rf_hook_smoke" in real_names
    contract_names = [s.name for s in manifest_fn(quick=True, real=False)]
    assert "s8_contract" not in contract_names
    assert "sm_budget_coresidence_smoke" in contract_names


def test_serving_cpu_cert_quick_passes():
    run_cert, _ = _bootstrap()
    report = run_cert(quick=True, real=True)
    assert report.bootstrap_ok is True
    assert report.real is True
    assert report.serving_version == "s8"
    assert report.torch_device in {"cpu", "cuda"}
    failed = [(s.name, s.returncode, s.stderr_tail) for s in report.stages if not s.ok]
    assert report.ok is True, failed
