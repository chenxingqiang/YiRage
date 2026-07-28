# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Integration: Serving Loop cert runner (torch only)."""

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
    names = [s.name for s in manifest_fn(quick=True)]
    assert "s1_contract" in names
    assert "s5_contract" in names
    assert "s6_contract" in names
    assert "s7_contract" in names
    assert "s8_contract" in names
    assert "s13_contract" in names
    assert "s15_contract" in names
    assert "s16_contract" in names
    assert "s17_contract" in names
    assert "s18_contract" in names
    assert "torch_e2e" in names
    assert "qwen05b_contract" in names
    assert "segment_torch_bench" in names
    assert "vllm_mlp_e2e" in names
    assert "qwen05b_cpu_e2e" in names


def test_serving_cpu_cert_quick_passes():
    run_cert, _ = _bootstrap()
    report = run_cert(quick=True)
    assert report.bootstrap_ok is True
    assert report.serving_version == "s18"
    assert report.torch_device in {"cpu", "cuda"}
    failed = [(s.name, s.returncode, s.stderr_tail) for s in report.stages if not s.ok]
    assert report.ok is True, failed
