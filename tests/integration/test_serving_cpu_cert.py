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


def test_serving_cpu_cert_gate_installs_pytest_from_pypi():
    """The unconditional cert gate must resolve pytest from PyPI.

    ``--index-url`` replaces the default index. The PyTorch CPU wheel index
    does not host pytest, so the install step exits before any cert test runs
    (Actions run 44114956773). The vLLM setup script must also use the
    interpreter's pip: ubuntu-22.04's ``/usr/bin/pip`` is 22.0.2 and rejects
    ``--break-system-packages``.
    """
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github/workflows/serving-cpu-cert-gate.yml").read_text()
    install_lines = [
        line
        for line in workflow.splitlines()
        if "pip install" in line and "pytest" in line
    ]
    assert install_lines, "cert gate workflow must install pytest"
    for line in install_lines:
        assert "--index-url" not in line, line

    script = (root / "scripts/setup_serving_vllm_cpu.sh").read_text()
    assert "python3 -m pip install --upgrade" in script
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("pip install") or stripped.startswith("pip3 install"):
            raise AssertionError(
                "setup_serving_vllm_cpu.sh must call python3 -m pip, not bare pip: "
                + stripped
            )


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
    assert "s19_contract" in names
    assert "s20_contract" in names
    assert "s21_contract" in names
    assert "segment_torch_bench" in names
    assert "vllm_mlp_e2e" in names
    assert "qwen05b_cpu_e2e" in names


def test_serving_cpu_cert_quick_passes():
    run_cert, _ = _bootstrap()
    report = run_cert(quick=True)
    assert report.bootstrap_ok is True
    assert report.serving_version == "s50"
    assert report.torch_device in {"cpu", "cuda"}
    failed = [(s.name, s.returncode, s.stderr_tail) for s in report.stages if not s.ok]
    assert report.ok is True, failed
