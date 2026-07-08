"""Subprocess smoke for MACA Qwen3 PersistentKernel scaffold demo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "demo" / "maca" / "qwen3_persistent_kernel_demo.py"


def _maca_vm_available() -> bool:
    if os.environ.get("YIRAGE_MACA_INTEGRATION", "") == "1":
        return True
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return "MetaX" in torch.cuda.get_device_name(0)
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_compile_plan():
    """Cloud-safe contract: compile-plan validates minimal embed PK prerequisites."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--compile-plan", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_plan"
    assert payload["compile_plan"]["variant"] == "embed_only"
    assert payload["compile_plan"]["minimal_task_graph"] == "embed_layer"


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_compile_plan_one_layer():
    """Cloud-safe contract: one-layer PK compile plan prerequisites."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [
        sys.executable,
        str(_DEMO),
        "--compile-plan",
        "--compile-plan-variant",
        "one_layer",
        "--json",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_plan"
    assert payload["compile_plan"]["variant"] == "one_layer"
    assert "layer[0]" in payload["compile_plan"]["minimal_task_graph"]


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_compile_inspect():
    """Cloud-safe contract: compile-inspect validates mxcc PK flags without GPU."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--compile-inspect", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_inspect"
    assert payload["compile_contract"]["compile_ready"] is True


@pytest.mark.integration
@pytest.mark.maca
def test_maca_qwen3_persistent_kernel_demo_inspect_only():
    """Cloud-safe contract: inspect-only exits 0 without MetaX GPU."""
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--inspect-only", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "inspect_only"
    assert payload["scaffold"]["cuda_reference"] == "demo/qwen3/demo.py --use-yirage"


@pytest.mark.integration
@pytest.mark.maca
@pytest.mark.slow
def test_maca_qwen3_persistent_kernel_demo_compile_only():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "maca")
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    env.setdefault("YIRAGE_HOME", str(_REPO))

    cmd = [sys.executable, str(_DEMO), "--compile-only", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_only"
    assert payload["compile"]["compiled"] is True
    assert payload["compile"]["compiler"] == "mxcc"


@pytest.mark.integration
@pytest.mark.maca
@pytest.mark.slow
def test_maca_qwen3_persistent_kernel_demo_compile_one_layer():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "maca")
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")
    env.setdefault("YIRAGE_HOME", str(_REPO))

    cmd = [sys.executable, str(_DEMO), "--compile-one-layer", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "compile_one_layer"
    assert payload["compile"]["compiled"] is True
    assert payload["compile"]["compiler"] == "mxcc"
    assert "paged_attention" in payload["compile"]["tasks"]


@pytest.mark.integration
@pytest.mark.maca
@pytest.mark.slow
def test_maca_qwen3_persistent_kernel_demo_quick():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "maca")
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")

    cmd = [sys.executable, str(_DEMO), "--quick", "--json"]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "pass"
    assert payload["pk_runtime"]["backend"] == "maca"


def test_maca_qwen3_persistent_kernel_demo_script_exists():
    assert _DEMO.is_file()
    text = _DEMO.read_text(encoding="utf-8")
    assert "qwen3_pk_utils" in text
    assert "--inspect-only" in text
    assert "--compile-one-layer" in text
