"""Subprocess smoke for MACA attention demo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "demo" / "maca" / "attention_smoke.py"


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
def test_maca_attention_smoke_inspect_only():
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
    assert payload["scaffold"]["kernel_file_exists"] is True


@pytest.mark.integration
@pytest.mark.maca
def test_maca_attention_smoke_bench_plan():
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_BACKEND", "maca")

    cmd = [sys.executable, str(_DEMO), "--bench-plan", "--json"]
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
    assert payload["status"] == "bench_plan"
    assert payload["bench_plan"]["bench_plan_ready"] is True


@pytest.mark.integration
@pytest.mark.maca
@pytest.mark.slow
def test_maca_attention_smoke_bench():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "maca")
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")

    cmd = [sys.executable, str(_DEMO), "--bench", "--quick", "--json"]
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
    assert payload["status"] == "bench"
    assert payload["bench"]["bench_ok"] is True


@pytest.mark.integration
@pytest.mark.maca
@pytest.mark.slow
def test_maca_attention_smoke_quick():
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
    assert payload["superoptimize"]["backend"] == "maca"
