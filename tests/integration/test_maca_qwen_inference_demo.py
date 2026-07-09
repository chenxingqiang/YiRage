"""Subprocess smoke for MACA Qwen full-chain demo (MetaX GPU required)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_DEMO = _REPO / "demo" / "maca" / "qwen_inference_demo.py"


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
@pytest.mark.slow
def test_maca_qwen_inference_demo_quick():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")

    env = os.environ.copy()
    env.setdefault("YIRAGE_BACKEND", "maca")
    env.setdefault("PYTHONPATH", str(_REPO / "python") + os.pathsep + str(_REPO))
    env.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")

    cmd = [
        sys.executable,
        str(_DEMO),
        "--quick",
        "--decode-steps",
        "2",
        "--prefill-len",
        "4",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(_REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "PASS" in result.stdout


def test_maca_qwen_demo_script_exists():
    assert _DEMO.is_file()
    text = _DEMO.read_text(encoding="utf-8")
    assert "demo/qwen2.5/demo.py" in text
    assert 'backend="maca"' in text or "backend='maca'" in text
    assert "--model" in text
    assert "qwen_hf_utils" in text
    assert "qwen_kernel_utils" in text


def test_maca_qwen_from_pretrained_demo_exists():
    demo = _REPO / "demo" / "maca" / "qwen_from_pretrained_demo.py"
    assert demo.is_file()
    text = demo.read_text(encoding="utf-8")
    assert "modeling_qwen2_maca" in text
    assert "--max-layers" in text
