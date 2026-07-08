"""Contract tests for MACA attention smoke scaffold."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_attention_smoke_demo_exists():
    demo = _REPO / "demo" / "maca" / "attention_smoke.py"
    assert demo.is_file()
    text = demo.read_text(encoding="utf-8")
    assert "chameleon_maca" in text
    assert "--inspect-only" in text
    assert "attention_utils" in text


def test_attention_utils_scaffold_contract():
    utils = _load_module("attention_utils", _REPO / "demo" / "maca" / "attention_utils.py")
    report = utils.inspect_maca_attention_scaffold()
    assert report["cuda_reference"].startswith("benchmark/end-to-end/maca/chameleon_maca.py")
    assert report["maca_kernel"] == "src/kernel/maca/attention_kernel.maca"
    assert report["kernel_file_exists"] is True
    assert report["warp_size"] == 64
    assert report["search_config"] == "attention"
    assert report["head_dim"] == 128


def test_attention_kernel_maca_has_warp64():
    kernel = _REPO / "src" / "kernel" / "maca" / "attention_kernel.maca"
    text = kernel.read_text(encoding="utf-8")
    assert "WARP_SIZE = 64" in text
    assert "YIRAGE_BACKEND_MACA_ENABLED" in text
