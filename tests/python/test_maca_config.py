"""Tests for MACA config (loads config module without yirage.core)."""

import importlib.util
import sys
from pathlib import Path


def _load_maca_backend_config():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    path = pkg_root / "yirage" / "backends" / "maca" / "config.py"
    spec = importlib.util.spec_from_file_location("maca_backend_config", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_maca_config_shim_file_reexports_backend():
    shim = Path(__file__).resolve().parents[2] / "python" / "yirage" / "maca_config.py"
    text = shim.read_text(encoding="utf-8")
    assert "from yirage.backends.maca.config import" in text


def test_maca_search_config_quick_smaller_than_full():
    cfg = _load_maca_backend_config()
    full = cfg.get_maca_search_config()
    quick = cfg.get_maca_search_config_quick()
    assert cfg.MACA_WARP_SIZE == 64
    assert len(full["grid_dims_to_explore"]) > len(quick["grid_dims_to_explore"])
    assert len(full["block_dims_to_explore"]) > len(quick["block_dims_to_explore"])
    assert cfg.resolve_maca_search_config(quick=True) == quick
    assert cfg.resolve_maca_search_config(quick=False) == full


def test_resolve_maca_search_config_env(monkeypatch):
    cfg = _load_maca_backend_config()
    monkeypatch.setenv("YIRAGE_MACA_SEARCH_QUICK", "0")
    assert cfg.resolve_maca_search_config() == cfg.get_maca_search_config()
    monkeypatch.setenv("YIRAGE_MACA_SEARCH_QUICK", "1")
    assert cfg.resolve_maca_search_config() == cfg.get_maca_search_config_quick()
