"""Contract tests for scripts/bench_ray_search.py MACA backend support."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "bench_ray_search.py"
_PKG = _REPO / "python"


def _load_bench_ray_search():
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    spec = importlib.util.spec_from_file_location("bench_ray_search", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_bench_ray_search_script_has_backend_arg():
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "--backend" in text
    assert '"maca"' in text or "'maca'" in text


def test_resolve_bench_backend_accepts_maca():
    mod = _load_bench_ray_search()
    assert mod.resolve_bench_backend("maca") == "maca"
    assert mod.resolve_bench_backend("cpu") == "cpu"


def test_maca_griddim_sets_quick_are_tractable():
    mod = _load_bench_ray_search()
    suites = mod.griddim_sets("maca", quick=True)
    assert "maca_quick_1" in suites
    assert all(len(g) == 3 for suite in suites.values() for g in suite)


def test_maca_bench_search_kwargs_blockdim_warp_multiple():
    mod = _load_bench_ray_search()
    kwargs = mod.bench_search_kwargs("maca", quick=True)
    blockdims = kwargs.get("blockdims") or []
    assert blockdims, "MACA bench must set blockdims"
    for block in blockdims:
        threads = block[0] * block[1] * block[2]
        assert threads % 64 == 0, f"block {block} not multiple of warp 64"
