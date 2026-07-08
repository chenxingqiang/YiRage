"""VerifierPool backend=maca contract (local path, no MetaX GPU required)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_PKG = _REPO / "python"
_RL = _PKG / "yirage" / "rl"
_VERIFIER = _RL / "verifier"


def _load_verifier_pool():
    """Load VerifierPool without importing yirage.core or yirage.rl.env."""
    if str(_PKG) not in sys.path:
        sys.path.insert(0, str(_PKG))

    for name, path in (
        ("yirage", _PKG / "yirage"),
        ("yirage.rl", _RL),
        ("yirage.rl.verifier", _VERIFIER),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            mod.__package__ = name
            sys.modules[name] = mod

    gpu_spec = importlib.util.spec_from_file_location(
        "yirage.rl.verifier.gpu_verifier", _VERIFIER / "gpu_verifier.py"
    )
    gpu_mod = importlib.util.module_from_spec(gpu_spec)
    assert gpu_spec.loader is not None
    sys.modules["yirage.rl.verifier.gpu_verifier"] = gpu_mod
    gpu_spec.loader.exec_module(gpu_mod)

    pool_spec = importlib.util.spec_from_file_location(
        "yirage.rl.verifier.verifier_pool", _VERIFIER / "verifier_pool.py"
    )
    pool_mod = importlib.util.module_from_spec(pool_spec)
    assert pool_spec.loader is not None
    sys.modules["yirage.rl.verifier.verifier_pool"] = pool_mod
    pool_spec.loader.exec_module(pool_mod)
    return pool_mod.VerifierPool


def test_verifier_pool_constructor_accepts_maca_backend():
    VerifierPool = _load_verifier_pool()
    pool = VerifierPool(num_gpus=1, verifiers_per_gpu=1, use_ray=False, backend="maca")
    try:
        pool._ensure_initialized()
        assert pool.backend == "maca"
        assert len(pool.verifiers) == 1
        assert pool.verifiers[0].backend == "maca"
    finally:
        pool.shutdown()


def test_verifier_pool_defaults_to_yirage_backend_env(monkeypatch):
    VerifierPool = _load_verifier_pool()
    monkeypatch.setenv("YIRAGE_BACKEND", "maca")
    pool = VerifierPool(num_gpus=1, verifiers_per_gpu=1, use_ray=False)
    try:
        assert pool.backend == "maca"
    finally:
        pool.shutdown()


def test_verifier_pool_local_maca_verify_returns_result():
    VerifierPool = _load_verifier_pool()
    pool = VerifierPool(num_gpus=1, verifiers_per_gpu=1, use_ray=False, backend="maca")
    try:
        graph = '{"operators": []}'
        result = pool.verify(graph, graph)
        assert result.verified is True
        assert result.fingerprint_time_ms >= 0
    finally:
        pool.shutdown()
