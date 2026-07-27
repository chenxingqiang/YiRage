# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Real PyTorch execution + latency measurement for RuntimeFusion (no mock)."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _import_serving():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if "yirage" not in sys.modules or not hasattr(sys.modules["yirage"], "__path__"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    for key in list(sys.modules):
        if key == "yirage.serving" or key.startswith("yirage.serving."):
            del sys.modules[key]
    return importlib.import_module("yirage.serving")


@pytest.fixture(scope="module")
def serving():
    return _import_serving()


def _require_torch(serving):
    serving.require_torch()
    import torch

    return torch


def test_torch_mlp_capsule_real_forward(serving):
    torch = _require_torch(serving)
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=32,
        intermediate_size=64,
        seed=2,
        backend=serving.BACKEND_TORCH,
    )
    assert cap.plan.backend == serving.BACKEND_TORCH
    x = torch.randn(4, 32, dtype=torch.float32)
    out = cap.execute({"hidden": x})["hidden"]
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_torch_hybrid_matches_engine_full(serving):
    torch = _require_torch(serving)
    model = serving.TorchEngineModel(4, hidden_size=16, intermediate_size=32, seed=5)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    x = torch.randn(2, 16, dtype=torch.float32, device=model.device)
    result = hybrid.forward(x)
    ref = model.forward_engine_full(x)
    assert result.rf_layer_ids == [0, 1]
    assert torch.allclose(result.hidden, ref, rtol=1e-5, atol=1e-5)


def test_torch_sm_budget_skip_still_correct(serving):
    torch = _require_torch(serving)
    model = serving.TorchEngineModel(2, hidden_size=8, intermediate_size=16, seed=1)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    x = torch.randn(2, 8, dtype=torch.float32, device=model.device)
    out = hybrid.forward(
        x,
        rf_meta={"sm_budget": 0, "extras": {"total_sms": 8, "reserved_aux_sms": 2}},
    )
    ref = model.forward_engine_full(x)
    assert out.rf_layer_ids == []
    assert torch.allclose(out.hidden, ref, rtol=1e-5, atol=1e-5)


def test_torch_forward_bench_runs(serving):
    torch = _require_torch(serving)
    model = serving.TorchEngineModel(2, hidden_size=64, intermediate_size=128, seed=0)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=1)
    x = torch.randn(8, 64, dtype=torch.float32, device=model.device)

    def _engine():
        model.forward_engine_full(x)

    def _hybrid():
        hybrid.forward(x)

    eng = serving.bench_forward(_engine, name="engine_full", warmup=2, iters=10)
    hyb = serving.bench_forward(_hybrid, name="hybrid_rf_k1", warmup=2, iters=10)
    assert eng.mean_ms > 0
    assert hyb.mean_ms > 0
    assert eng.device == model.device
