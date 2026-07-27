# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Real PyTorch execution + latency measurement for RuntimeFusion (no mock)."""

from __future__ import annotations

from serving_real_test_utils import serving, torch  # noqa: F401


def test_torch_mlp_capsule_real_forward(serving, torch):
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


def test_torch_hybrid_matches_engine_full(serving, torch):
    model = serving.TorchEngineModel(4, hidden_size=16, intermediate_size=32, seed=5)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    x = torch.randn(2, 16, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        result = hybrid.forward(x)
        ref = model.forward_engine_full(x)
    assert result.rf_layer_ids == [0, 1]
    assert torch.allclose(result.hidden, ref, rtol=1e-5, atol=1e-5)


def test_torch_sm_budget_skip_still_correct(serving, torch):
    model = serving.TorchEngineModel(2, hidden_size=8, intermediate_size=16, seed=1)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    x = torch.randn(2, 8, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        out = hybrid.forward(
            x,
            rf_meta={"sm_budget": 0, "extras": {"total_sms": 8, "reserved_aux_sms": 2}},
        )
        ref = model.forward_engine_full(x)
    assert out.rf_layer_ids == []
    assert torch.allclose(out.hidden, ref, rtol=1e-5, atol=1e-5)


def test_torch_forward_bench_runs(serving, torch):
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
