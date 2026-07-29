# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S16: MetaX tiers — yirage_maca capsule + SGLang-metax e2e (torch)."""

from __future__ import annotations

import os

import pytest

from serving_test_utils import maca_integration_enabled, serving, torch  # noqa: F401


def test_rf_step_meta_for_sglang_metax_merges_maca(serving):
    spec = serving.MacaServingRfSpec()
    fb = serving.SglangForwardBatchSpec(
        extend_seq_lens=[0, 2, 0],
        seq_lens=[16, 18, 20],
    )
    meta = serving.rf_step_meta_for_sglang_metax(fb, spec=spec, sm_budget=96)
    assert meta["sm_budget"] == 96
    assert serving.maca_serving_present(meta)
    payload = meta["extras"]["maca_serving"]
    assert payload["warp_size"] == serving.MACA_SERVING_WARP_SIZE


def test_torch_sglang_metax_mlp_rf_e2e(serving):
    report = serving.run_torch_sglang_metax_mlp_rf_e2e(
        hidden_size=16,
        intermediate_size=32,
        batch=4,
        bench=False,
    )
    assert report.parity_ok
    assert report.maca_meta_bridged
    assert report.plugin == "SglangMetaxBatchTorchMlpRfHook"
    assert report.warp_size == 64


def test_torch_sglang_metax_hybrid_full_e2e(serving):
    report = serving.run_torch_sglang_metax_hybrid_full_e2e(
        num_layers=3,
        max_rf_mlp_layers=2,
        hidden_size=16,
        intermediate_size=32,
        batch=4,
        bench=False,
    )
    assert report.parity_ok
    assert report.maca_meta_bridged
    assert report.rf_layer_ids == [0, 1]
    assert report.plugin == "HybridModelOverride+SglangMetaxForwardBatch+MacaServingMeta"


def test_sglang_metax_auto_entry(serving):
    report = serving.run_sglang_metax_mlp_rf_e2e_auto(
        hidden_size=8,
        intermediate_size=16,
        batch=2,
        bench=False,
    )
    assert report.parity_ok
    assert report.maca_meta_bridged


def test_is_sglang_metax_available_bool(serving):
    assert isinstance(serving.is_sglang_metax_available(), bool)


def test_sglang_metax_hook_raises_without_tier(serving):
    if serving.is_sglang_metax_available():
        pytest.skip("sglang-metax tier available; skip negative test")
    with pytest.raises(RuntimeError, match="SGLang-metax"):
        serving.require_sglang_metax()


@pytest.mark.skipif(
    not maca_integration_enabled(),
    reason="requires sglang on MetaX host or YIRAGE_MACA_INTEGRATION=1",
)
def test_sglang_metax_plugin_requires_tier(serving):
    serving.require_sglang_metax()
    assert serving.is_sglang_metax_available()


def test_yirage_maca_tier_inspect(serving):
    info = serving.inspect_maca_serving_yirage_tier()
    assert "yirage_maca_available" in info
    assert isinstance(info["yirage_maca_available"], bool)


@pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)
def test_hybrid_mlp_backend_yirage_maca(serving, torch):
    if not serving.is_yirage_maca_available():
        pytest.skip("yirage_maca not available (requires YIRAGE_BACKEND=maca)")
    model = serving.TorchEngineModel(1, hidden_size=16, intermediate_size=32, seed=1)
    hybrid = serving.HybridModelOverride(
        model,
        max_rf_mlp_layers=1,
        mlp_backend=serving.BACKEND_YIRAGE_MACA,
    )
    assert hybrid.mlp_backend == serving.BACKEND_YIRAGE_MACA
    cap = hybrid.rf.capsules[0]
    assert cap.plan.backend == serving.BACKEND_YIRAGE_MACA
    x = torch.randn(1, 16, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        ref = model.forward_engine_full(x)
        out = hybrid.forward(x)
    assert out.rf_layer_ids == [0]
    assert torch.allclose(out.hidden, ref, rtol=0.05, atol=0.05)


@pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)
def test_yirage_maca_full_layer_e2e(serving):
    if not serving.is_yirage_maca_available():
        pytest.skip("yirage_maca not available (requires YIRAGE_BACKEND=maca on MetaX VM)")
    report = serving.run_yirage_maca_full_layer_e2e(
        num_layers=2,
        hidden_size=16,
        intermediate_size=32,
        batch=1,
        bench=False,
    )
    assert report.parity_ok
    assert report.all_layers_rf
    assert report.yirage_maca_used
    assert report.maca_meta_bridged
    assert report.rf_layer_ids == [0, 1]
    assert report.plugin == "HybridModelOverride+YirageMacaMlpCapsule+MacaServingMeta"


def test_rf_inspect_version_s16(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s21"
