# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S15: MACA serving meta bridge + vLLM-metax plugin tier (real torch)."""

from __future__ import annotations

import os

import pytest

from serving_real_test_utils import serving, torch  # noqa: F401


def test_maca_serving_rf_spec_meta(serving):
    spec = serving.MacaServingRfSpec()
    meta = spec.as_rf_meta(sm_budget=96)
    assert meta["sm_budget"] == 96
    payload = meta["extras"]["maca_serving"]
    assert payload["warp_size"] == serving.MACA_SERVING_WARP_SIZE
    assert payload["sm_count"] == serving.MACA_SERVING_SM_COUNT_C500
    assert payload["block_dim"] == list(serving.MACA_SERVING_DEFAULT_BLOCK_DIM)


def test_validate_maca_block_dim_rejects_bad_warp(serving):
    with pytest.raises(ValueError, match="multiple of warp_size"):
        serving.validate_maca_block_dim((32, 1, 1))


def test_attach_maca_serving_to_step_meta(serving):
    merged = serving.attach_maca_serving_to_step_meta({"enabled": {"mlp_layer_0"}})
    assert serving.maca_serving_present(merged)
    assert merged["extras"]["total_sms"] == serving.MACA_SERVING_SM_COUNT_C500


def test_torch_maca_serving_full_layer_e2e(serving):
    report = serving.run_torch_maca_serving_full_layer_e2e(
        num_layers=3,
        hidden_size=16,
        intermediate_size=32,
        batch=4,
        bench=False,
    )
    assert report.parity_ok
    assert report.all_layers_rf
    assert report.maca_meta_bridged
    assert report.rf_layer_ids == [0, 1, 2]
    assert report.warp_size == 64
    assert report.plugin == "HybridModelOverride+MacaServingMeta+TorchDecoderMlpRfHook"


def test_maca_serving_auto_entry(serving):
    report = serving.run_maca_serving_full_layer_e2e_auto(
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        batch=2,
        bench=False,
    )
    assert report.parity_ok
    assert report.all_layers_rf


def test_is_metax_torch_bool(serving):
    assert isinstance(serving.is_metax_torch(), bool)


def test_is_vllm_metax_available_bool(serving):
    assert isinstance(serving.is_vllm_metax_available(), bool)


def test_vllm_metax_hook_raises_without_tier(serving):
    if serving.is_vllm_metax_available():
        pytest.skip("vllm-metax tier available; skip negative test")
    with pytest.raises(RuntimeError, match="vLLM-metax"):
        serving.require_vllm_metax()


@pytest.mark.skipif(
    not __import__(
        "yirage.serving.vllm_metax_plugin",
        fromlist=["is_vllm_metax_available"],
    ).is_vllm_metax_available(),
    reason="requires vllm on MetaX host or YIRAGE_MACA_INTEGRATION=1",
)
def test_vllm_metax_plugin_requires_tier(serving):
    serving.require_vllm_metax()
    assert serving.is_vllm_metax_available()


def test_yirage_maca_tier_inspect(serving):
    info = serving.inspect_maca_serving_yirage_tier()
    assert "yirage_maca_available" in info
    assert isinstance(info["yirage_maca_available"], bool)


def test_rf_inspect_version_s15(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s17"


def test_backend_yirage_maca_constant(serving):
    assert serving.BACKEND_YIRAGE_MACA == "yirage_maca"
    assert serving.is_maca_serving_backend(serving.BACKEND_YIRAGE_MACA)
