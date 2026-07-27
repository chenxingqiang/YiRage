# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S11: vLLM full-path MLP RF e2e (real torch; real vllm when installed)."""

from __future__ import annotations

import pytest

from serving_real_test_utils import serving, torch  # noqa: F401


def test_torch_vllm_mlp_rf_e2e_parity(serving):
    report = serving.run_torch_vllm_mlp_rf_e2e(
        hidden_size=16,
        intermediate_size=32,
        batch=3,
        bench=False,
    )
    assert report.parity_ok
    assert report.used_rf_mlp
    assert report.plugin == "TorchDecoderMlpRfHook"


def test_torch_vllm_hybrid_full_e2e_parity(serving):
    report = serving.run_torch_vllm_hybrid_full_e2e(
        num_layers=3,
        max_rf_mlp_layers=2,
        hidden_size=16,
        intermediate_size=32,
        batch=4,
        bench=False,
    )
    assert report.parity_ok
    assert report.rf_layer_ids == [0, 1]
    assert report.plugin == "HybridModelOverride+TorchDecoderMlpRfHook"


def test_torch_vllm_hybrid_with_paged_kv_meta(serving, torch):
    report = serving.run_torch_vllm_hybrid_full_e2e(
        num_layers=2,
        max_rf_mlp_layers=1,
        hidden_size=8,
        intermediate_size=16,
        batch=2,
        bench=False,
        rf_meta={
            "block_tables": [[1, 2, -1], [3, 4, -1]],
            "seq_lens": [32, 18],
            "page_size": 16,
        },
    )
    assert report.parity_ok


def test_vllm_e2e_auto_returns_report(serving):
    report = serving.run_vllm_mlp_rf_e2e_auto(
        hidden_size=16,
        intermediate_size=32,
        batch=2,
        bench=False,
    )
    assert hasattr(report, "parity_ok")
    assert report.parity_ok


@pytest.mark.skipif(
    not __import__("yirage.serving.vllm_plugin", fromlist=["is_vllm_available"]).is_vllm_available(),
    reason="requires installed vllm",
)
def test_vllm_qwen2_mlp_rf_e2e_real(serving):
    pytest.importorskip("transformers")
    report = serving.run_vllm_qwen2_mlp_rf_e2e(
        hidden_size=64,
        intermediate_size=128,
        batch=2,
        bench=False,
    )
    assert report.plugin == "VllmQwen2MlpRfHook"
    assert report.parity_ok
    assert report.used_rf_mlp


def test_rf_inspect_version_s11(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s13"
