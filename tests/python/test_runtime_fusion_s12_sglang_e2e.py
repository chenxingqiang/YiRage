# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S12: SGLang ForwardBatch full-path MLP RF e2e (real torch; real sglang when installed)."""

from __future__ import annotations

import pytest

from serving_real_test_utils import serving, torch  # noqa: F401


def test_torch_sglang_mlp_rf_e2e_partial_radix(serving):
    batch = serving.SglangForwardBatchSpec(
        extend_seq_lens=[0, 3, 0],
        seq_lens=[8, 12, 16],
    )
    report = serving.run_torch_sglang_mlp_rf_e2e(
        forward_batch=batch,
        hidden_size=16,
        intermediate_size=32,
        batch=3,
        bench=False,
    )
    assert report.parity_ok
    assert report.radix_partial
    assert report.used_rf_mlp


def test_torch_sglang_mlp_rf_e2e_all_radix_hit(serving):
    batch = serving.SglangForwardBatchSpec(
        extend_seq_lens=[0, 0],
        seq_lens=[4, 4],
    )
    report = serving.run_torch_sglang_mlp_rf_e2e(
        forward_batch=batch,
        hidden_size=8,
        intermediate_size=16,
        batch=2,
        bench=False,
    )
    assert report.parity_ok
    assert report.radix_all_hit
    assert not report.used_rf_mlp


def test_torch_sglang_hybrid_full_e2e_with_kv(serving):
    batch = serving.SglangForwardBatchSpec(
        extend_seq_lens=[0, 3, 0, 0],
        seq_lens=[32, 18, 24, 16],
        block_tables=[[1, 2, -1], [3, 4, -1], [5, 6, -1], [7, 8, -1]],
        page_size=16,
    )
    report = serving.run_torch_sglang_hybrid_full_e2e(
        forward_batch=batch,
        num_layers=3,
        max_rf_mlp_layers=2,
        hidden_size=16,
        intermediate_size=32,
        batch=4,
        bench=False,
    )
    assert report.parity_ok
    assert report.rf_layer_ids == [0, 1]
    meta = batch.as_meta(enabled={"mlp_layer_0"})
    assert "paged_kv" in meta["extras"]


def test_sglang_e2e_auto_returns_report(serving):
    report = serving.run_sglang_mlp_rf_e2e_auto(
        hidden_size=16,
        intermediate_size=32,
        batch=2,
        bench=False,
    )
    assert report.parity_ok


@pytest.mark.skipif(
    not __import__("yirage.serving.sglang_plugin", fromlist=["is_sglang_available"]).is_sglang_available(),
    reason="requires installed sglang",
)
def test_sglang_qwen2_mlp_rf_e2e_real(serving):
    pytest.importorskip("transformers")
    report = serving.run_sglang_qwen2_mlp_rf_e2e(
        hidden_size=64,
        intermediate_size=128,
        batch=2,
        bench=False,
    )
    assert report.plugin == "SglangQwen2MlpRfHook"
    assert report.parity_ok


def test_rf_inspect_version_s12(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s14"
