# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S13: vLLM PagedAttention + full-layer MLP RF hook e2e (real torch)."""

from __future__ import annotations

import pytest

from serving_real_test_utils import serving, torch  # noqa: F401


def test_vllm_paged_kv_batch_spec_meta(serving):
    spec = serving.VllmPagedKvBatchSpec(
        block_tables=[[1, 2, -1], [3, 4, -1]],
        seq_lens=[32, 18],
        page_size=16,
    )
    meta = spec.as_rf_meta()
    assert "paged_kv" in meta["extras"]
    assert meta["extras"]["paged_kv"]["paged_kv_indptr"] == [0, 2, 4]


def test_torch_vllm_paged_full_layer_e2e(serving):
    report = serving.run_torch_vllm_paged_full_layer_e2e(
        num_layers=3,
        hidden_size=16,
        intermediate_size=32,
        batch=4,
        bench=False,
    )
    assert report.parity_ok
    assert report.all_layers_rf
    assert report.paged_kv_bridged
    assert report.rf_layer_ids == [0, 1, 2]
    assert report.plugin == "HybridModelOverride+PagedKv+TorchDecoderMlpRfHook"


def test_vllm_paged_auto_entry(serving):
    report = serving.run_vllm_paged_full_layer_e2e_auto(
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        batch=2,
        bench=False,
    )
    assert report.parity_ok
    assert report.all_layers_rf


def test_rf_inspect_version_s13(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s18"
