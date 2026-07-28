# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU full-model Qwen2-0.5B e2e: HF generate + RF MLP prefill/decode parity."""

from __future__ import annotations

import pytest

from serving_test_utils import import_serving


@pytest.fixture(scope="module")
def hf_qwen_e2e():
    serving = import_serving()
    from yirage.serving.hf_qwen_cpu_e2e import (
        DEFAULT_QWEN05B_MODEL,
        is_transformers_available,
        run_hf_qwen05b_cpu_e2e,
    )

    if not is_transformers_available():
        pytest.skip("transformers not installed")
    return serving, DEFAULT_QWEN05B_MODEL, run_hf_qwen05b_cpu_e2e


def test_hf_qwen05b_cpu_e2e_quick(hf_qwen_e2e):
    _, default_model, run_e2e = hf_qwen_e2e
    report = run_e2e(quick=True)
    assert report.model_id == default_model
    assert report.num_layers == 24
    assert report.hidden_size == 896
    assert report.prefill_parity_ok is True
    assert report.decode_parity_ok is True
    assert report.parity_ok is True
    assert len(report.generated_text) > len(report.prompt)


def test_hf_qwen05b_cpu_e2e_contract(hf_qwen_e2e):
    _, _, run_e2e = hf_qwen_e2e
    report = run_e2e(quick=True)
    payload = report.to_dict()
    assert payload["plugin"] == "HfQwen2MlpRfHook+transformers"
    assert payload["device"] == "cpu"
    assert payload["used_rf_mlp_layers"] >= 1
