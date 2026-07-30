# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S28: Multi-layer Qwen decode-step bench — native HF vs YiRage RF fused MLP."""

from __future__ import annotations

import json
import os

import pytest

from serving_test_utils import serving


pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


@pytest.fixture(scope="module")
def yirage_serving(serving):
    if not serving.is_yirage_core_available():
        pytest.skip("yirage.core not available")
    serving.require_yirage_core()
    return serving


def test_runtime_fusion_version_s28(yirage_serving):
    """S28 multilayer decode bench API remains callable with version=s28."""
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, is_transformers_available
    from yirage.serving.qwen_decode_bench import run_qwen_multilayer_decode_bench

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report = run_qwen_multilayer_decode_bench(
        model_id=DEFAULT_QWEN05B_MODEL,
        max_rf_mlp_layers=2,
        quick=True,
        version="s28",
    )
    assert report.to_dict()["version"] == "s28"


def test_multilayer_decode_bench_json_contract(yirage_serving):
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, is_transformers_available
    from yirage.serving.qwen_decode_bench import run_qwen_multilayer_decode_bench

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report = run_qwen_multilayer_decode_bench(
        model_id=DEFAULT_QWEN05B_MODEL,
        max_rf_mlp_layers=2,
        quick=True,
    )
    payload = report.to_dict()
    assert payload["serving_qwen_decode_bench"] is True
    assert payload["version"] == "s28"
    assert payload["max_rf_mlp_layers"] == 2
    assert payload["num_layers"] == 24
    assert payload["all_rf_layers"] is False
    assert payload["parity_ok"] is True
    assert len(payload["rows"]) == 2
    assert len(payload["per_layer_superopt"]) == 2
    assert payload["speedup_yirage_vs_native"] > 0
    json.dumps(payload)


def test_multilayer_decode_bench_per_layer_superopt(yirage_serving):
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, is_transformers_available
    from yirage.serving.qwen_decode_bench import (
        qwen_decode_bench_per_layer_superopt,
        run_qwen_multilayer_decode_bench,
    )

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report = run_qwen_multilayer_decode_bench(
        model_id=DEFAULT_QWEN05B_MODEL,
        max_rf_mlp_layers=2,
        quick=True,
    )
    layers = qwen_decode_bench_per_layer_superopt(report)
    assert len(layers) == 2
    assert report.superopt_elapsed_s_total > 0.0


@pytest.mark.slow
def test_all_rf_layers_decode_bench(yirage_serving):
    """All 24 Qwen layers: decode-step bench with per-layer superopt stats (slow)."""
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, is_transformers_available
    from yirage.serving.qwen_decode_bench import run_qwen_multilayer_decode_bench

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report = run_qwen_multilayer_decode_bench(
        model_id=DEFAULT_QWEN05B_MODEL,
        all_rf_layers=True,
        quick=True,
    )
    payload = report.to_dict()
    assert report.max_rf_mlp_layers == report.num_layers == 24
    assert payload["all_rf_layers"] is True
    assert len(payload["per_layer_superopt"]) == 24
    assert report.parity_ok is True
    assert report.speedup_yirage_vs_native > 0


def test_cpu_cert_manifest_s28_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s28_contract" in names
