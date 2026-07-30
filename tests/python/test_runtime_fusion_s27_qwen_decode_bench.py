# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S27: Qwen decode-step bench — native HF vs YiRage RF fused MLP."""

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


def test_runtime_fusion_version_s27(yirage_serving):
    assert yirage_serving.RuntimeFusion([]).inspect()["version"] == "s27"


def test_qwen_decode_bench_json_contract(yirage_serving):
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, is_transformers_available
    from yirage.serving.qwen_decode_bench import run_qwen_decode_bench

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report = run_qwen_decode_bench(
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=True,
    )
    payload = report.to_dict()
    assert payload["serving_qwen_decode_bench"] is True
    assert payload["version"] == "s27"
    assert payload["parity_ok"] is True
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["name"] == "native_decode_step"
    assert payload["rows"][1]["name"] == "yirage_rf_decode_step"
    assert payload["speedup_yirage_vs_native"] > 0
    json.dumps(payload)


def test_qwen_decode_bench_speedup_sane(yirage_serving):
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, is_transformers_available
    from yirage.serving.qwen_decode_bench import run_qwen_decode_bench

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report = run_qwen_decode_bench(
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=True,
    )
    native_ms = report.rows[0].mean_ms
    yirage_ms = report.rows[1].mean_ms
    assert native_ms > 0
    assert yirage_ms > 0
    # MuGraph cache hot: YiRage path should be same order of magnitude as native.
    assert report.speedup_yirage_vs_native < 10.0


def test_qwen_decode_bench_per_layer_superopt(yirage_serving):
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, is_transformers_available
    from yirage.serving.qwen_decode_bench import (
        qwen_decode_bench_per_layer_superopt,
        run_qwen_decode_bench,
    )

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report = run_qwen_decode_bench(
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=True,
    )
    layers = qwen_decode_bench_per_layer_superopt(report)
    assert len(layers) == report.max_rf_mlp_layers
    assert report.superopt_elapsed_s_total >= 0


def test_cpu_cert_manifest_s27_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s27_contract" in names
