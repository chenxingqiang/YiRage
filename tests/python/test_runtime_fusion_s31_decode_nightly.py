# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S31: Qwen decode-step bench nightly archive validate + CI."""

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


def test_runtime_fusion_version_s31(yirage_serving):
    payload = _synthetic_decode_archive(version="s31")
    assert payload["version"] == "s31"


def _synthetic_decode_archive(*, version: str = "s31") -> dict:
    return {
        "serving_qwen_decode_bench": True,
        "version": version,
        "model_id": "Qwen/Qwen2-0.5B",
        "device": "cpu",
        "max_rf_mlp_layers": 1,
        "num_layers": 24,
        "all_rf_layers": False,
        "parity_ok": True,
        "speedup_yirage_vs_native": 1.05,
        "superopt_elapsed_s_total": 0.5,
        "serving_search_tier": "seed_verify",
        "rows": [
            {"name": "native_decode_step", "mean_ms": 10.0, "iters": 8, "device": "cpu"},
            {"name": "yirage_rf_decode_step", "mean_ms": 9.5, "iters": 8, "device": "cpu"},
        ],
    }


def test_validate_serving_qwen_decode_bench_archive_synthetic(yirage_serving):
    from yirage.serving.decode_bench_archive import validate_serving_qwen_decode_bench_archive

    payload = _synthetic_decode_archive()
    assert validate_serving_qwen_decode_bench_archive(payload) == []


def test_validate_decode_archive_rejects_bad_parity(yirage_serving):
    from yirage.serving.decode_bench_archive import validate_serving_qwen_decode_bench_archive

    bad = _synthetic_decode_archive()
    bad["parity_ok"] = False
    errors = validate_serving_qwen_decode_bench_archive(bad)
    assert any("parity_ok" in e for e in errors)


def test_serving_qwen_decode_bench_archive_metadata(yirage_serving):
    from yirage.serving.decode_bench_archive import serving_qwen_decode_bench_archive_metadata

    payload = _synthetic_decode_archive()
    meta = serving_qwen_decode_bench_archive_metadata(
        payload, archive_path="artifacts/decode.json", validation_ok=True, quick=True
    )
    assert meta["serving_qwen_decode_bench_archive_metadata"] is True
    assert meta["parity_ok"] is True
    assert meta["quick"] is True
    json.dumps(meta)


def test_cpu_cert_manifest_s31_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s31_contract" in names


@pytest.mark.slow
def test_run_serving_qwen_decode_bench_archive_quick(yirage_serving):
    from yirage.serving.decode_bench_archive import run_serving_qwen_decode_bench_archive
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, is_transformers_available

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    payload = run_serving_qwen_decode_bench_archive(
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=True,
        version="s31",
    )
    assert payload["parity_ok"] is True
    assert payload["version"] == "s31"
