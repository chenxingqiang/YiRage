# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S25: Full-model all-layer RF + search tier archive JSON."""

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


def test_runtime_fusion_version_s25(yirage_serving):
    """S25 archive API remains callable with explicit archive_version=s25."""
    from yirage.serving.exec_backend import BACKEND_YIRAGE_CPU
    from yirage.serving.hf_qwen_cpu_e2e import (
        DEFAULT_QWEN05B_MODEL,
        is_transformers_available,
        run_hf_qwen05b_search_tier_bench_archive,
    )

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    _report, archive = run_hf_qwen05b_search_tier_bench_archive(
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=True,
        mlp_backend=BACKEND_YIRAGE_CPU,
        archive_version="s25",
    )
    assert archive.version == "s25"


def test_search_tier_archive_json_contract(yirage_serving):
    from yirage.serving.exec_backend import BACKEND_YIRAGE_CPU
    from yirage.serving.hf_qwen_cpu_e2e import (
        DEFAULT_QWEN05B_MODEL,
        is_transformers_available,
        run_hf_qwen05b_search_tier_bench_archive,
    )

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report, archive = run_hf_qwen05b_search_tier_bench_archive(
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=True,
        mlp_backend=BACKEND_YIRAGE_CPU,
        archive_version="s25",
    )
    payload = archive.to_dict()
    assert payload["serving_bench_archive"] is True
    assert payload["version"] == "s25"
    assert "search_tier" in payload
    assert payload["search_tier"]["tier"] == report.serving_search_tier
    assert any(r["name"] == "qwen05b_yirage_e2e" for r in payload["rows"])
    assert report.parity_ok is True
    json.dumps(payload)


def test_per_layer_superopt_rows(yirage_serving):
    from yirage.serving.exec_backend import BACKEND_YIRAGE_CPU
    from yirage.serving.hf_qwen_cpu_e2e import (
        DEFAULT_QWEN05B_MODEL,
        is_transformers_available,
        run_hf_qwen05b_search_tier_bench_archive,
    )

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report, archive = run_hf_qwen05b_search_tier_bench_archive(
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=True,
        mlp_backend=BACKEND_YIRAGE_CPU,
        archive_version="s25",
    )
    layer_rows = [r for r in archive.rows if r.name.startswith("superopt_layer_")]
    assert len(layer_rows) == report.used_rf_mlp_layers
    assert report.superopt_elapsed_s_total > 0.0


@pytest.mark.slow
def test_all_rf_layers_search_tier_archive(yirage_serving):
    """All 24 Qwen layers: seed-verify superopt + archive JSON (slow)."""
    from yirage.serving.exec_backend import BACKEND_YIRAGE_CPU
    from yirage.serving.hf_qwen_cpu_e2e import (
        DEFAULT_QWEN05B_MODEL,
        is_transformers_available,
        run_hf_qwen05b_search_tier_bench_archive,
    )

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report, archive = run_hf_qwen05b_search_tier_bench_archive(
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=True,
        max_new_tokens=4,
        mlp_backend=BACKEND_YIRAGE_CPU,
        all_rf_layers=True,
        archive_version="s25",
    )
    payload = archive.to_dict()
    assert report.used_rf_mlp_layers == report.num_layers == 24
    assert payload["rows"][0]["all_rf_layers"] is True
    layer_rows = [r for r in payload["rows"] if r["name"].startswith("superopt_layer_")]
    assert len(layer_rows) == 24
    assert report.parity_ok is True


def test_cpu_cert_manifest_s25_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s25_contract" in names
