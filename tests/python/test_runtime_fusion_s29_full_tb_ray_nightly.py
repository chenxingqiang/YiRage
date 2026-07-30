# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S29: full_tb_ray nightly multi-tier search archive."""

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


def test_runtime_fusion_version_s29(yirage_serving):
    assert yirage_serving.RuntimeFusion([]).inspect()["version"] == "s29"


def test_validate_serving_multi_tier_bench_archive_synthetic(yirage_serving):
    from yirage.serving.search_tier_archive import (
        build_multi_tier_bench_archive,
        validate_serving_multi_tier_bench_archive,
        validate_serving_search_tier_archive,
    )
    from test_runtime_fusion_s26_multi_tier_archive import _synthetic_archive

    tiers = {
        "seed_verify": _synthetic_archive(tier="seed_verify", superopt_s=0.01),
        "full_tb_ray": _synthetic_archive(tier="full_tb_ray", superopt_s=20.0),
    }
    multi = build_multi_tier_bench_archive(tiers, version="s29")
    payload = multi.to_dict()
    assert validate_serving_multi_tier_bench_archive(payload) == []
    assert validate_serving_search_tier_archive(payload) == []
    assert payload["compare"]["ok"] is True
    assert payload["compare"]["candidate"]["tier"] == "full_tb_ray"


def test_serving_multi_tier_bench_archive_metadata(yirage_serving):
    from yirage.serving.search_tier_archive import (
        build_multi_tier_bench_archive,
        serving_multi_tier_bench_archive_metadata,
    )
    from test_runtime_fusion_s26_multi_tier_archive import _synthetic_archive

    tiers = {
        "seed_verify": _synthetic_archive(tier="seed_verify", superopt_s=0.01),
        "full_tb_ray": _synthetic_archive(tier="full_tb_ray", superopt_s=20.0),
    }
    multi = build_multi_tier_bench_archive(tiers, version="s29")
    payload = multi.to_dict()
    meta = serving_multi_tier_bench_archive_metadata(payload, quick=True)
    assert meta["serving_multi_tier_bench_archive_metadata"] is True
    assert meta["compare_ok"] is True
    assert "full_tb_ray" in meta["tier_names"]


@pytest.mark.slow
def test_multi_tier_quick_archive_full_tb_ray(yirage_serving):
    from yirage.serving.exec_backend import BACKEND_YIRAGE_CPU
    from yirage.serving.hf_qwen_cpu_e2e import DEFAULT_QWEN05B_MODEL, is_transformers_available
    from yirage.serving.search_tier_archive import (
        run_serving_multi_tier_bench_archive,
        serving_search_tier_preset_names,
        validate_serving_search_tier_archive,
    )

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    multi = run_serving_multi_tier_bench_archive(
        tier_names=serving_search_tier_preset_names(),
        quick=True,
        mlp_backend=BACKEND_YIRAGE_CPU,
        archive_version="s29",
    )
    payload = multi.to_dict()
    assert validate_serving_search_tier_archive(payload) == []
    assert payload["version"] == "s29"
    assert payload["compare"]["ok"] is True
    assert payload["compare"]["baseline"]["tier"] == "seed_verify"
    assert payload["compare"]["candidate"]["tier"] == "full_tb_ray"
    assert payload["compare"]["superopt_slowdown_vs_baseline"] > 1.0
    json.dumps(payload)


def test_cpu_cert_manifest_s29_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s29_contract" in names
