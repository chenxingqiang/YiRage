# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S26: multi-tier search archive validate + nightly compare."""

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


def test_runtime_fusion_version_s26(yirage_serving):
    """S26 multi-tier archive API remains callable with archive_version=s26."""
    from yirage.serving.search_tier_archive import serving_search_tier_preset_names

    assert "seed_verify" in serving_search_tier_preset_names()


def test_serving_search_tier_preset_names(yirage_serving):
    from yirage.serving.search_tier_archive import serving_search_tier_preset_names

    names = serving_search_tier_preset_names()
    assert "seed_verify" in names
    assert "full_tb_ray" in names


def test_serving_search_tier_preset_env(yirage_serving):
    from yirage.serving.search_tier_archive import serving_search_tier_preset
    from yirage.serving.yirage_exec import resolve_serving_search_tier

    with serving_search_tier_preset("seed_verify"):
        assert resolve_serving_search_tier() == "seed_verify"
    with serving_search_tier_preset("full_tb_ray"):
        assert resolve_serving_search_tier() == "full_tb_ray"


def _synthetic_archive(*, tier: str, superopt_s: float, layers: int = 1) -> dict:
    from yirage.serving.bench_archive import ServingBenchArchive, ServingBenchArchiveRow

    archive = ServingBenchArchive(
        version="s26",
        device="cpu",
        search_tier={"tier": tier, "full_tb_search": tier.startswith("full_tb")},
    )
    archive.rows.append(
        ServingBenchArchiveRow(
            name="qwen05b_yirage_e2e",
            mean_ms=12.0,
            iters=1,
            device="cpu",
            parity_ok=True,
            extras={
                "serving_search_tier": tier,
                "used_rf_mlp_layers": layers,
                "num_layers": 24,
                "all_rf_layers": layers >= 24,
                "superopt_elapsed_s_total": superopt_s,
            },
        )
    )
    for layer_id in range(layers):
        archive.rows.append(
            ServingBenchArchiveRow(
                name=f"superopt_layer_{layer_id}",
                mean_ms=superopt_s * 1000.0 / max(layers, 1),
                iters=1,
                device="cpu",
                parity_ok=True,
                extras={"layer_id": layer_id, "superopt_elapsed_s": superopt_s / max(layers, 1)},
            )
        )
    return archive.to_dict()


def test_compare_serving_search_tier_archives(yirage_serving):
    from yirage.serving.search_tier_archive import (
        compare_serving_search_tier_archives,
        extract_tier_summary,
        validate_serving_bench_archive,
    )

    baseline = _synthetic_archive(tier="seed_verify", superopt_s=0.005)
    candidate = _synthetic_archive(tier="full_tb_ray", superopt_s=18.0)
    assert validate_serving_bench_archive(baseline) == []
    base_summary = extract_tier_summary(baseline)
    assert base_summary["tier"] == "seed_verify"
    report = compare_serving_search_tier_archives(baseline, candidate)
    assert report["ok"] is True
    assert report["tier_changed"] is True
    assert report["superopt_slowdown_vs_baseline"] > 100.0


def test_build_multi_tier_bench_archive(yirage_serving):
    from yirage.serving.search_tier_archive import build_multi_tier_bench_archive

    tiers = {
        "seed_verify": _synthetic_archive(tier="seed_verify", superopt_s=0.01),
        "full_tb_ray": _synthetic_archive(tier="full_tb_ray", superopt_s=20.0),
    }
    multi = build_multi_tier_bench_archive(tiers, version="s26")
    payload = multi.to_dict()
    assert payload["serving_multi_tier_bench_archive"] is True
    assert "compare" in payload
    assert payload["compare"]["baseline"]["tier"] == "seed_verify"
    assert payload["compare"]["candidate"]["tier"] == "full_tb_ray"


def test_validate_serving_bench_archive_rejects_invalid(yirage_serving):
    from yirage.serving.search_tier_archive import validate_serving_bench_archive

    errors = validate_serving_bench_archive({"version": "s26"})
    assert errors


def test_seed_verify_tier_archive_json_contract(yirage_serving):
    from yirage.serving.exec_backend import BACKEND_YIRAGE_CPU
    from yirage.serving.hf_qwen_cpu_e2e import (
        DEFAULT_QWEN05B_MODEL,
        is_transformers_available,
    )
    from yirage.serving.search_tier_archive import (
        run_serving_search_tier_bench_archive_for_preset,
        validate_serving_bench_archive,
    )

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    report, archive = run_serving_search_tier_bench_archive_for_preset(
        "seed_verify",
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=True,
        mlp_backend=BACKEND_YIRAGE_CPU,
        archive_version="s26",
    )
    payload = archive.to_dict()
    assert validate_serving_bench_archive(payload) == []
    assert payload["version"] == "s26"
    assert payload["search_tier"]["tier"] == "seed_verify"
    assert report.parity_ok is True
    json.dumps(payload)


def test_cpu_cert_manifest_s26_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s26_contract" in names
