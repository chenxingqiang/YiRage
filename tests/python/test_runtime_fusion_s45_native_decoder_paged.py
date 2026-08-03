# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S45: Native vLLM full decoder paged multistep tier (G7 chain C paged native decoder)."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving, vllm_available
from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard
from test_runtime_fusion_s42_nightly_bundle_validate import _synthetic_bundle_artifacts


def _synthetic_paged_multistep_native_decoder(*, version: str = "s45") -> dict:
    steps = 3
    ok = [True] * steps
    tok = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    return {
        "serving_vllm_paged_multistep_bench": True,
        "version": version,
        "parity_ok": True,
        "token_match_ok": True,
        "decode_steps": steps,
        "paged_kv_bridged": True,
        "functional_chain": "chain_c_vllm_paged_multistep",
        "num_layers": 2,
        "step_parity_ok": ok,
        "step_token_match_ok": ok,
        "engine_token_ids": tok,
        "hybrid_token_ids": tok,
        "vllm_native_available": False,
        "native_parity_ok": None,
        "native_step_parity_ok": [],
        "native_full_layer_parity_ok": None,
        "native_full_layer_step_parity_ok": [],
        "native_decoder_parity_ok": None,
        "native_decoder_token_match_ok": None,
        "native_decoder_step_parity_ok": [],
        "native_decoder_step_token_match_ok": [],
        "native_decoder_ref_token_ids": [],
        "native_decoder_rf_token_ids": [],
    }


def test_runtime_fusion_version_s45(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s45"


def test_paged_multistep_native_decoder_fields_contract(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        quick=True,
        try_native=False,
        try_native_full_layer=False,
        try_native_decoder=False,
        version="s45",
    )
    payload = report.to_dict()
    assert "native_decoder_parity_ok" in payload
    assert "native_decoder_token_match_ok" in payload
    assert "native_decoder_step_parity_ok" in payload
    assert "native_decoder_step_token_match_ok" in payload
    assert payload["native_decoder_parity_ok"] is None
    assert payload["native_decoder_token_match_ok"] is None
    json.dumps(payload)


def test_paged_multistep_torch_gate_unchanged_s45(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=False,
        try_native_full_layer=False,
        try_native_decoder=False,
        version="s45",
    )
    assert report.parity_ok
    assert report.token_match_ok
    assert report.paged_kv_bridged


@pytest.mark.skipif(not vllm_available(), reason="vllm not installed")
def test_paged_multistep_native_decoder_when_vllm(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=True,
        try_native_full_layer=True,
        try_native_decoder=True,
        version="s45",
    )
    assert report.vllm_native_available is True
    assert report.native_decoder_parity_ok is True
    assert report.native_decoder_token_match_ok is True
    assert len(report.native_decoder_step_parity_ok) == 3
    assert len(report.native_decoder_step_token_match_ok) == 3
    assert report.native_decoder_ref_token_ids == report.native_decoder_rf_token_ids


def test_combined_archive_metadata_native_decoder_fields(serving):
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )

    payload = {
        "serving_combined_nightly_archive": True,
        "version": "s45",
        "parity_ok": True,
        "quick": True,
        "decode": {"parity_ok": True},
        "engine_g1": {"parity_ok": True},
        "multistep": {"parity_ok": True, "token_match_ok": True},
        "engine_multistep": {"parity_ok": True},
        "paged_multistep": _synthetic_paged_multistep_native_decoder(),
    }
    meta = serving_combined_nightly_archive_metadata(
        payload, archive_path="artifacts/combined.json", validation_ok=True, quick=True
    )
    assert "paged_multistep_native_decoder_parity_ok" in meta
    assert "paged_multistep_native_decoder_token_match_ok" in meta
    assert meta["paged_multistep_native_decoder_parity_ok"] is None
    assert meta["paged_multistep_native_decoder_token_match_ok"] is None


def test_bundle_metadata_includes_native_decoder_fields(serving):
    from yirage.serving.serving_nightly_bundle import (
        serving_combined_nightly_bundle_metadata,
        validate_serving_combined_nightly_bundle_metadata,
    )

    archive = _synthetic_combined_for_dashboard(version="s45")
    archive["paged_multistep"] = _synthetic_paged_multistep_native_decoder()
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        serving_dashboard_artifact_metadata,
    )

    archive_meta = serving_combined_nightly_archive_metadata(
        archive,
        archive_path="artifacts/serving-combined-nightly.json",
        validation_ok=True,
        quick=True,
    )
    dashboard = build_serving_dashboard_from_combined_archive(archive).to_dict()
    dashboard_meta = serving_dashboard_artifact_metadata(
        dashboard,
        json_path="artifacts/serving-dashboard.json",
        validation_ok=True,
    )
    meta = serving_combined_nightly_bundle_metadata(
        archive_payload=archive,
        archive_metadata=archive_meta,
        dashboard_metadata=dashboard_meta,
        validation_ok=True,
        archive_path="artifacts/serving-combined-nightly.json",
        archive_meta_path="artifacts/serving-combined-nightly.meta.json",
        dashboard_json_path="artifacts/serving-dashboard.json",
        dashboard_meta_path="artifacts/serving-dashboard.meta.json",
    )
    assert "paged_multistep_native_decoder_parity_ok" in meta
    assert "paged_multistep_native_decoder_token_match_ok" in meta
    assert validate_serving_combined_nightly_bundle_metadata(meta) == []


def test_bundle_validate_rejects_native_decoder_meta_mismatch(serving):
    from yirage.serving.serving_nightly_bundle import validate_serving_combined_nightly_bundle

    bundle = _synthetic_bundle_artifacts(version="s45")
    bundle["archive"]["paged_multistep"] = _synthetic_paged_multistep_native_decoder()
    bundle["archive_meta"] = serving_combined_nightly_archive_metadata_from_bundle(bundle)
    bad_meta = dict(bundle["archive_meta"])
    bad_meta["paged_multistep_native_decoder_parity_ok"] = True
    errors = validate_serving_combined_nightly_bundle(
        archive_payload=bundle["archive"],
        archive_metadata=bad_meta,
        dashboard_payload=bundle["dashboard"],
        dashboard_metadata=bundle["dashboard_meta"],
        html_document=bundle["html"],
        markdown_document=bundle["markdown"],
    )
    assert any("native_decoder_parity_ok mismatch" in e for e in errors)


def serving_combined_nightly_archive_metadata_from_bundle(bundle: dict) -> dict:
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )

    return serving_combined_nightly_archive_metadata(
        bundle["archive"],
        archive_path="artifacts/serving-combined-nightly.json",
        validation_ok=True,
        quick=True,
    )


def test_dashboard_paged_multistep_native_decoder_metric(serving):
    from yirage.serving.serving_dashboard import build_serving_dashboard_from_combined_archive

    archive = _synthetic_combined_for_dashboard(version="s45")
    archive["paged_multistep"] = _synthetic_paged_multistep_native_decoder()
    archive["paged_multistep"]["native_decoder_parity_ok"] = True
    archive["paged_multistep"]["native_decoder_token_match_ok"] = True
    report = build_serving_dashboard_from_combined_archive(archive)
    paged = [r for r in report.rows if r.section == "paged_multistep"]
    assert len(paged) == 1
    assert paged[0].metrics.get("native_decoder_parity_ok") is True
    assert paged[0].metrics.get("native_decoder_token_match_ok") is True


def test_cpu_cert_manifest_s45_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s45_contract" in names
