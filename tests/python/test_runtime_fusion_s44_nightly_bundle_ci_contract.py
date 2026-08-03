# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S44: Nightly bundle CI contract + dashboard S43 metrics closure."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from serving_test_utils import serving
from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard
from test_runtime_fusion_s42_nightly_bundle_validate import _synthetic_bundle_artifacts


def test_runtime_fusion_version_s44(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s44"


def test_nightly_bundle_ci_contract_manifest(serving):
    from yirage.serving.serving_nightly_bundle import (
        serving_nightly_bundle_ci_contract,
        validate_serving_nightly_bundle_ci_contract,
    )

    contract = serving_nightly_bundle_ci_contract()
    assert validate_serving_nightly_bundle_ci_contract(contract) == []
    assert contract["make_profile_target"] == "test-serving-combined-nightly-profile"
    assert "serving-nightly-bundle.meta.json" in contract["artifact_files"][-1]
    json.dumps(contract)


def test_bundle_metadata_includes_paged_native_fields(serving):
    from yirage.serving.serving_nightly_bundle import (
        serving_combined_nightly_bundle_metadata,
        validate_serving_combined_nightly_bundle_metadata,
    )

    bundle = _synthetic_bundle_artifacts(version="s44")
    meta = serving_combined_nightly_bundle_metadata(
        archive_payload=bundle["archive"],
        archive_metadata=bundle["archive_meta"],
        dashboard_metadata=bundle["dashboard_meta"],
        validation_ok=True,
        archive_path="artifacts/serving-combined-nightly.json",
        archive_meta_path="artifacts/serving-combined-nightly.meta.json",
        dashboard_json_path="artifacts/serving-dashboard.json",
        dashboard_meta_path="artifacts/serving-dashboard.meta.json",
    )
    assert "paged_multistep_native_full_layer_parity_ok" in meta
    assert validate_serving_combined_nightly_bundle_metadata(meta) == []


def test_bundle_validate_archive_native_full_layer_meta(serving):
    from yirage.serving.serving_nightly_bundle import validate_serving_combined_nightly_bundle

    bundle = _synthetic_bundle_artifacts(version="s44")
    errors = validate_serving_combined_nightly_bundle(
        archive_payload=bundle["archive"],
        archive_metadata=bundle["archive_meta"],
        dashboard_payload=bundle["dashboard"],
        dashboard_metadata=bundle["dashboard_meta"],
        html_document=bundle["html"],
        markdown_document=bundle["markdown"],
    )
    assert errors == []


def test_bundle_validate_rejects_native_full_layer_meta_mismatch(serving):
    from yirage.serving.serving_nightly_bundle import validate_serving_combined_nightly_bundle

    bundle = _synthetic_bundle_artifacts(version="s44")
    bad_meta = dict(bundle["archive_meta"])
    bad_meta["paged_multistep_native_full_layer_parity_ok"] = True
    errors = validate_serving_combined_nightly_bundle(
        archive_payload=bundle["archive"],
        archive_metadata=bad_meta,
        dashboard_payload=bundle["dashboard"],
        dashboard_metadata=bundle["dashboard_meta"],
        html_document=bundle["html"],
        markdown_document=bundle["markdown"],
    )
    assert any("native_full_layer_parity_ok mismatch" in e for e in errors)


def test_dashboard_paged_multistep_native_full_layer_metric(serving):
    from yirage.serving.serving_dashboard import build_serving_dashboard_from_combined_archive

    archive = _synthetic_combined_for_dashboard(version="s44")
    archive["paged_multistep"]["native_full_layer_parity_ok"] = True
    report = build_serving_dashboard_from_combined_archive(archive)
    paged = [r for r in report.rows if r.section == "paged_multistep"]
    assert len(paged) == 1
    assert paged[0].metrics.get("native_full_layer_parity_ok") is True


def test_makefile_profile_includes_bundle_validate():
    repo = Path(__file__).resolve().parents[2]
    makefile = (repo / "Makefile").read_text(encoding="utf-8")
    assert "validate_serving_combined_nightly_bundle.py" in makefile
    assert "serving-nightly-bundle.meta.json" in makefile


def test_workflow_uploads_bundle_meta():
    repo = Path(__file__).resolve().parents[2]
    workflow = (repo / ".github/workflows/serving-combined-nightly-archive.yml").read_text(
        encoding="utf-8"
    )
    assert "serving-nightly-bundle.meta.json" in workflow
    assert "test_runtime_fusion_s44_nightly_bundle_ci_contract.py" in workflow


def test_cpu_cert_manifest_s44_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s44_contract" in names
