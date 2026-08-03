# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S42: Combined nightly archive + dashboard artifact bundle validation (G7 closure).

Validates the full nightly artifact set produced by ``make test-serving-combined-nightly-profile``:
combined archive JSON/meta plus dashboard JSON/meta/HTML/markdown with cross-field parity.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Mapping, Optional

from .combined_nightly_archive import validate_serving_combined_nightly_archive
from .serving_dashboard import (
    build_serving_dashboard_from_combined_archive,
    validate_serving_dashboard_artifact_bundle,
)


def _archive_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def serving_nightly_bundle_ci_contract() -> Dict[str, Any]:
    """CI/nightly artifact manifest for combined archive + dashboard bundle (S44)."""
    return {
        "serving_nightly_bundle_ci_contract": True,
        "make_profile_target": "test-serving-combined-nightly-profile",
        "validate_cli": "scripts/validate_serving_combined_nightly_bundle.py",
        "workflow_path": ".github/workflows/serving-combined-nightly-archive.yml",
        "artifact_files": [
            "artifacts/serving-combined-nightly.json",
            "artifacts/serving-combined-nightly.meta.json",
            "artifacts/serving-dashboard.json",
            "artifacts/serving-dashboard.meta.json",
            "artifacts/serving-dashboard.html",
            "artifacts/serving-dashboard.md",
            "artifacts/serving-nightly-bundle.meta.json",
        ],
        "contract_pytests": [
            "tests/python/test_runtime_fusion_s42_nightly_bundle_validate.py",
            "tests/python/test_runtime_fusion_s44_nightly_bundle_ci_contract.py",
        ],
        "bench_smokes": [
            "benchmark/serving_combined_nightly_bundle_validate.py",
            "benchmark/serving_dashboard_validate.py",
        ],
    }


def validate_serving_nightly_bundle_ci_contract(payload: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if not payload.get("serving_nightly_bundle_ci_contract"):
        errors.append("missing serving_nightly_bundle_ci_contract marker")
    files = payload.get("artifact_files")
    if not isinstance(files, list) or len(files) < 7:
        errors.append("artifact_files must list all nightly bundle outputs")
    if not isinstance(payload.get("make_profile_target"), str):
        errors.append("make_profile_target must be a string")
    return errors


def validate_serving_combined_nightly_bundle(
    *,
    archive_payload: Mapping[str, Any],
    archive_metadata: Optional[Mapping[str, Any]] = None,
    dashboard_payload: Optional[Mapping[str, Any]] = None,
    dashboard_metadata: Optional[Mapping[str, Any]] = None,
    html_document: Optional[str] = None,
    markdown_document: Optional[str] = None,
    allow_partial: bool = False,
) -> List[str]:
    """Validate combined archive and dashboard artifacts as one nightly bundle."""
    errors: List[str] = list(validate_serving_combined_nightly_archive(dict(archive_payload)))
    if allow_partial:
        errors = [
            e
            for e in errors
            if not e.endswith("subsection must be a dict")
            and "functional_chains must" not in e
        ]

    archive_version = archive_payload.get("version")
    archive_parity = archive_payload.get("parity_ok")
    expected_sha = _archive_sha256(archive_payload)

    if archive_metadata is not None:
        if not archive_metadata.get("serving_combined_nightly_archive_metadata"):
            errors.append("archive_metadata missing serving_combined_nightly_archive_metadata marker")
        if archive_metadata.get("validation_ok") is not True:
            errors.append("archive_metadata validation_ok must be true")
        if archive_metadata.get("version") != archive_version:
            errors.append("archive_metadata version mismatch with combined archive")
        if archive_metadata.get("parity_ok") != archive_parity:
            errors.append("archive_metadata parity_ok mismatch with combined archive")
        paged_native_full = (
            archive_payload.get("paged_multistep")
            if isinstance(archive_payload.get("paged_multistep"), dict)
            else {}
        ).get("native_full_layer_parity_ok")
        meta_native_full = archive_metadata.get("paged_multistep_native_full_layer_parity_ok")
        if meta_native_full != paged_native_full:
            errors.append(
                "archive_metadata paged_multistep_native_full_layer_parity_ok mismatch"
            )
        meta_sha = archive_metadata.get("archive_sha256")
        if isinstance(meta_sha, str) and meta_sha != expected_sha:
            errors.append("archive_metadata archive_sha256 mismatch with combined archive payload")

    if dashboard_payload is not None:
        rebuilt = build_serving_dashboard_from_combined_archive(
            archive_payload, allow_partial=allow_partial
        ).to_dict()
        if dashboard_payload.get("archive_version") != archive_version:
            errors.append("dashboard archive_version mismatch with combined archive")
        if dashboard_payload.get("parity_ok") != archive_parity and not allow_partial:
            errors.append("dashboard parity_ok mismatch with combined archive")
        if dashboard_payload.get("merge_gate_ok") != rebuilt.get("merge_gate_ok"):
            errors.append("dashboard merge_gate_ok mismatch with archive rebuild")
        dash_errors = validate_serving_dashboard_artifact_bundle(
            json_payload=dashboard_payload,
            html_document=html_document,
            markdown_document=markdown_document,
        )
        if allow_partial:
            dash_errors = [
                e
                for e in dash_errors
                if e != "merge_gate_ok must be true when dashboard built from valid archive"
            ]
        errors.extend(f"dashboard.{e}" for e in dash_errors)

    if dashboard_metadata is not None:
        if not dashboard_metadata.get("serving_dashboard_artifact_metadata"):
            errors.append("dashboard_metadata missing serving_dashboard_artifact_metadata marker")
        if dashboard_metadata.get("validation_ok") is not True:
            errors.append("dashboard_metadata validation_ok must be true")
        if dashboard_metadata.get("archive_version") != archive_version:
            errors.append("dashboard_metadata archive_version mismatch with combined archive")
        if dashboard_payload is not None:
            if dashboard_metadata.get("parity_ok") != dashboard_payload.get("parity_ok"):
                errors.append("dashboard_metadata parity_ok mismatch with dashboard json")
            meta_sha = dashboard_metadata.get("json_sha256")
            if isinstance(meta_sha, str) and meta_sha != _archive_sha256(dashboard_payload):
                errors.append("dashboard_metadata json_sha256 mismatch with dashboard json")

    if dashboard_payload is not None and archive_metadata is not None and dashboard_metadata is not None:
        if archive_metadata.get("quick") != dashboard_metadata.get("quick"):
            errors.append("archive_metadata quick mismatch with dashboard_metadata quick")

    return errors


def serving_combined_nightly_bundle_metadata(
    *,
    archive_payload: Mapping[str, Any],
    archive_metadata: Mapping[str, Any],
    dashboard_metadata: Mapping[str, Any],
    validation_ok: bool,
    archive_path: str,
    archive_meta_path: str,
    dashboard_json_path: str,
    dashboard_meta_path: str,
    dashboard_html_path: str = "",
    dashboard_markdown_path: str = "",
) -> Dict[str, Any]:
    return {
        "serving_combined_nightly_bundle_metadata": True,
        "validation_ok": validation_ok,
        "archive_path": archive_path,
        "archive_meta_path": archive_meta_path,
        "dashboard_json_path": dashboard_json_path,
        "dashboard_meta_path": dashboard_meta_path,
        "dashboard_html_path": dashboard_html_path or None,
        "dashboard_markdown_path": dashboard_markdown_path or None,
        "version": archive_payload.get("version"),
        "parity_ok": archive_payload.get("parity_ok"),
        "quick": archive_payload.get("quick"),
        "archive_sha256": archive_metadata.get("archive_sha256"),
        "dashboard_json_sha256": dashboard_metadata.get("json_sha256"),
        "merge_gate_ok": dashboard_metadata.get("merge_gate_ok"),
        "row_count": dashboard_metadata.get("row_count"),
        "paged_multistep_native_parity_ok": archive_metadata.get("paged_multistep_native_parity_ok"),
        "paged_multistep_native_full_layer_parity_ok": archive_metadata.get(
            "paged_multistep_native_full_layer_parity_ok"
        ),
        "created_unix": time.time(),
    }


def validate_serving_combined_nightly_bundle_metadata(payload: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    if not payload.get("serving_combined_nightly_bundle_metadata"):
        errors.append("missing serving_combined_nightly_bundle_metadata marker")
    if payload.get("validation_ok") is not True:
        errors.append("validation_ok must be true")
    if not isinstance(payload.get("archive_sha256"), str):
        errors.append("archive_sha256 must be present")
    if not isinstance(payload.get("dashboard_json_sha256"), str):
        errors.append("dashboard_json_sha256 must be present")
    if "paged_multistep_native_full_layer_parity_ok" not in payload:
        errors.append("paged_multistep_native_full_layer_parity_ok must be present")
    return errors
