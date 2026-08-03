# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S42: Combined nightly archive + dashboard artifact bundle validation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from serving_test_utils import serving
from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard


def _sha256_payload(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _synthetic_bundle_artifacts(*, version: str = "s44") -> dict:
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        render_serving_dashboard_html,
        render_serving_dashboard_markdown,
        serving_dashboard_artifact_metadata,
    )

    archive = _synthetic_combined_for_dashboard(version=version)
    archive_meta = serving_combined_nightly_archive_metadata(
        archive,
        archive_path="artifacts/serving-combined-nightly.json",
        validation_ok=True,
        quick=True,
    )
    report = build_serving_dashboard_from_combined_archive(archive)
    dashboard = report.to_dict()
    html_doc = render_serving_dashboard_html(report)
    md_doc = render_serving_dashboard_markdown(report)
    dashboard_meta = serving_dashboard_artifact_metadata(
        dashboard,
        json_path="artifacts/serving-dashboard.json",
        validation_ok=True,
        html_path="artifacts/serving-dashboard.html",
        markdown_path="artifacts/serving-dashboard.md",
        html_ok=True,
        markdown_ok=True,
    )
    return {
        "archive": archive,
        "archive_meta": archive_meta,
        "dashboard": dashboard,
        "dashboard_meta": dashboard_meta,
        "html": html_doc,
        "markdown": md_doc,
    }


def test_runtime_fusion_version_s42(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s44"


def test_validate_nightly_bundle_synthetic(serving):
    from yirage.serving.serving_nightly_bundle import (
        validate_serving_combined_nightly_bundle,
    )

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


def test_nightly_bundle_metadata(serving):
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
        dashboard_html_path="artifacts/serving-dashboard.html",
        dashboard_markdown_path="artifacts/serving-dashboard.md",
    )
    assert meta["serving_combined_nightly_bundle_metadata"] is True
    assert meta["validation_ok"] is True
    assert meta["version"] == "s44"
    assert validate_serving_combined_nightly_bundle_metadata(meta) == []
    json.dumps(meta)


def test_validate_nightly_bundle_rejects_mismatch(serving):
    from yirage.serving.serving_nightly_bundle import (
        validate_serving_combined_nightly_bundle,
    )

    bundle = _synthetic_bundle_artifacts(version="s44")
    bad_meta = dict(bundle["archive_meta"])
    bad_meta["version"] = "s99"
    errors = validate_serving_combined_nightly_bundle(
        archive_payload=bundle["archive"],
        archive_metadata=bad_meta,
        dashboard_payload=bundle["dashboard"],
        dashboard_metadata=bundle["dashboard_meta"],
        html_document=bundle["html"],
        markdown_document=bundle["markdown"],
    )
    assert any("archive_metadata version mismatch" in e for e in errors)


def test_validate_nightly_bundle_script_cli(tmp_path: Path):
    from yirage.serving.serving_nightly_bundle import (
        serving_combined_nightly_bundle_metadata,
        validate_serving_combined_nightly_bundle_metadata,
    )

    bundle = _synthetic_bundle_artifacts(version="s44")
    archive_path = tmp_path / "archive.json"
    archive_meta_path = tmp_path / "archive.meta.json"
    dashboard_path = tmp_path / "dashboard.json"
    dashboard_meta_path = tmp_path / "dashboard.meta.json"
    html_path = tmp_path / "dashboard.html"
    md_path = tmp_path / "dashboard.md"
    bundle_meta_path = tmp_path / "bundle.meta.json"

    archive_path.write_text(json.dumps(bundle["archive"], indent=2), encoding="utf-8")
    archive_meta_path.write_text(json.dumps(bundle["archive_meta"], indent=2), encoding="utf-8")
    dashboard_path.write_text(json.dumps(bundle["dashboard"], indent=2), encoding="utf-8")
    dashboard_meta_path.write_text(json.dumps(bundle["dashboard_meta"], indent=2), encoding="utf-8")
    html_path.write_text(bundle["html"], encoding="utf-8")
    md_path.write_text(bundle["markdown"], encoding="utf-8")

    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "validate_serving_combined_nightly_bundle.py"),
            str(archive_path),
            "--archive-meta",
            str(archive_meta_path),
            "--dashboard",
            str(dashboard_path),
            "--dashboard-meta",
            str(dashboard_meta_path),
            "--html",
            str(html_path),
            "--markdown",
            str(md_path),
            "--metadata-output",
            str(bundle_meta_path),
        ],
        cwd=str(repo),
        env={"PYTHONPATH": "python", "YIRAGE_BACKEND": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    meta = json.loads(bundle_meta_path.read_text(encoding="utf-8"))
    assert meta["validation_ok"] is True
    assert validate_serving_combined_nightly_bundle_metadata(meta) == []


def test_cpu_cert_manifest_s42_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s42_contract" in names
