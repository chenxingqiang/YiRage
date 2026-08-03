# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S41: Serving dashboard artifact validate CLI (JSON + HTML + markdown bundle)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from serving_test_utils import serving
from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard


def test_runtime_fusion_version_s41(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s45"


def test_validate_dashboard_artifact_bundle(serving):
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        render_serving_dashboard_html,
        render_serving_dashboard_markdown,
        validate_serving_dashboard_artifact_bundle,
    )

    report = build_serving_dashboard_from_combined_archive(_synthetic_combined_for_dashboard())
    payload = report.to_dict()
    html_doc = render_serving_dashboard_html(report)
    md_doc = render_serving_dashboard_markdown(report)
    errors = validate_serving_dashboard_artifact_bundle(
        json_payload=payload,
        html_document=html_doc,
        markdown_document=md_doc,
    )
    assert errors == []


def test_dashboard_artifact_metadata(serving):
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        serving_dashboard_artifact_metadata,
    )

    payload = build_serving_dashboard_from_combined_archive(
        _synthetic_combined_for_dashboard()
    ).to_dict()
    meta = serving_dashboard_artifact_metadata(
        payload,
        json_path="artifacts/serving-dashboard.json",
        validation_ok=True,
        html_path="artifacts/serving-dashboard.html",
        markdown_path="artifacts/serving-dashboard.md",
        html_ok=True,
        markdown_ok=True,
    )
    assert meta["serving_dashboard_artifact_metadata"] is True
    assert meta["row_count"] == 5
    assert meta["merge_gate_ok"] is True
    json.dumps(meta)


def test_validate_dashboard_markdown_contract(serving):
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        render_serving_dashboard_markdown,
        validate_serving_dashboard_markdown,
    )

    report = build_serving_dashboard_from_combined_archive(_synthetic_combined_for_dashboard())
    md = render_serving_dashboard_markdown(report)
    assert validate_serving_dashboard_markdown(md) == []


def test_validate_dashboard_script_cli(tmp_path: Path):
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        render_serving_dashboard_html,
        render_serving_dashboard_markdown,
    )

    report = build_serving_dashboard_from_combined_archive(_synthetic_combined_for_dashboard())
    json_path = tmp_path / "dashboard.json"
    html_path = tmp_path / "dashboard.html"
    md_path = tmp_path / "dashboard.md"
    meta_path = tmp_path / "dashboard.meta.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_serving_dashboard_html(report), encoding="utf-8")
    md_path.write_text(render_serving_dashboard_markdown(report), encoding="utf-8")

    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "validate_serving_dashboard.py"),
            str(json_path),
            "--html",
            str(html_path),
            "--markdown",
            str(md_path),
            "--metadata-output",
            str(meta_path),
        ],
        cwd=str(repo),
        env={"PYTHONPATH": "python", "YIRAGE_BACKEND": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["validation_ok"] is True
    assert meta["html_ok"] is True


def test_cpu_cert_manifest_s41_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s41_contract" in names
