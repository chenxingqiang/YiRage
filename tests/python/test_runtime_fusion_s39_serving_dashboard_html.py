# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S39: Serving dashboard HTML rendering."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from serving_test_utils import serving
from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard


def test_runtime_fusion_version_s39(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s42"


def test_dashboard_html_renders(serving):
    from yirage.serving.serving_dashboard import (
        build_serving_dashboard_from_combined_archive,
        render_serving_dashboard_html,
        validate_serving_dashboard_html,
    )

    report = build_serving_dashboard_from_combined_archive(_synthetic_combined_for_dashboard())
    doc = render_serving_dashboard_html(report)
    assert validate_serving_dashboard_html(doc) == []
    assert "chain_c_vllm_paged_multistep" in doc
    assert 'class="badge pass"' in doc
    assert "<table>" in doc


def test_dashboard_html_escapes_metrics(serving):
    from yirage.serving.serving_dashboard import (
        ServingDashboardReport,
        ServingDashboardRow,
        render_serving_dashboard_html,
    )

    report = ServingDashboardReport(
        version="s39",
        archive_version="s39",
        parity_ok=True,
        merge_gate_ok=True,
        quick=True,
        rows=[
            ServingDashboardRow(
                section="decode",
                functional_chain="chain_b",
                parity_ok=True,
                metrics={"note": "<script>alert(1)</script>"},
            )
        ],
    )
    doc = render_serving_dashboard_html(report)
    assert "<script>" not in doc
    assert "&lt;script&gt;" in doc


def test_render_script_html_output(tmp_path: Path):
    archive = _synthetic_combined_for_dashboard()
    archive_path = tmp_path / "combined.json"
    html_path = tmp_path / "dashboard.html"
    archive_path.write_text(json.dumps(archive), encoding="utf-8")

    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "render_serving_dashboard.py"),
            str(archive_path),
            "--html-output",
            str(html_path),
        ],
        cwd=str(repo),
        env={"PYTHONPATH": "python", "YIRAGE_BACKEND": "cpu"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    doc = html_path.read_text(encoding="utf-8")
    assert "Serving Loop Dashboard" in doc
    assert html_path.stat().st_size > 100


def test_cpu_cert_manifest_s39_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s39_contract" in names
