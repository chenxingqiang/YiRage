# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S38: Wire vLLM paged multistep into combined archive + serving dashboard."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving
from test_runtime_fusion_s34_combined_nightly import _synthetic_combined_archive


def test_runtime_fusion_version_s38(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s43"


def test_combined_archive_includes_paged_multistep_subsection(serving):
    from yirage.serving.combined_nightly_archive import validate_serving_combined_nightly_archive

    payload = _synthetic_combined_archive(version="s38")
    assert "paged_multistep" in payload
    assert payload["paged_multistep"]["token_match_ok"] is True
    assert validate_serving_combined_nightly_archive(payload) == []


def test_combined_archive_paged_multistep_torch_live(serving):
    serving.require_torch()
    payload = serving.run_serving_vllm_paged_multistep_archive(quick=True, version="s38")
    errors = serving.validate_serving_vllm_paged_multistep_bench(payload)
    assert errors == []
    assert payload["functional_chain"] == "chain_c_vllm_paged_multistep"


def test_dashboard_includes_paged_multistep_row(serving):
    from yirage.serving.serving_dashboard import build_serving_dashboard_from_combined_archive

    report = build_serving_dashboard_from_combined_archive(_synthetic_combined_archive(version="s38"))
    sections = {r.section for r in report.rows}
    assert "paged_multistep" in sections
    paged_rows = [r for r in report.rows if r.section == "paged_multistep"]
    assert len(paged_rows) == 1
    assert paged_rows[0].parity_ok is True
    assert paged_rows[0].functional_chain == "chain_c_vllm_paged_multistep"


def test_combined_metadata_paged_multistep_fields(serving):
    from yirage.serving.combined_nightly_archive import serving_combined_nightly_archive_metadata

    payload = _synthetic_combined_archive(version="s38")
    meta = serving_combined_nightly_archive_metadata(
        payload, archive_path="artifacts/combined.json", validation_ok=True, quick=True
    )
    assert meta["paged_multistep_parity_ok"] is True
    assert meta["paged_multistep_token_match_ok"] is True
    json.dumps(meta)


def test_cpu_cert_manifest_s38_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s38_contract" in names
