# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S48: Native SGLang MLP generation parity on the multilayer ForwardBatch path."""

from __future__ import annotations

import pytest

from serving_test_utils import serving, sglang_available
from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard


def _run_sglang_native_generation(serving):
    serving.require_torch()
    return serving.run_sglang_multilayer_generation(
        decode_steps=3,
        quick=True,
        try_native=True,
        version="s48",
    )


def test_runtime_fusion_version_s48(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s49"


def test_sglang_native_generation_absent_without_sglang(serving):
    serving.require_torch()
    if sglang_available():
        pytest.skip("sglang installed")
    report = serving.run_sglang_multilayer_generation(
        decode_steps=3,
        quick=True,
        try_native=True,
        version="s48",
    )
    assert report.parity_ok
    assert report.token_match_ok
    assert report.num_layers >= 2
    assert report.native_generation_parity_ok is None
    assert report.native_generation_token_match_ok is None
    assert report.native_generation_step_parity_ok == []


def test_cpu_cert_manifest_s48_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [stage.name for stage in serving_cpu_cert_manifest(quick=True)]
    assert "s48_contract" in names


def test_archive_metadata_native_generation_fields(serving):
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )

    archive = _synthetic_combined_for_dashboard(version="s48")
    archive["sglang_generation"] = {
        "parity_ok": True,
        "token_match_ok": True,
        "native_generation_parity_ok": None,
        "native_generation_token_match_ok": None,
    }
    meta = serving_combined_nightly_archive_metadata(
        archive,
        archive_path="artifacts/combined.json",
        validation_ok=True,
        quick=True,
    )
    assert meta["sglang_native_generation_parity_ok"] is None
    assert meta["sglang_native_generation_token_match_ok"] is None


@pytest.mark.skipif(not sglang_available(), reason="sglang not installed")
def test_sglang_native_mlp_generation_when_installed(serving):
    report = _run_sglang_native_generation(serving)
    assert report.native_generation_parity_ok is True
    assert report.native_generation_token_match_ok is True
    assert report.native_generation_ref_token_ids == report.native_generation_rf_token_ids
    assert len(report.native_generation_step_parity_ok) == 3
