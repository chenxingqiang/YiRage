# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S47: Multi-layer SGLang ForwardBatch generation parity (G7 chain D)."""

from __future__ import annotations

from serving_test_utils import serving
from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard


def _synthetic_combined_with_sglang_generation() -> dict:
    archive = _synthetic_combined_for_dashboard(version="s47")
    archive["sglang_generation"] = {
        "parity_ok": True,
        "token_match_ok": True,
    }
    return archive


def test_runtime_fusion_version_s47(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s50"


def test_sglang_multilayer_generation_parity(serving):
    serving.require_torch()
    decode_steps = 3
    report = serving.run_sglang_multilayer_generation(
        decode_steps=decode_steps,
        quick=True,
        version="s47",
    )
    assert report.num_layers >= 2
    assert report.max_rf_mlp_layers == report.num_layers
    assert report.parity_ok
    assert report.token_match_ok
    assert report.functional_chain == "chain_d_sglang_forward_batch_generation"
    assert report.engine_token_ids == report.hybrid_token_ids
    assert len(report.engine_token_ids) == decode_steps
    assert set(report.radix_step_kinds) >= {"partial", "miss", "all_hit"}
    assert report.native_generation_parity_ok is (True if serving.is_sglang_available() else None)
    assert serving.validate_sglang_multilayer_generation(report.to_dict()) == []


def test_cpu_cert_manifest_s47_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s47_contract" in names


def test_archive_metadata_sglang_generation_fields(serving):
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )

    archive = _synthetic_combined_with_sglang_generation()
    meta = serving_combined_nightly_archive_metadata(
        archive,
        archive_path="artifacts/combined.json",
        validation_ok=True,
        quick=True,
    )
    assert meta["sglang_generation_parity_ok"] is True
    assert meta["sglang_generation_token_match_ok"] is True


def test_dashboard_sglang_generation_row(serving):
    from yirage.serving.serving_dashboard import build_serving_dashboard_from_combined_archive

    archive = _synthetic_combined_with_sglang_generation()
    report = build_serving_dashboard_from_combined_archive(archive)
    rows = [row for row in report.rows if row.section == "sglang_generation"]
    assert len(rows) == 1
    assert rows[0].parity_ok is True
    assert rows[0].metrics.get("token_match_ok") is True
