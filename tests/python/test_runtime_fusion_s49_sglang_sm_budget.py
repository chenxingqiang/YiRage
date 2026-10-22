# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S49: SGLang generation alternates SM skip and SM run, tokens still match."""

from __future__ import annotations

from serving_test_utils import serving
from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard


def test_runtime_fusion_version_s49(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s50"


def test_sglang_generation_sm_budget_skip_and_run(serving):
    serving.require_torch()
    report = serving.run_sglang_generation_sm_budget(quick=True)
    validate = serving.validate_sglang_sm_budget_generation
    payload = report.to_dict()
    validate(payload)
    assert payload["version"] == "s49"
    assert payload["functional_chain"] == "chain_d_sglang_generation_sm_budget"
    assert payload["sm_skip_observed"] is True
    assert payload["sm_run_observed"] is True
    assert payload["step_sm_budget"][0] == 0
    assert payload["step_used_rf"][0] is False
    assert payload["step_used_rf"][1] is True
    assert payload["parity_ok"] is True
    assert payload["token_match_ok"] is True
    assert payload["engine_token_ids"] == payload["hybrid_token_ids"]


def test_cpu_cert_manifest_s49_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [stage.name for stage in serving_cpu_cert_manifest(quick=True)]
    assert "s49_contract" in names


def test_archive_metadata_sm_budget_fields(serving):
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )
    from yirage.serving.serving_dashboard import build_serving_dashboard_from_combined_archive

    archive = _synthetic_combined_for_dashboard(version="s49")
    archive["sglang_sm_budget"] = {
        "parity_ok": True,
        "token_match_ok": True,
        "sm_skip_observed": True,
        "sm_run_observed": True,
        "functional_chain": "chain_d_sglang_generation_sm_budget",
        "decode_steps": 2,
        "num_layers": 2,
    }
    meta = serving_combined_nightly_archive_metadata(
        archive,
        archive_path="artifacts/serving-combined-nightly.json",
        validation_ok=True,
    )
    assert meta["sglang_sm_budget_parity_ok"] is True
    assert meta["sglang_sm_budget_token_match_ok"] is True
    assert meta["sglang_sm_budget_skip_observed"] is True
    assert meta["sglang_sm_budget_run_observed"] is True
    dash = build_serving_dashboard_from_combined_archive(archive, allow_partial=True)
    sections = [row.section for row in dash.rows]
    assert "sglang_sm_budget" in sections


def test_sglang_sm_budget_same_seed_is_deterministic(serving):
    serving.require_torch()
    validate = serving.validate_sglang_sm_budget_generation
    first = serving.run_sglang_generation_sm_budget(quick=True, seed=11).to_dict()
    second = serving.run_sglang_generation_sm_budget(quick=True, seed=11).to_dict()
    validate(first)
    validate(second)
    assert first["sm_skip_observed"] is True
    assert first["sm_run_observed"] is True
    assert second["sm_skip_observed"] is True
    assert second["sm_run_observed"] is True
    assert first["engine_token_ids"] == second["engine_token_ids"]
    vocab = int(first["vocab_size"])
    for row in first["engine_token_ids"]:
        for token_id in row:
            assert type(token_id) is int
            assert 0 <= token_id < vocab
