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


def test_sglang_generation_try_native_false_leaves_native_unset(serving):
    serving.require_torch()
    report = serving.run_sglang_multilayer_generation(
        quick=True,
        try_native=False,
        version="s47",
    )
    assert report.parity_ok is True
    assert report.token_match_ok is True
    assert report.engine_token_ids == report.hybrid_token_ids
    assert report.native_generation_parity_ok is None
    assert report.native_generation_token_match_ok is None
    assert set(report.radix_step_kinds) >= {"partial", "miss", "all_hit"}


def test_sglang_generation_first_k_keeps_partial_radix_when_batch_is_one(serving):
    """Quick generation must keep a miss row, and only the first K MLPs fuse.

    ``batch=1`` would collapse the partial step to an all-hit (``extend=[0]``),
    so the quick path floors the batch at 2. ``parity_ok`` also requires
    ``rf_layer_ids == [0]`` on miss/partial steps, which fails if the loop
    ignores ``max_rf_mlp_layers`` and fuses every layer.
    """
    serving.require_torch()
    report = serving.run_sglang_multilayer_generation(
        decode_steps=3,
        num_layers=2,
        max_rf_mlp_layers=1,
        batch=1,
        quick=True,
        try_native=False,
        seed=4,
        version="s47",
    )
    assert report.batch == 2
    assert report.num_layers == 2
    assert report.max_rf_mlp_layers == 1
    assert report.decode_steps == 3
    assert report.radix_step_kinds == ["partial", "miss", "all_hit"]
    assert report.step_parity_ok == [True, True, True]
    assert report.step_token_match_ok == [True, True, True]
    assert report.parity_ok is True
    assert report.token_match_ok is True
    assert report.engine_token_ids == report.hybrid_token_ids
    assert report.native_generation_parity_ok is None
    assert serving.validate_sglang_multilayer_generation(report.to_dict()) == []
    vocab = int(report.vocab_size)
    for row in report.engine_token_ids:
        assert len(row) == report.batch
        for token_id in row:
            assert type(token_id) is int
            assert 0 <= token_id < vocab


def test_sglang_generation_max_rf_clamps_into_layer_range(serving):
    """Non-positive and oversized ``max_rf_mlp_layers`` stay inside ``[1, num_layers]``."""
    serving.require_torch()
    low = serving.run_sglang_multilayer_generation(
        decode_steps=3,
        num_layers=2,
        max_rf_mlp_layers=0,
        quick=True,
        try_native=False,
        seed=5,
        version="s47",
    )
    high = serving.run_sglang_multilayer_generation(
        decode_steps=3,
        num_layers=2,
        max_rf_mlp_layers=9,
        quick=True,
        try_native=False,
        seed=5,
        version="s47",
    )
    assert low.max_rf_mlp_layers == 1
    assert low.num_layers == 2
    assert low.parity_ok is True
    assert low.token_match_ok is True
    assert low.engine_token_ids == low.hybrid_token_ids
    assert high.max_rf_mlp_layers == 2
    assert high.parity_ok is True
    assert high.token_match_ok is True
    assert high.engine_token_ids == high.hybrid_token_ids
    # Same seed, different fused-layer counts: the recorded cap must not collapse
    # to a single value, or the clamp is a no-op that both runs share.
    assert low.max_rf_mlp_layers != high.max_rf_mlp_layers
