# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S46: Multi-layer decoder generation parity vs torch and native vLLM (G7 chain C)."""

from __future__ import annotations

import pytest

from serving_test_utils import serving, vllm_available
from test_runtime_fusion_s36_serving_dashboard import _synthetic_combined_for_dashboard


def _synthetic_paged_multistep_generation(*, version: str = "s46") -> dict:
    steps = 3
    ok = [True] * steps
    tok = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    return {
        "serving_vllm_paged_multistep_bench": True,
        "version": version,
        "parity_ok": True,
        "token_match_ok": True,
        "decode_steps": steps,
        "paged_kv_bridged": True,
        "functional_chain": "chain_c_vllm_paged_multistep",
        "num_layers": 2,
        "step_parity_ok": ok,
        "step_token_match_ok": ok,
        "engine_token_ids": tok,
        "hybrid_token_ids": tok,
        "vllm_native_available": False,
        "native_parity_ok": None,
        "native_step_parity_ok": [],
        "native_full_layer_parity_ok": None,
        "native_full_layer_step_parity_ok": [],
        "native_decoder_parity_ok": None,
        "native_decoder_token_match_ok": None,
        "native_decoder_step_parity_ok": [],
        "native_decoder_step_token_match_ok": [],
        "native_decoder_ref_token_ids": [],
        "native_decoder_rf_token_ids": [],
        "generation_parity_ok": True,
        "generation_token_match_ok": True,
        "generation_functional_chain": "chain_c_multilayer_decoder_generation_vs_torch",
        "generation_engine_token_ids": tok,
        "generation_hybrid_token_ids": tok,
        "native_decoder_generation_parity_ok": None,
        "native_decoder_generation_token_match_ok": None,
        "native_decoder_generation_step_parity_ok": [],
        "native_decoder_generation_step_token_match_ok": [],
        "native_decoder_generation_ref_token_ids": [],
        "native_decoder_generation_rf_token_ids": [],
    }


def test_runtime_fusion_version_s46(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s50"


def test_multilayer_torch_generation_parity(serving):
    serving.require_torch()
    decode_steps = 3
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=decode_steps,
        quick=True,
        try_native=False,
        try_native_full_layer=False,
        try_native_decoder=False,
        version="s46",
    )
    assert report.num_layers >= 2
    assert report.generation_parity_ok is True
    assert report.generation_token_match_ok is True
    assert (
        report.generation_functional_chain
        == "chain_c_multilayer_decoder_generation_vs_torch"
    )
    assert len(report.generation_engine_token_ids) == decode_steps
    assert report.generation_engine_token_ids == report.generation_hybrid_token_ids
    assert report.native_decoder_generation_parity_ok is None
    assert report.native_decoder_generation_token_match_ok is None
    assert report.parity_ok
    assert report.token_match_ok
    assert report.paged_kv_bridged
    assert report.functional_chain == "chain_c_vllm_paged_multistep"
    assert serving.validate_serving_vllm_paged_multistep_bench(report.to_dict()) == []


def test_cpu_cert_manifest_s46_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s46_contract" in names


def test_combined_archive_metadata_generation_fields(serving):
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )

    payload = {
        "serving_combined_nightly_archive": True,
        "version": "s46",
        "parity_ok": True,
        "quick": True,
        "decode": {"parity_ok": True},
        "engine_g1": {"parity_ok": True},
        "multistep": {"parity_ok": True, "token_match_ok": True},
        "engine_multistep": {"parity_ok": True},
        "paged_multistep": _synthetic_paged_multistep_generation(),
    }
    meta = serving_combined_nightly_archive_metadata(
        payload, archive_path="artifacts/combined.json", validation_ok=True, quick=True
    )
    assert "paged_multistep_generation_parity_ok" in meta
    assert "paged_multistep_generation_token_match_ok" in meta
    assert "paged_multistep_native_decoder_generation_parity_ok" in meta
    assert "paged_multistep_native_decoder_generation_token_match_ok" in meta
    assert meta["paged_multistep_generation_parity_ok"] is True
    assert meta["paged_multistep_generation_token_match_ok"] is True
    assert meta["paged_multistep_native_decoder_generation_parity_ok"] is None
    assert meta["paged_multistep_native_decoder_generation_token_match_ok"] is None


def test_dashboard_generation_metrics(serving):
    from yirage.serving.serving_dashboard import build_serving_dashboard_from_combined_archive

    archive = _synthetic_combined_for_dashboard(version="s46")
    archive["paged_multistep"]["parity_ok"] = True
    archive["paged_multistep"]["token_match_ok"] = True
    archive["paged_multistep"]["generation_parity_ok"] = True
    archive["paged_multistep"]["generation_token_match_ok"] = True
    archive["paged_multistep"]["native_decoder_generation_parity_ok"] = True
    archive["paged_multistep"]["native_decoder_generation_token_match_ok"] = True
    report = build_serving_dashboard_from_combined_archive(archive)
    paged = [r for r in report.rows if r.section == "paged_multistep"]
    assert len(paged) == 1
    assert paged[0].metrics.get("generation_parity_ok") is True
    assert paged[0].metrics.get("generation_token_match_ok") is True
    assert paged[0].metrics.get("native_decoder_generation_parity_ok") is True
    assert paged[0].metrics.get("native_decoder_generation_token_match_ok") is True


@pytest.mark.skipif(not vllm_available(), reason="vllm not installed")
def test_native_decoder_generation_when_vllm(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native_decoder=True,
        version="s46",
    )
    assert report.native_decoder_generation_parity_ok is True
    assert report.native_decoder_generation_token_match_ok is True
    assert (
        report.native_decoder_generation_ref_token_ids
        == report.native_decoder_generation_rf_token_ids
    )
    assert len(report.native_decoder_generation_step_parity_ok) == 3
    assert len(report.native_decoder_generation_step_token_match_ok) == 3


def test_decoder_generation_quick_tokens_without_native(serving):
    report = serving.run_vllm_paged_multistep_bench(
        quick=True,
        try_native=False,
        try_native_full_layer=False,
        try_native_decoder=False,
    )
    assert report.generation_parity_ok is True
    assert report.generation_token_match_ok is True
    assert report.generation_engine_token_ids == report.generation_hybrid_token_ids
    assert report.native_decoder_generation_parity_ok is None
