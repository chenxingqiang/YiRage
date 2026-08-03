# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S40: Native vLLM tier for paged multistep bench (G7 chain C paged native)."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving, vllm_available


def _synthetic_paged_multistep_native(*, version: str = "s40") -> dict:
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
        "step_parity_ok": ok,
        "step_token_match_ok": ok,
        "engine_token_ids": tok,
        "hybrid_token_ids": tok,
        "vllm_native_available": False,
        "native_parity_ok": None,
        "native_step_parity_ok": [],
    }


def test_runtime_fusion_version_s40(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s45"


def test_paged_multistep_native_fields_contract(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        quick=True,
        try_native=False,
        version="s40",
    )
    payload = report.to_dict()
    assert "vllm_native_available" in payload
    assert "native_parity_ok" in payload
    assert "native_step_parity_ok" in payload
    assert payload["vllm_native_available"] is False
    assert payload["native_parity_ok"] is None
    json.dumps(payload)


def test_paged_multistep_torch_gate_unchanged(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=False,
        version="s40",
    )
    assert report.parity_ok
    assert report.token_match_ok
    assert report.paged_kv_bridged


@pytest.mark.skipif(not vllm_available(), reason="vllm not installed")
def test_paged_multistep_native_tier_when_vllm(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=True,
        version="s40",
    )
    assert report.vllm_native_available is True
    assert report.native_parity_ok is True
    assert len(report.native_step_parity_ok) == 3
    assert all(report.native_step_parity_ok)


def test_combined_metadata_native_paged_fields(serving):
    from yirage.serving.combined_nightly_archive import serving_combined_nightly_archive_metadata
    from test_runtime_fusion_s34_combined_nightly import _synthetic_combined_archive

    payload = _synthetic_combined_archive(version="s40")
    payload["paged_multistep"]["vllm_native_available"] = False
    payload["paged_multistep"]["native_parity_ok"] = None
    meta = serving_combined_nightly_archive_metadata(
        payload, archive_path="artifacts/combined.json", validation_ok=True, quick=True
    )
    assert "paged_multistep_native_parity_ok" in meta


def test_cpu_cert_manifest_s40_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s40_contract" in names
