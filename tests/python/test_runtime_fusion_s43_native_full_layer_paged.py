# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S43: Full-layer native vLLM paged multistep tier (G7 chain C paged native full)."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving, vllm_available


def _synthetic_paged_multistep_full_layer(*, version: str = "s43") -> dict:
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
    }


def test_runtime_fusion_version_s43(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s43"


def test_paged_multistep_full_layer_fields_contract(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        quick=True,
        try_native=False,
        try_native_full_layer=False,
        version="s43",
    )
    payload = report.to_dict()
    assert "native_full_layer_parity_ok" in payload
    assert "native_full_layer_step_parity_ok" in payload
    assert payload["native_full_layer_parity_ok"] is None
    assert payload["native_full_layer_step_parity_ok"] == []
    json.dumps(payload)


def test_paged_multistep_torch_gate_unchanged_s43(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=False,
        try_native_full_layer=False,
        version="s43",
    )
    assert report.parity_ok
    assert report.token_match_ok
    assert report.paged_kv_bridged


@pytest.mark.skipif(not vllm_available(), reason="vllm not installed")
def test_paged_multistep_native_full_layer_when_vllm(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=True,
        try_native_full_layer=True,
        version="s43",
    )
    assert report.vllm_native_available is True
    assert report.native_parity_ok is True
    assert report.native_full_layer_parity_ok is True
    assert len(report.native_full_layer_step_parity_ok) == 3


def test_combined_archive_metadata_full_layer_field(serving):
    from yirage.serving.combined_nightly_archive import (
        serving_combined_nightly_archive_metadata,
    )

    payload = {
        "serving_combined_nightly_archive": True,
        "version": "s43",
        "parity_ok": True,
        "quick": True,
        "decode": {"parity_ok": True},
        "engine_g1": {"parity_ok": True},
        "multistep": {"parity_ok": True, "token_match_ok": True},
        "engine_multistep": {"parity_ok": True},
        "paged_multistep": _synthetic_paged_multistep_full_layer(),
    }
    meta = serving_combined_nightly_archive_metadata(
        payload, archive_path="artifacts/combined.json", validation_ok=True, quick=True
    )
    assert "paged_multistep_native_full_layer_parity_ok" in meta
    assert meta["paged_multistep_native_full_layer_parity_ok"] is None


def test_cpu_cert_manifest_s43_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s43_contract" in names
