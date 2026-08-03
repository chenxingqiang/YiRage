# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S37: vLLM PagedAttention multistep decode bench (G7 chain C paged extension)."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving


def _synthetic_paged_multistep(*, version: str = "s37") -> dict:
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
    }


def test_runtime_fusion_version_s37(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s42"


def test_validate_vllm_paged_multistep_synthetic(serving):
    from yirage.serving.vllm_paged_multistep_bench import (
        validate_serving_vllm_paged_multistep_bench,
    )

    payload = _synthetic_paged_multistep()
    assert validate_serving_vllm_paged_multistep_bench(payload) == []


def test_vllm_paged_multistep_torch_parity(serving):
    serving.require_torch()
    report = serving.run_vllm_paged_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=False,
        version="s37",
    )
    assert report.parity_ok
    assert report.token_match_ok
    assert report.paged_kv_bridged
    assert len(report.step_parity_ok) == 3
    assert len(report.engine_token_ids) == 3
    assert report.engine_token_ids == report.hybrid_token_ids
    json.dumps(report.to_dict())


def test_vllm_paged_multistep_json_contract(serving):
    serving.require_torch()
    payload = serving.run_vllm_paged_multistep_bench(quick=True, try_native=False, version="s37").to_dict()
    errors = serving.validate_serving_vllm_paged_multistep_bench(payload)
    assert errors == []
    assert payload["serving_vllm_paged_multistep_bench"] is True
    assert payload["functional_chain"] == "chain_c_vllm_paged_multistep"


def test_vllm_paged_multistep_archive(serving):
    serving.require_torch()
    payload = serving.run_serving_vllm_paged_multistep_archive(quick=True, version="s37")
    assert payload["parity_ok"] is True
    assert payload["token_match_ok"] is True


def test_cpu_cert_manifest_s37_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s37_contract" in names
