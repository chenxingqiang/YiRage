# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S35: Engine-native multistep MLP bench (G7 chains C/D multistep extension)."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving


def _synthetic_engine_multistep(*, version: str = "s35") -> dict:
    steps = 3
    ok = [True] * steps
    return {
        "serving_engine_native_multistep_bench": True,
        "version": version,
        "parity_ok": True,
        "decode_steps": steps,
        "functional_chain": "chain_c_d_engine_multistep",
        "vllm_native_available": False,
        "sglang_native_available": False,
        "native_parity_ok": None,
        "chains": [
            {
                "chain_id": "chain_c_vllm_torch_multistep",
                "functional_chain": "chain_c_vllm_plugin",
                "engine": "torch_surrogate",
                "decode_steps": steps,
                "step_parity_ok": ok,
                "parity_ok": True,
            },
            {
                "chain_id": "chain_d_sglang_torch_multistep",
                "functional_chain": "chain_d_sglang_forward_batch",
                "engine": "torch_surrogate",
                "decode_steps": steps,
                "step_parity_ok": ok,
                "parity_ok": True,
            },
        ],
    }


def test_runtime_fusion_version_s35(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s35"


def test_validate_engine_native_multistep_synthetic(serving):
    from yirage.serving.engine_native_multistep_bench import (
        validate_serving_engine_native_multistep_bench,
    )

    payload = _synthetic_engine_multistep()
    assert validate_serving_engine_native_multistep_bench(payload) == []


def test_engine_native_multistep_torch_parity(serving):
    serving.require_torch()
    report = serving.run_engine_native_multistep_bench(
        decode_steps=3,
        quick=True,
        try_native=False,
        version="s35",
    )
    assert report.parity_ok
    torch_chains = [c for c in report.chains if c.engine == "torch_surrogate"]
    assert len(torch_chains) == 2
    assert all(c.parity_ok for c in torch_chains)
    assert all(len(c.step_parity_ok) == 3 for c in torch_chains)
    json.dumps(report.to_dict())


def test_engine_native_multistep_json_contract(serving):
    serving.require_torch()
    report = serving.run_engine_native_multistep_bench(quick=True, try_native=False, version="s35")
    payload = report.to_dict()
    errors = serving.validate_serving_engine_native_multistep_bench(payload)
    assert errors == []
    assert payload["serving_engine_native_multistep_bench"] is True


def test_engine_g1_regression_native_tier_fields(serving):
    serving.require_torch()
    report = serving.run_engine_g1_regression(quick=True, try_native=False, version="s35")
    payload = report.to_dict()
    assert "native_parity_ok" in payload
    assert payload["parity_ok"] is True
    ids = {c.chain_id for c in report.chains}
    assert "chain_c_vllm_torch" in ids
    assert "chain_d_sglang_torch" in ids


def test_cpu_cert_manifest_s35_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s35_contract" in names
