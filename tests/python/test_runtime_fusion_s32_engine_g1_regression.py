# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S32: vLLM + SGLang G1 engine-cooperative regression (G7 chains C/D)."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving


def test_runtime_fusion_version_s32(serving):
    report = serving.run_engine_g1_regression(quick=True, version="s32")
    assert report.version == "s32"


def test_engine_g1_regression_parity(serving):
    serving.require_torch()
    report = serving.run_engine_g1_regression(quick=True, version="s32")
    assert report.parity_ok
    torch_chains = [c for c in report.chains if c.engine == "torch_surrogate"]
    assert len(torch_chains) == 2
    assert all(c.parity_ok for c in torch_chains)
    ids = {c.chain_id for c in torch_chains}
    assert ids == {"chain_c_vllm_torch", "chain_d_sglang_torch"}


def test_engine_g1_regression_json_contract(serving):
    serving.require_torch()
    report = serving.run_engine_g1_regression(quick=True, version="s32")
    payload = report.to_dict()
    assert payload["serving_engine_g1_regression"] is True
    assert payload["version"] == "s32"
    assert payload["parity_ok"] is True
    assert payload["vllm_hybrid"]["parity_ok"] is True
    assert payload["sglang_hybrid"]["parity_ok"] is True
    errors = serving.validate_serving_engine_g1_regression(payload)
    assert errors == []
    json.dumps(payload)


def test_engine_g1_regression_vllm_chain_uses_paged_kv(serving):
    serving.require_torch()
    report = serving.run_engine_g1_regression(quick=True)
    vllm = report.vllm_hybrid or {}
    assert vllm.get("parity_ok") is True
    assert vllm.get("rf_layer_ids")


def test_engine_g1_regression_sglang_chain_uses_forward_batch_meta(serving):
    serving.require_torch()
    report = serving.run_engine_g1_regression(quick=True)
    sglang = report.sglang_hybrid or {}
    assert sglang.get("parity_ok") is True
    assert sglang.get("rf_layer_ids")


def test_validate_engine_g1_regression_rejects_bad_parity(serving):
    serving.require_torch()
    report = serving.run_engine_g1_regression(quick=True)
    bad = report.to_dict()
    bad["parity_ok"] = False
    errors = serving.validate_serving_engine_g1_regression(bad)
    assert any("parity_ok" in e for e in errors)


def test_cpu_cert_manifest_s32_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s32_contract" in names
