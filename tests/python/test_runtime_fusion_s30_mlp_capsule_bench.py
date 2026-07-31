# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S30: MLP FusionCapsule micro-bench — G7 chain A parity + timing."""

from __future__ import annotations

import json

import pytest

from serving_test_utils import serving


def test_runtime_fusion_version_s30(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s30"


def test_mlp_capsule_bench_json_contract(serving):
    serving.require_torch()
    report = serving.run_mlp_capsule_bench(quick=True, version="s30")
    payload = report.to_dict()
    assert payload["serving_mlp_capsule_bench"] is True
    assert payload["version"] == "s30"
    assert payload["functional_chain"] == "chain_a_mlp_capsule_min"
    assert payload["parity_ok"] is True
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["name"] == "eager_mlp_torch"
    assert payload["rows"][1]["name"] == "rf_step_mlp_capsule"
    assert payload["speedup_rf_vs_eager"] > 0
    json.dumps(payload)


def test_mlp_capsule_bench_parity_before_timing(serving):
    """G5/G7: RF.step output must match eager mlp_torch before bench rows are trusted."""
    serving.require_torch()
    import torch

    report = serving.run_mlp_capsule_bench(
        hidden_size=32,
        intermediate_size=64,
        batch=4,
        quick=True,
        version="s30",
    )
    assert report.parity_ok
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=32, intermediate_size=64, seed=0, backend=serving.BACKEND_TORCH
    )
    rf = serving.RuntimeFusion([cap])
    x = torch.randn(4, 32, dtype=torch.float32)
    rms_w, w_g, w_u, w_d = cap.weights()
    with torch.no_grad():
        ref = serving.mlp_torch(x, rms_weight=rms_w, w_gate=w_g, w_up=w_u, w_down=w_d)
        step = rf.step({"hidden": x}, meta={"enabled": {cap.name}})
    assert torch.allclose(step.outputs["hidden"], ref, rtol=1e-5, atol=1e-5)


def test_cpu_cert_manifest_s30_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s30_contract" in names


def test_g7_chain_a_documented_in_report(serving):
    report = serving.run_mlp_capsule_bench(quick=True)
    assert report.functional_chain == "chain_a_mlp_capsule_min"
    assert report.to_dict()["parity_ok"] is True
