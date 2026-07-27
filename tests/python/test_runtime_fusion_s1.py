# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S1: FusionPlan / MLP FusionCapsule / RuntimeFusion.step (real torch)."""

from __future__ import annotations

from serving_real_test_utils import serving, torch  # noqa: F401


def test_fusion_plan_mlp_dict_has_standard_identity(serving):
    plan = serving.FusionPlan.mlp(hidden_size=32, intermediate_size=64)
    d = plan.to_dict()
    assert d["kind"] == "mlp"
    assert d["name"] == "mlp_rms_gated_residual"
    assert "mugraph_mlp" in d["legacy_aliases"]
    assert "mirage" not in plan.name.lower()
    assert plan.kind != "megakernel"


def test_mlp_capsule_matches_torch_oracle(serving, torch):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=48, intermediate_size=96, seed=7, backend=serving.BACKEND_TORCH
    )
    x = torch.randn(4, 48, dtype=torch.float32)
    with torch.no_grad():
        out = cap.execute({"hidden": x})["hidden"]
        rms_w, w_g, w_u, w_d = cap.weights()
        ref = serving.mlp_torch(x, rms_weight=rms_w, w_gate=w_g, w_up=w_u, w_down=w_d)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_rf_step_selects_mlp_capsule(serving, torch):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=32, intermediate_size=64, seed=3, backend=serving.BACKEND_TORCH
    )
    rf = serving.RuntimeFusion([cap])
    x = torch.randn(2, 32, dtype=torch.float32)
    result = rf.step({"hidden": x}, meta={"enabled": {cap.name}})
    assert result.ran == [cap.name]
    assert result.skipped == []
    assert tuple(result.outputs["hidden"].shape) == x.shape
    assert not torch.allclose(result.outputs["hidden"], x)


def test_rf_step_can_skip_capsule_identity(serving, torch):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=32, intermediate_size=64, seed=3, backend=serving.BACKEND_TORCH
    )
    rf = serving.RuntimeFusion([cap])
    x = torch.randn(2, 32, dtype=torch.float32)
    result = rf.step({"hidden": x}, meta={"force_skip_all": True})
    assert result.ran == []
    assert result.skipped == [cap.name]
    assert torch.allclose(result.outputs["hidden"], x)


def test_rf_step_disabled_list(serving, torch):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=16, intermediate_size=32, seed=1, name="mlp_a", backend=serving.BACKEND_TORCH
    )
    rf = serving.RuntimeFusion([cap])
    x = torch.zeros(1, 16, dtype=torch.float32)
    result = rf.step({"hidden": x}, meta={"disabled": {"mlp_a"}})
    assert result.skipped == ["mlp_a"]
    assert result.ran == []


def test_runtime_fusion_inspect_lists_capsule(serving):
    cap = serving.MlpFusionCapsule.from_random(hidden_size=8, intermediate_size=16, seed=0)
    rf = serving.RuntimeFusion([cap])
    info = rf.inspect()
    assert info["runtime"] == "RuntimeFusion"
    assert info["version"] == "s10"
    assert info["capsules"][0]["plan"]["backend"] == serving.BACKEND_TORCH
