# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S5 contracts: SM worker quota + RF.step co-residence with aux streams."""

from __future__ import annotations

import pytest

from serving_test_utils import serving, torch  # noqa: F401


def test_resolve_sm_worker_quota_defaults(serving):
    q = serving.resolve_sm_worker_quota()
    assert q.total_sms > 0
    assert q.reserved_aux_sms > 0
    assert q.capsule_budget_sms == q.total_sms - q.reserved_aux_sms
    assert q.capsule_budget_sms > 0


def test_resolve_sm_budget_explicit_override(serving):
    q = serving.resolve_sm_worker_quota(total_sms=64, reserved_aux_sms=8, sm_budget=40)
    assert q.total_sms == 64
    assert q.reserved_aux_sms == 8
    assert q.capsule_budget_sms == 40


def test_resolve_sm_budget_rejects_eating_aux(serving):
    with pytest.raises(ValueError, match="aux"):
        serving.resolve_sm_worker_quota(total_sms=16, reserved_aux_sms=8, sm_budget=12)


def test_capsule_sm_cost_from_plan_extras(serving):
    plan = serving.FusionPlan(
        name="mlp_costly",
        kind="mlp",
        hidden_size=8,
        intermediate_size=16,
        extras={"sm_cost": 4},
    )
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=8, intermediate_size=16, seed=0, name="mlp_costly", plan=plan
    )
    assert serving.capsule_sm_cost(cap) == 4


def test_rf_step_skips_when_sm_budget_exhausted(serving, torch):
    def _cap(name: str, cost: int):
        plan = serving.FusionPlan(
            name=name,
            kind="mlp",
            hidden_size=8,
            intermediate_size=16,
            extras={"sm_cost": cost},
        )
        return serving.MlpFusionCapsule.from_random(
            hidden_size=8,
            intermediate_size=16,
            seed=hash(name) % 10_000,
            name=name,
            plan=plan,
            backend=serving.BACKEND_TORCH,
        )

    rf = serving.RuntimeFusion([_cap("mlp_a", 3), _cap("mlp_b", 3)])
    x = torch.zeros(1, 8, dtype=torch.float32)
    result = rf.step(
        {"hidden": x},
        meta={"sm_budget": 4, "extras": {"total_sms": 16, "reserved_aux_sms": 4}},
    )
    assert result.ran == ["mlp_a"]
    assert "mlp_b" in result.skipped
    assert result.sm_allocation is not None
    assert result.sm_allocation.skipped_budget == ["mlp_b"]
    assert result.sm_allocation.remaining_sms == 1


def test_rf_step_aux_coresidence_never_consumed(serving, torch):
    plan = serving.FusionPlan(
        name="mlp_a",
        kind="mlp",
        hidden_size=8,
        intermediate_size=16,
        extras={"sm_cost": 2},
    )
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=8,
        intermediate_size=16,
        seed=0,
        name="mlp_a",
        plan=plan,
        backend=serving.BACKEND_TORCH,
    )
    rf = serving.RuntimeFusion([cap])
    x = torch.zeros(1, 8, dtype=torch.float32)
    result = rf.step(
        {"hidden": x},
        meta={"extras": {"total_sms": 32, "reserved_aux_sms": 8}},
    )
    assert result.ran == ["mlp_a"]
    alloc = result.sm_allocation
    assert alloc is not None
    serving.assert_aux_coresidence(alloc)
    assert alloc.quota.reserved_aux_sms == 8
    assert alloc.used_sms + alloc.remaining_sms == alloc.quota.capsule_budget_sms


def test_hybrid_forward_respects_sm_budget_skip(serving, torch):
    model = serving.TorchEngineModel(2, hidden_size=8, intermediate_size=16, seed=1)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    x = torch.randn(2, 8, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        out = hybrid.forward(
            x,
            rf_meta={"sm_budget": 0, "extras": {"total_sms": 8, "reserved_aux_sms": 2}},
        )
        ref = model.forward_engine_full(x)
    assert out.rf_layer_ids == []
    assert out.engine_mlp_layer_ids == [0, 1]
    assert torch.allclose(out.hidden, ref, rtol=1e-5, atol=1e-5)


def test_serving_cpu_cert_manifest_includes_s5(serving):
    names = [s.name for s in serving.serving_cpu_cert_manifest(quick=True)]
    assert "s5_contract" in names
    assert "torch_e2e" in names
    assert "s8_contract" in names
