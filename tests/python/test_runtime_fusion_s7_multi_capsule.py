# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S7 contracts: multi-Capsule pipeline + decoder segment override."""

from __future__ import annotations

import pytest

from serving_test_utils import serving, torch  # noqa: F401


def test_split_mlp_pipeline_names(serving):
    gate, down = serving.split_mlp_pipeline_names(3)
    assert gate == "mlp_layer_3_gate_up"
    assert down == "mlp_layer_3_down"


def test_resolve_capsule_pipeline_explicit_order(serving):
    a = serving.MlpFusionCapsule.from_random(hidden_size=8, intermediate_size=16, seed=1, name="a")
    b = serving.MlpFusionCapsule.from_random(hidden_size=8, intermediate_size=16, seed=2, name="b")
    rf = serving.RuntimeFusion([a, b])
    meta = serving.StepMeta.from_mapping({"pipeline": ["b", "a"], "enabled": {"a", "b"}})
    ordered = serving.resolve_capsule_pipeline(rf.capsules, meta)
    assert [c.name for c in ordered] == ["b", "a"]


def _two_capsules(serving):
    a = serving.MlpFusionCapsule.from_random(
        hidden_size=8, intermediate_size=16, seed=1, name="a", backend=serving.BACKEND_TORCH
    )
    b = serving.MlpFusionCapsule.from_random(
        hidden_size=8, intermediate_size=16, seed=2, name="b", backend=serving.BACKEND_TORCH
    )
    return a, b


def test_resolve_capsule_pipeline_extras_and_unknown(serving):
    a, b = _two_capsules(serving)
    rf = serving.RuntimeFusion([a, b])
    from_extras = serving.StepMeta.from_mapping(
        {"extras": {"capsule_pipeline": ["b", "a"]}}
    )
    assert [c.name for c in serving.resolve_capsule_pipeline(rf.capsules, from_extras)] == [
        "b",
        "a",
    ]
    explicit_wins = serving.StepMeta.from_mapping(
        {
            "pipeline": ["a", "b"],
            "extras": {"capsule_pipeline": ["b", "a"]},
        }
    )
    assert [c.name for c in serving.resolve_capsule_pipeline(rf.capsules, explicit_wins)] == [
        "a",
        "b",
    ]
    with pytest.raises(KeyError, match="unknown capsule"):
        serving.resolve_capsule_pipeline(
            rf.capsules,
            serving.StepMeta.from_mapping({"pipeline": ["missing"]}),
        )


def test_rf_step_pipeline_subset_and_enabled_filter(serving, torch):
    a, b = _two_capsules(serving)
    rf = serving.RuntimeFusion([a, b])
    x = torch.randn(2, 8, dtype=torch.float32)
    only_b = rf.step(
        {"hidden": x},
        meta={"pipeline": ["b"], "enabled": {"a", "b"}},
    )
    assert only_b.ran == ["b"]
    assert only_b.skipped == []
    with torch.no_grad():
        expected_b = b.execute({"hidden": x})["hidden"]
    assert torch.allclose(only_b.outputs["hidden"], expected_b, rtol=1e-5, atol=1e-5)

    only_a = rf.step(
        {"hidden": x},
        meta={"pipeline": ["b", "a"], "enabled": {"a"}},
    )
    assert only_a.ran == ["a"]
    assert only_a.skipped == ["b"]
    with torch.no_grad():
        expected_a = a.execute({"hidden": x})["hidden"]
    assert torch.allclose(only_a.outputs["hidden"], expected_a, rtol=1e-5, atol=1e-5)


def test_rf_step_runs_pipeline_gate_up_then_down(serving, torch):
    layer = serving.TorchDecoderLayer(0, hidden_size=16, intermediate_size=32, seed=4)
    rf = serving.build_split_mlp_runtime_fusion(layer, backend=serving.BACKEND_TORCH)
    x = torch.randn(2, 16, dtype=torch.float32, device=layer.device)
    meta = serving.pipeline_meta_for_layer(0)
    with torch.no_grad():
        result = rf.step({"hidden": x}, meta=meta)
        ref = serving.mlp_torch(
            x,
            rms_weight=layer.rms_weight,
            w_gate=layer.w_gate,
            w_up=layer.w_up,
            w_down=layer.w_down,
        )
    assert result.ran == [
        serving.split_mlp_gate_up_name(0),
        serving.split_mlp_down_name(0),
    ]
    assert torch.allclose(result.outputs["hidden"], ref, rtol=1e-5, atol=1e-6)


def test_split_mlp_parity_oracle(serving, torch):
    layer = serving.TorchDecoderLayer(0, hidden_size=12, intermediate_size=24, seed=6)
    x = torch.randn(3, 12, dtype=torch.float32, device=layer.device)
    assert serving.split_mlp_matches_fused(
        x,
        rms_weight=layer.rms_weight,
        w_gate=layer.w_gate,
        w_up=layer.w_up,
        w_down=layer.w_down,
        backend=serving.BACKEND_TORCH,
    )


def test_decoder_segment_override_matches_engine(serving, torch):
    model = serving.TorchEngineModel(4, hidden_size=16, intermediate_size=32, seed=8)
    seg = serving.DecoderSegmentOverride(
        model, layer_start=1, layer_end=3, backend=serving.BACKEND_TORCH
    )
    x = torch.randn(2, 16, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        got = seg.forward_segment(x)
        ref = x
        for lid in [1, 2]:
            ref = model.layers[lid].forward_engine_full(ref)
    assert torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6)
    assert all(r.used_rf_mlp for r in got.layer_results)
    assert got.capsules_per_step == 2


def test_segment_hybrid_mixed_paths(serving, torch):
    model = serving.TorchEngineModel(4, hidden_size=16, intermediate_size=32, seed=10)
    hybrid = serving.TorchSegmentHybridModelOverride(
        model,
        segment_layer_ids=[1, 2],
        rf_mlp_layer_ids=[0],
    )
    x = torch.randn(2, 16, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        got = hybrid.forward(x)
        ref = model.forward_engine_full(x)
    assert torch.allclose(got.hidden, ref, rtol=1e-5, atol=1e-6)
    used = {r.layer_id for r in got.layer_results if r.used_rf_mlp}
    assert used == {0, 1, 2}


def test_rf_inspect_version_s7(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s50"


def test_pipeline_sm_budget_zero_skips_both(serving, torch):
    layer = serving.TorchDecoderLayer(0, hidden_size=16, intermediate_size=32, seed=4)
    rf = serving.build_split_mlp_runtime_fusion(layer, backend=serving.BACKEND_TORCH)
    torch.manual_seed(11)
    x = torch.randn(2, 16, dtype=torch.float32, device=layer.device)
    assert torch.count_nonzero(x) > 0
    names = [
        serving.split_mlp_gate_up_name(0),
        serving.split_mlp_down_name(0),
    ]
    extras = {"total_sms": 16, "reserved_aux_sms": 4}
    with torch.no_grad():
        skipped = rf.step(
            {"hidden": x},
            meta=serving.pipeline_meta_for_layer(0, {"sm_budget": 0, "extras": extras}),
        )
    assert skipped.skipped == names
    assert skipped.ran == []
    # Skipped capsules are not applied: hidden stays the step input.
    assert torch.equal(skipped.outputs["hidden"], x)

    with torch.no_grad():
        ran = rf.step(
            {"hidden": x},
            meta=serving.pipeline_meta_for_layer(0, {"sm_budget": 8, "extras": extras}),
        )
    assert ran.ran == names
    hidden = ran.outputs["hidden"]
    assert bool(torch.isfinite(hidden).all())
    assert not torch.equal(hidden, x)
