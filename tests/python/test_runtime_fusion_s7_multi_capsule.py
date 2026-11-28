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


def test_split_mlp_partial_radix_keeps_hit_rows(serving, torch):
    """Hit rows stay at the MLP input; only miss rows match the fused MLP."""
    layer = serving.TorchDecoderLayer(0, hidden_size=8, intermediate_size=16, seed=5)
    rf = serving.build_split_mlp_runtime_fusion(layer, backend=serving.BACKEND_TORCH)
    torch.manual_seed(3)
    x = torch.randn(3, 8, dtype=torch.float32, device=layer.device)
    mask = [True, False, True]
    meta = serving.pipeline_meta_for_layer(0, {"radix_hit_mask": mask})
    with torch.no_grad():
        result = rf.step({"hidden": x}, meta=meta)
        fused = serving.mlp_torch(
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
    assert result.skipped_radix == []
    hidden = result.outputs["hidden"]
    assert torch.allclose(hidden[0], x[0])
    assert torch.allclose(hidden[2], x[2])
    assert torch.allclose(hidden[1], fused[1], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(hidden[1], x[1])


def test_split_override_partial_budget_falls_back_to_engine_mlp(serving, torch):
    """A budget that finishes only gate_up must not publish that half-step."""
    layer = serving.TorchDecoderLayer(1, hidden_size=8, intermediate_size=16, seed=7)
    override = serving.RuntimeFusionSplitMlpLayerOverride(
        layer, backend=serving.BACKEND_TORCH
    )
    torch.manual_seed(9)
    x = torch.randn(2, 8, dtype=torch.float32, device=layer.device)
    extras = {"total_sms": 16, "reserved_aux_sms": 4}
    with torch.no_grad():
        result = override.forward(
            x,
            rf_meta={"sm_budget": 1, "extras": extras},
        )
        full = layer.forward_engine_full(x)
        post_attn = layer.attention_forward(x)
    assert result.used_rf_mlp is False
    assert result.rf is not None
    assert result.rf.ran == [serving.split_mlp_gate_up_name(1)]
    assert serving.split_mlp_down_name(1) in result.rf.skipped
    assert torch.allclose(result.hidden, full, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(result.hidden, post_attn)


def test_split_override_all_hit_radix_is_post_attention_identity(serving, torch):
    layer = serving.TorchDecoderLayer(0, hidden_size=8, intermediate_size=16, seed=8)
    override = serving.RuntimeFusionSplitMlpLayerOverride(
        layer, backend=serving.BACKEND_TORCH
    )
    torch.manual_seed(4)
    x = torch.randn(2, 8, dtype=torch.float32, device=layer.device)
    with torch.no_grad():
        result = override.forward(x, rf_meta={"radix_hit_mask": [True, True]})
        post_attn = layer.attention_forward(x)
        full = layer.forward_engine_full(x)
    assert result.used_rf_mlp is False
    assert result.rf is not None
    assert result.rf.ran == []
    assert set(result.rf.skipped_radix) == {
        serving.split_mlp_gate_up_name(0),
        serving.split_mlp_down_name(0),
    }
    assert torch.allclose(result.hidden, post_attn)
    assert not torch.allclose(result.hidden, full)


def test_resolve_segment_layer_ids_rejects_bad_ranges(serving):
    with pytest.raises(ValueError, match="invalid segment"):
        serving.resolve_segment_layer_ids(4, layer_start=2, layer_end=2)
    with pytest.raises(ValueError, match="invalid segment"):
        serving.resolve_segment_layer_ids(4, layer_start=3, layer_end=1)
    with pytest.raises(ValueError, match="invalid segment"):
        serving.resolve_segment_layer_ids(4, layer_start=-1, layer_end=2)
    with pytest.raises(ValueError, match="invalid segment"):
        serving.resolve_segment_layer_ids(4, layer_start=1, layer_end=5)
    with pytest.raises(ValueError, match="provide layer_ids"):
        serving.resolve_segment_layer_ids(4)
    with pytest.raises(ValueError, match="out of range"):
        serving.resolve_segment_layer_ids(4, layer_ids=[4])
    with pytest.raises(ValueError, match="out of range"):
        serving.resolve_segment_layer_ids(4, layer_ids=[-1])


def test_resolve_segment_layer_ids_explicit_ids_win(serving):
    assert serving.resolve_segment_layer_ids(4, layer_ids=[2, 0, 2]) == [0, 2]
    assert serving.resolve_segment_layer_ids(4, layer_ids=[]) == []
    # An explicit list is the selection, even when the unused range is illegal.
    assert serving.resolve_segment_layer_ids(
        4, layer_ids=[1], layer_start=9, layer_end=-1
    ) == [1]
    assert serving.resolve_segment_layer_ids(4, layer_start=1, layer_end=3) == [1, 2]
