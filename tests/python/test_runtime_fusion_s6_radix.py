# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S6 contracts: Radix hit meta → skip/shrink FusionCapsule."""

from __future__ import annotations

import numpy as np
import pytest

from serving_test_utils import serving, torch  # noqa: F401


def test_parse_radix_hit_mask_basic(serving):
    radix = serving.parse_radix_hit_mask([True, False, True], batch_size=3)
    assert radix is not None
    assert radix.batch_size == 3
    assert radix.all_hit is False
    assert radix.needs_shrink() is True
    np.testing.assert_array_equal(radix.active_row_indices(), [1])


def test_attach_radix_to_step_meta(serving):
    meta = serving.attach_radix_to_step_meta(
        {"enabled": {"mlp_layer_0"}},
        radix_hit_mask=[False, True],
        batch_size=2,
    )
    assert "radix_hit" in meta["extras"]
    assert meta["extras"]["radix_hit"]["batch_size"] == 2


def test_rf_step_skips_capsule_when_all_radix_hit(serving, torch):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=8, intermediate_size=16, seed=1, name="mlp_a", backend=serving.BACKEND_TORCH
    )
    rf = serving.RuntimeFusion([cap])
    x = torch.randn(2, 8, dtype=torch.float32)
    result = rf.step(
        {"hidden": x},
        meta={"enabled": {"mlp_a"}, "radix_hit_mask": [True, True]},
    )
    assert result.ran == []
    assert result.skipped == ["mlp_a"]
    assert result.skipped_radix == ["mlp_a"]
    assert torch.allclose(result.outputs["hidden"], x)


def test_mlp_capsule_radix_shrink_partial(serving, torch):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=8,
        intermediate_size=16,
        seed=2,
        backend=serving.BACKEND_TORCH,
    )
    x = torch.randn(3, 8, dtype=torch.float32)
    with torch.no_grad():
        full = cap.execute({"hidden": x})["hidden"]
        mask = [True, False, True]
        shrunk = cap.execute({"hidden": x}, meta={"radix_hit_mask": mask})["hidden"]
    assert torch.allclose(shrunk[0], x[0])
    assert torch.allclose(shrunk[2], x[2])
    assert torch.allclose(shrunk[1], full[1])


def test_layer_override_radix_all_hit_identity(serving, torch):
    layer = serving.TorchDecoderLayer(0, hidden_size=8, intermediate_size=16, seed=4)
    cap = serving.build_layer_mlp_capsule(layer, backend=serving.BACKEND_TORCH)
    rf = serving.RuntimeFusion([cap])
    override = serving.RuntimeFusionMlpLayerOverride(layer, rf)
    x = torch.randn(2, 8, dtype=torch.float32, device=layer.device)
    with torch.no_grad():
        post_attn = layer.attention_forward(x)
        result = override.forward(x, rf_meta={"radix_hit_mask": [True, True]})
    assert result.used_rf_mlp is False
    assert result.rf is not None
    assert result.rf.skipped_radix == [cap.name]
    assert torch.allclose(result.hidden, post_attn)


def test_hybrid_partial_radix_shrink_matches_manual(serving, torch):
    model = serving.TorchEngineModel(1, hidden_size=8, intermediate_size=16, seed=6)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=1)
    x = torch.randn(3, 8, dtype=torch.float32, device=model.device)
    mask = [True, False, True]

    layer = model.layers[0]
    with torch.no_grad():
        h = layer.attention_forward(x)
        expected = h.clone()
        expected[1] = layer.mlp_forward(h[1:2])[0]
        out = hybrid.forward(x, rf_meta={"radix_hit_mask": mask})
    assert out.rf_layer_ids == [0]
    assert torch.allclose(out.hidden, expected, rtol=1e-5, atol=1e-6)


def test_parse_radix_hit_mask_coerces_and_rejects(serving):
    assert serving.parse_radix_hit_mask(None) is None
    ints = serving.parse_radix_hit_mask([0, 1, 0])
    np.testing.assert_array_equal(ints.hit_mask, [False, True, False])
    assert ints.active_row_indices().tolist() == [0, 2]
    scalar_miss = serving.parse_radix_hit_mask(0)
    assert scalar_miss.batch_size == 1
    assert scalar_miss.hit_mask.tolist() == [False]
    assert scalar_miss.skip_capsule_entirely() is False
    scalar_hit = serving.parse_radix_hit_mask(True)
    assert scalar_hit.all_hit is True
    with pytest.raises(ValueError, match="rank-1"):
        serving.parse_radix_hit_mask([[True, False]])
    with pytest.raises(ValueError, match="batch_size"):
        serving.parse_radix_hit_mask([True, False], batch_size=3)


def test_radix_mask_length_follows_batch_not_hidden_width(serving, torch):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=8, intermediate_size=16, seed=1, name="mlp_a", backend=serving.BACKEND_TORCH
    )
    rf = serving.RuntimeFusion([cap])
    row = torch.randn(8, dtype=torch.float32)
    with pytest.raises(ValueError, match="batch_size"):
        rf.step({"hidden": row}, meta={"radix_hit_mask": [False] * 8})
    batch = torch.randn(2, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="batch_size"):
        rf.step({"hidden": batch}, meta={"radix_hit_mask": [False] * 8})


def test_all_hit_radix_does_not_consume_sm_budget(serving, torch):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=8, intermediate_size=16, seed=2, name="mlp_a", backend=serving.BACKEND_TORCH
    )
    rf = serving.RuntimeFusion([cap])
    x = torch.randn(2, 8, dtype=torch.float32)
    extras = {"total_sms": 16, "reserved_aux_sms": 4}
    result = rf.step(
        {"hidden": x},
        meta={
            "enabled": {"mlp_a"},
            "radix_hit_mask": [True, True],
            "sm_budget": 4,
            "extras": extras,
        },
    )
    assert result.ran == []
    assert result.skipped_radix == ["mlp_a"]
    assert result.sm_allocation.skipped_budget == []
    assert result.sm_allocation.used_sms == 0
    assert result.sm_allocation.remaining_sms == 4
    assert torch.equal(result.outputs["hidden"], x)


def test_partial_radix_still_charges_capsule_sm_cost(serving, torch):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=8, intermediate_size=16, seed=3, name="mlp_a", backend=serving.BACKEND_TORCH
    )
    rf = serving.RuntimeFusion([cap])
    x = torch.randn(2, 8, dtype=torch.float32)
    result = rf.step(
        {"hidden": x},
        meta={
            "enabled": {"mlp_a"},
            "radix_hit_mask": [True, False],
            "sm_budget": 4,
            "extras": {"total_sms": 16, "reserved_aux_sms": 4},
        },
    )
    cost = serving.capsule_sm_cost(cap)
    assert result.ran == ["mlp_a"]
    assert result.skipped_radix == []
    assert result.sm_allocation.skipped_budget == []
    assert result.sm_allocation.used_sms == cost
    assert result.sm_allocation.remaining_sms == 4 - cost
    assert not torch.equal(result.outputs["hidden"], x)


def test_rf_inspect_version_s6(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s50"


def test_partial_radix_batch4_preserves_shape(serving, torch):
    model = serving.TorchEngineModel(2, hidden_size=8, intermediate_size=16, seed=6)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    x = torch.randn(4, 8, dtype=torch.float32, device=model.device)
    mask = [True, False, True, False]

    with torch.no_grad():
        expected = x
        for layer in model.layers:
            h = layer.attention_forward(expected)
            expected = h.clone()
            for idx, hit in enumerate(mask):
                if not hit:
                    expected[idx] = layer.mlp_forward(h[idx : idx + 1])[0]
        out = hybrid.forward(x, rf_meta={"radix_hit_mask": mask})
    assert out.rf_layer_ids == [0, 1]
    assert tuple(out.hidden.shape) == tuple(x.shape) == (4, 8)
    assert bool(torch.isfinite(out.hidden).all())
    assert torch.allclose(out.hidden, expected, rtol=1e-5, atol=1e-6)
