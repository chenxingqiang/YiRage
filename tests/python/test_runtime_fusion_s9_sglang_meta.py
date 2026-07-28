# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S9: SGLang ForwardBatch-style meta → RF StepMeta bridge (real torch)."""

from __future__ import annotations

import numpy as np

from serving_real_test_utils import serving, torch  # noqa: F401


def test_extend_lens_to_radix_hit_mask(serving):
    mask = serving.radix_hit_mask_from_sglang_extend_lens([0, 4, 0, 2])
    np.testing.assert_array_equal(mask, [True, False, True, False])


def test_build_sglang_rf_step_meta_kv_and_radix(serving):
    meta = serving.build_sglang_rf_step_meta(
        block_tables=[[1, 2, -1], [5, -1, -1]],
        seq_lens=[20, 16],
        extend_lens=[0, 3],
        page_size=16,
        enabled={"mlp_layer_0"},
    )
    np.testing.assert_array_equal(meta["radix_hit_mask"], [True, False])
    assert "paged_kv" in meta["extras"]
    assert meta["extras"]["paged_kv"]["paged_kv_indptr"] == [0, 2, 3]
    assert "sglang" in meta["extras"]
    assert meta["extras"]["sglang"]["extend_lens"] == [0, 3]


def test_rf_step_skips_all_hit_rows_from_sglang_meta(serving, torch):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=8,
        intermediate_size=16,
        seed=3,
        name="mlp_a",
        backend=serving.BACKEND_TORCH,
    )
    rf = serving.RuntimeFusion([cap])
    x = torch.randn(3, 8, dtype=torch.float32)
    meta = serving.build_sglang_rf_step_meta(
        extend_lens=[0, 0, 0],
        enabled={"mlp_a"},
    )
    result = rf.step({"hidden": x}, meta=meta)
    assert result.ran == []
    assert result.skipped_radix == ["mlp_a"]
    assert torch.allclose(result.outputs["hidden"], x)


def test_hybrid_partial_sglang_extend_lens(serving, torch):
    model = serving.TorchEngineModel(1, hidden_size=8, intermediate_size=16, seed=5)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=1)
    x = torch.randn(3, 8, dtype=torch.float32, device=model.device)
    meta = serving.build_sglang_rf_step_meta(extend_lens=[0, 3, 0])
    with torch.no_grad():
        layer = model.layers[0]
        h = layer.attention_forward(x)
        expected = h.clone()
        expected[1] = layer.mlp_forward(h[1:2])[0]
        out = hybrid.forward(x, rf_meta=meta)
    assert out.rf_layer_ids == [0]
    assert torch.allclose(out.hidden, expected, rtol=1e-5, atol=1e-6)


def test_rf_inspect_version_s9(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s18"
