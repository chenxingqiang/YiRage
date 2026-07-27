# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S2/S3: MLP Override + first-K hybrid (real torch)."""

from __future__ import annotations

from serving_real_test_utils import serving, torch  # noqa: F401


def test_qwen2_mlp_hf_attach_map(serving):
    assert "mlp.gate_proj.weight" in serving.QWEN2_MLP_HF_ATTACH.values()
    layer = serving.TorchDecoderLayer(0, hidden_size=16, intermediate_size=32, seed=0)
    assert layer.hf_attach["w_gate"] == "layers.0.mlp.gate_proj.weight"


def test_s2_override_attn_engine_mlp_rf_parity(serving, torch):
    layer = serving.TorchDecoderLayer(0, hidden_size=24, intermediate_size=48, seed=2)
    cap = serving.build_layer_mlp_capsule(layer, backend=serving.BACKEND_TORCH)
    rf = serving.RuntimeFusion([cap])
    ov = serving.RuntimeFusionMlpLayerOverride(layer, rf)
    x = torch.randn(3, 24, dtype=torch.float32, device=layer.device)
    with torch.no_grad():
        ref = layer.forward_engine_full(x)
        out = ov.forward(x, rf_meta={"enabled": {cap.name}})
    assert out.used_rf_mlp is True
    assert torch.allclose(out.hidden, ref, rtol=1e-5, atol=1e-5)


def test_s2_override_skip_falls_back_to_engine_mlp(serving, torch):
    layer = serving.TorchDecoderLayer(1, hidden_size=24, intermediate_size=48, seed=2)
    cap = serving.build_layer_mlp_capsule(layer, backend=serving.BACKEND_TORCH)
    rf = serving.RuntimeFusion([cap])
    ov = serving.RuntimeFusionMlpLayerOverride(layer, rf)
    x = torch.randn(2, 24, dtype=torch.float32, device=layer.device)
    with torch.no_grad():
        ref = layer.forward_engine_full(x)
        out = ov.forward(x, rf_meta={"force_skip_all": True})
    assert out.used_rf_mlp is False
    assert torch.allclose(out.hidden, ref, rtol=1e-5, atol=1e-5)


def test_s3_first_k_layers_use_rf(serving, torch):
    model = serving.TorchEngineModel(4, hidden_size=16, intermediate_size=32, seed=5)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    assert hybrid.rf_layer_ids == {0, 1}
    x = torch.randn(2, 16, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        result = hybrid.forward(x)
        ref = model.forward_engine_full(x)
    assert result.rf_layer_ids == [0, 1]
    assert result.engine_mlp_layer_ids == [2, 3]
    assert torch.allclose(result.hidden, ref, rtol=1e-5, atol=1e-5)


def test_s3_explicit_layer_ids_and_k_values(serving, torch):
    model = serving.TorchEngineModel(4, hidden_size=8, intermediate_size=16, seed=0)
    for k in (1, 2, 4):
        hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=k)
        assert hybrid.rf_layer_ids == set(range(k))
    hybrid = serving.HybridModelOverride(model, rf_mlp_layer_ids=[0, 3])
    assert hybrid.rf_layer_ids == {0, 3}
    x = torch.zeros(1, 8, dtype=torch.float32, device=model.device)
    r = hybrid.forward(x)
    assert set(r.rf_layer_ids) == {0, 3}


def test_s3_force_engine_mlp_flag(serving, torch):
    model = serving.TorchEngineModel(3, hidden_size=8, intermediate_size=16, seed=1)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    x = torch.randn(1, 8, dtype=torch.float32, device=model.device)
    r = hybrid.forward(x, force_engine_mlp=True)
    assert r.rf_layer_ids == []
    assert r.engine_mlp_layer_ids == [0, 1, 2]
