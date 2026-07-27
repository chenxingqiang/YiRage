# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S2/S3 contracts: vLLM-shaped MLP Override + first-K hybrid (no vLLM install)."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _import_serving():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    if "yirage" not in sys.modules or not hasattr(sys.modules["yirage"], "__path__"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub
    for key in list(sys.modules):
        if key == "yirage.serving" or key.startswith("yirage.serving."):
            del sys.modules[key]
    return importlib.import_module("yirage.serving")


@pytest.fixture(scope="module")
def serving():
    return _import_serving()


def test_qwen2_mlp_hf_attach_map(serving):
    assert "mlp.gate_proj.weight" in serving.QWEN2_MLP_HF_ATTACH.values()
    layer = serving.EngineDecoderLayerStub(0, hidden_size=16, intermediate_size=32, seed=0)
    assert layer.hf_attach["w_gate"] == "layers.0.mlp.gate_proj.weight"


def test_s2_override_attn_engine_mlp_rf_parity(serving):
    layer = serving.EngineDecoderLayerStub(0, hidden_size=24, intermediate_size=48, seed=2)
    cap = serving.build_layer_mlp_capsule(layer)
    rf = serving.RuntimeFusion([cap])
    ov = serving.RuntimeFusionMlpLayerOverride(layer, rf)

    x = np.random.default_rng(0).normal(0, 1, size=(3, 24)).astype(np.float32)
    # Engine-only reference path.
    ref = layer.forward_engine_full(x)
    # RF path with capsule enabled (same weights → same math).
    out = ov.forward(x, rf_meta={"enabled": {cap.name}})
    assert out.used_rf_mlp is True
    np.testing.assert_allclose(out.hidden, ref, rtol=1e-5, atol=1e-6)


def test_s2_override_skip_falls_back_to_engine_mlp(serving):
    layer = serving.EngineDecoderLayerStub(1, hidden_size=24, intermediate_size=48, seed=2)
    cap = serving.build_layer_mlp_capsule(layer)
    rf = serving.RuntimeFusion([cap])
    ov = serving.RuntimeFusionMlpLayerOverride(layer, rf)

    x = np.random.default_rng(1).normal(0, 1, size=(2, 24)).astype(np.float32)
    ref = layer.forward_engine_full(x)
    out = ov.forward(x, rf_meta={"force_skip_all": True})
    assert out.used_rf_mlp is False
    np.testing.assert_allclose(out.hidden, ref, rtol=1e-5, atol=1e-6)


def test_s3_first_k_layers_use_rf(serving):
    model = serving.EngineModelStub(4, hidden_size=16, intermediate_size=32, seed=5)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    assert hybrid.rf_layer_ids == {0, 1}

    x = np.random.default_rng(3).normal(0, 1, size=(2, 16)).astype(np.float32)
    result = hybrid.forward(x)
    assert result.rf_layer_ids == [0, 1]
    assert result.engine_mlp_layer_ids == [2, 3]

    # Full engine path for parity on layers that share weights via capsules 0/1.
    # Hybrid with RF on 0,1 should match engine_full because capsules share weights.
    ref = model.forward_engine_full(x)
    np.testing.assert_allclose(result.hidden, ref, rtol=1e-5, atol=1e-6)


def test_s3_explicit_layer_ids_and_k_values(serving):
    model = serving.EngineModelStub(4, hidden_size=8, intermediate_size=16, seed=0)
    for k in (1, 2, 4):
        hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=k)
        assert hybrid.rf_layer_ids == set(range(k))
    hybrid = serving.HybridModelOverride(model, rf_mlp_layer_ids=[0, 3])
    assert hybrid.rf_layer_ids == {0, 3}
    x = np.zeros((1, 8), dtype=np.float32)
    r = hybrid.forward(x)
    assert set(r.rf_layer_ids) == {0, 3}


def test_s3_force_engine_mlp_flag(serving):
    model = serving.EngineModelStub(3, hidden_size=8, intermediate_size=16, seed=1)
    hybrid = serving.HybridModelOverride(model, max_rf_mlp_layers=2)
    x = np.random.default_rng(9).normal(0, 1, size=(1, 8)).astype(np.float32)
    r = hybrid.forward(x, force_engine_mlp=True)
    assert r.rf_layer_ids == []
    assert r.engine_mlp_layer_ids == [0, 1, 2]
