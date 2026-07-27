# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S7 contracts: multi-Capsule pipeline + decoder segment override."""

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


def test_rf_step_runs_pipeline_gate_up_then_down(serving):
    model = serving.EngineModelStub(1, hidden_size=16, intermediate_size=32, seed=4)
    layer = model.layers[0]
    rf = serving.build_split_mlp_runtime_fusion(layer, backend=serving.BACKEND_NUMPY_REF)
    x = np.random.default_rng(5).normal(size=(2, 16)).astype(np.float32)
    meta = serving.pipeline_meta_for_layer(0)
    result = rf.step({"hidden": x}, meta=meta)
    assert result.ran == [
        serving.split_mlp_gate_up_name(0),
        serving.split_mlp_down_name(0),
    ]
    ref = serving.mlp_eager_numpy(
        x,
        rms_weight=layer.rms_weight,
        w_gate=layer.w_gate,
        w_up=layer.w_up,
        w_down=layer.w_down,
    )
    np.testing.assert_allclose(result.outputs["hidden"], ref, rtol=1e-5, atol=1e-6)


def test_split_mlp_parity_oracle(serving):
    model = serving.EngineModelStub(1, hidden_size=12, intermediate_size=24, seed=6)
    layer = model.layers[0]
    x = np.random.default_rng(7).normal(size=(3, 12)).astype(np.float32)
    assert serving.split_mlp_matches_fused(
        x,
        rms_weight=layer.rms_weight,
        w_gate=layer.w_gate,
        w_up=layer.w_up,
        w_down=layer.w_down,
        backend=serving.BACKEND_NUMPY_REF,
    )


def test_decoder_segment_override_matches_engine(serving):
    model = serving.EngineModelStub(4, hidden_size=16, intermediate_size=32, seed=8)
    seg = serving.DecoderSegmentOverride(model, layer_start=1, layer_end=3)
    x = np.random.default_rng(9).normal(size=(2, 16)).astype(np.float32)
    got = seg.forward_segment(x)
    ref = x
    for lid in [1, 2]:
        ref = model.layers[lid].forward_engine_full(ref)
    np.testing.assert_allclose(got.hidden, ref, rtol=1e-5, atol=1e-6)
    assert all(r.used_rf_mlp for r in got.layer_results)
    assert got.capsules_per_step == 2


def test_segment_hybrid_mixed_paths(serving):
    model = serving.EngineModelStub(4, hidden_size=16, intermediate_size=32, seed=10)
    hybrid = serving.SegmentHybridModelOverride(
        model,
        segment_layer_ids=[1, 2],
        rf_mlp_layer_ids=[0],
    )
    x = np.random.default_rng(11).normal(size=(2, 16)).astype(np.float32)
    got = hybrid.forward(x)
    ref = model.forward_engine_full(x)
    np.testing.assert_allclose(got.hidden, ref, rtol=1e-5, atol=1e-6)
    used = {r.layer_id for r in got.layer_results if r.used_rf_mlp}
    assert used == {0, 1, 2}


def test_rf_inspect_version_s7(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s8"
