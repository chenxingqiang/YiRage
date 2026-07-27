# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S1 contract: FusionPlan / MLP FusionCapsule / RuntimeFusion.step (no yirage.core)."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _import_serving():
    """Load ``yirage.serving`` without executing ``yirage/__init__.py`` (needs core)."""
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "python"
    yirage_dir = pkg_root / "yirage"
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    if "yirage" not in sys.modules or not hasattr(sys.modules["yirage"], "__path__"):
        stub = types.ModuleType("yirage")
        stub.__path__ = [str(yirage_dir)]  # type: ignore[attr-defined]
        sys.modules["yirage"] = stub

    # Drop cached serving modules so edits reload cleanly in long sessions.
    for key in list(sys.modules):
        if key == "yirage.serving" or key.startswith("yirage.serving."):
            del sys.modules[key]

    return importlib.import_module("yirage.serving")


@pytest.fixture(scope="module")
def serving():
    return _import_serving()


def test_fusion_plan_mlp_dict_has_standard_identity(serving):
    plan = serving.FusionPlan.mlp(hidden_size=32, intermediate_size=64)
    d = plan.to_dict()
    assert d["kind"] == "mlp"
    assert d["name"] == "mlp_rms_gated_residual"
    assert d["hidden_size"] == 32
    assert d["intermediate_size"] == 64
    assert "mugraph_mlp" in d["legacy_aliases"]
    # Product narrative must not claim Mirage identity on the plan itself.
    assert "mirage" not in plan.name.lower()
    assert plan.kind != "megakernel"


def test_mlp_capsule_matches_unfused_oracle(serving):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=48, intermediate_size=96, seed=7, backend=serving.BACKEND_NUMPY_REF
    )
    x = np.random.default_rng(1).normal(0, 1, size=(4, 48)).astype(np.float32)
    out = cap.execute({"hidden": x})["hidden"]
    rms_w, w_g, w_u, w_d = cap.weights()
    ref = serving.mlp_eager_numpy(
        x, rms_weight=rms_w, w_gate=w_g, w_up=w_u, w_down=w_d
    )
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_rf_step_selects_mlp_capsule(serving):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=32, intermediate_size=64, seed=3, backend=serving.BACKEND_NUMPY_REF
    )
    rf = serving.RuntimeFusion([cap])
    x = np.random.default_rng(0).normal(0, 1, size=(2, 32)).astype(np.float32)
    result = rf.step({"hidden": x}, meta={"enabled": {cap.name}})
    assert result.ran == [cap.name]
    assert result.skipped == []
    assert result.outputs["hidden"].shape == x.shape
    assert not np.allclose(result.outputs["hidden"], x)


def test_rf_step_can_skip_capsule_identity(serving):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=32, intermediate_size=64, seed=3, backend=serving.BACKEND_NUMPY_REF
    )
    rf = serving.RuntimeFusion([cap])
    x = np.random.default_rng(0).normal(0, 1, size=(2, 32)).astype(np.float32)
    result = rf.step({"hidden": x}, meta={"force_skip_all": True})
    assert result.ran == []
    assert result.skipped == [cap.name]
    np.testing.assert_array_equal(result.outputs["hidden"], x)


def test_rf_step_disabled_list(serving):
    cap = serving.MlpFusionCapsule.from_random(
        hidden_size=16, intermediate_size=32, seed=1, name="mlp_a", backend=serving.BACKEND_NUMPY_REF
    )
    rf = serving.RuntimeFusion([cap])
    x = np.zeros((1, 16), dtype=np.float32)
    result = rf.step({"hidden": x}, meta={"disabled": {"mlp_a"}})
    assert result.skipped == ["mlp_a"]
    assert result.ran == []


def test_runtime_fusion_inspect_lists_capsule(serving):
    cap = serving.MlpFusionCapsule.from_random(hidden_size=8, intermediate_size=16, seed=0)
    rf = serving.RuntimeFusion([cap])
    info = rf.inspect()
    assert info["runtime"] == "RuntimeFusion"
    assert info["version"] in {"s1", "s2", "s3", "s4", "s5"}
    assert info["capsules"][0]["kind"] == "mlp"
    assert info["capsules"][0]["plan"]["kind"] == "mlp"


def test_serving_public_exports(serving):
    for name in (
        "FusionPlan",
        "FusionCapsule",
        "MlpFusionCapsule",
        "RuntimeFusion",
        "StepMeta",
        "StepResult",
        "mlp_eager_numpy",
    ):
        assert hasattr(serving, name)
