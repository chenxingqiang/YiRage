# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S14: yirage.core MLP capsule full-layer hybrid e2e."""

from __future__ import annotations

import os

import pytest

from serving_real_test_utils import serving, torch  # noqa: F401


pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


@pytest.fixture(scope="module")
def yirage_serving(serving):
    if not serving.is_yirage_core_available():
        pytest.skip("yirage.core not available")
    return serving


def test_hybrid_mlp_backend_yirage_cpu(yirage_serving, torch):
    model = yirage_serving.TorchEngineModel(1, hidden_size=16, intermediate_size=32, seed=1)
    hybrid = yirage_serving.HybridModelOverride(
        model,
        max_rf_mlp_layers=1,
        mlp_backend=yirage_serving.BACKEND_YIRAGE_CPU,
    )
    assert hybrid.mlp_backend == yirage_serving.BACKEND_YIRAGE_CPU
    cap = hybrid.rf.capsules[0]
    assert cap.plan.backend == yirage_serving.BACKEND_YIRAGE_CPU
    x = torch.randn(1, 16, dtype=torch.float32, device=model.device)
    with torch.no_grad():
        ref = model.forward_engine_full(x)
        out = hybrid.forward(x)
    assert out.rf_layer_ids == [0]
    assert torch.allclose(out.hidden, ref, rtol=0.05, atol=0.05)


def test_yirage_core_full_layer_e2e(yirage_serving):
    report = yirage_serving.run_yirage_core_full_layer_e2e(
        num_layers=2,
        hidden_size=16,
        intermediate_size=32,
        batch=1,
        bench=False,
    )
    assert report.parity_ok
    assert report.all_layers_rf
    assert report.yirage_core_used
    assert report.rf_layer_ids == [0, 1]
    assert report.superopt_elapsed_s_total >= 0.0
    assert report.plugin == "HybridModelOverride+YirageCoreMlpCapsule"


def test_yirage_core_full_layer_auto_entry(yirage_serving):
    report = yirage_serving.run_yirage_core_full_layer_e2e_auto(
        num_layers=2,
        hidden_size=8,
        intermediate_size=16,
        bench=False,
    )
    assert report.parity_ok
    assert report.all_layers_rf


def test_rf_inspect_version_s14(yirage_serving):
    assert yirage_serving.RuntimeFusion([]).inspect()["version"] == "s17"
