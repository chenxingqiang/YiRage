# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S18: mcPytorch baseline generation archive + SGLang-metax multi-layer fork."""

from __future__ import annotations

import os

import pytest

from serving_test_utils import maca_integration_enabled, serving, torch  # noqa: F401


def test_mcpytorch_baseline_name_constant(serving):
    assert serving.MCPYTORCH_BASELINE_NAME == "mcPytorch_torch_engine"


def test_yirage_maca_generation_mcpytorch_baseline_archive(serving):
    baseline = serving.run_yirage_maca_generation_mcpytorch_baseline_archive(
        num_layers=2,
        hidden_size=16,
        intermediate_size=32,
        decode_steps=3,
        warmup=1,
        iters=4,
        backend=serving.BACKEND_TORCH,
    )
    assert baseline.version == "s20"
    assert baseline.summary.baseline_name == serving.MCPYTORCH_BASELINE_NAME
    assert baseline.summary.parity_ok
    assert baseline.summary.baseline_decode_step_ms > 0
    assert baseline.summary.hybrid_decode_step_ms > 0
    assert baseline.summary.speedup_vs_baseline > 0
    payload = baseline.to_dict()
    assert payload["generation_baseline_archive"] is True
    assert "summary" in payload
    assert payload["summary"]["speedup_vs_baseline"] > 0


def test_sglang_metax_multilayer_fork_auto(serving):
    fork = serving.run_sglang_metax_multilayer_fork_auto(
        layer_ids=[0, 1],
        hidden_size=16,
        intermediate_size=32,
        batch=4,
        bench=False,
    )
    assert fork.parity_ok
    assert fork.layer_ids == [0, 1]
    assert len(fork.layer_reports) == 2
    assert fork.hybrid.parity_ok


@pytest.mark.skipif(
    not maca_integration_enabled(),
    reason="requires sglang on MetaX host or YIRAGE_MACA_INTEGRATION=1",
)
def test_sglang_metax_multilayer_fork_e2e(serving):
    fork = serving.run_sglang_metax_multilayer_fork_e2e(
        layer_ids=[0, 1],
        hidden_size=16,
        intermediate_size=32,
        batch=2,
        bench=False,
    )
    assert fork.fork
    assert fork.parity_ok
    assert all(r.plugin == "SglangMetaxQwen2MlpRfHook" for r in fork.layer_reports)


@pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)
def test_mcpytorch_baseline_with_maca_backend(serving):
    if not serving.is_yirage_maca_available():
        pytest.skip("yirage_maca not available (MetaX VM)")
    baseline = serving.run_yirage_maca_generation_mcpytorch_baseline_archive(
        num_layers=2,
        hidden_size=16,
        intermediate_size=32,
        decode_steps=2,
        warmup=1,
        iters=3,
        backend=serving.BACKEND_YIRAGE_MACA,
    )
    assert baseline.summary.parity_ok
    assert baseline.summary.backend_used == serving.BACKEND_YIRAGE_MACA


def test_rf_inspect_version_s18(serving):
    assert serving.RuntimeFusion([]).inspect()["version"] == "s20"
