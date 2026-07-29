# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S24: Multi-layer full TB+Ray e2e + AccelForge prescreen reject-path contract."""

from __future__ import annotations

import os

import pytest

from serving_test_utils import serving
from tests.python.conftest import RAY_AVAILABLE


pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


@pytest.fixture(scope="module")
def yirage_serving(serving):
    if not serving.is_yirage_core_available():
        pytest.skip("yirage.core not available")
    serving.require_yirage_core()
    return serving


def test_runtime_fusion_version_s24(yirage_serving):
    assert yirage_serving.RuntimeFusion([]).inspect()["version"] == "s26"


def test_accelforge_latency_budget_env(yirage_serving, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_ACCELFORGE_LATENCY_BUDGET_MS", raising=False)
    assert yirage_serving.resolve_serving_accelforge_latency_budget_ms() is None
    monkeypatch.setenv("YIRAGE_SERVING_ACCELFORGE_LATENCY_BUDGET_MS", "0.5")
    assert yirage_serving.resolve_serving_accelforge_latency_budget_ms() == 0.5


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
def test_accelforge_prescreen_reject_path(yirage_serving, monkeypatch):
    """Tight latency budget rejects coordinator payloads via prescreen bench."""
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    monkeypatch.setenv("YIRAGE_SERVING_USE_COORDINATOR", "1")
    monkeypatch.setenv("YIRAGE_SERVING_RAY_WORKERS", "2")
    monkeypatch.delenv("YIRAGE_SERVING_ACCELFORGE_PRESCREEN", raising=False)
    yirage_serving.apply_serving_kn_down_matmul_tractability(use_ray=True)

    import yirage as yr

    graph = yr.new_kernel_graph()
    mid = graph.new_input(dims=(1, 32), dtype=yr.float32)
    w = graph.new_input(dims=(32, 64), strides=(1, 32), dtype=yr.float32)
    graph.mark_output(graph.matmul(mid, w))

    opt = yirage_serving.superoptimize_down_matmul_via_coordinator(graph, quick=True)
    assert opt is not None

    from yirage.core import cy_to_json
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        cy_to_json(opt.cygraph, path)
        with open(path, "r", encoding="utf-8") as f:
            graph_json = f.read()
    finally:
        os.unlink(path)

    monkeypatch.setenv("YIRAGE_SERVING_ACCELFORGE_LATENCY_BUDGET_MS", "0.000001")
    stats = yirage_serving.bench_serving_accelforge_prescreen(
        [{"graph_json": graph_json}],
        enabled=True,
    )
    assert stats["verifier_available"] is True
    assert stats["rejected_count"] >= 1
    assert stats["accepted_count"] == 0
    assert stats["sample"][0]["rejections"]


@pytest.mark.slow
@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
def test_hf_qwen05b_multilayer_full_tb_ray_e2e(yirage_serving, monkeypatch):
    """Qwen e2e: 2 RF layers with full TB + Ray (quick=False to allow multi-layer)."""
    from yirage.serving.exec_backend import BACKEND_YIRAGE_CPU
    from yirage.serving.hf_qwen_cpu_e2e import (
        DEFAULT_QWEN05B_MODEL,
        is_transformers_available,
        run_hf_qwen05b_cpu_e2e,
    )

    if not is_transformers_available():
        pytest.skip("transformers not installed")

    monkeypatch.setenv("YIRAGE_SERVING_FULL_TB_SEARCH", "1")
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    monkeypatch.setenv("YIRAGE_SERVING_USE_COORDINATOR", "1")
    monkeypatch.setenv("YIRAGE_SERVING_RAY_WORKERS", "2")
    yirage_serving.apply_serving_full_tb_search_tractability(use_ray=True)

    report = run_hf_qwen05b_cpu_e2e(
        model_id=DEFAULT_QWEN05B_MODEL,
        quick=False,
        max_new_tokens=4,
        max_rf_mlp_layers=2,
        mlp_backend=BACKEND_YIRAGE_CPU,
    )
    assert report.serving_search_tier == "full_tb_ray"
    assert report.used_rf_mlp_layers == 2
    assert report.parity_ok is True
    assert report.superopt_elapsed_s_total > 0.0


def test_cpu_cert_manifest_s24_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s24_contract" in names
