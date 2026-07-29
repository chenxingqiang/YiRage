# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S23: Qwen full TB+Ray e2e smoke + AccelForge prescreen bench on coordinator payloads."""

from __future__ import annotations

import os

import pytest

from serving_test_utils import import_serving, serving
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


def test_runtime_fusion_version_s23(yirage_serving):
    assert yirage_serving.RuntimeFusion([]).inspect()["version"] == "s23"


def test_resolve_serving_search_tier_matrix(yirage_serving, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_FULL_TB_SEARCH", raising=False)
    monkeypatch.delenv("YIRAGE_SERVING_USE_RAY", raising=False)
    monkeypatch.delenv("YIRAGE_SERVING_ACCELFORGE_PRESCREEN", raising=False)
    assert yirage_serving.resolve_serving_search_tier() == "seed_verify"

    monkeypatch.setenv("YIRAGE_SERVING_FULL_TB_SEARCH", "1")
    assert yirage_serving.resolve_serving_search_tier() == "full_tb"

    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    monkeypatch.setenv("YIRAGE_SERVING_USE_COORDINATOR", "1")
    assert yirage_serving.resolve_serving_search_tier() == "full_tb_ray"

    monkeypatch.setenv("YIRAGE_SERVING_ACCELFORGE_PRESCREEN", "1")
    assert yirage_serving.resolve_serving_search_tier() == "full_tb_ray_accelforge"


def test_inspect_serving_search_tier_json(yirage_serving, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_FULL_TB_SEARCH", "1")
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    snap = yirage_serving.inspect_serving_search_tier()
    assert snap["tier"] == "full_tb_ray"
    assert snap["full_tb_search"] is True
    assert snap["use_ray"] is True


def test_bench_serving_accelforge_prescreen_disabled(yirage_serving):
    entries = [{"graph_json": "{}"}]
    stats = yirage_serving.bench_serving_accelforge_prescreen(entries, enabled=False)
    assert stats["enabled"] is False
    assert stats["input_count"] == 1
    assert stats["accepted_count"] == 1


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
def test_bench_accelforge_prescreen_on_coordinator_payload(yirage_serving, monkeypatch):
    """AccelForge prescreen bench on real coordinator graph_json payloads (tiny shape)."""
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    monkeypatch.setenv("YIRAGE_SERVING_USE_COORDINATOR", "1")
    monkeypatch.setenv("YIRAGE_SERVING_RAY_WORKERS", "2")
    monkeypatch.setenv("YIRAGE_SERVING_ACCELFORGE_PRESCREEN", "1")
    yirage_serving.apply_serving_kn_down_matmul_tractability(use_ray=True)

    import yirage as yr

    graph = yr.new_kernel_graph()
    mid = graph.new_input(dims=(1, 32), dtype=yr.float32)
    w = graph.new_input(dims=(32, 64), strides=(1, 32), dtype=yr.float32)
    graph.mark_output(graph.matmul(mid, w))

    opt = yirage_serving.superoptimize_down_matmul_via_coordinator(graph, quick=True)
    assert opt is not None

    stats = yirage_serving.last_serving_accelforge_prescreen_stats()
    assert stats is not None
    assert stats["enabled"] is True
    assert stats["input_count"] >= 1
    assert stats["accepted_count"] >= 1


@pytest.mark.slow
@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
def test_hf_qwen05b_full_tb_ray_e2e(yirage_serving, monkeypatch):
    """Qwen2-0.5B e2e with full TB + Ray coordinator down matmul (1 RF layer, quick)."""
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
        quick=True,
        max_rf_mlp_layers=1,
        mlp_backend=BACKEND_YIRAGE_CPU,
    )
    assert report.serving_search_tier == "full_tb_ray"
    assert report.yirage_core_used is True
    assert report.parity_ok is True
    assert report.hidden_size == 896
    assert report.superopt_elapsed_s_total > 0.0


def test_cpu_cert_manifest_s23_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s23_contract" in names
