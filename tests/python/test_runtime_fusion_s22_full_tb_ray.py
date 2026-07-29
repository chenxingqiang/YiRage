# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S22: Full TB + Ray coordinator combo for serving down matmul (Qwen-scale)."""

from __future__ import annotations

import os

import pytest

from serving_test_utils import serving


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


def test_runtime_fusion_version_s22(yirage_serving):
    assert yirage_serving.RuntimeFusion([]).inspect()["version"] == "s23"


def test_snapshot_serving_env_full_tb(yirage_serving, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_FULL_TB_SEARCH", "1")
    monkeypatch.delenv("YIRAGE_SERVING_KN_MATMUL_ONLY", raising=False)
    yirage_serving.apply_serving_full_tb_search_tractability(use_ray=True)
    env = yirage_serving.snapshot_serving_env()
    assert env.get("YIRAGE_SERVING_FULL_TB_SEARCH") == "1"
    assert env.get("YIRAGE_SERVING_KN_MATMUL_ONLY") is None
    assert env.get("YIRAGE_CPU_BENCH_MINIMAL_EXPLORE") == "1"
    assert env.get("YIRAGE_SERVING_USE_RAY") == "1"


def test_apply_serving_env_roundtrip(yirage_serving, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_FULL_TB_SEARCH", raising=False)
    monkeypatch.setenv("YIRAGE_SERVING_KN_MATMUL_ONLY", "1")
    snap = yirage_serving.snapshot_serving_env()
    monkeypatch.delenv("YIRAGE_SERVING_KN_MATMUL_ONLY", raising=False)
    monkeypatch.delenv("YIRAGE_SERVING_FULL_TB_SEARCH", raising=False)
    yirage_serving.apply_serving_env(snap)
    assert os.environ.get("YIRAGE_SERVING_KN_MATMUL_ONLY") == "1"
    assert "YIRAGE_SERVING_FULL_TB_SEARCH" not in os.environ


def test_build_serving_cpu_search_config_full_tb_single_point(yirage_serving, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_FULL_TB_SEARCH", "1")
    import yirage as yr

    graph = yr.new_kernel_graph()
    mid = graph.new_input(dims=(1, 32), dtype=yr.float32)
    w = graph.new_input(dims=(32, 64), strides=(1, 32), dtype=yr.float32)
    graph.mark_output(graph.matmul(mid, w))

    cfg = yirage_serving.build_serving_cpu_search_config(graph)
    assert cfg["griddims"] == [(1, 1, 1)]
    assert cfg["blockdims"] == [(128, 1, 1)]
    assert cfg["franges"] == [1]
    assert "serving_env" in cfg
    assert cfg["serving_env"].get("YIRAGE_SERVING_FULL_TB_SEARCH") == "1"


def test_qwen_down_matmul_search_config_contract(yirage_serving, monkeypatch):
    """Qwen2-0.5B decode down matmul (H=896, I=4864) uses capped full-TB search space."""
    monkeypatch.setenv("YIRAGE_SERVING_FULL_TB_SEARCH", "1")
    yirage_serving.apply_serving_full_tb_search_tractability(use_ray=True)
    import yirage as yr

    graph = yr.new_kernel_graph()
    mid = graph.new_input(dims=(1, 4864), dtype=yr.float32)
    w = graph.new_input(dims=(4864, 896), strides=(1, 4864), dtype=yr.float32)
    graph.mark_output(graph.matmul(mid, w))
    cfg = yirage_serving.build_serving_cpu_search_config(graph)
    assert cfg["griddims"] == [(1, 1, 1)]
    assert cfg["blockdims"] == [(128, 1, 1)]
    assert cfg["franges"] == [1]
    assert cfg["serving_env"]["YIRAGE_CPU_MAX_KN_GRAPH_OP"] == "4"


def test_resolve_serving_accelforge_prescreen_env(yirage_serving, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_ACCELFORGE_PRESCREEN", raising=False)
    assert yirage_serving.resolve_serving_accelforge_prescreen() is False
    monkeypatch.setenv("YIRAGE_SERVING_ACCELFORGE_PRESCREEN", "1")
    assert yirage_serving.resolve_serving_accelforge_prescreen() is True


def test_cpu_cert_manifest_s22_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s22_contract" in names
