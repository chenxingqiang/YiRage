# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Serving CPU superoptimize Ray opt-in contract tests."""

from __future__ import annotations

import pytest

from serving_test_utils import import_serving


@pytest.fixture(scope="module")
def yirage_exec():
    import_serving()
    from yirage.serving import yirage_exec as mod

    return mod


def test_resolve_serving_use_ray_default_off(yirage_exec, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_USE_RAY", raising=False)
    assert yirage_exec.resolve_serving_use_ray() is False
    assert yirage_exec.serving_superoptimize_ray_kwargs() == {"use_ray": False}


def test_resolve_serving_use_ray_env_on(yirage_exec, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    assert yirage_exec.resolve_serving_use_ray() is True
    assert yirage_exec.serving_superoptimize_ray_kwargs()["use_ray"] is True


def test_superoptimize_kwargs_ray_uses_auto_search_space(yirage_exec, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    kwargs = yirage_exec.superoptimize_kwargs(quick=True)
    assert kwargs["use_ray"] is True
    assert "griddims" not in kwargs
    assert "blockdims" not in kwargs


def test_apply_serving_tractability_ray_disables_seed_verify_env(yirage_exec, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_USE_RAY", raising=False)
    monkeypatch.delenv("YIRAGE_SERVING_KN_MATMUL_ONLY", raising=False)
    yirage_exec.apply_serving_kn_down_matmul_tractability(use_ray=True)
    import os

    assert os.environ.get("YIRAGE_SERVING_USE_RAY") == "1"
    assert "YIRAGE_SERVING_KN_MATMUL_ONLY" not in os.environ


def test_apply_serving_tractability_full_tb_disables_seed_verify(yirage_exec, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_FULL_TB_SEARCH", "1")
    monkeypatch.setenv("YIRAGE_SERVING_KN_MATMUL_ONLY", "1")
    yirage_exec.apply_serving_kn_down_matmul_tractability(use_ray=False)
    import os

    assert os.environ.get("YIRAGE_SERVING_FULL_TB_SEARCH") == "1"
    assert "YIRAGE_SERVING_KN_MATMUL_ONLY" not in os.environ


def test_apply_serving_tractability_default_sets_seed_verify_env(yirage_exec, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_USE_RAY", raising=False)
    monkeypatch.delenv("YIRAGE_SERVING_KN_MATMUL_ONLY", raising=False)
    yirage_exec.apply_serving_kn_down_matmul_tractability(use_ray=False)
    import os

    assert os.environ.get("YIRAGE_SERVING_KN_MATMUL_ONLY") == "1"
    assert "YIRAGE_SERVING_USE_RAY" not in os.environ


def test_resolve_serving_use_coordinator_follows_ray(yirage_exec, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_USE_COORDINATOR", raising=False)
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    assert yirage_exec.resolve_serving_use_coordinator() is True
    monkeypatch.setenv("YIRAGE_SERVING_USE_COORDINATOR", "0")
    assert yirage_exec.resolve_serving_use_coordinator() is False


def test_build_serving_cpu_search_config_has_blockdims(yirage_exec):
    from tests.python._yirage_test_support import native_core_available

    if not native_core_available():
        pytest.skip("yirage.core not built")
    import yirage as yr

    graph = yr.new_kernel_graph()
    a = graph.new_input(dims=(1, 32), dtype=yr.float32)
    b = graph.new_input(dims=(32, 64), dtype=yr.float32)
    graph.mark_output(graph.matmul(a, b))
    cfg = yirage_exec.build_serving_cpu_search_config(graph)
    assert cfg["griddims"]
    assert len(cfg["blockdims"]) >= 1
    assert cfg["franges"]


@pytest.mark.slow
def test_superoptimize_down_matmul_coordinator_local_smoke(yirage_exec, monkeypatch):
    from tests.python._yirage_test_support import native_core_available

    if not native_core_available():
        pytest.skip("yirage.core not built")
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    monkeypatch.setenv("YIRAGE_SERVING_USE_COORDINATOR", "1")
    monkeypatch.setenv("YIRAGE_SERVING_RAY_WORKERS", "2")
    import yirage as yr

    graph = yr.new_kernel_graph()
    a = graph.new_input(dims=(1, 32), dtype=yr.float32)
    b = graph.new_input(dims=(32, 64), dtype=yr.float32)
    graph.mark_output(graph.matmul(a, b))
    # Local coordinator (no Ray cluster) for contract smoke
    monkeypatch.setattr(
        yirage_exec,
        "resolve_serving_use_ray",
        lambda **_: False,
    )
    opt = yirage_exec.superoptimize_down_matmul_via_coordinator(graph, quick=True)
    assert opt is not None
    assert opt.backend == "cpu"
