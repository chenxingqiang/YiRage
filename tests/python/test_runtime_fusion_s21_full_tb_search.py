# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S21: Tractable full TB matmul search tier (no seed-verify shortcut)."""

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


def test_runtime_fusion_version_s21(yirage_serving):
    assert yirage_serving.RuntimeFusion([]).inspect()["version"] == "s22"


def test_resolve_serving_full_tb_search_env(yirage_serving, monkeypatch):
    monkeypatch.delenv("YIRAGE_SERVING_FULL_TB_SEARCH", raising=False)
    assert yirage_serving.resolve_serving_full_tb_search() is False
    monkeypatch.setenv("YIRAGE_SERVING_FULL_TB_SEARCH", "1")
    assert yirage_serving.resolve_serving_full_tb_search() is True


def test_apply_full_tb_disables_seed_verify_env(yirage_serving, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_KN_MATMUL_ONLY", "1")
    yirage_serving.apply_serving_full_tb_search_tractability(use_ray=False)
    assert os.environ.get("YIRAGE_SERVING_FULL_TB_SEARCH") == "1"
    assert "YIRAGE_SERVING_KN_MATMUL_ONLY" not in os.environ
    assert os.environ.get("YIRAGE_CPU_BENCH_MINIMAL_EXPLORE") == "1"


def test_superoptimize_kwargs_full_tb_single_point(yirage_serving, monkeypatch):
    monkeypatch.setenv("YIRAGE_SERVING_FULL_TB_SEARCH", "1")
    monkeypatch.delenv("YIRAGE_SERVING_USE_RAY", raising=False)
    kwargs = yirage_serving.superoptimize_kwargs(quick=True)
    assert kwargs["griddims"] == [(1, 1, 1)]
    assert kwargs["blockdims"] == [(128, 1, 1)]
    assert kwargs["franges"] == [1]


@pytest.mark.slow
def test_full_tb_search_tiny_shape(yirage_serving, monkeypatch):
    """Tractable TB search on tiny matmul; must not use seed-verify shortcut."""
    monkeypatch.setenv("YIRAGE_SERVING_FULL_TB_SEARCH", "1")
    monkeypatch.delenv("YIRAGE_SERVING_USE_RAY", raising=False)
    monkeypatch.delenv("YIRAGE_SERVING_USE_COORDINATOR", raising=False)
    import yirage as yr

    graph = yr.new_kernel_graph()
    mid = graph.new_input(dims=(1, 32), dtype=yr.float32)
    w = graph.new_input(dims=(32, 64), strides=(1, 32), dtype=yr.float32)
    graph.mark_output(graph.matmul(mid, w))

    yirage_serving.apply_serving_kn_down_matmul_tractability(use_ray=False)
    opt = graph.superoptimize(**yirage_serving.superoptimize_kwargs(quick=True))
    assert opt is not None
    assert opt.backend == "cpu"


def test_cpu_cert_manifest_s21_contract():
    from yirage.serving.cpu_cert import serving_cpu_cert_manifest

    names = [s.name for s in serving_cpu_cert_manifest(quick=True)]
    assert "s21_contract" in names
