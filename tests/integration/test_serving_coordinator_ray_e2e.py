#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""S20: Serving down-matmul coordinator with Ray cluster (when Ray installed)."""

from __future__ import annotations

import os

import pytest

from tests.python._yirage_test_support import ensure_native_library_path, native_core_available
from tests.python.conftest import RAY_AVAILABLE


pytestmark = [
    pytest.mark.cpu,
    pytest.mark.ray,
    pytest.mark.integration,
    pytest.mark.slow,
]


@pytest.fixture(autouse=True)
def _native_ld_path():
    ensure_native_library_path()


@pytest.fixture(scope="module")
def yirage_exec():
    if not native_core_available():
        pytest.skip("yirage.core not built")
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pkg = root / "python"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from yirage.serving import yirage_exec as mod

    mod.require_yirage_core()
    return mod


@pytest.fixture(scope="module")
def tiny_down_graph(yirage_exec):
    import yirage as yr

    graph = yr.new_kernel_graph()
    mid = graph.new_input(dims=(1, 32), dtype=yr.float32)
    w = graph.new_input(dims=(32, 64), strides=(1, 32), dtype=yr.float32)
    graph.mark_output(graph.matmul(mid, w))
    return graph


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
def test_serving_coordinator_ray_down_matmul(yirage_exec, tiny_down_graph, monkeypatch):
    """Ray-backed DistributedSearchCoordinator for serving down matmul (tiny shape)."""
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    monkeypatch.setenv("YIRAGE_SERVING_USE_COORDINATOR", "1")
    monkeypatch.setenv("YIRAGE_SERVING_RAY_WORKERS", "2")
    yirage_exec.apply_serving_kn_down_matmul_tractability(use_ray=True)

    opt = yirage_exec.superoptimize_down_matmul_via_coordinator(
        tiny_down_graph,
        quick=True,
    )
    assert opt is not None
    assert opt.backend == "cpu"


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not installed")
def test_serving_superoptimize_ray_blockdim_partition(yirage_exec, tiny_down_graph, monkeypatch):
    """KNGraph.superoptimize(use_ray=True) partitions blockdims when griddims=1."""
    monkeypatch.setenv("YIRAGE_SERVING_USE_RAY", "1")
    monkeypatch.setenv("YIRAGE_SERVING_USE_COORDINATOR", "0")
    yirage_exec.apply_serving_kn_down_matmul_tractability(use_ray=True)

    kwargs = yirage_exec.superoptimize_kwargs(quick=True)
    opt = tiny_down_graph.superoptimize(**kwargs)
    assert opt is not None
    assert opt.backend == "cpu"
