#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
CPU backend Ray end-to-end smoke tests.

Verifies distributed search paths on hosts without CUDA/MPS:
- KNGraph.superoptimize(backend="cpu", use_ray=True)
- RayDistributedEngine with gpus_per_worker=0
- DistributedSearchCoordinator.parallel_search(backend="cpu")
"""

from __future__ import annotations

import pytest

from tests.python._yirage_test_support import ensure_native_library_path, native_core_available
from tests.python.conftest import RAY_AVAILABLE


pytestmark = [
    pytest.mark.cpu,
    pytest.mark.ray,
    pytest.mark.integration,
]


@pytest.fixture(autouse=True)
def _native_ld_path():
    ensure_native_library_path()


@pytest.fixture
def yirage_core():
    if not native_core_available():
        pytest.skip("yirage.core not built")
    import yirage as yr

    return yr


@pytest.fixture
def tiny_matmul_graph(yirage_core):
    graph = yirage_core.new_kernel_graph()
    a = graph.new_input(dims=(16, 32), dtype=yirage_core.float16)
    b = graph.new_input(dims=(32, 64), dtype=yirage_core.float16)
    graph.mark_output(graph.matmul(a, b))
    return graph


_TINY_SEARCH = {
    "griddims": [(1, 1, 1), (2, 1, 1)],
    "blockdims": [(128, 1, 1)],
}


@pytest.mark.slow
def test_superoptimize_cpu_with_ray(tiny_matmul_graph):
    if not RAY_AVAILABLE:
        pytest.skip("Ray not installed")

    result = tiny_matmul_graph.superoptimize(
        backend="cpu",
        use_ray=True,
        num_workers=2,
        use_graph_dataset=False,
        use_cached_graphs=False,
        use_persistent_cache=False,
        verbose=False,
        **_TINY_SEARCH,
    )
    assert result is not None


def test_ray_distributed_engine_cpu():
    if not RAY_AVAILABLE:
        pytest.skip("Ray not installed")

    from yirage.ray.ray_distributed import (
        DistributedConfig,
        GPUPlacementConfig,
        RayDistributedEngine,
    )

    engine = RayDistributedEngine(
        DistributedConfig(
            num_workers=2,
            backend="cpu",
            max_search_time_s=120,
            gpu_placement=GPUPlacementConfig(gpus_per_worker=0, cpus_per_worker=1),
        )
    )
    result = engine.optimize(
        {"nodes": [], "edges": []},
        {
            "grid_dims": [(1, 1, 1), (2, 1, 1)],
            "block_dims": [(128, 1, 1)],
        },
    )
    assert result.num_workers == 2
    assert result.total_candidates_searched >= 1


@pytest.mark.slow
def test_distributed_search_coordinator_cpu(tiny_matmul_graph):
    if not RAY_AVAILABLE:
        pytest.skip("Ray not installed")

    from yirage.ray import DistributedSearchCoordinator

    coord = DistributedSearchCoordinator(num_workers=2, use_ray=True)
    try:
        out = coord.parallel_search(
            computation_graph=tiny_matmul_graph,
            config={
                "griddims": [(1, 1, 1)],
                "blockdims": [(128, 1, 1)],
            },
            backend="cpu",
            collect_feedback=False,
            verbose=False,
        )
    finally:
        coord.shutdown()

    stats = out.get("statistics", {})
    assert stats.get("num_workers") == 2
    assert "elapsed_seconds" in stats
