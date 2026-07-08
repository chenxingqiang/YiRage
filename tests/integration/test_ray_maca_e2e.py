#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
MACA backend Ray end-to-end smoke tests (MetaX GPU required).

Mirrors ``test_ray_cpu_e2e.py`` with ``backend="maca"`` and block sizes that are
multiples of ``MACA_WARP_SIZE`` (64). Cloud CPU VMs skip GPU tests; contract
tests run without MetaX hardware.
"""

from __future__ import annotations

import os

import pytest

from tests.python._yirage_test_support import ensure_native_library_path, native_core_available
from tests.python.conftest import RAY_AVAILABLE


pytestmark = [
    pytest.mark.maca,
    pytest.mark.ray,
    pytest.mark.integration,
]


def _maca_vm_available() -> bool:
    if os.environ.get("YIRAGE_MACA_INTEGRATION", "") == "1":
        return True
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        return "MetaX" in torch.cuda.get_device_name(0)
    except Exception:
        return False


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
    a = graph.new_input(dims=(64, 32), dtype=yirage_core.float16)
    b = graph.new_input(dims=(32, 64), dtype=yirage_core.float16)
    graph.mark_output(graph.matmul(a, b))
    return graph


# Tractable MACA search: single grid, 64-thread warp block (256 = 4 warps).
_TINY_MACA_SEARCH = {
    "griddims": [(4, 1, 1)],
    "blockdims": [(256, 1, 1)],
    "franges": [8],
}


def test_ray_maca_e2e_contract_blockdims_are_warp_multiple():
    """Contract: MACA Ray e2e uses block sizes aligned to warp=64 (no GPU)."""
    import importlib.util
    import sys
    from pathlib import Path

    pkg = Path(__file__).resolve().parents[2] / "python"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    spec = importlib.util.spec_from_file_location(
        "maca_config_contract", pkg / "yirage" / "backends" / "maca" / "config.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    for block in _TINY_MACA_SEARCH["blockdims"]:
        threads = block[0] * block[1] * block[2]
        assert threads % mod.MACA_WARP_SIZE == 0
        assert mod.validate_block_size(threads)


@pytest.mark.slow
def test_superoptimize_maca_with_ray(tiny_matmul_graph):
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")
    if not RAY_AVAILABLE:
        pytest.skip("Ray not installed")

    os.environ.setdefault("YIRAGE_BACKEND", "maca")
    os.environ.setdefault("YIRAGE_MACA_SEARCH_QUICK", "1")

    result = tiny_matmul_graph.superoptimize(
        backend="maca",
        use_ray=True,
        num_workers=2,
        use_graph_dataset=False,
        use_cached_graphs=False,
        use_persistent_cache=False,
        verbose=False,
        **_TINY_MACA_SEARCH,
    )
    assert result is not None
    assert result.backend == "maca"


def test_ray_distributed_engine_maca():
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")
    if not RAY_AVAILABLE:
        pytest.skip("Ray not installed")

    from yirage.ray.ray_distributed import (
        DistributedConfig,
        GPUPlacementConfig,
        RayDistributedEngine,
    )
    from yirage.backends.maca.config import resolve_maca_gpus_per_worker

    gpus = resolve_maca_gpus_per_worker()

    engine = RayDistributedEngine(
        DistributedConfig(
            num_workers=2,
            backend="maca",
            max_search_time_s=120,
            gpu_placement=GPUPlacementConfig(gpus_per_worker=gpus, cpus_per_worker=1),
        )
    )
    result = engine.optimize(
        {"nodes": [], "edges": []},
        {
            "grid_dims": [(4, 1, 1)],
            "block_dims": [(256, 1, 1)],
        },
    )
    assert result.num_workers == 2
    assert result.total_candidates_searched >= 1


@pytest.mark.slow
def test_distributed_search_coordinator_maca(tiny_matmul_graph):
    if not _maca_vm_available():
        pytest.skip("MetaX MACA GPU not available (set YIRAGE_MACA_INTEGRATION=1 to force)")
    if not RAY_AVAILABLE:
        pytest.skip("Ray not installed")

    from yirage.ray import DistributedSearchCoordinator

    os.environ.setdefault("YIRAGE_BACKEND", "maca")

    coord = DistributedSearchCoordinator(num_workers=2, use_ray=True)
    try:
        out = coord.parallel_search(
            computation_graph=tiny_matmul_graph,
            config={
                "griddims": [(4, 1, 1)],
                "blockdims": [(256, 1, 1)],
                "franges": [8],
            },
            backend="maca",
            collect_feedback=False,
            verbose=False,
        )
    finally:
        coord.shutdown()

    stats = out.get("statistics", {})
    assert stats.get("num_workers") == 2
    assert "elapsed_seconds" in stats
