#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Persistent MuGraph save/restore across superoptimize sessions."""

from __future__ import annotations

import os
import time

import pytest

from tests.python._yirage_test_support import ensure_native_library_path, native_core_available


pytestmark = [pytest.mark.cpu, pytest.mark.integration]


@pytest.fixture(autouse=True)
def _ld_path():
    ensure_native_library_path()


@pytest.fixture
def yirage():
    if not native_core_available():
        pytest.skip("yirage.core not built")
    import yirage as yr

    return yr


def _tiny_matmul(yr):
    g = yr.new_kernel_graph()
    a = g.new_input(dims=(8, 16), dtype=yr.float16)
    b = g.new_input(dims=(16, 32), dtype=yr.float16)
    g.mark_output(g.matmul(a, b))
    return g


_SEARCH = {
    "backend": "cpu",
    "griddims": [(1, 1, 1), (2, 1, 1)],
    "blockdims": [(128, 1, 1)],
    "use_graph_dataset": False,
    "use_cached_graphs": False,
    "use_ray": True,
    "num_workers": 2,
    "verbose": False,
}


def test_graph_serde_roundtrip(yirage):
    from yirage.storage.graph_serde import deserialize_cygraph, serialize_optimized_graph

    g = _tiny_matmul(yirage)
    payload = serialize_optimized_graph(g)
    assert payload is not None
    assert len(payload) > 32

    cy = deserialize_cygraph(payload)
    assert cy is not None


def test_persistent_cache_skips_second_search(yirage, tmp_path, monkeypatch):
    """Ray superoptimize saves graph_json; a new process-like call restores without search."""
    from yirage.kernel.graph import KNGraph
    from yirage.storage import get_mugraph_store

    store_root = tmp_path / "mugraphs"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        "yirage.storage.mugraph_store.DEFAULT_STORE_ROOT",
        str(store_root),
    )
    monkeypatch.setattr("yirage.storage.mugraph_store._default_store", None)

    g1 = _tiny_matmul(yirage)
    g1.superoptimize(use_persistent_cache=True, **_SEARCH)

    gh = hex(g1.cygraph.get_owner_independent_hash())[2:]
    entry = get_mugraph_store().find_best(gh, "cpu")
    assert entry is not None
    assert entry.graph_json is not None

    restored = KNGraph.from_persistent_entry(entry, "cpu")
    assert restored is not None

    import ray

    if ray.is_initialized():
        ray.shutdown()

    g2 = _tiny_matmul(yirage)
    t0 = time.perf_counter()
    out = g2.superoptimize(use_persistent_cache=True, **_SEARCH)
    elapsed = time.perf_counter() - t0

    assert out is not None
    assert elapsed < 5.0, f"expected cache restore, took {elapsed:.1f}s"
