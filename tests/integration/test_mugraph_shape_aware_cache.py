# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Shape-aware MuGraph persistent cache (runtime dynamism P0)."""

from __future__ import annotations

import pytest

from yirage.storage.mugraph_store import (
    MuGraphStore,
    bucket_dim,
    bucket_input_shapes,
    input_shapes_bucket_match,
    input_shapes_match,
    mugraph_require_shape_match,
    mugraph_shape_bucket_enabled,
    normalize_input_shapes,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "mugraphs"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("yirage.storage.mugraph_store.DEFAULT_STORE_ROOT", str(root))
    monkeypatch.setattr("yirage.storage.mugraph_store._default_store", None)
    return MuGraphStore(root_path=str(root))


def test_normalize_input_shapes():
    assert normalize_input_shapes([[8, 16], [16, 32]]) == [[8, 16], [16, 32]]
    assert normalize_input_shapes(()) == []
    assert input_shapes_match([[4, 8]], [[4, 8]]) is True
    assert input_shapes_match([[4, 8]], [[8, 4]]) is False


def test_find_best_prefers_matching_shape(store):
    dummy = {"type": "matmul"}
    gh = "shape_test_hash"
    store.save(
        graph_hash=gh,
        optimized_graph=dummy,
        backend="cpu",
        latency_ms=1.0,
        input_shapes=[[8, 16], [16, 32]],
        griddims=[[1, 1, 1]],
    )
    store.save(
        graph_hash=gh,
        optimized_graph=dummy,
        backend="cpu",
        latency_ms=0.5,
        input_shapes=[[4, 8], [8, 16]],
        griddims=[[2, 1, 1]],
    )

    entry = store.find_best(gh, "cpu", input_shapes=[[8, 16], [16, 32]])
    assert entry is not None
    assert entry.metadata.latency_ms == 1.0
    assert entry.metadata.input_shapes == [[8, 16], [16, 32]]


def test_find_best_shape_mismatch_falls_back_without_require(store):
    dummy = {"type": "matmul"}
    gh = "shape_fallback_hash"
    store.save(
        graph_hash=gh,
        optimized_graph=dummy,
        backend="cpu",
        latency_ms=2.0,
        input_shapes=[[8, 16], [16, 32]],
        griddims=[[1, 1, 1]],
    )

    entry = store.find_best(gh, "cpu", input_shapes=[[99, 99], [99, 99]])
    assert entry is not None
    assert entry.metadata.latency_ms == 2.0


def test_bucket_dim_pow2_ceil():
    assert bucket_dim(1) == 1
    assert bucket_dim(8) == 8
    assert bucket_dim(9) == 16
    assert bucket_dim(12) == 16
    assert bucket_dim(50) == 64


def test_find_best_bucket_match_before_global_fallback(store):
    dummy = {"type": "matmul"}
    gh = "shape_bucket_hash"
    store.save(
        graph_hash=gh,
        optimized_graph=dummy,
        backend="cpu",
        latency_ms=1.0,
        input_shapes=[[16, 32], [32, 64]],
        griddims=[[1, 1, 1]],
    )
    store.save(
        graph_hash=gh,
        optimized_graph=dummy,
        backend="cpu",
        latency_ms=0.1,
        input_shapes=[[512, 512], [512, 512]],
        griddims=[[2, 1, 1]],
    )

    assert mugraph_shape_bucket_enabled() is True
    entry = store.find_best(gh, "cpu", input_shapes=[[12, 32], [32, 50]])
    assert entry is not None
    assert entry.metadata.latency_ms == 1.0
    assert bucket_input_shapes([[12, 32], [32, 50]]) == [[16, 32], [32, 64]]


def test_find_best_bucket_disabled_falls_back_global(store, monkeypatch):
    monkeypatch.setenv("YIRAGE_MUGraph_SHAPE_BUCKET", "0")
    assert mugraph_shape_bucket_enabled() is False

    dummy = {"type": "matmul"}
    gh = "shape_bucket_off_hash"
    store.save(
        graph_hash=gh,
        optimized_graph=dummy,
        backend="cpu",
        latency_ms=1.0,
        input_shapes=[[16, 32], [32, 64]],
        griddims=[[1, 1, 1]],
    )
    store.save(
        graph_hash=gh,
        optimized_graph=dummy,
        backend="cpu",
        latency_ms=0.1,
        input_shapes=[[512, 512], [512, 512]],
        griddims=[[2, 1, 1]],
    )

    entry = store.find_best(gh, "cpu", input_shapes=[[12, 32], [32, 50]])
    assert entry.metadata.latency_ms == 0.1


def test_find_best_require_shape_match_returns_none(store, monkeypatch):
    monkeypatch.setenv("YIRAGE_MUGraph_REQUIRE_SHAPE_MATCH", "1")
    assert mugraph_require_shape_match() is True

    dummy = {"type": "matmul"}
    gh = "shape_strict_hash"
    store.save(
        graph_hash=gh,
        optimized_graph=dummy,
        backend="cpu",
        latency_ms=1.0,
        input_shapes=[[8, 16], [16, 32]],
        griddims=[[1, 1, 1]],
    )

    entry = store.find_best(gh, "cpu", input_shapes=[[99, 99]])
    assert entry is None
