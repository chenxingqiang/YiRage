# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.python._yirage_test_support import native_core_available


@pytest.mark.cpu
def test_serialize_requires_cygraph():
    from yirage.storage.graph_serde import serialize_optimized_graph

    assert serialize_optimized_graph(object()) is None


@pytest.mark.cpu
def test_deserialize_empty():
    from yirage.storage.graph_serde import deserialize_cygraph

    assert deserialize_cygraph(None) is None
    assert deserialize_cygraph("") is None


@pytest.mark.cpu
def test_kngraph_from_persistent_entry(yirage_core):
    import yirage as yr
    from yirage.kernel.graph import KNGraph
    from yirage.storage.graph_serde import serialize_optimized_graph
    from yirage.storage.mugraph_store import MuGraphEntry, MuGraphMetadata

    g = yr.new_kernel_graph()
    a = g.new_input(dims=(4, 8), dtype=yr.float16)
    b = g.new_input(dims=(8, 16), dtype=yr.float16)
    g.mark_output(g.matmul(a, b))

    entry = MuGraphEntry(
        metadata=MuGraphMetadata(graph_hash="abc", config_hash="cfg", backend="cpu"),
        graph_json=serialize_optimized_graph(g),
    )
    restored = KNGraph.from_persistent_entry(entry, "cpu")
    assert restored is not None
    assert restored.backend == "cpu"


@pytest.fixture
def yirage_core():
    if not native_core_available():
        pytest.skip("yirage.core not built")
    import yirage  # noqa: F401

    return True
