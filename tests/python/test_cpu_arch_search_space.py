# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU architecture-aware search space resolution."""

import pytest

from yirage.backends.cpu.config import (
    build_arch_aware_block_dims,
    build_arch_aware_grid_dims,
    detect_simd_support,
    get_cpu_search_config,
    resolve_cpu_search_space,
)


def test_block_dims_align_to_simd():
    simd = detect_simd_support()
    cfg = get_cpu_search_config()
    blocks = build_arch_aware_block_dims(cfg["vector_width"], simd)
    assert len(blocks) >= 2
    assert all(b[1] == 1 and b[2] == 1 for b in blocks)


def test_grid_dims_respect_core_count():
    cfg = get_cpu_search_config()
    grids = build_arch_aware_grid_dims(cfg["num_cores"], m_dim=8)
    xs = [g[0] for g in grids]
    assert 1 in xs
    assert all(x <= cfg["num_cores"] for x in xs)
    assert all(m == 8 or 8 % m == 0 or 8 >= m for m in xs)


@pytest.mark.skipif(
    __import__("os").environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)
def test_resolve_cpu_search_space_from_graph():
    import yirage as yr

    g = yr.new_kernel_graph()
    a = g.new_input(dims=(8, 32), dtype=yr.float16)
    b = g.new_input(dims=(32, 64), dtype=yr.float16)
    g.mark_output(g.matmul(a, b))

    space = resolve_cpu_search_space(g.cygraph)
    assert space["problem_mnk"] == (8, 64, 32)
    assert len(space["grid_dims_to_explore"]) >= 1
    assert len(space["block_dims_to_explore"]) >= 1
    assert space["search_thread"] <= space["num_cores"]
