# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""TB chunk layout ops are supported after Loop R54."""

from __future__ import annotations

import pytest

from tests.integration.cpu_op_builders import LAYOUT_EXPLORE_BUILDERS
from tests.integration.cpu_tb_op_builders import (
    TB_LAYOUT_CHUNK_DEFERRED_PATTERNS,
    TB_LAYOUT_EXPLORE_BUILDERS,
)
from yirage.backends.cpu.support_matrix import (
    cpu_search_yaml_explore,
    tb_op_contracts,
)

_TB_CHUNK_OPS = (
    "tb_chunk_0_op",
    "tb_chunk_1_op",
    "tb_chunk_2_op",
)


def test_tb_chunk_ops_supported_in_matrix():
    contracts = tb_op_contracts()
    for op in _TB_CHUNK_OPS:
        assert op in contracts
        assert contracts[op].tier == "supported"
        assert contracts[op].layer == "layout"


def test_tb_chunk_in_cpu_search_explore():
    explored = set(cpu_search_yaml_explore(layer="tb"))
    assert _TB_CHUNK_OPS <= explored


def test_tb_chunk_layout_patterns_active_in_builders():
    tb_chunk_active = [k for k in TB_LAYOUT_EXPLORE_BUILDERS if "chunk" in k]
    assert len(tb_chunk_active) >= 9
    kn_chunk = [k for k in LAYOUT_EXPLORE_BUILDERS if "chunk" in k]
    assert len(kn_chunk) >= 9


def test_tb_chunk_deferred_patterns_placeholder_empty():
    assert TB_LAYOUT_CHUNK_DEFERRED_PATTERNS == frozenset()


@pytest.mark.parametrize(
    "pattern",
    sorted(
        k
        for k in TB_LAYOUT_EXPLORE_BUILDERS
        if "chunk" in k or "split_chunk" in k
    ),
)
def test_tb_chunk_layout_pattern_in_active_builders(pattern: str):
    assert pattern in TB_LAYOUT_EXPLORE_BUILDERS
