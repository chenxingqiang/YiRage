# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Inventory planned value-verify count vs registry (Loop R66)."""

from __future__ import annotations

from tests.integration.cpu_inventory import planned_value_verify_count, registry_sizes
from yirage.backends.cpu.support_matrix import cpu_layout_explore_gap_table


def test_planned_value_verify_count_is_346():
    assert planned_value_verify_count() == 346


def test_layout_explore_registry_symmetric_16_each():
    sizes = registry_sizes()
    assert sizes["layout_explore_builders"] == 16
    assert sizes["tb_layout_explore_builders"] == 16


def test_layout_explore_chunk_gap_table_does_not_raise():
    table = cpu_layout_explore_gap_table()
    assert len(table) == 3
    assert all(row["gap_kind"] == "none" for row in table)
