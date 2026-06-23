# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU search explore list must match C++ GeneratorConfig::get_cpu_search_config."""

from __future__ import annotations

from yirage.backends.cpu.support_matrix import (
    cpu_layout_explore_gap_meta,
    cpu_layout_explore_gap_table,
    cpu_search_explore_not_supported,
    cpu_search_yaml_explore,
)

# Mirror src/search/config.cc get_cpu_search_config() knop/tbop_to_explore.
_CPP_CPU_KN_EXPLORE = sorted(
    [
        "kn_matmul_op",
        "kn_exp_op",
        "kn_square_op",
        "kn_sqrt_op",
        "kn_silu_op",
        "kn_gelu_op",
        "kn_relu_op",
        "kn_sigmoid_op",
        "kn_log_op",
        "kn_clamp_op",
        "kn_mul_scalar_op",
        "kn_reduction_0_op",
        "kn_reduction_1_op",
        "kn_reduction_2_op",
        "kn_add_op",
        "kn_sub_op",
        "kn_mul_op",
        "kn_div_op",
        "kn_pow_op",
        "kn_concat_0_op",
        "kn_concat_1_op",
        "kn_concat_2_op",
        "kn_split_0_op",
        "kn_split_1_op",
        "kn_split_2_op",
        "kn_chunk_0_op",
        "kn_chunk_1_op",
        "kn_chunk_2_op",
        "kn_transpose_01_op",
        "kn_customized_op",
    ]
)
_CPP_CPU_TB_EXPLORE = sorted(
    [
        "tb_matmul_op",
        "tb_exp_op",
        "tb_square_op",
        "tb_sqrt_op",
        "tb_silu_op",
        "tb_gelu_op",
        "tb_relu_op",
        "tb_sigmoid_op",
        "tb_log_op",
        "tb_clamp_op",
        "tb_mul_scalar_op",
        "tb_add_op",
        "tb_sub_op",
        "tb_mul_op",
        "tb_div_op",
        "tb_pow_op",
        "tb_rms_norm_op",
        "tb_concat_0_op",
        "tb_concat_1_op",
        "tb_concat_2_op",
        "tb_split_0_op",
        "tb_split_1_op",
        "tb_split_2_op",
        "tb_chunk_0_op",
        "tb_chunk_1_op",
        "tb_chunk_2_op",
        "tb_reduction_0_op",
        "tb_reduction_1_op",
        "tb_reduction_2_op",
        "tb_reduction_0_to_dimx_op",
        "tb_reduction_1_to_dimx_op",
        "tb_reduction_2_to_dimx_op",
        "tb_reduction_0_max_op",
        "tb_reduction_1_max_op",
        "tb_reduction_2_max_op",
        "tb_forloop_accum_no_red_op",
        "tb_forloop_accum_red_ld_sum_op",
        "tb_forloop_accum_red_ld_mean_op",
        "tb_forloop_accum_red_ld_rms_op",
        "tb_forloop_accum_redtox_ld_sum_op",
        "tb_forloop_accum_max_op",
    ]
)


def test_search_yaml_explore_matches_cpp_cpu_config():
    assert cpu_search_yaml_explore(layer="kn") == _CPP_CPU_KN_EXPLORE
    assert cpu_search_yaml_explore(layer="tb") == _CPP_CPU_TB_EXPLORE


def test_layout_explore_concat_matmul_symmetric_kn_tb():
    from tests.integration.cpu_op_builders import LAYOUT_EXPLORE_BUILDERS
    from tests.integration.cpu_tb_op_builders import TB_LAYOUT_EXPLORE_BUILDERS

    assert "kn_layout_concat_matmul" in LAYOUT_EXPLORE_BUILDERS
    assert "tb_layout_concat_matmul" in TB_LAYOUT_EXPLORE_BUILDERS


def test_search_explore_ops_are_cpu_supported():
    assert cpu_search_explore_not_supported() == []


def test_layout_explore_chunk_gap_table_documents_symmetry():
    table = cpu_layout_explore_gap_table()
    assert len(table) == 3
    for row in table:
        assert row["kn_in_search_explore"] is True
        assert row["tb_in_search_explore"] is True
        assert row["kn_matrix_tier"] == "supported"
        assert row["tb_matrix_tier"] == "supported"
        assert row["gap_kind"] == "none"


def test_layout_explore_chunk_gap_meta_empty_when_resolved():
    meta = cpu_layout_explore_gap_meta()
    assert meta == {}


def test_tb_chunk_layout_explore_builders_cover_all_dims():
    from tests.integration.cpu_tb_op_builders import TB_LAYOUT_EXPLORE_BUILDERS

    tb_chunk_ops = {f"tb_chunk_{d}_op" for d in (0, 1, 2)}
    assert tb_chunk_ops.issubset(set(cpu_search_yaml_explore(layer="tb")))
    tb_chunk_patterns = [k for k in TB_LAYOUT_EXPLORE_BUILDERS if "chunk" in k]
    assert len(tb_chunk_patterns) >= 9
    for dim in (0, 1, 2):
        dim_patterns = [p for p in tb_chunk_patterns if p.endswith(f"dim{dim}")]
        assert len(dim_patterns) >= 3


def test_kn_chunk_layout_explore_builders_cover_all_dims():
    from tests.integration.cpu_op_builders import LAYOUT_EXPLORE_BUILDERS

    kn_chunk_ops = {f"kn_chunk_{d}_op" for d in (0, 1, 2)}
    assert kn_chunk_ops.issubset(set(cpu_search_yaml_explore(layer="kn")))
    kn_chunk_patterns = [k for k in LAYOUT_EXPLORE_BUILDERS if "chunk" in k]
    assert len(kn_chunk_patterns) >= 9
    for dim in (0, 1, 2):
        dim_patterns = [p for p in kn_chunk_patterns if p.endswith(f"dim{dim}")]
        assert len(dim_patterns) >= 3


def test_tb_chunk_deferred_patterns_placeholder_removed():
    from tests.integration.cpu_tb_op_builders import (
        TB_LAYOUT_CHUNK_DEFERRED_PATTERNS,
        TB_LAYOUT_EXPLORE_BUILDERS,
    )

    assert TB_LAYOUT_CHUNK_DEFERRED_PATTERNS == frozenset()
    table = cpu_layout_explore_gap_table()
    for row in table:
        dim = row["dim"]
        dim_patterns = [
            p for p in TB_LAYOUT_EXPLORE_BUILDERS if f"dim{dim}" in p and "chunk" in p
        ]
        assert len(dim_patterns) >= 3
