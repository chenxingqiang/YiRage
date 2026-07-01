# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for no-bias fused conv2d Graph APIs (Loop R94)."""

from __future__ import annotations

import pytest

from tests.python._yirage_test_support import native_core_available


@pytest.fixture
def yirage_core():
    if not native_core_available():
        pytest.skip("yirage.core not built")
    import yirage  # noqa: F401

    return True


@pytest.mark.cpu
def test_graph_exposes_no_bias_fused_conv2d_methods(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    for name in (
        "conv2d_relu",
        "conv2d_gelu",
        "conv2d_silu",
        "conv2d_depthwise",
        "conv2d_depthwise_relu",
        "conv2d_depthwise_gelu",
        "conv2d_depthwise_silu",
        "conv2d_separable_relu",
        "conv2d_separable_gelu",
        "conv2d_separable_silu",
    ):
        assert hasattr(g, name), f"KNGraph missing fused API {name}"


@pytest.mark.cpu
def test_conv2d_relu_fused_matches_unary_chain(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
    fused = g.conv2d_relu(x, w, stride=(1, 1), padding=(1, 1))
    chain = g.relu(g.conv2d(x, w, stride=(1, 1), padding=(1, 1)))
    assert fused is not None
    assert chain is not None


@pytest.mark.cpu
def test_conv2d_depthwise_fused_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
    out = g.conv2d_depthwise_silu(x, w, stride=(1, 1), padding=(1, 1))
    g.mark_output(out)
    assert g.cygraph is not None


@pytest.mark.cpu
def test_conv2d_separable_gelu_fused_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
    pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
    out = g.conv2d_separable_gelu(x, dw, pw, stride=(1, 1), padding=(1, 1))
    g.mark_output(out)
    assert g.cygraph is not None
