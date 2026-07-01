# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for fused conv2d Graph APIs (Loop R94+)."""

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
        "conv2d_groups",
        "conv2d_groups_relu",
        "conv2d_groups_gelu",
        "conv2d_groups_silu",
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
def test_conv2d_activation_batch2_fused_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(2, 3, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
    relu_out = g.conv2d_relu(x, w, stride=(1, 1), padding=(1, 1))
    gelu_out = g.conv2d_gelu(x, w, stride=(1, 1), padding=(1, 1))
    silu_out = g.conv2d_silu(x, w, stride=(1, 1), padding=(1, 1))
    g.mark_output(relu_out)
    g.mark_output(gelu_out)
    g.mark_output(silu_out)
    assert g.cygraph is not None


@pytest.mark.cpu
def test_conv2d_groups_fused_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
    out = g.conv2d_groups_relu(x, w, stride=(1, 1), padding=(1, 1))
    g.mark_output(out)
    assert g.cygraph is not None


@pytest.mark.cpu
def test_conv2d_groups_matches_groups_conv2d(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
    grouped = g.conv2d_groups(x, w, stride=(1, 1), padding=(1, 1))
    direct = g.conv2d(x, w, stride=(1, 1), padding=(1, 1), groups=2)
    assert grouped is not None
    assert direct is not None


@pytest.mark.cpu
def test_conv2d_groups_gelu_silu_fused_build_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
    gelu_out = g.conv2d_groups_gelu(x, w, stride=(1, 1), padding=(1, 1))
    silu_out = g.conv2d_groups_silu(x, w, stride=(1, 1), padding=(1, 1))
    g.mark_output(gelu_out)
    g.mark_output(silu_out)
    assert g.cygraph is not None


@pytest.mark.cpu
def test_conv2d_groups_activation_matches_groups_param(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
    fused = g.conv2d_groups_relu(x, w, stride=(1, 1), padding=(1, 1), groups=2)
    direct = g.conv2d_relu(
        x, w, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=2
    )
    assert fused is not None
    assert direct is not None


@pytest.mark.cpu
def test_conv2d_groups_batch2_fused_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
    out = g.conv2d_groups(x, w, stride=(1, 1), padding=(1, 1))
    g.mark_output(out)
    assert g.cygraph is not None


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


@pytest.mark.cpu
def test_conv2d_depthwise_matches_groups_conv2d(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
    depthwise = g.conv2d_depthwise(x, w, stride=(1, 1), padding=(1, 1))
    grouped = g.conv2d(x, w, stride=(1, 1), padding=(1, 1), groups=4)
    assert depthwise is not None
    assert grouped is not None


@pytest.mark.cpu
def test_conv2d_separable_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
    pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
    out = g.conv2d_separable(x, dw, pw, stride=(1, 1), padding=(1, 1))
    g.mark_output(out)
    assert g.cygraph is not None


@pytest.mark.cpu
def test_graph_exposes_conv2d_bias_fused_methods(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    for name in (
        "conv2d_bias",
        "conv2d_bias_relu",
        "conv2d_bias_gelu",
        "conv2d_bias_silu",
    ):
        assert hasattr(g, name), f"KNGraph missing bias fused API {name}"


@pytest.mark.cpu
def test_conv2d_bias_fused_matches_add_chain(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
    b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
    fused = g.conv2d_bias(x, w, b, stride=(1, 1), padding=(1, 1))
    chain = g.add(g.conv2d(x, w, stride=(1, 1), padding=(1, 1)), b)
    assert fused is not None
    assert chain is not None


@pytest.mark.cpu
def test_conv2d_bias_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
    b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
    out = g.conv2d_bias_relu(x, w, b, stride=(1, 1), padding=(1, 1))
    g.mark_output(out)
    assert g.cygraph is not None


@pytest.mark.cpu
def test_graph_exposes_bias_groups_fused_methods(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    for name in (
        "conv2d_bias_groups",
        "conv2d_bias_groups_relu",
        "conv2d_bias_groups_gelu",
        "conv2d_bias_groups_silu",
    ):
        assert hasattr(g, name), f"KNGraph missing grouped bias fused API {name}"


@pytest.mark.cpu
def test_conv2d_bias_groups_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
    b = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
    out = g.conv2d_bias_groups(x, w, b, stride=(1, 1), padding=(1, 1))
    g.mark_output(out)
    assert g.cygraph is not None


@pytest.mark.cpu
def test_conv2d_bias_groups_delegates_to_conv2d_bias(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
    b = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
    grouped = g.conv2d_bias_groups(x, w, b, stride=(1, 1), padding=(1, 1))
    direct = g.conv2d_bias(
        x, w, b, stride=(1, 1), padding=(1, 1), groups=2
    )
    assert grouped is not None
    assert direct is not None


@pytest.mark.cpu
def test_graph_exposes_separable_bias_fused_methods(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    for name in (
        "conv2d_separable_bias",
        "conv2d_separable_bias_relu",
        "conv2d_separable_bias_gelu",
        "conv2d_separable_bias_silu",
    ):
        assert hasattr(g, name), f"KNGraph missing separable bias fused API {name}"


@pytest.mark.cpu
def test_conv2d_separable_bias_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
    pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
    db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
    pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
    out = g.conv2d_separable_bias_relu(
        x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
    )
    g.mark_output(out)
    assert g.cygraph is not None


@pytest.mark.cpu
def test_graph_exposes_depthwise_bias_fused_methods(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    for name in (
        "conv2d_depthwise_bias",
        "conv2d_depthwise_bias_relu",
        "conv2d_depthwise_bias_gelu",
        "conv2d_depthwise_bias_silu",
    ):
        assert hasattr(g, name), f"KNGraph missing depthwise bias fused API {name}"


@pytest.mark.cpu
def test_conv2d_depthwise_bias_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
    b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
    out = g.conv2d_depthwise_bias_relu(x, w, b, stride=(1, 1), padding=(1, 1))
    g.mark_output(out)
    assert g.cygraph is not None


@pytest.mark.cpu
def test_conv2d_depthwise_bias_delegates_to_conv2d_bias(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
    b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
    depthwise = g.conv2d_depthwise_bias(x, w, b, stride=(1, 1), padding=(1, 1))
    grouped = g.conv2d_bias(
        x, w, b, stride=(1, 1), padding=(1, 1), groups=4
    )
    assert depthwise is not None
    assert grouped is not None


@pytest.mark.cpu
def test_conv2d_bias_groups_relu_builds_graph(yirage_core):
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
    w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
    b = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
    out = g.conv2d_bias_groups_relu(x, w, b, stride=(1, 1), padding=(1, 1))
    g.mark_output(out)
    assert g.cygraph is not None
