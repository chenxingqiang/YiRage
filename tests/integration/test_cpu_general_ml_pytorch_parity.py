# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""PyTorch parity for general-ML ops (softmax, layer_norm, rescale accum, compounds)."""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

import yirage as yr

from tests.integration.cpu_op_builders import (
    CUSTOMIZED_OP_BUILDERS,
    _f16,
)
from tests.integration.cpu_tb_op_builders import TB_OP_BUILDERS
from yirage.search.verifier_config import runtime_verify_mugraph

pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


@pytest.fixture(autouse=True)
def _force_kn_interpreter(monkeypatch):
    monkeypatch.setenv("YIRAGE_CPU_RMS_MATMUL_FAST", "0")


@pytest.mark.parametrize(
    "pattern",
    [
        "kn_softmax",
        "kn_softmax_3d",
        "kn_softmax_3d_batch1",
        "kn_layer_norm",
        "kn_rms_norm_3d",
        "kn_rms_norm_3d_batch1",
        "kn_layer_norm_3d",
        "kn_layer_norm_3d_batch1",
        "gemm_softmax",
        "gemm_softmax_batch1",
        "gemm_softmax_scaled",
        "gemm_softmax_scaled_batch1",
        "gemm_softmax_scaled_batched",
        "gemm_softmax_scaled_batched_batch1",
        "gemm_softmax_3d",
        "gemm_softmax_3d_batch1",
        "gemm_softmax_scaled_3d",
        "gemm_softmax_scaled_3d_batch1",
        "gemm_layernorm",
        "gemm_layernorm_batch1",
        "gemm_layernorm_gelu",
        "gemm_layernorm_gelu_batch1",
        "gemm_layernorm_relu",
        "gemm_layernorm_relu_batch1",
        "gemm_layernorm_silu",
        "gemm_layernorm_silu_batch1",
        "gemm_layernorm_3d",
        "gemm_layernorm_3d_batch1",
        "gemm_layernorm_3d_gelu",
        "gemm_layernorm_3d_gelu_batch1",
        "gemm_layernorm_3d_relu",
        "gemm_layernorm_3d_relu_batch1",
        "gemm_layernorm_3d_silu",
        "gemm_layernorm_3d_silu_batch1",
        "gemm_gelu",
        "gemm_gelu_batch1",
        "gemm_gelu_3d",
        "gemm_gelu_3d_batch1",
        "gemm_silu",
        "gemm_silu_batch1",
        "gemm_silu_3d",
        "gemm_silu_3d_batch1",
        "gemm_relu",
        "gemm_relu_batch1",
        "gemm_relu_3d",
        "gemm_relu_3d_batch1",
        "gemm_bias",
        "gemm_bias_batch1",
        "gemm_bias_relu_batch1",
        "gemm_bias_relu",
        "gemm_bias_gelu",
        "gemm_bias_gelu_batch1",
        "gemm_bias_silu",
        "gemm_bias_silu_batch1",
        "gemm_bias_3d",
        "gemm_bias_3d_batch1",
        "gemm_bias_3d_relu",
        "gemm_bias_3d_relu_batch1",
        "gemm_bias_3d_gelu_batch1",
        "gemm_bias_3d_silu_batch1",
        "gemm_bias_3d_gelu",
        "gemm_bias_3d_silu",
        "gated_mlp",
        "gated_mlp_gelu",
        "gated_mlp_batch1",
        "gated_mlp_gelu_batch1",
        "gated_mlp_batched",
        "gated_mlp_batched_gelu",
        "gated_mlp_3d",
        "gated_mlp_3d_gelu",
        "gated_mlp_3d_batch1",
        "gated_mlp_3d_gelu_batch1",
        "gated_mlp_batched_batch1",
        "gated_mlp_batched_gelu_batch1",
        "rms_norm_linear",
        "rms_norm_linear_batch1",
        "rms_norm_linear_3d",
        "rms_norm_linear_3d_batch1",
        "rms_norm_linear_3d_gelu_batch1",
        "rms_norm_linear_3d_relu_batch1",
        "rms_norm_linear_3d_silu_batch1",
        "rms_norm_linear_3d_gelu",
        "rms_norm_linear_3d_relu",
        "rms_norm_linear_3d_silu",
        "rms_norm_linear_gelu",
        "rms_norm_linear_gelu_batch1",
        "rms_norm_linear_relu",
        "rms_norm_linear_relu_batch1",
        "rms_norm_linear_silu",
        "rms_norm_linear_silu_batch1",
        "self_attention",
        "self_attention_batch1",
        "self_attention_scaled",
        "self_attention_scaled_batch1",
        "self_attention_online",
        "self_attention_online_batch1",
        "self_attention_multi_head",
        "self_attention_multi_head_batch1",
        "self_attention_batched",
        "self_attention_batched_batch1",
        "self_attention_3d",
        "self_attention_3d_batch1",
        "self_attention_scaled_3d",
        "self_attention_scaled_3d_batch1",
        "conv2d_bias",
        "conv2d_bias_batch2",
        "conv2d_bias_relu",
        "conv2d_bias_relu_batch2",
        "conv2d_bias_gelu",
        "conv2d_bias_gelu_batch2",
        "conv2d_bias_silu",
        "conv2d_bias_silu_batch2",
        "conv2d_groups",
        "conv2d_groups_batch2",
        "conv2d_bias_groups",
        "conv2d_bias_groups_batch2",
        "conv2d_depthwise_bias",
        "conv2d_depthwise_bias_batch2",
        "conv2d_depthwise_bias_relu",
        "conv2d_depthwise_bias_relu_batch2",
        "conv2d_depthwise_bias_gelu",
        "conv2d_depthwise_bias_gelu_batch2",
        "conv2d_depthwise_bias_silu",
        "conv2d_depthwise_bias_silu_batch2",
        "conv2d_separable",
        "conv2d_separable_batch2",
        "conv2d_separable_bias",
        "conv2d_separable_bias_batch2",
        "conv2d_separable_bias_relu",
        "conv2d_separable_bias_gelu",
        "conv2d_separable_bias_silu",
    ],
)
def test_general_ml_compound_patterns_match_pytorch(pattern: str):
    kg, inputs, ref = CUSTOMIZED_OP_BUILDERS[pattern]()
    ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"{pattern} max_err={max_err}"


@pytest.mark.parametrize(
    "op_type",
    [
        "tb_forloop_accum_no_red_rescale_op",
        "tb_forloop_accum_red_ld_sum_rescale_op",
    ],
)
def test_online_softmax_rescale_accum_matches_reference(op_type: str):
    kg, inputs, ref = TB_OP_BUILDERS[op_type]()
    ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"{op_type} max_err={max_err}"


def test_kn_softmax_stable_vs_f_softmax():
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(16, 32), dtype=yr.float16)
    g.mark_output(g.softmax(x, dim=-1))
    inp = _f16((16, 32))
    ref = F.softmax(inp.float(), dim=-1).to(torch.float16)
    ok, max_err, err = runtime_verify_mugraph(g, [inp], ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"softmax max_err={max_err}"


def test_kn_layer_norm_vs_f_layer_norm_eps_zero():
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(16, 32), dtype=yr.float16)
    g.mark_output(g.layer_norm(x, normalized_shape=(32,), eps=0.0))
    inp = _f16((16, 32))
    ref = F.layer_norm(inp.float(), (32,), eps=0.0).to(torch.float16)
    ok, max_err, err = runtime_verify_mugraph(g, [inp], ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"layer_norm max_err={max_err}"
