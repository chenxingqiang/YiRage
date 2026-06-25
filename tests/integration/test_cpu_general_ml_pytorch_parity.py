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
        "kn_layer_norm",
        "gemm_softmax",
        "gemm_layernorm",
        "self_attention",
        "self_attention_scaled",
        "self_attention_multi_head",
        "self_attention_batched",
        "conv2d_bias",
        "conv2d_groups",
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
