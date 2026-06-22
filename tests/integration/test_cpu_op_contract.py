# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU op contract tests driven by docs/cpu_support_matrix.yaml."""

from __future__ import annotations

import os

import pytest
import torch

from tests.integration.cpu_op_builders import CUSTOMIZED_OP_BUILDERS, KN_OP_BUILDERS

pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


@pytest.fixture(autouse=True)
def _force_interpreter_for_unary_kn_ops(monkeypatch):
    """Exercise KN interpreter for ops that also have cpu_call fast paths."""
    monkeypatch.setenv("YIRAGE_CPU_RMS_MATMUL_FAST", "0")


@pytest.mark.parametrize(
    "op_type",
    sorted(KN_OP_BUILDERS),
)
def test_supported_kn_op_matches_torch(op_type: str):
    from yirage.backends.cpu.support_matrix import kn_op_contracts
    from yirage.search.verifier_config import runtime_verify_mugraph

    tier = kn_op_contracts()[op_type].tier
    assert tier in ("supported", "fast_path")

    kg, inputs, ref = KN_OP_BUILDERS[op_type]()
    ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"{op_type} max_err={max_err}"


def test_support_matrix_covers_kn_builders():
    from yirage.backends.cpu.support_matrix import kn_op_contracts

    contracts = kn_op_contracts()
    for op in KN_OP_BUILDERS:
        assert op in contracts
        assert contracts[op].tier in ("supported", "fast_path")


def test_search_explore_aligned_with_cpu_matrix():
    from yirage.backends.cpu.support_matrix import cpu_search_explore_not_supported

    gaps = cpu_search_explore_not_supported()
    assert gaps == [], f"CPU search still explores unsupported ops: {gaps}"


@pytest.mark.parametrize(
    "pattern_name",
    sorted(CUSTOMIZED_OP_BUILDERS),
)
def test_customized_op_interpreter_matches_torch(pattern_name: str):
    from yirage.search.verifier_config import runtime_verify_mugraph

    kg, inputs, ref = CUSTOMIZED_OP_BUILDERS[pattern_name]()
    ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"{pattern_name} max_err={max_err}"


def test_plain_matmul_fast_path_when_enabled():
    from yirage.kernel.graph import KNGraph, _is_plain_matmul_mugraph

    kg, inputs, ref = KN_OP_BUILDERS["kn_matmul_op"]()
    assert _is_plain_matmul_mugraph(kg.cygraph)
    out = kg(inputs=inputs)[0]
    assert torch.allclose(out, ref, rtol=0.02, atol=0.08)
