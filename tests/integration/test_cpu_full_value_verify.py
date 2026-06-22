# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Matrix-driven CPU value verification: every supported KN/TB op and host primitive
vs torch reference (docs/cpu_support_matrix.yaml).
"""

from __future__ import annotations

import os

import pytest
import torch

from tests.integration.cpu_op_builders import (
    CUSTOMIZED_OP_BUILDERS,
    FAST_PATH_BUILDERS,
    KN_OP_BUILDERS,
    LAYOUT_EXPLORE_BUILDERS,
)
from tests.integration.cpu_tb_op_builders import (
    TB_LAYOUT_EXPLORE_BUILDERS,
    TB_OP_BUILDERS,
    TB_UNSUPPORTED_BUILDERS,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


@pytest.fixture(autouse=True)
def _force_kn_interpreter_for_dual_path_ops(monkeypatch):
    monkeypatch.setenv("YIRAGE_CPU_RMS_MATMUL_FAST", "0")


@pytest.mark.parametrize("op_type", sorted(KN_OP_BUILDERS))
def test_kn_op_value_matches_torch(op_type: str):
    from yirage.backends.cpu.support_matrix import kn_op_contracts
    from yirage.search.verifier_config import runtime_verify_mugraph

    assert kn_op_contracts()[op_type].tier in ("supported", "fast_path")
    kg, inputs, ref = KN_OP_BUILDERS[op_type]()
    ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"{op_type} max_err={max_err}"


@pytest.mark.parametrize("op_type", sorted(TB_OP_BUILDERS))
def test_tb_op_value_matches_torch(op_type: str):
    from yirage.backends.cpu.support_matrix import tb_op_contracts
    from yirage.search.verifier_config import runtime_verify_mugraph

    tier = tb_op_contracts()[op_type].tier
    assert tier in ("supported", "experimental")
    tol = 0.15 if tier == "experimental" else 0.12
    kg, inputs, ref = TB_OP_BUILDERS[op_type]()
    ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=tol)
    assert err is None, err
    assert ok, f"{op_type} max_err={max_err}"


@pytest.mark.parametrize("pattern", sorted(LAYOUT_EXPLORE_BUILDERS))
def test_kn_layout_explore_graph_value_matches_torch(pattern: str):
    from yirage.search.verifier_config import runtime_verify_mugraph

    kg, inputs, ref = LAYOUT_EXPLORE_BUILDERS[pattern]()
    ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"{pattern} max_err={max_err}"


@pytest.mark.parametrize("pattern", sorted(TB_LAYOUT_EXPLORE_BUILDERS))
def test_tb_layout_explore_graph_value_matches_torch(pattern: str):
    from yirage.search.verifier_config import runtime_verify_mugraph

    kg, inputs, ref = TB_LAYOUT_EXPLORE_BUILDERS[pattern]()
    ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"{pattern} max_err={max_err}"


@pytest.mark.parametrize("pattern", sorted(CUSTOMIZED_OP_BUILDERS))
def test_customized_pattern_value_matches_torch(pattern: str):
    from yirage.search.verifier_config import runtime_verify_mugraph

    kg, inputs, ref = CUSTOMIZED_OP_BUILDERS[pattern]()
    ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"{pattern} max_err={max_err}"


@pytest.mark.parametrize("name", sorted(FAST_PATH_BUILDERS))
def test_fast_path_value_matches_torch(name: str):
    from yirage.search.verifier_config import runtime_verify_mugraph

    kg, inputs, ref = FAST_PATH_BUILDERS[name]()
    ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=0.12)
    assert err is None, err
    assert ok, f"{name} max_err={max_err}"


def test_fast_path_rms_matmul_with_blas_enabled():
    from yirage.search.verifier_config import runtime_verify_mugraph

    kg, inputs, ref = FAST_PATH_BUILDERS["kn_unfused_rms_matmul"]()
    os.environ["YIRAGE_CPU_RMS_MATMUL_FAST"] = "1"
    try:
        ok, max_err, err = runtime_verify_mugraph(kg, inputs, ref, max_abs_tol=0.12)
    finally:
        os.environ.pop("YIRAGE_CPU_RMS_MATMUL_FAST", None)
    assert err is None, err
    assert ok, f"blas rms_matmul max_err={max_err}"


def test_cpu_native_primitives_match_torch():
    from yirage.kernel.cpu_native import cpu_matmul, cpu_rms_matmul, cpu_rms_norm

    x = torch.randn(12, 48, dtype=torch.float16)
    w = torch.randn(48, 24, dtype=torch.float16)
    a = torch.randn(8, 32, dtype=torch.float16)
    b = torch.randn(32, 16, dtype=torch.float16)

    ref_norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    out_norm = cpu_rms_norm(x)
    assert torch.allclose(out_norm.float(), ref_norm, atol=0.08)

    ref_mm = torch.matmul(a, b)
    assert torch.allclose(cpu_matmul(a, b), ref_mm, atol=0.08)

    ref_rms_mm = torch.matmul(
        x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6),
        w.float(),
    )
    assert torch.allclose(cpu_rms_matmul(x, w), ref_rms_mm.to(x.dtype), atol=0.1)


@pytest.mark.skipif(not TB_UNSUPPORTED_BUILDERS, reason="no unsupported tb ops in matrix")
@pytest.mark.parametrize("op_type", sorted(TB_UNSUPPORTED_BUILDERS))
def test_unsupported_tb_op_raises(op_type: str):
    from yirage.backends.cpu.support_matrix import tb_op_contracts

    assert tb_op_contracts()[op_type].tier == "unsupported"
    kg, inputs, _ = TB_UNSUPPORTED_BUILDERS[op_type]()
    with pytest.raises(NotImplementedError, match="CPU .* does not support"):
        kg(inputs=inputs)


def test_matrix_kn_coverage_complete():
    from yirage.backends.cpu.support_matrix import cpu_verifiable_kn_ops

    expected = set(cpu_verifiable_kn_ops()) - {"kn_customized_op"}
    covered = set(KN_OP_BUILDERS) | set(FAST_PATH_BUILDERS)
  # kn_matmul covered by both KN_OP_BUILDERS and FAST_PATH
    missing = expected - set(KN_OP_BUILDERS)
    assert not missing, f"KN verifiers missing for: {sorted(missing)}"


def test_matrix_tb_coverage_complete():
    from yirage.backends.cpu.support_matrix import cpu_verifiable_tb_ops

    expected = set(cpu_verifiable_tb_ops())
    missing = expected - set(TB_OP_BUILDERS)
    assert not missing, f"TB verifiers missing for: {sorted(missing)}"
