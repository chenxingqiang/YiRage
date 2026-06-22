# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""E2E: MLIR JIT correctness and speed vs Python interpreter."""

from __future__ import annotations

import os

import pytest
import torch

from yirage.kernel.cpu_mlir_jit import (
    bench_jit_vs_interpreter,
    compare_hand_tiled_vs_dialect_lowered_jit,
    emit_rms_matmul_mlir,
    extract_rms_matmul_tiling,
    is_mlir_jit_available,
    rms_matmul_mlir_emit_path,
    try_rms_matmul_jit,
)
from yirage.kernel.graph import KNGraph, _interpret_mugraph_on_cpu_impl

pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


from scripts.cpu_cert_utils import apply_rms_matmul_search_tractability


def _apply_rms_matmul_search_tractability() -> None:
    """Cap CPU search for fused MLIR e2e (mirrors bench ``rms_norm_matmul`` tractability)."""
    apply_rms_matmul_search_tractability()


def _superoptimize_rms_matmul_fused(
    m: int = 32,
    k: int = 64,
    n: int = 128,
):
    """Return fused rms_norm+matmul µGraph with M-grid and K-forloop tiling."""
    import tempfile

    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(x, normalized_shape=(k,)), w))

    prev_home = os.environ.get("HOME")
    _apply_rms_matmul_search_tractability()
    with tempfile.TemporaryDirectory(prefix="yirage_mlir_fuse_") as tmp:
        os.environ["HOME"] = tmp
        try:
            import yirage.storage.mugraph_store as ms

            ms._default_store = None
            opt = g.superoptimize(
                backend="cpu",
                griddims=[(2, 1, 1)],
                blockdims=[(32, 1, 1)],
                franges=[2],
                use_ray=False,
                use_graph_dataset=False,
                use_cached_graphs=False,
                use_persistent_cache=False,
                warmup_iters=1,
                profile_iters=2,
            )
        finally:
            if prev_home is not None:
                os.environ["HOME"] = prev_home
            else:
                os.environ.pop("HOME", None)

    assert opt is not None
    return opt


def test_tiled_mlir_emission_has_scf_for():
    tiling = __import__(
        "yirage.kernel.cpu_mlir_jit", fromlist=["RmsMatmulTiling"]
    ).RmsMatmulTiling(4, 1, 1, 1, 32, 32, 128)
    text = emit_rms_matmul_mlir(128, 256, 128, tiling=tiling)
    assert "scf.for" in text
    assert "scf.for %bx" in text
    assert "arith.addi %m0, %mi" in text


def test_extract_tiling_from_unfused_graph():
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(64, 32), dtype=yr.float16)
    w = g.new_input(dims=(32, 16), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(x, normalized_shape=(32,)), w))
    t = extract_rms_matmul_tiling(g.cygraph)
    assert t is not None
    assert t.grid_m == 1
    assert t.m_tile == 64


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_mlir_jit_correctness_unfused():
    import yirage as yr

    m, k, n = 32, 64, 48
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(x, normalized_shape=(k,)), w))

    xt = torch.randn(m, k, dtype=torch.float16)
    wt = torch.randn(k, n, dtype=torch.float16)
    ref = _interpret_mugraph_on_cpu_impl(g.cygraph, [xt, wt])[0]

    os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
    out = try_rms_matmul_jit(g.cygraph, [xt, wt])
    assert out is not None
    assert torch.allclose(out[0], ref, rtol=0.05, atol=0.1)


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_mlir_jit_fused_customized_op_correctness():
    """Fused kn_customized_op from superoptimize works with dialect + hand JIT."""
    from yirage.kernel import cpu_mlir_jit as cmj

    cmj._JIT_CACHE.clear()
    m, k, n = 32, 64, 128
    opt = _superoptimize_rms_matmul_fused(m, k, n)
    cy = opt.cygraph
    ops = {o["op_type"] for o in cy.get_graph_structure()}
    assert "kn_customized_op" in ops

    from yirage.kernel.cpu_mlir_jit import (
        emit_dialect_mlir_from_cygraph,
        is_rms_matmul_mugraph,
    )

    assert is_rms_matmul_mugraph(cy)
    dialect = emit_dialect_mlir_from_cygraph(cy)
    assert dialect is not None
    assert "yirage.rms_norm" in dialect
    assert "yirage.matmul" in dialect

    xt = torch.randn(m, k, dtype=torch.float16)
    wt = torch.randn(k, n, dtype=torch.float16)
    ref = _interpret_mugraph_on_cpu_impl(cy, [xt, wt])[0]

    os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
    os.environ["YIRAGE_CPU_MLIR_JIT_DIALECT"] = "1"
    try:
        out_dialect = try_rms_matmul_jit(cy, [xt, wt])
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_DIALECT", None)

    assert out_dialect is not None
    assert torch.allclose(out_dialect[0], ref, rtol=0.05, atol=0.12)

    cmj._JIT_CACHE.clear()
    out_hand = try_rms_matmul_jit(cy, [xt, wt])
    assert out_hand is not None
    assert torch.allclose(out_hand[0], ref, rtol=0.05, atol=0.12)

    cmj._JIT_CACHE.clear()
    os.environ["YIRAGE_CPU_MLIR_JIT_BLAS"] = "0"
    try:
        out_blas_off = try_rms_matmul_jit(cy, [xt, wt])
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_BLAS", None)
    assert out_blas_off is not None
    assert torch.allclose(out_blas_off[0], ref, rtol=0.05, atol=0.12)

    align = compare_hand_tiled_vs_dialect_lowered_jit(cy, [xt, wt])
    assert align.get("error") is None, align
    assert align["aligned"] is True
    assert align["max_abs_diff"] < 0.2


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_superoptimized_rms_matmul_emit_path_is_hand_bgrid_tiled():
    """Fused bgrid tiling from superoptimize selects hand_bgrid_tiled JIT emit (R48)."""
    from yirage.kernel import cpu_mlir_jit as cmj

    cmj._JIT_CACHE.clear()
    opt = _superoptimize_rms_matmul_fused()
    cy = opt.cygraph
    tiling = extract_rms_matmul_tiling(cy)
    assert tiling is not None
    assert tiling.uses_loops

    os.environ.pop("YIRAGE_CPU_MLIR_JIT_DIALECT", None)
    os.environ["YIRAGE_CPU_MLIR_JIT_PRESERVE_TILING"] = "1"
    try:
        assert rms_matmul_mlir_emit_path(cy) == "hand_bgrid_tiled"
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_PRESERVE_TILING", None)


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_superoptimized_rms_matmul_emit_path_is_dialect_lowered_when_dialect_enabled():
    """Fused superoptimize + DIALECT=1 selects dialect_lowered over hand_bgrid_tiled (R50)."""
    from yirage.kernel.cpu_mlir_jit import (
        _yirage_cpu_opt_path,
    )
    from yirage.kernel import cpu_mlir_jit as cmj

    if _yirage_cpu_opt_path() is None:
        pytest.skip("yirage-cpu-opt not built")

    cmj._JIT_CACHE.clear()
    opt = _superoptimize_rms_matmul_fused()
    cy = opt.cygraph

    os.environ["YIRAGE_CPU_MLIR_JIT_DIALECT"] = "1"
    os.environ["YIRAGE_CPU_MLIR_JIT_PRESERVE_TILING"] = "1"
    try:
        assert rms_matmul_mlir_emit_path(cy) == "dialect_lowered"
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_DIALECT", None)
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_PRESERVE_TILING", None)


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_mlir_jit_speed_report():
    """Record JIT vs interpreter latency; JIT should run without crashing."""
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(64, 128), dtype=yr.float16)
    w = g.new_input(dims=(128, 256), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(x, normalized_shape=(128,)), w))
    kg = KNGraph(g.cygraph, backend="cpu")

    ins = [
        torch.randn(64, 128, dtype=torch.float16),
        torch.randn(128, 256, dtype=torch.float16),
    ]
    report = bench_jit_vs_interpreter(kg.cygraph, ins, warmup=5, iters=40)
    assert report["ok"], report.get("error", "jit bench failed")
    assert report["interpreter_ms"] > 0
    assert report["mlir_jit_ms"] > 0
    # On small shapes JIT compile amortization may lose; we only assert both paths ran.
    assert report["speedup_interp_over_mlir_jit"] > 0
