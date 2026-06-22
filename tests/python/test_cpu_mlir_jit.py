# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Tests for CPU MLIR JIT helpers (no native MLIR required for most cases)."""

from __future__ import annotations

import os

import pytest
import torch

from yirage.kernel.cpu_mlir_jit import (
    RmsMatmulTiling,
    bench_hand_vs_dialect_lowered_jit,
    blas_fast_path_enabled,
    compare_hand_tiled_vs_dialect_lowered_jit,
    emit_rms_matmul_mlir,
    extract_rms_matmul_tiling,
    is_mlir_jit_available,
    is_rms_matmul_mugraph,
    mlir_jit_enabled,
    rms_matmul_shapes_from_cygraph,
    should_use_blas_fast_path,
    try_rms_matmul_jit,
)


def test_tiled_emit_includes_scf_for():
    tiling = RmsMatmulTiling(2, 1, 1, 1, 64, 64, 128)
    text = emit_rms_matmul_mlir(128, 256, 128, tiling=tiling)
    assert "scf.for" in text
    assert "yirage.grid_m" in text
    assert "scf.for %bx" in text
    assert "arith.addi %m0, %mi" in text
    assert "memref.subview" not in text


def test_k_tiled_emit_includes_forloop_attr():
    tiling = RmsMatmulTiling(1, 1, 1, 2, 64, 32, 128)
    text = emit_rms_matmul_mlir(64, 64, 128, tiling=tiling)
    assert "yirage.forloop_k" in text
    assert "%cFk" in text


def test_blas_fast_path_threshold():
    assert should_use_blas_fast_path(128, 256, 512)
    assert not should_use_blas_fast_path(8, 16, 32)
    assert blas_fast_path_enabled()


def test_extract_tiling_defaults():
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(8, 16), dtype=yr.float16)
    w = g.new_input(dims=(16, 32), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(x, normalized_shape=(16,)), w))
    t = extract_rms_matmul_tiling(g.cygraph)
    assert t is not None
    assert t.grid_m == 1


def test_emit_rms_matmul_mlir_contains_ops():
    text = emit_rms_matmul_mlir(64, 128, 256)
    assert "func.func @mugraph" in text
    assert "scf.for" in text
    assert "arith.extf" in text  # fp16 f32 accum matmul
    assert "math.rsqrt" in text
    assert "memref<64x128xf16>" in text
    assert "memref<128x256xf16>" in text
    assert "memref<64x256xf16>" in text


def test_mlir_jit_availability_smoke():
    """CPU MLIR JIT smoke: import helpers; native ext optional (USE_MLIR=1)."""
    from yirage.kernel.cpu_mlir_jit import is_mlir_jit_available, mlir_jit_enabled

    assert isinstance(is_mlir_jit_available(), bool)
    assert isinstance(mlir_jit_enabled(), bool)


def test_is_rms_matmul_unfused_graph():
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(32, 64), dtype=yr.float16)
    w = g.new_input(dims=(64, 128), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(64,))
    g.mark_output(g.matmul(normed, w))
    assert is_rms_matmul_mugraph(g.cygraph)
    assert rms_matmul_shapes_from_cygraph(g.cygraph) == (32, 64, 128)


def test_try_jit_disabled_by_default():
    import yirage as yr

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(4, 8), dtype=yr.float16)
    w = g.new_input(dims=(8, 16), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(8,))
    g.mark_output(g.matmul(normed, w))
    ins = [
        torch.randn(4, 8, dtype=torch.float16),
        torch.randn(8, 16, dtype=torch.float16),
    ]
    prev = os.environ.pop("YIRAGE_CPU_MLIR_JIT", None)
    try:
        assert try_rms_matmul_jit(g.cygraph, ins) is None
    finally:
        if prev is not None:
            os.environ["YIRAGE_CPU_MLIR_JIT"] = prev


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_tiled_hand_mlir_compiles_in_jit_kernel():
    """Bgrid M-tile hand emit must parse and JIT-compile (not only flat fallback)."""
    from yirage import _yirage_mlir as mlir

    tiling = RmsMatmulTiling(2, 1, 1, 1, 16, 64, 128)
    text = emit_rms_matmul_mlir(32, 64, 128, tiling=tiling)
    assert "attributes {yirage.grid_m" in text
    assert "memref<32x128xf16> attributes" not in text
    kernel = mlir.CPUJITKernel()
    assert kernel.compile_mlir(text, "mugraph"), kernel.last_error()


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_rms_matmul_jit_medium_shape_fp16_accum():
    """Hand JIT with f32 accumulators (BLAS fast path disabled)."""
    import yirage as yr

    m, k, n = 64, 128, 256
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(k,))
    g.mark_output(g.matmul(normed, w))

    xt = torch.randn(m, k, dtype=torch.float16)
    wt = torch.randn(k, n, dtype=torch.float16)
    ref = torch.matmul(
        xt.float() * torch.rsqrt(xt.float().pow(2).mean(-1, keepdim=True) + 1e-6),
        wt.float(),
    ).to(torch.float16)

    os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
    os.environ["YIRAGE_CPU_MLIR_JIT_BLAS"] = "0"
    try:
        out = try_rms_matmul_jit(g.cygraph, [xt, wt])
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_BLAS", None)
        os.environ.pop("YIRAGE_CPU_MLIR_JIT", None)

    assert out is not None
    assert torch.allclose(out[0].float(), ref.float(), rtol=0.02, atol=0.15)


def test_rms_matmul_jit_large_shape_uses_blas_fast_path():
    import yirage as yr

    from yirage.kernel.cpu_mlir_jit import _torch_rms_matmul

    m, k, n = 128, 256, 512
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(x, normalized_shape=(k,)), w))

    xt = torch.randn(m, k, dtype=torch.float16)
    wt = torch.randn(k, n, dtype=torch.float16)
    ref = _torch_rms_matmul(xt, wt)

    os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
    try:
        out = try_rms_matmul_jit(g.cygraph, [xt, wt])
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT", None)

    assert out is not None
    assert torch.allclose(out[0], ref, rtol=0.02, atol=0.15)


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_rms_matmul_jit_e2e():
    import yirage as yr

    m, k, n = 16, 32, 48
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(k,))
    g.mark_output(g.matmul(normed, w))

    xt = torch.randn(m, k, dtype=torch.float16)
    wt = torch.randn(k, n, dtype=torch.float16)
    ref = torch.matmul(
        xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-6),
        wt,
    )

    os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
    out = try_rms_matmul_jit(g.cygraph, [xt, wt])
    assert out is not None
    assert torch.allclose(out[0], ref, rtol=0.05, atol=0.1)


def test_fused_customized_split_kn_matmul_exports_dialect():
    """kn_customized (RMS bgraph) + kn_matmul at KN level exports rms then matmul."""
    from yirage.kernel.cpu_mlir_jit import emit_dialect_mlir_from_cygraph

    class _CyGraph:
        def __init__(self, ops, inputs):
            self._ops = ops
            self._inputs = inputs

        def get_graph_structure(self):
            return self._ops

        def get_input_dtensors(self):
            return self._inputs

        def get_input_dtensor_shape_and_stride(self, dt):
            dims = [int(d) for d in dt["dim"] if int(d) > 0]
            return tuple(dims), tuple([1] * len(dims))

    x_dt = {"guid": 1, "dim": [8, 16, 0, 0], "num_dims": 2, "dtype": "fp16"}
    w_dt = {"guid": 2, "dim": [16, 32, 0, 0], "num_dims": 2, "dtype": "fp16"}
    norm_dt = {"guid": 10, "dim": [8, 16, 0, 0], "num_dims": 2, "dtype": "fp16"}
    out_dt = {"guid": 3, "dim": [8, 32, 0, 0], "num_dims": 2, "dtype": "fp16"}
    ops = [
        {"op_type": "kn_input_op", "output_tensors": [x_dt]},
        {"op_type": "kn_input_op", "output_tensors": [w_dt]},
        {
            "op_type": "kn_customized_op",
            "input_tensors": [x_dt, w_dt],
            "output_tensors": [norm_dt],
            "bgraph": {
                "operators": [
                    {"op_type": "tb_input_op"},
                    {"op_type": "tb_rms_norm_op"},
                    {"op_type": "tb_output_op"},
                ],
            },
        },
        {
            "op_type": "kn_matmul_op",
            "input_tensors": [norm_dt, w_dt],
            "output_tensors": [out_dt],
        },
        {"op_type": "kn_output_op", "input_tensors": [out_dt]},
    ]
    g = type("G", (), {"cygraph": _CyGraph(ops, [x_dt, w_dt])})()
    text = emit_dialect_mlir_from_cygraph(g.cygraph)
    assert text is not None
    assert "yirage.rms_norm" in text
    assert "yirage.matmul" in text
    assert text.index("yirage.rms_norm") < text.index("yirage.matmul")


def test_fused_customized_bgraph_exports_yirage_dialect():
    from yirage.kernel.cpu_mlir_jit import emit_dialect_mlir_from_cygraph

    class _CyGraph:
        def __init__(self, ops, inputs):
            self._ops = ops
            self._inputs = inputs

        def get_graph_structure(self):
            return self._ops

        def get_input_dtensors(self):
            return self._inputs

        def get_input_dtensor_shape_and_stride(self, dt):
            dims = [int(d) for d in dt["dim"] if int(d) > 0]
            return tuple(dims), tuple([1] * len(dims))

    x_dt = {"guid": 1, "dim": [8, 16, 0, 0], "num_dims": 2, "dtype": "fp16"}
    w_dt = {"guid": 2, "dim": [16, 32, 0, 0], "num_dims": 2, "dtype": "fp16"}
    out_dt = {"guid": 3, "dim": [8, 32, 0, 0], "num_dims": 2, "dtype": "fp16"}
    ops = [
        {"op_type": "kn_input_op", "output_tensors": [x_dt]},
        {"op_type": "kn_input_op", "output_tensors": [w_dt]},
        {
            "op_type": "kn_customized_op",
            "input_tensors": [x_dt, w_dt],
            "output_tensors": [out_dt],
            "bgraph": {
                "grid_dim": {"x": 2, "y": 1, "z": 1},
                "forloop_range": 2,
                "operators": [
                    {"op_type": "tb_input_op"},
                    {"op_type": "tb_input_op"},
                    {"op_type": "tb_forloop_accum_red_ld_rms_op"},
                    {"op_type": "tb_matmul_op"},
                    {"op_type": "tb_output_op"},
                ],
            },
        },
        {"op_type": "kn_output_op", "input_tensors": [out_dt]},
    ]
    g = type("G", (), {"cygraph": _CyGraph(ops, [x_dt, w_dt])})()
    text = emit_dialect_mlir_from_cygraph(g.cygraph)
    assert text is not None
    assert "yirage.rms_norm" in text
    assert "yirage.matmul" in text
    assert "yirage.grid_m" in text
    assert "yirage.forloop_k" in text
    assert "kn_customized" not in text
    assert "yirage.custom" not in text


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="USE_MLIR=1 build with _yirage_mlir required",
)
def test_cpu_jit_pipeline_has_no_trailing_memref_copy():
    import os
    import subprocess
    from pathlib import Path

    from yirage.kernel.cpu_mlir_jit import emit_dialect_mlir_from_cygraph

    import yirage as yr

    m, k, n = 4, 8, 16
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(x, normalized_shape=(k,)), w))
    dialect = emit_dialect_mlir_from_cygraph(g.cygraph)
    assert dialect is not None

    opt = Path(__file__).resolve().parents[2] / "build" / "mlir" / "yirage-cpu-opt"
    if not opt.is_file():
        pytest.skip("yirage-cpu-opt not built")
    proc = subprocess.run(
        [str(opt), "-", "-yirage-cpu-jit-pipeline"],
        input=dialect,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "memref.copy" not in proc.stdout
    assert "memref<4x16xf16>" in proc.stdout
    assert "-> (" not in proc.stdout.split("func.func @mugraph")[1].split("{")[0]


def test_cygraph_exports_yirage_dialect_rms_matmul():
    import yirage as yr

    from yirage.kernel.cpu_mlir_jit import emit_dialect_mlir_from_cygraph

    m, k, n = 8, 16, 32
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(k,))
    g.mark_output(g.matmul(normed, w))
    text = emit_dialect_mlir_from_cygraph(g.cygraph)
    assert text is not None
    assert "yirage.rms_norm" in text
    assert "yirage.matmul" in text
    assert "tensor<8x16xf16>" in text
    assert "tensor<8x32xf16>" in text


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="USE_MLIR=1 build with _yirage_mlir required",
)
def test_dialect_pipeline_fp16_accum_medium_shape():
    """Dialect yirage-to-linalg promotes fp16 RMS+matmul to f32 internally."""
    import yirage as yr

    from yirage.kernel.cpu_mlir_jit import _torch_rms_matmul

    m, k, n = 64, 128, 256
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(k,))
    g.mark_output(g.matmul(normed, w))

    xt = torch.randn(m, k, dtype=torch.float16)
    wt = torch.randn(k, n, dtype=torch.float16)
    ref = _torch_rms_matmul(xt, wt)

    os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
    os.environ["YIRAGE_CPU_MLIR_JIT_DIALECT"] = "1"
    os.environ["YIRAGE_CPU_MLIR_JIT_BLAS"] = "0"
    try:
        out = try_rms_matmul_jit(g.cygraph, [xt, wt])
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_BLAS", None)
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_DIALECT", None)
        os.environ.pop("YIRAGE_CPU_MLIR_JIT", None)

    assert out is not None
    assert torch.allclose(out[0], ref, rtol=0.02, atol=0.15)


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="USE_MLIR=1 build with _yirage_mlir required",
)
def test_dialect_pipeline_rms_matmul_jit():
    import yirage as yr

    m, k, n = 8, 16, 32
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(k,))
    g.mark_output(g.matmul(normed, w))

    xt = torch.randn(m, k, dtype=torch.float16)
    wt = torch.randn(k, n, dtype=torch.float16)
    ref = torch.matmul(
        xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + 1e-6),
        wt,
    )

    os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
    os.environ["YIRAGE_CPU_MLIR_JIT_DIALECT"] = "1"
    try:
        out = try_rms_matmul_jit(g.cygraph, [xt, wt])
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_DIALECT", None)
        os.environ.pop("YIRAGE_CPU_MLIR_JIT", None)

    assert out is not None
    assert torch.allclose(out[0], ref, rtol=0.05, atol=0.1)


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_rms_matmul_emit_path_is_dialect_lowered_when_dialect_enabled():
    import yirage as yr

    from yirage.kernel.cpu_mlir_jit import (
        _yirage_cpu_opt_path,
        rms_matmul_mlir_emit_path,
    )

    if _yirage_cpu_opt_path() is None:
        pytest.skip("yirage-cpu-opt not built")

    m, k, n = 16, 32, 64
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(k,))
    g.mark_output(g.matmul(normed, w))

    os.environ["YIRAGE_CPU_MLIR_JIT_DIALECT"] = "1"
    try:
        assert rms_matmul_mlir_emit_path(g.cygraph) == "dialect_lowered"
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_DIALECT", None)


def test_bench_jit_vs_interpreter_reports_emit_path():
    import yirage as yr

    from yirage.kernel.cpu_mlir_jit import bench_jit_vs_interpreter

    m, k, n = 16, 32, 64
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(k,))
    g.mark_output(g.matmul(normed, w))
    inputs = [torch.randn(m, k, dtype=torch.float16), torch.randn(k, n, dtype=torch.float16)]

    report = bench_jit_vs_interpreter(g.cygraph, inputs, warmup=1, iters=3)
    assert report.get("ok") is True
    assert report.get("mlir_jit_emit_path") in {
        "dialect_lowered",
        "dialect_raw",
        "hand_bgrid_tiled",
        "hand_tiled",
        "hand_flat",
    }


def test_mlir_jit_enabled_env():
    os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
    try:
        assert mlir_jit_enabled()
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT", None)


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_bench_hand_vs_dialect_lowered_jit_reports_timing():
    """Hand vs dialect lowered emit paths are benchmarked (Loop R42)."""
    import yirage as yr

    m, k, n = 32, 64, 128
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(x, normalized_shape=(k,)), w))

    xt = torch.randn(m, k, dtype=torch.float16)
    wt = torch.randn(k, n, dtype=torch.float16)
    report = bench_hand_vs_dialect_lowered_jit(
        g.cygraph, [xt, wt], warmup=2, iters=10
    )
    assert report.get("ok") is True, report
    assert report["hand_mlir_jit_ms"] > 0
    assert report["dialect_lowered_jit_ms"] > 0
    assert report["speedup_hand_over_dialect_lowered"] > 0
    assert report["mlir_hand_dialect_aligned"] is True


def test_hand_m_tiled_vs_dialect_lowered_numerical_alignment():
    """Hand M-grid emit matches yirage-cpu-jit-pipeline lowered output (Loop R40)."""
    import yirage as yr

    m, k, n = 32, 64, 128
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(x, normalized_shape=(k,)), w))

    xt = torch.randn(m, k, dtype=torch.float16)
    wt = torch.randn(k, n, dtype=torch.float16)
    tiling = RmsMatmulTiling(2, 1, 1, 1, 16, 64, 128)
    report = compare_hand_tiled_vs_dialect_lowered_jit(
        g.cygraph, [xt, wt], hand_tiling=tiling
    )
    assert report.get("error") is None, report
    assert report["aligned"] is True
    assert report["hand_path"] == "hand_synthetic_tiling"
    assert report["dialect_path"] == "yirage-cpu-jit-pipeline"
    assert report["max_abs_diff"] < 0.2


def test_rms_matmul_compile_candidates_end_with_flat_fallback():
    from yirage.kernel.cpu_mlir_jit import (
        RmsMatmulTiling,
        _rms_matmul_mlir_compile_candidates,
        emit_rms_matmul_mlir,
    )

    tiling = RmsMatmulTiling(2, 1, 1, 1, 16, 64, 128)
    flat = emit_rms_matmul_mlir(32, 64, 128, dtype="f16", tiling=None)
    candidates = _rms_matmul_mlir_compile_candidates(
        32, 64, 128, "f16", tiling, cygraph=None
    )
    assert candidates
    assert candidates[-1][0] == "hand_flat"
    assert candidates[-1][1] == flat


@pytest.mark.skipif(
    not is_mlir_jit_available(),
    reason="yirage._yirage_mlir not built (USE_MLIR=1 required)",
)
def test_dialect_enabled_prioritizes_lowered_before_hand_tiled():
    """YIRAGE_CPU_MLIR_JIT_DIALECT=1 tries lowered dialect before hand tiled emit."""
    import yirage as yr

    from yirage.kernel.cpu_mlir_jit import (
        RmsMatmulTiling,
        _rms_matmul_mlir_compile_candidates,
        emit_bgrid_tiled_mlir_from_cygraph,
        emit_dialect_mlir_from_cygraph,
        lower_dialect_mlir_via_cpu_opt,
    )

    m, k, n = 32, 64, 128
    g = yr.new_kernel_graph()
    x = g.new_input(dims=(m, k), dtype=yr.float16)
    w = g.new_input(dims=(k, n), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(k,))
    g.mark_output(g.matmul(normed, w))

    tiling = RmsMatmulTiling(2, 1, 1, 1, 16, k, n)
    os.environ["YIRAGE_CPU_MLIR_JIT_DIALECT"] = "1"
    os.environ["YIRAGE_CPU_MLIR_JIT_PRESERVE_TILING"] = "1"
    try:
        candidates = _rms_matmul_mlir_compile_candidates(
            m, k, n, "f16", tiling, g.cygraph
        )
        paths = [path for path, _ in candidates]
        dialect = emit_dialect_mlir_from_cygraph(g.cygraph)
        lowered = lower_dialect_mlir_via_cpu_opt(dialect) if dialect else None
        tiled_hand = emit_bgrid_tiled_mlir_from_cygraph(g.cygraph)
        assert dialect is not None
        if lowered is not None:
            assert candidates[0][1] == lowered
            assert paths[0] == "dialect_lowered"
            if tiled_hand is not None:
                assert paths.index("dialect_lowered") < paths.index("hand_bgrid_tiled")
        else:
            assert candidates[0][1] == dialect
            assert paths[0] == "dialect_raw"
    finally:
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_DIALECT", None)
        os.environ.pop("YIRAGE_CPU_MLIR_JIT_PRESERVE_TILING", None)
