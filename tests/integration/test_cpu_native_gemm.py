# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""CPU matmul uses host BLAS (MKL) by default; profile path matches runtime."""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.cpu_cert_utils import apply_rms_matmul_search_tractability

pytestmark = pytest.mark.skipif(
    os.environ.get("YIRAGE_SKIP_NATIVE") == "1",
    reason="native yirage.core not built",
)


def test_cpu_matmul_defaults_to_host_blas():
    from yirage.kernel.cpu_native import cpu_matmul, uses_host_blas

    assert uses_host_blas()
    a = torch.randn(8, 32, dtype=torch.float16)
    b = torch.randn(32, 64, dtype=torch.float16)
    ref = torch.matmul(a, b)
    out = cpu_matmul(a, b)
    assert torch.allclose(out, ref)


def test_cpu_rms_matmul_matches_torch_baseline():
    from yirage.kernel.cpu_native import cpu_rms_matmul

    x = torch.randn(16, 64, dtype=torch.float16)
    w = torch.randn(64, 32, dtype=torch.float16)
    ref = torch.matmul(
        x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6),
        w.float(),
    ).to(x.dtype)
    out = cpu_rms_matmul(x, w)
    assert torch.allclose(out, ref, rtol=0.01, atol=0.08)


def test_unfused_rms_matmul_mugraph_detection():
    import yirage as yr
    from yirage.kernel.graph import (
        _has_fused_customized_op,
        _is_plain_matmul_mugraph,
        _is_unfused_rms_matmul_mugraph,
    )

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(8, 32), dtype=yr.float16)
    w = g.new_input(dims=(32, 64), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(32,))
    g.mark_output(g.matmul(normed, w))
    assert not _is_plain_matmul_mugraph(g.cygraph)
    assert _is_unfused_rms_matmul_mugraph(g.cygraph)
    assert not _has_fused_customized_op(g.cygraph)


def test_cpu_call_skips_jit_without_experimental_flag():
    """P0: YIRAGE_CPU_MLIR_JIT=1 alone must not route cpu_call through LLVM JIT."""
    import os

    import yirage as yr
    from yirage.kernel.cpu_mlir_jit import mlir_jit_experimental_enabled, try_rms_matmul_jit
    from yirage.kernel.graph import KNGraph

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(8, 32), dtype=yr.float16)
    w = g.new_input(dims=(32, 64), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(32,))
    g.mark_output(g.matmul(normed, w))
    kg = KNGraph(g.cygraph, backend="cpu")

    prev_jit = os.environ.pop("YIRAGE_CPU_MLIR_JIT", None)
    prev_exp = os.environ.pop("YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL", None)
    try:
        os.environ["YIRAGE_CPU_MLIR_JIT"] = "1"
        assert not mlir_jit_experimental_enabled()
        ref_x = torch.randn(8, 32, dtype=torch.float16)
        ref_w = torch.randn(32, 64, dtype=torch.float16)
        assert try_rms_matmul_jit(g.cygraph, [ref_x, ref_w], require_experimental=True) is None
        out = kg(inputs=[ref_x, ref_w])[0]
        ref = torch.matmul(
            ref_x.float()
            * torch.rsqrt(ref_x.float().pow(2).mean(-1, keepdim=True) + 1e-6),
            ref_w.float(),
        ).to(ref_x.dtype)
        assert torch.allclose(out, ref, rtol=0.01, atol=0.08)
    finally:
        if prev_jit is not None:
            os.environ["YIRAGE_CPU_MLIR_JIT"] = prev_jit
        else:
            os.environ.pop("YIRAGE_CPU_MLIR_JIT", None)
        if prev_exp is not None:
            os.environ["YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL"] = prev_exp
        else:
            os.environ.pop("YIRAGE_CPU_MLIR_JIT_EXPERIMENTAL", None)


def test_plain_matmul_fast_path_not_used_for_rms_matmul():
    """rms_norm+matmul must not hit the plain-matmul MKL fast path."""
    import yirage as yr
    from yirage.kernel.graph import KNGraph, _is_plain_matmul_mugraph

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(8, 32), dtype=yr.float16)
    w = g.new_input(dims=(32, 64), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(32,))
    g.mark_output(g.matmul(normed, w))
    kg = KNGraph(g.cygraph, backend="cpu")
    assert not _is_plain_matmul_mugraph(g.cygraph)

    ref_x = torch.randn(8, 32, dtype=torch.float16)
    ref_w = torch.randn(32, 64, dtype=torch.float16)
    ref = torch.matmul(
        ref_x * torch.rsqrt(ref_x.pow(2).mean(-1, keepdim=True) + 1e-6),
        ref_w,
    )
    out = kg(inputs=[ref_x, ref_w])[0]
    assert torch.allclose(out, ref, rtol=0.01, atol=0.08)


def test_production_rms_matmul_detects_fused_customized():
    import os
    import tempfile

    import yirage as yr
    from yirage.kernel.cpu_mlir_jit import is_production_rms_matmul_mugraph

    rm, rk, rn = 32, 64, 128
    g = yr.new_kernel_graph()
    xi = g.new_input(dims=(rm, rk), dtype=yr.float16)
    wi = g.new_input(dims=(rk, rn), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(xi, (rk,)), wi))

    prev_home = os.environ.get("HOME")
    apply_rms_matmul_search_tractability()
    with tempfile.TemporaryDirectory(prefix="yirage_p1_det_") as tmp:
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
                profile_iters=3,
            )
        finally:
            if prev_home is not None:
                os.environ["HOME"] = prev_home
            else:
                os.environ.pop("HOME", None)

    assert opt is not None
    assert is_production_rms_matmul_mugraph(opt.cygraph)


def test_fused_rms_matmul_customized_op_correctness():
    """Fused kn_customized_op (RMS accum + matmul + div) matches torch baseline."""
    import os
    import tempfile

    import yirage as yr

    rm, rk, rn = 32, 64, 128
    x = torch.randn(rm, rk, dtype=torch.float16)
    w = torch.randn(rk, rn, dtype=torch.float16)
    ref = torch.matmul(
        x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6),
        w,
    )

    g = yr.new_kernel_graph()
    xi = g.new_input(dims=(rm, rk), dtype=yr.float16)
    wi = g.new_input(dims=(rk, rn), dtype=yr.float16)
    normed = g.rms_norm(xi, normalized_shape=(rk,))
    g.mark_output(g.matmul(normed, wi))

    prev_home = os.environ.get("HOME")
    apply_rms_matmul_search_tractability()
    with tempfile.TemporaryDirectory(prefix="yirage_rms_fuse_") as tmp:
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
                profile_iters=3,
            )
        finally:
            if prev_home is not None:
                os.environ["HOME"] = prev_home
            else:
                os.environ.pop("HOME", None)

    assert opt is not None
    out = opt(inputs=[x, w])[0]
    assert torch.allclose(out, ref, rtol=0.02, atol=0.08)


def test_fused_rms_matmul_near_mkl_baseline():
    """Fused customized µGraph should use host-BLAS path, not slow TB interpreter."""
    import os
    import tempfile

    import yirage as yr

    rm, rk, rn = 32, 64, 128
    x = torch.randn(rm, rk, dtype=torch.float16)
    w = torch.randn(rk, rn, dtype=torch.float16)

    def mkl_baseline():
        x32 = x.float()
        scale = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
        return torch.matmul(x32 * scale, w.float())

    g = yr.new_kernel_graph()
    xi = g.new_input(dims=(rm, rk), dtype=yr.float16)
    wi = g.new_input(dims=(rk, rn), dtype=yr.float16)
    g.mark_output(g.matmul(g.rms_norm(xi, (rk,)), wi))

    prev_home = os.environ.get("HOME")
    apply_rms_matmul_search_tractability()
    with tempfile.TemporaryDirectory(prefix="yirage_p1_bench_") as tmp:
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
                profile_iters=3,
            )
        finally:
            if prev_home is not None:
                os.environ["HOME"] = prev_home
            else:
                os.environ.pop("HOME", None)

    assert opt is not None

    def bench(fn, iters=60):
        for _ in range(10):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1000

    mkl_ms = bench(mkl_baseline)
    fused_ms = bench(lambda: opt(inputs=[x, w]))
    # Host-BLAS fast path should track MKL baseline (not ~0.3ms+ TB interpreter).
    assert fused_ms < mkl_ms * 1.4, (
        f"fused µGraph {fused_ms:.4f}ms vs MKL baseline {mkl_ms:.4f}ms"
    )
    assert fused_ms < 0.15, (
        f"fused µGraph {fused_ms:.4f}ms — likely still on TB interpreter"
    )


def test_unfused_rms_matmul_near_torch_mkl():
    """Unfused rms+matmul fast path should track torch (MKL), not interpreter overhead."""
    import yirage as yr
    from yirage.kernel.cpu_native import cpu_rms_matmul
    from yirage.kernel.graph import KNGraph

    g = yr.new_kernel_graph()
    x = g.new_input(dims=(64, 128), dtype=yr.float16)
    w = g.new_input(dims=(128, 256), dtype=yr.float16)
    normed = g.rms_norm(x, normalized_shape=(128,))
    g.mark_output(g.matmul(normed, w))
    kg = KNGraph(g.cygraph, backend="cpu")

    ref_x = torch.randn(64, 128, dtype=torch.float16)
    ref_w = torch.randn(128, 256, dtype=torch.float16)

    def bench(fn, iters=80):
        for _ in range(10):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1000

    def torch_rms_matmul():
        x32 = ref_x.float()
        scale = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
        return torch.matmul(x32 * scale, ref_w.float())

    torch_ms = bench(torch_rms_matmul)
    native_ms = bench(lambda: cpu_rms_matmul(ref_x, ref_w))
    yirage_ms = bench(lambda: kg(inputs=[ref_x, ref_w]))
    assert native_ms < torch_ms * 2.0, (
        f"cpu_rms_matmul {native_ms:.4f}ms vs torch {torch_ms:.4f}ms"
    )
    # cpu_call should use the same BLAS primitive; allow KNGraph dispatch overhead.
    assert yirage_ms < native_ms * 1.6, (
        f"KNGraph unfused rms+matmul {yirage_ms:.4f}ms vs cpu_rms_matmul {native_ms:.4f}ms"
    )


def test_kngraph_plain_matmul_near_torch_mkl():
    """Plain kn_matmul_op should track torch.matmul (MKL), not orders-of-magnitude slower."""
    import yirage as yr
    from yirage.kernel.graph import KNGraph

    g = yr.new_kernel_graph()
    a = g.new_input(dims=(8, 32), dtype=yr.float16)
    b = g.new_input(dims=(32, 64), dtype=yr.float16)
    g.mark_output(g.matmul(a, b))
    kg = KNGraph(g.cygraph, backend="cpu")

    ref_a = torch.randn(8, 32, dtype=torch.float16)
    ref_b = torch.randn(32, 64, dtype=torch.float16)

    def bench(fn, iters=100):
        for _ in range(10):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters * 1000

    torch_ms = bench(lambda: torch.matmul(ref_a, ref_b))
    yirage_ms = bench(lambda: kg(inputs=[ref_a, ref_b]))
    # Interpreter dispatch overhead; allow 2.5x vs raw torch on tiny shapes
    assert yirage_ms < torch_ms * 2.5, (
        f"KNGraph {yirage_ms:.4f}ms vs torch {torch_ms:.4f}ms — MKL path not wired"
    )


def test_native_rms_matmul_f32_correctness():
    """P2 OpenMP + cblas fused kernel matches deferred torch path."""
    from yirage.kernel.cpu_native import (
        cpu_rms_matmul,
        cpu_rms_matmul_torch,
        native_rms_matmul_available,
    )

    if not native_rms_matmul_available():
        pytest.skip("cpu_rms_matmul_f32 not built (needs CPU backend + BLAS)")

    prev = os.environ.get("YIRAGE_CPU_RMS_MATMUL_NATIVE")
    os.environ["YIRAGE_CPU_RMS_MATMUL_NATIVE"] = "1"
    try:
        x = torch.randn(128, 256, dtype=torch.float16)
        w = torch.randn(256, 512, dtype=torch.float16)
        ref = cpu_rms_matmul_torch(x, w)
        out = cpu_rms_matmul(x, w)
        assert torch.allclose(out, ref, rtol=0.02, atol=0.1)
    finally:
        if prev is not None:
            os.environ["YIRAGE_CPU_RMS_MATMUL_NATIVE"] = prev
        else:
            os.environ.pop("YIRAGE_CPU_RMS_MATMUL_NATIVE", None)


def test_cpu_matmul_chain_matches_torch_baseline():
    from yirage.kernel.cpu_native import cpu_matmul_chain

    m, k, k2, n = 32, 64, 64, 128
    a = torch.randn(m, k, dtype=torch.float16)
    b = torch.randn(k, k2, dtype=torch.float16)
    c = torch.randn(k2, n, dtype=torch.float16)
    ref = torch.matmul(torch.matmul(a, b), c)
    out = cpu_matmul_chain(a, b, c)
    assert torch.allclose(out, ref, rtol=0.01, atol=0.08)


def test_unfused_matmul_chain_mugraph_detection():
    import yirage as yr
    from yirage.kernel.cpu_mlir_jit import (
        is_production_matmul_chain_mugraph,
        matmul_chain_shapes_from_cygraph,
    )

    m, k, k2, n = 16, 32, 32, 64
    g = yr.new_kernel_graph()
    a = g.new_input(dims=(m, k), dtype=yr.float16)
    b = g.new_input(dims=(k, k2), dtype=yr.float16)
    c = g.new_input(dims=(k2, n), dtype=yr.float16)
    t = g.matmul(a, b)
    g.mark_output(g.matmul(t, c))
    assert matmul_chain_shapes_from_cygraph(g.cygraph) == (m, k, k2, n)
    assert is_production_matmul_chain_mugraph(g.cygraph)


def test_unfused_matmul_chain_near_mkl_baseline():
    import time

    import yirage as yr
    from yirage.kernel.graph import KNGraph

    m, k, k2, n = 32, 64, 64, 128
    inputs = [
        torch.randn(m, k, dtype=torch.float16),
        torch.randn(k, k2, dtype=torch.float16),
        torch.randn(k2, n, dtype=torch.float16),
    ]

    def mkl_baseline():
        return torch.matmul(torch.matmul(inputs[0], inputs[1]), inputs[2])

    g = yr.new_kernel_graph()
    a = g.new_input(dims=(m, k), dtype=yr.float16)
    b = g.new_input(dims=(k, k2), dtype=yr.float16)
    c = g.new_input(dims=(k2, n), dtype=yr.float16)
    t = g.matmul(a, b)
    g.mark_output(g.matmul(t, c))
    kg = KNGraph(g.cygraph, backend="cpu")

    ref = mkl_baseline()
    out = kg(inputs=inputs)[0]
    assert torch.allclose(out, ref, rtol=0.02, atol=0.1)

    for _ in range(5):
        mkl_baseline()
    t0 = time.perf_counter()
    for _ in range(20):
        mkl_baseline()
    mkl_ms = (time.perf_counter() - t0) / 20 * 1000

    for _ in range(5):
        kg(inputs=inputs)
    t0 = time.perf_counter()
    for _ in range(20):
        kg(inputs=inputs)
    yirage_ms = (time.perf_counter() - t0) / 20 * 1000

    assert yirage_ms < mkl_ms * 2.5, (
        f"matmul_chain {yirage_ms:.4f}ms vs MKL {mkl_ms:.4f}ms — BLAS fast path not wired"
    )


def test_cpu_concat_matmul_matches_torch_baseline():
    from yirage.kernel.cpu_native import cpu_concat_matmul

    m, k1, k2, n = 32, 64, 64, 128
    a = torch.randn(m, k1, dtype=torch.float16)
    b = torch.randn(m, k2, dtype=torch.float16)
    c = torch.randn(k1, n, dtype=torch.float16)
    d = torch.randn(k2, n, dtype=torch.float16)
    ref = torch.matmul(torch.cat([a, b], dim=1), torch.cat([c, d], dim=0))
    out = cpu_concat_matmul(a, b, c, d)
    assert torch.allclose(out, ref, rtol=0.01, atol=0.08)


def test_unfused_concat_matmul_mugraph_detection():
    import yirage as yr
    from yirage.kernel.cpu_mlir_jit import (
        concat_matmul_shapes_from_cygraph,
        is_production_concat_matmul_mugraph,
    )

    m, k1, k2, n = 16, 32, 32, 64
    g = yr.new_kernel_graph()
    a = g.new_input(dims=(m, k1), dtype=yr.float16)
    b = g.new_input(dims=(m, k2), dtype=yr.float16)
    c = g.new_input(dims=(k1, n), dtype=yr.float16)
    d = g.new_input(dims=(k2, n), dtype=yr.float16)
    left = g.concat(a, b, dim=1)
    right = g.concat(c, d, dim=0)
    g.mark_output(g.matmul(left, right))
    assert concat_matmul_shapes_from_cygraph(g.cygraph) == (m, k1, k2, n)
    assert is_production_concat_matmul_mugraph(g.cygraph)


def test_concat_matmul_shapes_ignores_four_3d_inputs():
    """Four 3D inputs must not crash shape probing (unblocks 3D cpu_call graphs)."""
    import yirage as yr
    from yirage.kernel.cpu_mlir_jit import (
        concat_matmul_shapes_from_cygraph,
        is_production_concat_matmul_mugraph,
    )

    g = yr.new_kernel_graph()
    a = g.new_input(dims=(2, 4, 8), dtype=yr.float16)
    b = g.new_input(dims=(2, 4, 8), dtype=yr.float16)
    c = g.new_input(dims=(2, 8, 16), dtype=yr.float16)
    d = g.new_input(dims=(2, 8, 16), dtype=yr.float16)
    g.mark_output(g.matmul(a, c))
    assert concat_matmul_shapes_from_cygraph(g.cygraph) is None
    assert is_production_concat_matmul_mugraph(g.cygraph) is False


def test_unfused_concat_matmul_near_mkl_baseline():
    """LoRA concat_matmul µGraph should use host-BLAS blocked path, not TB interpreter."""
    import time

    import yirage as yr
    from yirage.kernel.graph import KNGraph

    m, k1, k2, n = 32, 64, 64, 128
    inputs = [
        torch.randn(m, k1, dtype=torch.float16),
        torch.randn(m, k2, dtype=torch.float16),
        torch.randn(k1, n, dtype=torch.float16),
        torch.randn(k2, n, dtype=torch.float16),
    ]

    def mkl_baseline():
        a, b, c, d = inputs
        return torch.matmul(torch.cat([a, b], dim=1), torch.cat([c, d], dim=0))

    g = yr.new_kernel_graph()
    a = g.new_input(dims=(m, k1), dtype=yr.float16)
    b = g.new_input(dims=(m, k2), dtype=yr.float16)
    c = g.new_input(dims=(k1, n), dtype=yr.float16)
    d = g.new_input(dims=(k2, n), dtype=yr.float16)
    left = g.concat(a, b, dim=1)
    right = g.concat(c, d, dim=0)
    g.mark_output(g.matmul(left, right))
    kg = KNGraph(g.cygraph, backend="cpu")

    ref = mkl_baseline()
    out = kg(inputs=inputs)[0]
    assert torch.allclose(out, ref, rtol=0.02, atol=0.1)

    for _ in range(5):
        mkl_baseline()
    t0 = time.perf_counter()
    for _ in range(20):
        mkl_baseline()
    mkl_ms = (time.perf_counter() - t0) / 20 * 1000

    for _ in range(5):
        kg(inputs=inputs)
    t0 = time.perf_counter()
    for _ in range(20):
        kg(inputs=inputs)
    yirage_ms = (time.perf_counter() - t0) / 20 * 1000

    assert yirage_ms < mkl_ms * 2.5, (
        f"concat_matmul {yirage_ms:.4f}ms vs MKL {mkl_ms:.4f}ms — BLAS fast path not wired"
    )


@pytest.mark.skipif(
    os.environ.get("YIRAGE_CPU_NATIVE") != "1",
    reason="experimental native GEMM only when YIRAGE_CPU_NATIVE=1",
)
def test_experimental_native_gemm_correctness():
    from yirage.kernel.cpu_native import cpu_matmul, native_matmul_available

    if not native_matmul_available():
        pytest.skip("cpu_gemm_f32 not exposed")

    a = torch.randn(64, 128, dtype=torch.float32)
    b = torch.randn(128, 256, dtype=torch.float32)
    ref = torch.matmul(a, b)
    out = cpu_matmul(a, b)
    assert torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
