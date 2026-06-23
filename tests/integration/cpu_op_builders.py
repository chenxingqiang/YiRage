# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Minimal KN graphs for CPU op contract tests."""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch

import yirage as yr


Builder = Callable[[], Tuple[object, List[torch.Tensor], torch.Tensor]]


def _f16(shape, *, positive: bool = False):
    t = torch.randn(*shape, dtype=torch.float16)
    if positive:
        t = t.abs() + 0.25
    return t


def build_kn_unary(op_name: str, *, positive: bool = False) -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        fn = getattr(g, op_name)
        g.mark_output(fn(x))
        inp = _f16((8, 16), positive=positive)
        ref_ops = {
            "exp": torch.exp,
            "square": lambda t: t * t,
            "sqrt": lambda t: torch.sqrt(t.float()).to(torch.float16),
            "silu": torch.nn.functional.silu,
            "gelu": torch.nn.functional.gelu,
            "relu": torch.nn.functional.relu,
            "sigmoid": torch.sigmoid,
            "log": lambda t: torch.log(t.float()).to(torch.float16),
        }
        ref = ref_ops[op_name](inp.float() if op_name in ("sqrt", "log") else inp)
        if op_name in ("sqrt", "log"):
            ref = ref.to(torch.float16) if ref.dtype != torch.float16 else ref
        else:
            ref = ref.to(torch.float16) if hasattr(ref, "to") else ref
        return g, [inp], ref

    return _build


def build_kn_binary(op_name: str) -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 16), dtype=yr.float16)
        b = g.new_input(dims=(8, 16), dtype=yr.float16)
        fn = getattr(g, op_name)
        g.mark_output(fn(a, b))
        ta = _f16((8, 16), positive=(op_name == "pow"))
        tb = _f16((8, 16), positive=True)
        ref_ops = {
            "add": lambda x, y: x + y,
            "sub": lambda x, y: x - y,
            "mul": lambda x, y: x * y,
            "div": lambda x, y: x / y,
            "pow": lambda x, y: torch.pow(x.float(), y.float()).to(torch.float16),
        }
        ref = ref_ops[op_name](ta.float(), tb.float())
        if ref.dtype != torch.float16:
            ref = ref.to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_kn_matmul() -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.matmul(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        return g, [ta, tb], torch.matmul(ta, tb)

    return _build


def build_kn_rms_norm() -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 32), dtype=yr.float16)
        g.mark_output(g.rms_norm(x, normalized_shape=(32,)))
        inp = _f16((8, 32))
        scale = torch.rsqrt(inp.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return g, [inp], (inp.float() * scale).to(torch.float16)

    return _build


def build_kn_reduction(dim: int) -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16, 4), dtype=yr.float16)
        g.mark_output(g.reduction(x, dim))
        inp = _f16((8, 16, 4))
        return g, [inp], inp.float().sum(dim=dim, keepdim=False).to(torch.float16)

    return _build


def build_customized_tb_matmul() -> Builder:
    """kn_customized_op: TB matmul (interpreter path, not rms+matmul fast path)."""

    def _build():
        m, k, n = 8, 32, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        tb = yr.new_threadblock_graph(
            grid_dim=(1, 1, 1),
            block_dim=(32, 1, 1),
            forloop_range=1,
            reduction_dimx=k,
        )
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tw = tb.new_input(dtensor=w, input_map=(-1, -1, -1), forloop_dim=0)
        tm = tb.matmul(tx, tw)
        tacc = tb.forloop_accum(tm)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x, w], tb)
        g.mark_output(out[0])
        ta, twt = _f16((m, k)), _f16((k, n))
        return g, [ta, twt], torch.matmul(ta, twt)

    return _build


def build_customized_tb_exp() -> Builder:
    """kn_customized_op: TB unary exp chain."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        tb = yr.new_threadblock_graph(
            grid_dim=(1, 1, 1),
            block_dim=(16, 1, 1),
            forloop_range=1,
            reduction_dimx=16,
        )
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        te = tb.exp(tx)
        tacc = tb.forloop_accum(te)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16((8, 16))
        return g, [inp], torch.exp(inp)

    return _build


def build_customized_tb_matmul_add_bias() -> Builder:
    """kn_customized_op matmul + KN add (mixed interpreter + elementwise)."""

    def _build():
        m, k, n = 8, 32, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        b = g.new_input(dims=(m, n), dtype=yr.float16)
        tb = yr.new_threadblock_graph(
            grid_dim=(1, 1, 1),
            block_dim=(32, 1, 1),
            forloop_range=1,
            reduction_dimx=k,
        )
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tw = tb.new_input(dtensor=w, input_map=(-1, -1, -1), forloop_dim=0)
        tm = tb.matmul(tx, tw)
        tacc = tb.forloop_accum(tm)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x, w], tb)
        g.mark_output(g.add(out[0], b))
        ta, twt, tbias = _f16((m, k)), _f16((k, n)), _f16((m, n))
        ref = torch.matmul(ta, twt) + tbias
        return g, [ta, twt, tbias], ref

    return _build


def build_kn_clamp() -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(4, 8), dtype=yr.float16)
        g.mark_output(g.clamp(x, -1.0, 1.0))
        inp = _f16((4, 8))
        return g, [inp], torch.clamp(inp.float(), -1.0, 1.0).to(torch.float16)

    return _build


def build_kn_mul_scalar() -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(4, 8), dtype=yr.float16)
        g.mark_output(g.mul_scalar(x, 0.5))
        inp = _f16((4, 8))
        return g, [inp], (inp.float() * 0.5).to(torch.float16)

    return _build


def build_kn_split(dim: int) -> Builder:
    def _build():
        if dim == 2:
            shape = (8, 16, 4)
        else:
            shape = (8, 16)
        split_size = shape[dim] // 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        parts = g.split(x, split_size, dim)
        g.mark_output(parts[0])
        inp = _f16(shape)
        ref = torch.split(inp, (split_size, shape[dim] - split_size), dim=dim)[0]
        return g, [inp], ref

    return _build


def build_kn_concat(dim: int) -> Builder:
    def _build():
        if dim == 0:
            shape_a, shape_b = (4, 16), (4, 16)
        elif dim == 1:
            shape_a, shape_b = (8, 8), (8, 8)
        else:
            shape_a, shape_b = (8, 16, 2), (8, 16, 2)
        g = yr.new_kernel_graph()
        a = g.new_input(dims=shape_a, dtype=yr.float16)
        b = g.new_input(dims=shape_b, dtype=yr.float16)
        g.mark_output(g.concat(a, b, dim))
        ta, tb = _f16(shape_a), _f16(shape_b)
        ref = torch.cat([ta, tb], dim=dim)
        return g, [ta, tb], ref

    return _build


def build_kn_chunk(dim: int) -> Builder:
    def _build():
        if dim == 2:
            shape = (4, 8, 4)
        else:
            shape = (8, 16)
        chunk_size = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        parts = g.chunk(x, chunk_size, dim)
        g.mark_output(parts[0])
        inp = _f16(shape)
        ref = torch.chunk(inp, chunk_size, dim=dim)[0]
        return g, [inp], ref

    return _build


def build_kn_transpose_01() -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        g.mark_output(g.transpose(x, dim0=0, dim1=1))
        inp = _f16((8, 16))
        ref = inp.transpose(0, 1).contiguous()
        return g, [inp], ref

    return _build


KN_OP_BUILDERS = {
    "kn_matmul_op": build_kn_matmul(),
    "kn_rms_norm_op": build_kn_rms_norm(),
    "kn_exp_op": build_kn_unary("exp"),
    "kn_square_op": build_kn_unary("square"),
    "kn_sqrt_op": build_kn_unary("sqrt", positive=True),
    "kn_silu_op": build_kn_unary("silu"),
    "kn_gelu_op": build_kn_unary("gelu"),
    "kn_relu_op": build_kn_unary("relu"),
    "kn_sigmoid_op": build_kn_unary("sigmoid"),
    "kn_log_op": build_kn_unary("log", positive=True),
    "kn_clamp_op": build_kn_clamp(),
    "kn_mul_scalar_op": build_kn_mul_scalar(),
    "kn_add_op": build_kn_binary("add"),
    "kn_sub_op": build_kn_binary("sub"),
    "kn_mul_op": build_kn_binary("mul"),
    "kn_div_op": build_kn_binary("div"),
    "kn_pow_op": build_kn_binary("pow"),
    "kn_reduction_0_op": build_kn_reduction(0),
    "kn_reduction_1_op": build_kn_reduction(1),
    "kn_reduction_2_op": build_kn_reduction(2),
    "kn_split_0_op": build_kn_split(0),
    "kn_split_1_op": build_kn_split(1),
    "kn_split_2_op": build_kn_split(2),
    "kn_concat_0_op": build_kn_concat(0),
    "kn_concat_1_op": build_kn_concat(1),
    "kn_concat_2_op": build_kn_concat(2),
    "kn_chunk_0_op": build_kn_chunk(0),
    "kn_chunk_1_op": build_kn_chunk(1),
    "kn_chunk_2_op": build_kn_chunk(2),
    "kn_transpose_01_op": build_kn_transpose_01(),
}

def build_kn_unfused_rms_matmul() -> Builder:
    """KN-level rms_norm + matmul (fast_path via cpu_rms_matmul when enabled)."""

    def _build():
        m, k, n = 16, 64, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        normed = g.rms_norm(x, normalized_shape=(k,))
        g.mark_output(g.matmul(normed, w))
        tx, tw = _f16((m, k)), _f16((k, n))
        ref = torch.matmul(
            tx.float() * torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6),
            tw.float(),
        ).to(torch.float16)
        return g, [tx, tw], ref

    return _build


    return _build


def build_kn_softmax() -> Builder:
    """KN softmax via stable TB reduction_max path (general ML, not LLM-only)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        g.mark_output(g.softmax(x, dim=-1))
        inp = _f16((8, 16))
        ref = torch.nn.functional.softmax(inp.float(), dim=-1).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_layer_norm() -> Builder:
    """KN layer_norm (elementwise_affine=False; eps=0 matches TB sqrt(var) path)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        g.mark_output(g.layer_norm(x, normalized_shape=(16,), eps=0.0))
        inp = _f16((8, 16))
        ref = torch.nn.functional.layer_norm(inp.float(), (16,), eps=0.0).to(torch.float16)
        return g, [inp], ref

    return _build


def build_gemm_softmax() -> Builder:
    """COMET gemm_softmax compound op vs torch matmul + F.softmax."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_softmax(a, b, dim=-1))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.softmax(c, dim=-1).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm() -> Builder:
    """COMET gemm_layernorm compound op vs torch matmul + F.layer_norm (eps=0)."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm(a, b, normalized_shape=(16,), eps=0.0))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.layer_norm(c, (16,), eps=0.0).to(torch.float16)
        return g, [ta, tb], ref

    return _build


CUSTOMIZED_OP_BUILDERS = {
    "customized_tb_matmul": build_customized_tb_matmul(),
    "customized_tb_exp": build_customized_tb_exp(),
    "customized_tb_matmul_add_bias": build_customized_tb_matmul_add_bias(),
    "kn_softmax": build_kn_softmax(),
    "kn_layer_norm": build_kn_layer_norm(),
    "gemm_softmax": build_gemm_softmax(),
    "gemm_layernorm": build_gemm_layernorm(),
}

FAST_PATH_BUILDERS = {
    "kn_matmul_op_fast": build_kn_matmul(),
    "kn_unfused_rms_matmul": build_kn_unfused_rms_matmul(),
}


def build_kn_layout_split_concat_roundtrip(dim: int) -> Builder:
    """Split then concat along the same dim (search explore round-trip)."""

    def _build():
        if dim == 2:
            shape = (8, 16, 4)
        else:
            shape = (8, 16)
        split_size = shape[dim] // 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        parts = g.split(x, split_size, dim)
        y = g.concat(parts[0], parts[1], dim)
        g.mark_output(y)
        inp = _f16(shape)
        return g, [inp], inp

    return _build


def build_kn_layout_concat_split_first(dim: int) -> Builder:
    """Concat two tensors then split to recover the first half."""

    def _build():
        if dim == 0:
            shape_a, shape_b = (4, 16), (4, 16)
        elif dim == 1:
            shape_a, shape_b = (8, 8), (8, 8)
        else:
            shape_a, shape_b = (8, 16, 2), (8, 16, 2)
        split_size = shape_a[dim]
        g = yr.new_kernel_graph()
        a = g.new_input(dims=shape_a, dtype=yr.float16)
        b = g.new_input(dims=shape_b, dtype=yr.float16)
        merged = g.concat(a, b, dim)
        parts = g.split(merged, split_size, dim)
        g.mark_output(parts[0])
        ta, tb = _f16(shape_a), _f16(shape_b)
        return g, [ta, tb], ta

    return _build


def build_kn_layout_chunk_concat_roundtrip(dim: int) -> Builder:
    """Chunk into two pieces then concat back (search explore round-trip)."""

    def _build():
        if dim == 2:
            shape = (4, 8, 4)
        else:
            shape = (8, 16)
        chunk_count = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        parts = g.chunk(x, chunk_count, dim)
        y = g.concat(parts[0], parts[1], dim)
        g.mark_output(y)
        inp = _f16(shape)
        return g, [inp], inp

    return _build


def build_kn_layout_chunk_split_first(dim: int) -> Builder:
    """Chunk into two, re-concat, then split to recover the first half."""

    def _build():
        if dim == 2:
            shape = (4, 8, 4)
        else:
            shape = (8, 16)
        split_size = shape[dim] // 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        parts = g.chunk(x, 2, dim)
        merged = g.concat(parts[0], parts[1], dim)
        out_parts = g.split(merged, split_size, dim)
        g.mark_output(out_parts[0])
        inp = _f16(shape)
        ref = torch.split(
            inp, (split_size, shape[dim] - split_size), dim=dim
        )[0]
        return g, [inp], ref

    return _build


def build_kn_layout_split_chunk_first(dim: int) -> Builder:
    """Split to first half, then chunk that piece along an orthogonal dim."""

    def _build():
        if dim == 2:
            shape = (8, 16, 4)
            chunk_sub_dim = 1
        elif dim == 0:
            shape = (8, 16)
            chunk_sub_dim = 1
        else:
            shape = (8, 16)
            chunk_sub_dim = 0
        split_size = shape[dim] // 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        sp = g.split(x, split_size, dim)
        chunks = g.chunk(sp[0], 2, chunk_sub_dim)
        g.mark_output(chunks[0])
        inp = _f16(shape)
        first = torch.split(
            inp, (split_size, shape[dim] - split_size), dim=dim
        )[0]
        ref = torch.chunk(first, 2, dim=chunk_sub_dim)[0]
        return g, [inp], ref

    return _build


def build_kn_layout_concat_matmul() -> Builder:
    """LoRA-style dual concat + matmul (KN layout search explore representative)."""

    def _build():
        m, k1, k2, n = 16, 32, 32, 64
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(m, k1), dtype=yr.float16)
        b = g.new_input(dims=(m, k2), dtype=yr.float16)
        c = g.new_input(dims=(k1, n), dtype=yr.float16)
        d = g.new_input(dims=(k2, n), dtype=yr.float16)
        left = g.concat(a, b, dim=1)
        right = g.concat(c, d, dim=0)
        g.mark_output(g.matmul(left, right))
        inputs = [
            _f16((m, k1)),
            _f16((m, k2)),
            _f16((k1, n)),
            _f16((k2, n)),
        ]
        ref = torch.matmul(
            torch.cat([inputs[0], inputs[1]], dim=1),
            torch.cat([inputs[2], inputs[3]], dim=0),
        )
        return g, inputs, ref

    return _build


LAYOUT_EXPLORE_BUILDERS = {
    "kn_layout_split_concat_roundtrip_dim0": build_kn_layout_split_concat_roundtrip(0),
    "kn_layout_split_concat_roundtrip_dim1": build_kn_layout_split_concat_roundtrip(1),
    "kn_layout_split_concat_roundtrip_dim2": build_kn_layout_split_concat_roundtrip(2),
    "kn_layout_concat_split_first_dim0": build_kn_layout_concat_split_first(0),
    "kn_layout_concat_split_first_dim1": build_kn_layout_concat_split_first(1),
    "kn_layout_concat_split_first_dim2": build_kn_layout_concat_split_first(2),
    "kn_layout_chunk_concat_roundtrip_dim0": build_kn_layout_chunk_concat_roundtrip(0),
    "kn_layout_chunk_concat_roundtrip_dim1": build_kn_layout_chunk_concat_roundtrip(1),
    "kn_layout_chunk_concat_roundtrip_dim2": build_kn_layout_chunk_concat_roundtrip(2),
    "kn_layout_chunk_split_first_dim0": build_kn_layout_chunk_split_first(0),
    "kn_layout_chunk_split_first_dim1": build_kn_layout_chunk_split_first(1),
    "kn_layout_chunk_split_first_dim2": build_kn_layout_chunk_split_first(2),
    "kn_layout_split_chunk_first_dim0": build_kn_layout_split_chunk_first(0),
    "kn_layout_split_chunk_first_dim1": build_kn_layout_split_chunk_first(1),
    "kn_layout_split_chunk_first_dim2": build_kn_layout_split_chunk_first(2),
    "kn_layout_concat_matmul": build_kn_layout_concat_matmul(),
}
