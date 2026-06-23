# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""TB op graphs (via kn_customized_op) for CPU value verification."""

from __future__ import annotations

from typing import Callable, List, Tuple

import torch

import yirage as yr

from tests.integration.cpu_op_builders import (
    Builder,
    _f16,
    build_customized_tb_matmul,
)


def _tb_graph(forloop_range: int = 1, block: int = 16, reduction_dimx: int = 16):
    return yr.new_threadblock_graph(
        grid_dim=(1, 1, 1),
        block_dim=(block, 1, 1),
        forloop_range=forloop_range,
        reduction_dimx=reduction_dimx,
    )


def _finish_customized_unary(
    g,
    x,
    tb,
    tx,
    out_stensor,
    inp: torch.Tensor,
    ref: torch.Tensor,
):
    tacc = tb.forloop_accum(out_stensor)
    tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
    out = g.customized([x], tb)
    g.mark_output(out[0])
    return g, [inp], ref


def build_customized_tb_unary(op_name: str, *, positive: bool = False) -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        tb = _tb_graph()
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        fn = getattr(tb, op_name)
        ty = fn(tx)
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
        if ref.dtype != torch.float16:
            ref = ref.to(torch.float16)
        return _finish_customized_unary(g, x, tb, tx, ty, inp, ref)

    return _build


def build_customized_tb_binary(op_name: str, *, positive: bool = False) -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 16), dtype=yr.float16)
        b = g.new_input(dims=(8, 16), dtype=yr.float16)
        tb = _tb_graph()
        ta = tb.new_input(dtensor=a, input_map=(-1, -1, -1), forloop_dim=1)
        tb_in = tb.new_input(dtensor=b, input_map=(-1, -1, -1), forloop_dim=1)
        fn = getattr(tb, op_name)
        ty = fn(ta, tb_in)
        ta_t = _f16((8, 16), positive=positive or op_name == "pow")
        tb_t = _f16((8, 16), positive=True)
        ref_ops = {
            "add": lambda x, y: x + y,
            "mul": lambda x, y: x * y,
            "div": lambda x, y: x / y,
            "sub": lambda x, y: x - y,
            "pow": lambda x, y: torch.pow(x.float(), y.float()).to(torch.float16),
        }
        ref = ref_ops[op_name](ta_t.float(), tb_t.float())
        if ref.dtype != torch.float16:
            ref = ref.to(torch.float16)
        tacc = tb.forloop_accum(ty)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([a, b], tb)
        g.mark_output(out[0])
        return g, [ta_t, tb_t], ref

    return _build


def build_customized_tb_rms_norm() -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 32), dtype=yr.float16)
        tb = _tb_graph(block=32, reduction_dimx=32)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tacc = tb.forloop_accum(tx)
        tn = tb.rms_norm(tacc)
        tb.new_output(stensor=tn, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16((8, 32))
        scale = torch.rsqrt(inp.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return g, [inp], (inp.float() * scale).to(torch.float16)

    return _build


def build_customized_tb_reduction(dim: int) -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16, 4), dtype=yr.float16)
        tb = _tb_graph(block=16, reduction_dimx=4)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tr = tb.reduction(tx, dim)
        tacc = tb.forloop_accum(tr)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16((8, 16, 4))
        ref = inp.float().sum(dim=dim, keepdim=True).to(torch.float16)
        return g, [inp], ref

    return _build


def _ref_reduction_to_dimx(x: torch.Tensor, dim: int, reduction_dimx: int) -> torch.Tensor:
    t = x.float().movedim(dim, -1)
    group = t.shape[-1] // reduction_dimx
    t = t.reshape(*t.shape[:-1], reduction_dimx, group)
    out = t.sum(dim=-1)
    return out.movedim(-1, dim).to(x.dtype)


def build_customized_tb_reduction_max(dim: int) -> Builder:
    def _build():
        if dim == 0:
            shape = (16, 8, 4)
        elif dim == 1:
            shape = (8, 16, 4)
        else:
            shape = (8, 4, 16)
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        tb = _tb_graph(block=16, reduction_dimx=4)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tmax, _tdiff = tb.reduction_max(tx, dim)
        tacc = tb.forloop_accum(tmax)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16(shape)
        ref = inp.float().max(dim=dim, keepdim=True).values.to(torch.float16)
        return g, [inp], ref

    return _build


def build_customized_tb_reduction_to_dimx(dim: int) -> Builder:
    def _build():
        rdx = 4
        if dim == 0:
            shape = (rdx * 3, 16, 4)
        elif dim == 1:
            shape = (8, rdx * 3, 4)
        else:
            shape = (8, 16, rdx * 3)
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        tb = _tb_graph(block=16, reduction_dimx=rdx)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tr = tb.reduction_to_dimx(tx, dim)
        tacc = tb.forloop_accum(tr)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16(shape)
        ref = _ref_reduction_to_dimx(inp, dim, rdx)
        return g, [inp], ref

    return _build


def build_customized_tb_forloop_accum_no_red() -> Builder:
    """TB forloop_accum_no_red via exp + accum (forloop_range=1)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        tb = _tb_graph()
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        te = tb.exp(tx)
        tacc = tb.forloop_accum(te)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16((8, 16))
        return g, [inp], torch.exp(inp)

    return _build


def build_customized_tb_forloop_accum_red_ld_sum() -> Builder:
    """TB forloop_accum(..., acc='sum'): reduce-sum along last dim per tile, then accumulate."""

    def _build():
        rows, cols, tile, fl = 4, 32, 16, 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(rows, cols), dtype=yr.float16)
        tb = _tb_graph(forloop_range=fl, block=tile, reduction_dimx=tile)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tacc = tb.forloop_accum(tx, "sum")
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16((rows, cols))
        ref = inp.float().sum(dim=-1, keepdim=True).to(torch.float16)
        return g, [inp], ref

    return _build


def build_customized_tb_forloop_accum_red_ld_mean() -> Builder:
    """TB forloop_accum(..., acc='mean'): reduce-sum per tile, mean on last iter."""

    def _build():
        rows, cols, tile, fl = 4, 32, 16, 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(rows, cols), dtype=yr.float16)
        tb = _tb_graph(forloop_range=fl, block=tile, reduction_dimx=tile)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tacc = tb.forloop_accum(tx, "mean")
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16((rows, cols))
        ref = inp.float().mean(dim=-1, keepdim=True).to(torch.float16)
        return g, [inp], ref

    return _build


def build_customized_tb_forloop_accum_redtox_ld_sum() -> Builder:
    """TB forloop_accum(..., acc='sum_todimx'): reduce last dim to reduction_dimx."""

    def _build():
        rows, cols, tile, fl, rdx = 4, 32, 16, 2, 4
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(rows, cols), dtype=yr.float16)
        tb = _tb_graph(forloop_range=fl, block=tile, reduction_dimx=rdx)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tacc = tb.forloop_accum(tx, "sum_todimx")
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16((rows, cols))
        acc = None
        for i in range(fl):
            sl = inp[:, i * tile : (i + 1) * tile].float()
            partial = sl.reshape(rows, rdx, tile // rdx).sum(dim=-1)
            acc = partial if acc is None else acc + partial
        return g, [inp], acc.to(torch.float16)

    return _build


def build_customized_tb_forloop_accum_max() -> Builder:
    """TB forloop_accum_max: elementwise max across forloop tiles."""

    def _build():
        rows, cols, tile, fl = 4, 16, 8, 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(rows, cols), dtype=yr.float16)
        tb = _tb_graph(forloop_range=fl, block=tile, reduction_dimx=tile)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tacc = tb.forloop_accum_max(tx)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16((rows, cols))
        ref = torch.maximum(inp[:, :tile].float(), inp[:, tile : tile * fl].float())
        return g, [inp], ref.to(torch.float16)

    return _build


def build_customized_tb_forloop_accum_no_red_rescale() -> Builder:
    """TB forloop_accum_rescale (no reduction): acc = acc * rescale + src."""

    def _build():
        rows, cols, tile, fl = 4, 32, 16, 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(rows, cols), dtype=yr.float16)
        r = g.new_input(dims=(rows, 1), dtype=yr.float16)
        tb = _tb_graph(forloop_range=fl, block=tile, reduction_dimx=tile)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tr = tb.new_input(dtensor=r, input_map=(-1, -1, -1), forloop_dim=-1)
        tacc = tb.forloop_accum_rescale(tx, tr)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x, r], tb)
        g.mark_output(out[0])
        inp = _f16((rows, cols))
        resc = (_f16((rows, 1), positive=True) * 0.5 + 0.25).to(torch.float16)
        acc = torch.zeros(rows, tile, dtype=torch.float32)
        for i in range(fl):
            sl = inp[:, i * tile : (i + 1) * tile].float()
            acc = acc * resc.float() + sl
        return g, [inp, resc], acc.to(torch.float16)

    return _build


def build_customized_tb_forloop_accum_red_ld_sum_rescale() -> Builder:
    """TB forloop_accum_rescale (sum): partial = sum(src); acc = acc * rescale + partial."""

    def _build():
        rows, cols, tile, fl = 4, 32, 16, 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(rows, cols), dtype=yr.float16)
        r = g.new_input(dims=(rows, 1), dtype=yr.float16)
        tb = _tb_graph(forloop_range=fl, block=tile, reduction_dimx=tile)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tr = tb.new_input(dtensor=r, input_map=(-1, -1, -1), forloop_dim=-1)
        tacc = tb.forloop_accum_rescale(tx, tr, "sum")
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x, r], tb)
        g.mark_output(out[0])
        inp = _f16((rows, cols))
        resc = (_f16((rows, 1), positive=True) * 0.5 + 0.25).to(torch.float16)
        acc = torch.zeros(rows, 1, dtype=torch.float32)
        for i in range(fl):
            sl = inp[:, i * tile : (i + 1) * tile].float()
            partial = sl.sum(dim=-1, keepdim=True)
            acc = acc * resc.float() + partial
        return g, [inp, resc], acc.to(torch.float16)

    return _build


def build_customized_tb_forloop_accum_red_ld_rms() -> Builder:
    """TB forloop_accum(..., acc='rms'): RMS epilogue on last forloop iteration."""

    def _build():
        rows, cols, tile, fl = 4, 32, 16, 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(rows, cols), dtype=yr.float16)
        tb = _tb_graph(forloop_range=fl, block=tile, reduction_dimx=tile)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tacc = tb.forloop_accum(tx, "rms")
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16((rows, cols))
        n = float(cols)
        ref = torch.sqrt(inp.float().pow(2).sum(dim=-1, keepdim=True) / n + 1e-6)
        return g, [inp], ref.to(torch.float16)

    return _build


def build_customized_tb_clamp() -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        tb = _tb_graph()
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tc = tb.clamp(tx, -1.0, 1.0)
        inp = _f16((8, 16))
        ref = torch.clamp(inp.float(), -1.0, 1.0).to(torch.float16)
        return _finish_customized_unary(g, x, tb, tx, tc, inp, ref)

    return _build


def build_customized_tb_split(dim: int) -> Builder:
    def _build():
        if dim == 2:
            shape = (8, 16, 4)
        else:
            shape = (8, 16)
        split_size = shape[dim] // 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        tb = _tb_graph(forloop_range=1)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        parts = tb.split(tx, split_size, dim)
        tacc = tb.forloop_accum(parts[0])
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16(shape)
        ref = torch.split(inp, (split_size, shape[dim] - split_size), dim=dim)[0]
        return g, [inp], ref

    return _build


def build_customized_tb_concat(dim: int) -> Builder:
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
        tb = _tb_graph(forloop_range=1)
        ta = tb.new_input(dtensor=a, input_map=(-1, -1, -1), forloop_dim=1)
        tb_b = tb.new_input(dtensor=b, input_map=(-1, -1, -1), forloop_dim=1)
        ty = tb.concat(ta, tb_b, dim)
        tacc = tb.forloop_accum(ty)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([a, b], tb)
        g.mark_output(out[0])
        ta_t = _f16(shape_a)
        tb_t = _f16(shape_b)
        ref = torch.cat([ta_t, tb_t], dim=dim)
        return g, [ta_t, tb_t], ref

    return _build


def build_customized_tb_mul_scalar() -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        tb = _tb_graph()
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        tm = tb.mul_scalar(tx, 0.5)
        inp = _f16((8, 16))
        ref = (inp.float() * 0.5).to(torch.float16)
        return _finish_customized_unary(g, x, tb, tx, tm, inp, ref)

    return _build


    return _build


def build_customized_tb_chunk(dim: int) -> Builder:
    def _build():
        if dim == 2:
            shape = (8, 16, 4)
        else:
            shape = (8, 16)
        chunk_count = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        tb = _tb_graph(forloop_range=1)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        parts = tb.chunk(tx, chunk_count, dim)
        tacc = tb.forloop_accum(parts[0])
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16(shape)
        ref = torch.chunk(inp, chunk_count, dim=dim)[0]
        return g, [inp], ref

    return _build


def build_customized_tb_layout_chunk_concat_roundtrip(dim: int) -> Builder:
    def _build():
        if dim == 2:
            shape = (4, 8, 4)
        else:
            shape = (8, 16)
        chunk_count = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        tb = _tb_graph(forloop_range=1)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        parts = tb.chunk(tx, chunk_count, dim)
        ty = tb.concat(parts[0], parts[1], dim)
        tacc = tb.forloop_accum(ty)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16(shape)
        return g, [inp], inp

    return _build


def build_customized_tb_layout_chunk_split_first(dim: int) -> Builder:
    def _build():
        if dim == 2:
            shape = (4, 8, 4)
        else:
            shape = (8, 16)
        split_size = shape[dim] // 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        tb = _tb_graph(forloop_range=1)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        parts = tb.chunk(tx, 2, dim)
        merged = tb.concat(parts[0], parts[1], dim)
        sp = tb.split(merged, split_size, dim)
        tacc = tb.forloop_accum(sp[0])
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16(shape)
        ref = torch.split(
            inp, (split_size, shape[dim] - split_size), dim=dim
        )[0]
        return g, [inp], ref

    return _build


def build_customized_tb_layout_split_chunk_first(dim: int) -> Builder:
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
        tb = _tb_graph(forloop_range=1)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        sp = tb.split(tx, split_size, dim)
        chunks = tb.chunk(sp[0], 2, chunk_sub_dim)
        tacc = tb.forloop_accum(chunks[0])
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16(shape)
        first = torch.split(
            inp, (split_size, shape[dim] - split_size), dim=dim
        )[0]
        ref = torch.chunk(first, 2, dim=chunk_sub_dim)[0]
        return g, [inp], ref

    return _build


TB_OP_BUILDERS = {
    "tb_matmul_op": build_customized_tb_matmul(),
    "tb_exp_op": build_customized_tb_unary("exp"),
    "tb_square_op": build_customized_tb_unary("square"),
    "tb_sqrt_op": build_customized_tb_unary("sqrt", positive=True),
    "tb_silu_op": build_customized_tb_unary("silu"),
    "tb_sigmoid_op": build_customized_tb_unary("sigmoid"),
    "tb_gelu_op": build_customized_tb_unary("gelu"),
    "tb_relu_op": build_customized_tb_unary("relu"),
    "tb_log_op": build_customized_tb_unary("log", positive=True),
    "tb_add_op": build_customized_tb_binary("add"),
    "tb_mul_op": build_customized_tb_binary("mul"),
    "tb_div_op": build_customized_tb_binary("div"),
    "tb_sub_op": build_customized_tb_binary("sub"),
    "tb_pow_op": build_customized_tb_binary("pow", positive=True),
    "tb_rms_norm_op": build_customized_tb_rms_norm(),
    "tb_reduction_0_op": build_customized_tb_reduction(0),
    "tb_reduction_1_op": build_customized_tb_reduction(1),
    "tb_reduction_2_op": build_customized_tb_reduction(2),
    "tb_reduction_0_to_dimx_op": build_customized_tb_reduction_to_dimx(0),
    "tb_reduction_1_to_dimx_op": build_customized_tb_reduction_to_dimx(1),
    "tb_reduction_2_to_dimx_op": build_customized_tb_reduction_to_dimx(2),
    "tb_reduction_0_max_op": build_customized_tb_reduction_max(0),
    "tb_reduction_1_max_op": build_customized_tb_reduction_max(1),
    "tb_reduction_2_max_op": build_customized_tb_reduction_max(2),
    "tb_forloop_accum_no_red_op": build_customized_tb_forloop_accum_no_red(),
    "tb_forloop_accum_red_ld_sum_op": build_customized_tb_forloop_accum_red_ld_sum(),
    "tb_forloop_accum_red_ld_mean_op": build_customized_tb_forloop_accum_red_ld_mean(),
    "tb_forloop_accum_red_ld_rms_op": build_customized_tb_forloop_accum_red_ld_rms(),
    "tb_forloop_accum_redtox_ld_sum_op": build_customized_tb_forloop_accum_redtox_ld_sum(),
    "tb_forloop_accum_max_op": build_customized_tb_forloop_accum_max(),
    "tb_forloop_accum_no_red_rescale_op": build_customized_tb_forloop_accum_no_red_rescale(),
    "tb_forloop_accum_red_ld_sum_rescale_op": build_customized_tb_forloop_accum_red_ld_sum_rescale(),
    "tb_clamp_op": build_customized_tb_clamp(),
    "tb_mul_scalar_op": build_customized_tb_mul_scalar(),
    "tb_concat_0_op": build_customized_tb_concat(0),
    "tb_concat_1_op": build_customized_tb_concat(1),
    "tb_concat_2_op": build_customized_tb_concat(2),
    "tb_split_0_op": build_customized_tb_split(0),
    "tb_split_1_op": build_customized_tb_split(1),
    "tb_split_2_op": build_customized_tb_split(2),
    "tb_chunk_0_op": build_customized_tb_chunk(0),
    "tb_chunk_1_op": build_customized_tb_chunk(1),
    "tb_chunk_2_op": build_customized_tb_chunk(2),
}

def build_customized_tb_layout_split_concat_roundtrip(dim: int) -> Builder:
    """TB split then concat along the same dim (search explore round-trip)."""

    def _build():
        if dim == 2:
            shape = (8, 16, 4)
        else:
            shape = (8, 16)
        split_size = shape[dim] // 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=shape, dtype=yr.float16)
        tb = _tb_graph(forloop_range=1)
        tx = tb.new_input(dtensor=x, input_map=(-1, -1, -1), forloop_dim=1)
        parts = tb.split(tx, split_size, dim)
        ty = tb.concat(parts[0], parts[1], dim)
        tacc = tb.forloop_accum(ty)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([x], tb)
        g.mark_output(out[0])
        inp = _f16(shape)
        return g, [inp], inp

    return _build


def build_customized_tb_layout_concat_split_first(dim: int) -> Builder:
    """TB concat two tensors then split to recover the first half."""

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
        tb = _tb_graph(forloop_range=1)
        ta = tb.new_input(dtensor=a, input_map=(-1, -1, -1), forloop_dim=1)
        tb_b = tb.new_input(dtensor=b, input_map=(-1, -1, -1), forloop_dim=1)
        merged = tb.concat(ta, tb_b, dim)
        parts = tb.split(merged, split_size, dim)
        tacc = tb.forloop_accum(parts[0])
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([a, b], tb)
        g.mark_output(out[0])
        ta_t, tb_t = _f16(shape_a), _f16(shape_b)
        return g, [ta_t, tb_t], ta_t

    return _build


    return _build


def build_customized_tb_layout_concat_matmul() -> Builder:
    """LoRA-style dual TB concat + matmul (symmetric kn_layout_concat_matmul)."""

    def _build():
        m, k1, k2, n = 16, 32, 32, 64
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(m, k1), dtype=yr.float16)
        b = g.new_input(dims=(m, k2), dtype=yr.float16)
        c = g.new_input(dims=(k1, n), dtype=yr.float16)
        d = g.new_input(dims=(k2, n), dtype=yr.float16)
        tb = _tb_graph(forloop_range=1, block=64, reduction_dimx=k1 + k2)
        ta = tb.new_input(dtensor=a, input_map=(-1, -1, -1), forloop_dim=1)
        tb_b = tb.new_input(dtensor=b, input_map=(-1, -1, -1), forloop_dim=1)
        tc = tb.new_input(dtensor=c, input_map=(-1, -1, -1), forloop_dim=0)
        td = tb.new_input(dtensor=d, input_map=(-1, -1, -1), forloop_dim=0)
        left = tb.concat(ta, tb_b, dim=1)
        right = tb.concat(tc, td, dim=0)
        tm = tb.matmul(left, right)
        tacc = tb.forloop_accum(tm)
        tb.new_output(stensor=tacc, output_map=(-1, -1, -1))
        out = g.customized([a, b, c, d], tb)
        g.mark_output(out[0])
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


TB_LAYOUT_EXPLORE_BUILDERS = {
    "tb_layout_split_concat_roundtrip_dim0": build_customized_tb_layout_split_concat_roundtrip(0),
    "tb_layout_split_concat_roundtrip_dim1": build_customized_tb_layout_split_concat_roundtrip(1),
    "tb_layout_split_concat_roundtrip_dim2": build_customized_tb_layout_split_concat_roundtrip(2),
    "tb_layout_concat_split_first_dim0": build_customized_tb_layout_concat_split_first(0),
    "tb_layout_concat_split_first_dim1": build_customized_tb_layout_concat_split_first(1),
    "tb_layout_concat_split_first_dim2": build_customized_tb_layout_concat_split_first(2),
    "tb_layout_chunk_concat_roundtrip_dim0": build_customized_tb_layout_chunk_concat_roundtrip(0),
    "tb_layout_chunk_concat_roundtrip_dim1": build_customized_tb_layout_chunk_concat_roundtrip(1),
    "tb_layout_chunk_concat_roundtrip_dim2": build_customized_tb_layout_chunk_concat_roundtrip(2),
    "tb_layout_chunk_split_first_dim0": build_customized_tb_layout_chunk_split_first(0),
    "tb_layout_chunk_split_first_dim1": build_customized_tb_layout_chunk_split_first(1),
    "tb_layout_chunk_split_first_dim2": build_customized_tb_layout_chunk_split_first(2),
    "tb_layout_split_chunk_first_dim0": build_customized_tb_layout_split_chunk_first(0),
    "tb_layout_split_chunk_first_dim1": build_customized_tb_layout_split_chunk_first(1),
    "tb_layout_split_chunk_first_dim2": build_customized_tb_layout_split_chunk_first(2),
    "tb_layout_concat_matmul": build_customized_tb_layout_concat_matmul(),
}

# Retained for inventory/docs; all patterns are now active in TB_LAYOUT_EXPLORE_BUILDERS (R54).
TB_LAYOUT_CHUNK_DEFERRED_PATTERNS = frozenset()

TB_UNSUPPORTED_BUILDERS: dict = {}
