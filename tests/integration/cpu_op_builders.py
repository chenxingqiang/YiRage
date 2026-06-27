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


def build_kn_matmul_batch1() -> Builder:
    """2D KN matmul batch=1 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.matmul(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        return g, [ta, tb], torch.matmul(ta, tb)

    return _build


def build_kn_matmul_batch2() -> Builder:
    """2D KN matmul batch=2 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.matmul(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        return g, [ta, tb], torch.matmul(ta, tb)

    return _build


def build_kn_matmul_3d_2d() -> Builder:
    """KN matmul with PyTorch-style broadcast: [B,M,K] @ [K,N] -> [B,M,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(2, 4, 8), dtype=yr.float16)
        b = g.new_input(dims=(8, 16), dtype=yr.float16)
        g.mark_output(g.matmul(a, b))
        ta, tb = _f16((2, 4, 8)), _f16((8, 16))
        return g, [ta, tb], torch.matmul(ta, tb)

    return _build


def build_kn_matmul_3d_2d_batch1() -> Builder:
    """KN matmul batch=1 broadcast: [1,M,K] @ [K,N] -> [1,M,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, 4, 8), dtype=yr.float16)
        b = g.new_input(dims=(8, 16), dtype=yr.float16)
        g.mark_output(g.matmul(a, b))
        ta, tb = _f16((1, 4, 8)), _f16((8, 16))
        return g, [ta, tb], torch.matmul(ta, tb)

    return _build


def build_kn_matmul_3d_2d_batch2() -> Builder:
    """KN matmul batch=2 broadcast: [2,M,K] @ [K,N] -> [2,M,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(2, 4, 8), dtype=yr.float16)
        b = g.new_input(dims=(8, 16), dtype=yr.float16)
        g.mark_output(g.matmul(a, b))
        ta, tb = _f16((2, 4, 8)), _f16((8, 16))
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


def build_kn_conv2d() -> Builder:
    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        g.mark_output(
            g.conv2d(x, w, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((1, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        ref = torch.nn.functional.conv2d(
            inp_x, inp_w, stride=(1, 1), padding=(1, 1), dilation=(1, 1)
        )
        return g, [inp_x, inp_w], ref

    return _build


def build_kn_conv2d_batch1() -> Builder:
    """KN conv2d batch=1 shape contract [1,C,H,W] NCHW."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        g.mark_output(
            g.conv2d(x, w, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((1, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        ref = torch.nn.functional.conv2d(
            inp_x, inp_w, stride=(1, 1), padding=(1, 1), dilation=(1, 1)
        )
        return g, [inp_x, inp_w], ref

    return _build


def build_kn_conv2d_batch2() -> Builder:
    """KN conv2d batch=2 shape contract [2,C,H,W] NCHW."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        g.mark_output(
            g.conv2d(x, w, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((2, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        ref = torch.nn.functional.conv2d(
            inp_x, inp_w, stride=(1, 1), padding=(1, 1), dilation=(1, 1)
        )
        return g, [inp_x, inp_w], ref

    return _build


def build_kn_conv2d_groups() -> Builder:
    """Grouped conv2d (groups=2) aligned with F.conv2d."""

    def _build():
        groups = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
        g.mark_output(
            g.conv2d(x, w, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=groups)
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((8, 2, 3, 3))
        ref = torch.nn.functional.conv2d(
            inp_x, inp_w, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=groups
        )
        return g, [inp_x, inp_w], ref

    return _build


def build_kn_conv2d_groups_batch1() -> Builder:
    """Grouped conv2d batch=1 [1,C,H,W] (groups=2) shape contract."""

    def _build():
        groups = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
        g.mark_output(
            g.conv2d(x, w, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=groups)
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((8, 2, 3, 3))
        ref = torch.nn.functional.conv2d(
            inp_x, inp_w, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=groups
        )
        return g, [inp_x, inp_w], ref

    return _build


def build_kn_conv2d_groups_batch2() -> Builder:
    """Grouped conv2d batch=2 [2,C,H,W] (groups=2) shape contract."""

    def _build():
        groups = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
        g.mark_output(
            g.conv2d(x, w, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=groups)
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_w = _f16((8, 2, 3, 3))
        ref = torch.nn.functional.conv2d(
            inp_x, inp_w, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=groups
        )
        return g, [inp_x, inp_w], ref

    return _build


def build_conv2d_groups_batch1() -> Builder:
    """Grouped conv2d batch=1 inference [1,C,H,W] (groups=2) NCHW."""

    def _build():
        groups = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
        g.mark_output(
            g.conv2d(x, w, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=groups)
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((8, 2, 3, 3))
        ref = torch.nn.functional.conv2d(
            inp_x, inp_w, stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=groups
        )
        return g, [inp_x, inp_w], ref

    return _build


def build_conv2d_bias() -> Builder:
    """Conv2d + broadcast bias (F.conv2d with bias vector parity)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((1, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.conv2d(
            inp_x,
            inp_w,
            bias=inp_b.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_batch1() -> Builder:
    """Conv2d + bias batch=1 inference [1,C,H,W] NCHW."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((1, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.conv2d(
            inp_x,
            inp_w,
            bias=inp_b.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_batch2() -> Builder:
    """Conv2d + bias batch=2 inference [2,C,H,W] NCHW."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((2, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.conv2d(
            inp_x,
            inp_w,
            bias=inp_b.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_relu() -> Builder:
    """Conv2d + bias + ReLU vs F.relu(F.conv2d(...))."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias_relu(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((1, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.relu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_relu_batch1() -> Builder:
    """Conv2d + bias + ReLU batch=1 inference [1,C,H,W] NCHW."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias_relu(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((1, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.relu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_relu_batch2() -> Builder:
    """Conv2d + bias + ReLU batch=2 inference [2,C,H,W] NCHW."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias_relu(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((2, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.relu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_gelu() -> Builder:
    """Conv2d + bias + GELU vs F.gelu(F.conv2d(...))."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias_gelu(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((1, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.gelu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_gelu_batch1() -> Builder:
    """Conv2d + bias + GELU batch=1 inference [1,C,H,W] NCHW."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias_gelu(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((1, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.gelu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_gelu_batch2() -> Builder:
    """Conv2d + bias + GELU batch=2 inference [2,C,H,W] NCHW."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias_gelu(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((2, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.gelu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_silu() -> Builder:
    """Conv2d + bias + SiLU vs F.silu(F.conv2d(...))."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias_silu(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((1, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.silu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_silu_batch1() -> Builder:
    """Conv2d + bias + SiLU batch=1 inference [1,C,H,W] NCHW."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias_silu(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((1, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.silu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_silu_batch2() -> Builder:
    """Conv2d + bias + SiLU batch=2 inference [2,C,H,W] NCHW."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 3, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 3, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias_silu(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
        )
        inp_x = _f16((2, 3, 8, 8))
        inp_w = _f16((4, 3, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.silu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_groups() -> Builder:
    """Grouped conv2d + bias (groups=2)."""

    def _build():
        groups = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias(
                x,
                w,
                b,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=groups,
            )
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((8, 2, 3, 3))
        inp_b = _f16((1, 8, 1, 1))
        ref = torch.nn.functional.conv2d(
            inp_x,
            inp_w,
            bias=inp_b.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=groups,
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_groups_batch1() -> Builder:
    """Grouped conv2d + bias batch=1 inference [1,C,H,W] (groups=2)."""

    def _build():
        groups = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias(
                x,
                w,
                b,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=groups,
            )
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((8, 2, 3, 3))
        inp_b = _f16((1, 8, 1, 1))
        ref = torch.nn.functional.conv2d(
            inp_x,
            inp_w,
            bias=inp_b.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=groups,
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_bias_groups_batch2() -> Builder:
    """Grouped conv2d + bias batch=2 inference [2,C,H,W] (groups=2)."""

    def _build():
        groups = 2
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(8, 2, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_bias(
                x,
                w,
                b,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=groups,
            )
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_w = _f16((8, 2, 3, 3))
        inp_b = _f16((1, 8, 1, 1))
        ref = torch.nn.functional.conv2d(
            inp_x,
            inp_w,
            bias=inp_b.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            groups=groups,
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias() -> Builder:
    """Depthwise conv2d + bias (groups = in_channels)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.conv2d(
            inp_x,
            inp_w,
            bias=inp_b.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_batch1() -> Builder:
    """Depthwise conv2d + bias batch=1 inference [1,C,H,W] (groups=C)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.conv2d(
            inp_x,
            inp_w,
            bias=inp_b.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_batch2() -> Builder:
    """Depthwise conv2d + bias batch=2 inference [2,C,H,W] (groups=C)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.conv2d(
            inp_x,
            inp_w,
            bias=inp_b.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_relu() -> Builder:
    """Depthwise conv2d + bias + ReLU vs F.relu(F.conv2d(..., groups=C))."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias_relu(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.relu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                groups=4,
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_relu_batch1() -> Builder:
    """Depthwise conv2d + bias + ReLU batch=1 inference [1,C,H,W] (groups=C)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias_relu(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.relu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                groups=4,
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_relu_batch2() -> Builder:
    """Depthwise conv2d + bias + ReLU batch=2 [2,C,H,W] (groups=C)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias_relu(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.relu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                groups=4,
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_gelu() -> Builder:
    """Depthwise conv2d + bias + GELU vs F.gelu(F.conv2d(..., groups=C))."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias_gelu(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.gelu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                groups=4,
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_silu() -> Builder:
    """Depthwise conv2d + bias + SiLU vs F.silu(F.conv2d(..., groups=C))."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias_silu(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.silu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                groups=4,
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_gelu_batch1() -> Builder:
    """Depthwise conv2d + bias + GELU batch=1 inference [1,C,H,W] (groups=C)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias_gelu(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.gelu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                groups=4,
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_gelu_batch2() -> Builder:
    """Depthwise conv2d + bias + GELU batch=2 [2,C,H,W] (groups=C)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias_gelu(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.gelu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                groups=4,
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_silu_batch1() -> Builder:
    """Depthwise conv2d + bias + SiLU batch=1 inference [1,C,H,W] (groups=C)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias_silu(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.silu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                groups=4,
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_depthwise_bias_silu_batch2() -> Builder:
    """Depthwise conv2d + bias + SiLU batch=2 [2,C,H,W] (groups=C)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        w = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        b = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_depthwise_bias_silu(x, w, b, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_w = _f16((4, 1, 3, 3))
        inp_b = _f16((1, 4, 1, 1))
        ref = torch.nn.functional.silu(
            torch.nn.functional.conv2d(
                inp_x,
                inp_w,
                bias=inp_b.reshape(-1),
                stride=(1, 1),
                padding=(1, 1),
                groups=4,
            )
        )
        return g, [inp_x, inp_w, inp_b], ref

    return _build


def build_conv2d_separable() -> Builder:
    """Depthwise + 1x1 pointwise separable conv (no bias)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable(x, dw, pw, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x, inp_dw, stride=(1, 1), padding=(1, 1), groups=4
        )
        ref = torch.nn.functional.conv2d(hidden, inp_pw)
        return g, [inp_x, inp_dw, inp_pw], ref

    return _build


def build_conv2d_separable_batch1() -> Builder:
    """Separable conv batch=1 inference [1,C,H,W] (depthwise + pointwise)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable(x, dw, pw, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x, inp_dw, stride=(1, 1), padding=(1, 1), groups=4
        )
        ref = torch.nn.functional.conv2d(hidden, inp_pw)
        return g, [inp_x, inp_dw, inp_pw], ref

    return _build


def build_conv2d_separable_batch2() -> Builder:
    """Separable conv batch=2 inference [2,C,H,W] (depthwise + pointwise)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable(x, dw, pw, stride=(1, 1), padding=(1, 1))
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x, inp_dw, stride=(1, 1), padding=(1, 1), groups=4
        )
        ref = torch.nn.functional.conv2d(hidden, inp_pw)
        return g, [inp_x, inp_dw, inp_pw], ref

    return _build


def build_conv2d_separable_bias() -> Builder:
    """Separable conv with depthwise and pointwise broadcast biases."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = mid + inp_pb
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_batch2() -> Builder:
    """Separable conv + biases batch=2 inference [2,C,H,W]."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = mid + inp_pb
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_relu() -> Builder:
    """Separable conv + biases + ReLU vs F.relu(separable reference)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias_relu(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = torch.nn.functional.relu(mid + inp_pb)
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_gelu() -> Builder:
    """Separable conv + biases + GELU vs F.gelu(separable reference)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias_gelu(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = torch.nn.functional.gelu(mid + inp_pb)
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_silu() -> Builder:
    """Separable conv + biases + SiLU vs F.silu(separable reference)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias_silu(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = torch.nn.functional.silu(mid + inp_pb)
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_relu_batch2() -> Builder:
    """Separable conv + biases + ReLU batch=2 [2,C,H,W]."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias_relu(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = torch.nn.functional.relu(mid + inp_pb)
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_gelu_batch2() -> Builder:
    """Separable conv + biases + GELU batch=2 [2,C,H,W]."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias_gelu(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = torch.nn.functional.gelu(mid + inp_pb)
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_silu_batch2() -> Builder:
    """Separable conv + biases + SiLU batch=2 [2,C,H,W]."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(2, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias_silu(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((2, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = torch.nn.functional.silu(mid + inp_pb)
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_batch1() -> Builder:
    """Separable conv + biases batch=1 inference [1,C,H,W]."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = mid + inp_pb
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_relu_batch1() -> Builder:
    """Separable conv + biases + ReLU batch=1 inference [1,C,H,W]."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias_relu(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = torch.nn.functional.relu(mid + inp_pb)
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_gelu_batch1() -> Builder:
    """Separable conv + biases + GELU batch=1 inference [1,C,H,W]."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias_gelu(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = torch.nn.functional.gelu(mid + inp_pb)
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


def build_conv2d_separable_bias_silu_batch1() -> Builder:
    """Separable conv + biases + SiLU batch=1 inference [1,C,H,W]."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, 4, 8, 8), dtype=yr.float16)
        dw = g.new_input(dims=(4, 1, 3, 3), dtype=yr.float16)
        pw = g.new_input(dims=(8, 4, 1, 1), dtype=yr.float16)
        db = g.new_input(dims=(1, 4, 1, 1), dtype=yr.float16)
        pb = g.new_input(dims=(1, 8, 1, 1), dtype=yr.float16)
        g.mark_output(
            g.conv2d_separable_bias_silu(
                x, dw, pw, db, pb, stride=(1, 1), padding=(1, 1)
            )
        )
        inp_x = _f16((1, 4, 8, 8))
        inp_dw = _f16((4, 1, 3, 3))
        inp_pw = _f16((8, 4, 1, 1))
        inp_db = _f16((1, 4, 1, 1))
        inp_pb = _f16((1, 8, 1, 1))
        hidden = torch.nn.functional.conv2d(
            inp_x,
            inp_dw,
            bias=inp_db.reshape(-1),
            stride=(1, 1),
            padding=(1, 1),
            groups=4,
        )
        mid = torch.nn.functional.conv2d(hidden, inp_pw)
        ref = torch.nn.functional.silu(mid + inp_pb)
        return g, [inp_x, inp_dw, inp_pw, inp_db, inp_pb], ref

    return _build


KN_OP_BUILDERS = {
    "kn_matmul_op": build_kn_matmul(),
    "kn_matmul_3d_2d_op": build_kn_matmul_3d_2d(),
    "kn_matmul_3d_2d_batch1_op": build_kn_matmul_3d_2d_batch1(),
    "kn_matmul_3d_2d_batch2_op": build_kn_matmul_3d_2d_batch2(),
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
    "kn_conv2d_op": build_kn_conv2d(),
    "kn_conv2d_batch1_op": build_kn_conv2d_batch1(),
    "kn_conv2d_batch2_op": build_kn_conv2d_batch2(),
    "kn_conv2d_groups_batch1_op": build_kn_conv2d_groups_batch1(),
    "kn_conv2d_groups_batch2_op": build_kn_conv2d_groups_batch2(),
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


def build_kn_rms_norm_batch1() -> Builder:
    """2D KN rms_norm batch=1 inference [M,D]."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 32), dtype=yr.float16)
        g.mark_output(g.rms_norm(x, normalized_shape=(32,)))
        inp = _f16((8, 32))
        scale = torch.rsqrt(inp.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = (inp.float() * scale).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_rms_norm_batch2() -> Builder:
    """2D KN rms_norm batch=2 inference [M,D]."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 32), dtype=yr.float16)
        g.mark_output(g.rms_norm(x, normalized_shape=(32,)))
        inp = _f16((8, 32))
        scale = torch.rsqrt(inp.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = (inp.float() * scale).to(torch.float16)
        return g, [inp], ref

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


def build_kn_softmax_batch1() -> Builder:
    """2D KN softmax batch=1 inference [M,N] vs F.softmax."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        g.mark_output(g.softmax(x, dim=-1))
        inp = _f16((8, 16))
        ref = torch.nn.functional.softmax(inp.float(), dim=-1).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_softmax_batch2() -> Builder:
    """2D KN softmax batch=2 inference [M,N] vs F.softmax."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        g.mark_output(g.softmax(x, dim=-1))
        inp = _f16((8, 16))
        ref = torch.nn.functional.softmax(inp.float(), dim=-1).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_softmax_3d() -> Builder:
    """3D softmax [B,S,N] vs F.softmax on last dim."""

    def _build():
        batch, seq, n = 2, 4, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, n), dtype=yr.float16)
        g.mark_output(g.softmax(x, dim=-1))
        inp = _f16((batch, seq, n))
        ref = torch.nn.functional.softmax(inp.float(), dim=-1).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_rms_norm_3d() -> Builder:
    """3D RMSNorm [B,S,D] vs torch rsqrt mean square scaling."""

    def _build():
        batch, seq, dim = 2, 4, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        g.mark_output(g.rms_norm(x, normalized_shape=(dim,)))
        inp = _f16((batch, seq, dim))
        scale = torch.rsqrt(inp.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = (inp.float() * scale).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_softmax_3d_batch1() -> Builder:
    """3D softmax batch=1 fast path [1,S,N] vs F.softmax."""

    def _build():
        seq, n = 4, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, n), dtype=yr.float16)
        g.mark_output(g.softmax(x, dim=-1))
        inp = _f16((1, seq, n))
        ref = torch.nn.functional.softmax(inp.float(), dim=-1).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_softmax_3d_batch2() -> Builder:
    """3D softmax batch=2 fast path [2,S,N] vs F.softmax."""

    def _build():
        batch, seq, n = 2, 4, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, n), dtype=yr.float16)
        g.mark_output(g.softmax(x, dim=-1))
        inp = _f16((batch, seq, n))
        ref = torch.nn.functional.softmax(inp.float(), dim=-1).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_rms_norm_3d_batch1() -> Builder:
    """3D RMSNorm batch=1 fast path [1,S,D]."""

    def _build():
        seq, dim = 4, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        g.mark_output(g.rms_norm(x, normalized_shape=(dim,)))
        inp = _f16((1, seq, dim))
        scale = torch.rsqrt(inp.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = (inp.float() * scale).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_rms_norm_3d_batch2() -> Builder:
    """3D RMSNorm batch=2 fast path [2,S,D]."""

    def _build():
        batch, seq, dim = 2, 4, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        g.mark_output(g.rms_norm(x, normalized_shape=(dim,)))
        inp = _f16((batch, seq, dim))
        scale = torch.rsqrt(inp.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = (inp.float() * scale).to(torch.float16)
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


def build_kn_layer_norm_batch1() -> Builder:
    """2D KN layer_norm batch=1 inference [M,N] (eps=0)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        g.mark_output(g.layer_norm(x, normalized_shape=(16,), eps=0.0))
        inp = _f16((8, 16))
        ref = torch.nn.functional.layer_norm(inp.float(), (16,), eps=0.0).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_layer_norm_batch2() -> Builder:
    """2D KN layer_norm batch=2 inference [M,N] (eps=0)."""

    def _build():
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(8, 16), dtype=yr.float16)
        g.mark_output(g.layer_norm(x, normalized_shape=(16,), eps=0.0))
        inp = _f16((8, 16))
        ref = torch.nn.functional.layer_norm(inp.float(), (16,), eps=0.0).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_layer_norm_3d() -> Builder:
    """3D layer_norm [B,S,N] vs F.layer_norm (eps=0)."""

    def _build():
        batch, seq, n = 2, 4, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, n), dtype=yr.float16)
        g.mark_output(g.layer_norm(x, normalized_shape=(n,), eps=0.0))
        inp = _f16((batch, seq, n))
        ref = torch.nn.functional.layer_norm(inp.float(), (n,), eps=0.0).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_layer_norm_3d_batch1() -> Builder:
    """3D layer_norm batch=1 fast path [1,S,N]."""

    def _build():
        seq, n = 4, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, n), dtype=yr.float16)
        g.mark_output(g.layer_norm(x, normalized_shape=(n,), eps=0.0))
        inp = _f16((1, seq, n))
        ref = torch.nn.functional.layer_norm(inp.float(), (n,), eps=0.0).to(torch.float16)
        return g, [inp], ref

    return _build


def build_kn_layer_norm_3d_batch2() -> Builder:
    """3D layer_norm batch=2 fast path [2,S,N]."""

    def _build():
        batch, seq, n = 2, 4, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, n), dtype=yr.float16)
        g.mark_output(g.layer_norm(x, normalized_shape=(n,), eps=0.0))
        inp = _f16((batch, seq, n))
        ref = torch.nn.functional.layer_norm(inp.float(), (n,), eps=0.0).to(torch.float16)
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


def build_gemm_softmax_batch1() -> Builder:
    """2D GEMM + softmax batch=1 inference [M,K] @ [K,N]."""

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


def build_gemm_softmax_batch2() -> Builder:
    """2D GEMM + softmax batch=2 inference [M,K] @ [K,N]."""

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


def build_gemm_softmax_scaled() -> Builder:
    """Scaled gemm_softmax vs softmax(matmul / sqrt(d)) (attention scores)."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(seq, dim), dtype=yr.float16)
        b = g.new_input(dims=(dim, seq), dtype=yr.float16)
        g.mark_output(g.gemm_softmax_scaled(a, b, dim=-1, head_dim=dim))
        ta, tb = _f16((seq, dim)), _f16((dim, seq))
        scale = dim ** -0.5
        scores = torch.matmul(ta.float(), tb.float()) * scale
        ref = torch.nn.functional.softmax(scores, dim=-1).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_scaled_batch1() -> Builder:
    """2D scaled GEMM + softmax batch=1 inference [S,D] @ [D,S]."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(seq, dim), dtype=yr.float16)
        b = g.new_input(dims=(dim, seq), dtype=yr.float16)
        g.mark_output(g.gemm_softmax_scaled(a, b, dim=-1, head_dim=dim))
        ta, tb = _f16((seq, dim)), _f16((dim, seq))
        scale = dim ** -0.5
        scores = torch.matmul(ta.float(), tb.float()) * scale
        ref = torch.nn.functional.softmax(scores, dim=-1).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_scaled_batch2() -> Builder:
    """Scaled gemm_softmax batch=2 inference [S,D] @ [D,S]."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(seq, dim), dtype=yr.float16)
        b = g.new_input(dims=(dim, seq), dtype=yr.float16)
        g.mark_output(g.gemm_softmax_scaled(a, b, dim=-1, head_dim=dim))
        ta, tb = _f16((seq, dim)), _f16((dim, seq))
        scale = dim ** -0.5
        scores = torch.matmul(ta.float(), tb.float()) * scale
        ref = torch.nn.functional.softmax(scores, dim=-1).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_scaled_batched() -> Builder:
    """Batched scaled gemm_softmax on 3D [B,S,D] / [B,D,S]."""

    def _build():
        batch, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        b = g.new_input(dims=(batch, dim, seq), dtype=yr.float16)
        g.mark_output(g.gemm_softmax_scaled_batched(a, b, dim=-1, head_dim=dim))
        ta = _f16((batch, seq, dim))
        tb = _f16((batch, dim, seq))
        scale = dim ** -0.5
        outs = []
        for bi in range(batch):
            scores = torch.matmul(ta[bi].float(), tb[bi].float()) * scale
            outs.append(torch.nn.functional.softmax(scores, dim=-1))
        ref = torch.stack(outs, dim=0).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_scaled_batched_batch1() -> Builder:
    """Batched scaled gemm_softmax batch=1 [1,S,D] / [1,D,S]."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        b = g.new_input(dims=(1, dim, seq), dtype=yr.float16)
        g.mark_output(g.gemm_softmax_scaled_batched(a, b, dim=-1, head_dim=dim))
        ta = _f16((1, seq, dim))
        tb = _f16((1, dim, seq))
        scale = dim ** -0.5
        scores = torch.matmul(ta[0].float(), tb[0].float()) * scale
        ref = torch.nn.functional.softmax(scores, dim=-1).unsqueeze(0).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_scaled_batched_batch2() -> Builder:
    """Batched scaled gemm_softmax batch=2 [2,S,D] / [2,D,S]."""

    def _build():
        batch, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        b = g.new_input(dims=(batch, dim, seq), dtype=yr.float16)
        g.mark_output(g.gemm_softmax_scaled_batched(a, b, dim=-1, head_dim=dim))
        ta = _f16((batch, seq, dim))
        tb = _f16((batch, dim, seq))
        scale = dim ** -0.5
        outs = []
        for bi in range(batch):
            scores = torch.matmul(ta[bi].float(), tb[bi].float()) * scale
            outs.append(torch.nn.functional.softmax(scores, dim=-1))
        ref = torch.stack(outs, dim=0).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_3d() -> Builder:
    """3D GEMM + softmax [B,S,K] @ [K,N] vs matmul + F.softmax."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_softmax(a, b, dim=-1))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.softmax(c, dim=-1).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_3d_batch2() -> Builder:
    """3D GEMM + softmax batch=2 [2,S,K] @ [K,N] vs matmul + F.softmax."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_softmax(a, b, dim=-1))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.softmax(c, dim=-1).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_scaled_3d() -> Builder:
    """Scaled 3D attention scores [B,S,D] @ [D,S] vs softmax(matmul / sqrt(d))."""

    def _build():
        batch, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        b = g.new_input(dims=(dim, seq), dtype=yr.float16)
        g.mark_output(g.gemm_softmax_scaled(a, b, dim=-1, head_dim=dim))
        ta = _f16((batch, seq, dim))
        tb = _f16((dim, seq))
        scale = dim ** -0.5
        c = torch.matmul(ta.float(), tb.float()) * scale
        ref = torch.nn.functional.softmax(c, dim=-1).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_3d_batch1() -> Builder:
    """3D GEMM + softmax batch=1 [1,S,K] @ [K,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_softmax(a, b, dim=-1))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.softmax(c, dim=-1).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_scaled_3d_batch1() -> Builder:
    """Scaled 3D attention scores batch=1 [1,S,D] @ [D,S]."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        b = g.new_input(dims=(dim, seq), dtype=yr.float16)
        g.mark_output(g.gemm_softmax_scaled(a, b, dim=-1, head_dim=dim))
        ta = _f16((1, seq, dim))
        tb = _f16((dim, seq))
        scale = dim ** -0.5
        c = torch.matmul(ta.float(), tb.float()) * scale
        ref = torch.nn.functional.softmax(c, dim=-1).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_softmax_scaled_3d_batch2() -> Builder:
    """Scaled 3D attention scores batch=2 [2,S,D] @ [D,S]."""

    def _build():
        batch, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        b = g.new_input(dims=(dim, seq), dtype=yr.float16)
        g.mark_output(g.gemm_softmax_scaled(a, b, dim=-1, head_dim=dim))
        ta = _f16((batch, seq, dim))
        tb = _f16((dim, seq))
        scale = dim ** -0.5
        c = torch.matmul(ta.float(), tb.float()) * scale
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


def build_gemm_layernorm_gelu() -> Builder:
    """GEMM + LayerNorm + GELU vs F.gelu(layer_norm(matmul))."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_gelu(a, b, normalized_shape=(16,), eps=0.0))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.gelu(
            torch.nn.functional.layer_norm(c, (16,), eps=0.0)
        ).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_relu() -> Builder:
    """GEMM + LayerNorm + ReLU vs F.relu(layer_norm(matmul))."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_relu(a, b, normalized_shape=(16,), eps=0.0))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.relu(
            torch.nn.functional.layer_norm(c, (16,), eps=0.0)
        ).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_silu() -> Builder:
    """GEMM + LayerNorm + SiLU vs F.silu(layer_norm(matmul))."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_silu(a, b, normalized_shape=(16,), eps=0.0))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.silu(
            torch.nn.functional.layer_norm(c, (16,), eps=0.0)
        ).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_batch1() -> Builder:
    """2D GEMM + LayerNorm batch=1 inference [M,K] @ [K,N]."""

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


def build_gemm_layernorm_batch2() -> Builder:
    """2D GEMM + LayerNorm batch=2 inference [M,K] @ [K,N]."""

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


def build_gemm_layernorm_gelu_batch1() -> Builder:
    """2D GEMM + LayerNorm + GELU batch=1 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_gelu(a, b, normalized_shape=(16,), eps=0.0))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.gelu(
            torch.nn.functional.layer_norm(c, (16,), eps=0.0)
        ).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_gelu_batch2() -> Builder:
    """2D GEMM + LayerNorm + GELU batch=2 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_gelu(a, b, normalized_shape=(16,), eps=0.0))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.gelu(
            torch.nn.functional.layer_norm(c, (16,), eps=0.0)
        ).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_relu_batch1() -> Builder:
    """2D GEMM + LayerNorm + ReLU batch=1 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_relu(a, b, normalized_shape=(16,), eps=0.0))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.relu(
            torch.nn.functional.layer_norm(c, (16,), eps=0.0)
        ).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_relu_batch2() -> Builder:
    """2D GEMM + LayerNorm + ReLU batch=2 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_relu(a, b, normalized_shape=(16,), eps=0.0))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.relu(
            torch.nn.functional.layer_norm(c, (16,), eps=0.0)
        ).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_silu_batch1() -> Builder:
    """2D GEMM + LayerNorm + SiLU batch=1 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_silu(a, b, normalized_shape=(16,), eps=0.0))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.silu(
            torch.nn.functional.layer_norm(c, (16,), eps=0.0)
        ).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_silu_batch2() -> Builder:
    """2D GEMM + LayerNorm + SiLU batch=2 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_silu(a, b, normalized_shape=(16,), eps=0.0))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.silu(
            torch.nn.functional.layer_norm(c, (16,), eps=0.0)
        ).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def _gemm_layernorm_3d_ref(
    ta: torch.Tensor,
    tb: torch.Tensor,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    c = torch.matmul(ta.float(), tb.float())
    out = torch.nn.functional.layer_norm(c, (c.shape[-1],), eps=0.0)
    if activation == "gelu":
        out = torch.nn.functional.gelu(out)
    elif activation == "relu":
        out = torch.nn.functional.relu(out)
    elif activation == "silu":
        out = torch.nn.functional.silu(out)
    return out.to(torch.float16)


def build_gemm_layernorm_3d() -> Builder:
    """3D GEMM + LayerNorm [B,S,K] @ [K,N] vs torch matmul + F.layer_norm."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_batch1() -> Builder:
    """3D GEMM + LayerNorm batch=1 [1,S,K] @ [K,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_batch2() -> Builder:
    """3D GEMM + LayerNorm batch=2 [2,S,K] @ [K,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb)
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_gelu() -> Builder:
    """3D GEMM + LayerNorm + GELU [B,S,K] @ [K,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_gelu(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb, activation="gelu")
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_relu() -> Builder:
    """3D GEMM + LayerNorm + ReLU [B,S,K] @ [K,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_relu(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb, activation="relu")
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_silu() -> Builder:
    """3D GEMM + LayerNorm + SiLU [B,S,K] @ [K,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_silu(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb, activation="silu")
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_gelu_batch1() -> Builder:
    """3D GEMM + LayerNorm + GELU batch=1 [1,S,K] @ [K,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_gelu(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb, activation="gelu")
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_relu_batch1() -> Builder:
    """3D GEMM + LayerNorm + ReLU batch=1 [1,S,K] @ [K,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_relu(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb, activation="relu")
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_silu_batch1() -> Builder:
    """3D GEMM + LayerNorm + SiLU batch=1 [1,S,K] @ [K,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_silu(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb, activation="silu")
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_gelu_batch2() -> Builder:
    """3D GEMM + LayerNorm + GELU batch=2 [2,S,K] @ [K,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_gelu(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb, activation="gelu")
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_relu_batch2() -> Builder:
    """3D GEMM + LayerNorm + ReLU batch=2 [2,S,K] @ [K,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_relu(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb, activation="relu")
        return g, [ta, tb], ref

    return _build


def build_gemm_layernorm_3d_silu_batch2() -> Builder:
    """3D GEMM + LayerNorm + SiLU batch=2 [2,S,K] @ [K,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_layernorm_silu(a, b, normalized_shape=(n,), eps=0.0))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        ref = _gemm_layernorm_3d_ref(ta, tb, activation="silu")
        return g, [ta, tb], ref

    return _build


def build_gemm_gelu() -> Builder:
    """COMET-style gemm_gelu compound op vs torch matmul + F.gelu."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_gelu(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.gelu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_gelu_batch1() -> Builder:
    """2D GEMM + GELU batch=1 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_gelu(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.gelu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_gelu_batch2() -> Builder:
    """2D GEMM + GELU batch=2 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_gelu(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.gelu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_gelu_3d() -> Builder:
    """3D GEMM + GELU [B,S,K] @ [K,N] vs matmul + F.gelu."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_gelu(a, b))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.gelu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_gelu_3d_batch1() -> Builder:
    """3D GEMM + GELU batch=1 [1,S,K] @ [K,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_gelu(a, b))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.gelu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_gelu_3d_batch2() -> Builder:
    """3D GEMM + GELU batch=2 [2,S,K] @ [K,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_gelu(a, b))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.gelu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_silu() -> Builder:
    """COMET-style gemm_silu compound op vs torch matmul + F.silu."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_silu(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.silu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_silu_batch1() -> Builder:
    """2D GEMM + SiLU batch=1 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_silu(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.silu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_silu_batch2() -> Builder:
    """2D GEMM + SiLU batch=2 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_silu(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.silu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_relu() -> Builder:
    """COMET-style gemm_relu compound op vs torch matmul + F.relu."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_relu(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.relu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_relu_batch1() -> Builder:
    """2D GEMM + ReLU batch=1 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_relu(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.relu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_relu_batch2() -> Builder:
    """2D GEMM + ReLU batch=2 inference [M,K] @ [K,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        g.mark_output(g.gemm_relu(a, b))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.relu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_relu_3d() -> Builder:
    """3D GEMM + ReLU [B,S,K] @ [K,N] vs matmul + F.relu."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_relu(a, b))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.relu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_relu_3d_batch1() -> Builder:
    """3D GEMM + ReLU batch=1 [1,S,K] @ [K,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_relu(a, b))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.relu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_relu_3d_batch2() -> Builder:
    """3D GEMM + ReLU batch=2 [2,S,K] @ [K,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_relu(a, b))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.relu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_silu_3d() -> Builder:
    """3D GEMM + SiLU [B,S,K] @ [K,N] vs matmul + F.silu."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_silu(a, b))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.silu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_silu_3d_batch1() -> Builder:
    """3D GEMM + SiLU batch=1 [1,S,K] @ [K,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_silu(a, b))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.silu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_silu_3d_batch2() -> Builder:
    """3D GEMM + SiLU batch=2 [2,S,K] @ [K,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.gemm_silu(a, b))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        c = torch.matmul(ta.float(), tb.float())
        ref = torch.nn.functional.silu(c).to(torch.float16)
        return g, [ta, tb], ref

    return _build


def build_gemm_bias() -> Builder:
    """GEMM + broadcast bias vs torch.matmul + bias row."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = (torch.matmul(ta.float(), tb.float()) + tbias.float()).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_batch1() -> Builder:
    """2D GEMM + bias batch=1 inference [M,K] @ [K,N] + [1,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = (torch.matmul(ta.float(), tb.float()) + tbias.float()).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_relu_batch1() -> Builder:
    """2D GEMM + bias + ReLU batch=1 inference [M,K] @ [K,N] + [1,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias_relu(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = torch.nn.functional.relu(
            torch.matmul(ta.float(), tb.float()) + tbias.float()
        ).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_relu() -> Builder:
    """GEMM + bias + ReLU vs F.relu(matmul + bias)."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias_relu(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = torch.nn.functional.relu(
            torch.matmul(ta.float(), tb.float()) + tbias.float()
        ).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_gelu() -> Builder:
    """GEMM + bias + GELU vs F.gelu(matmul + bias)."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias_gelu(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = torch.nn.functional.gelu(
            torch.matmul(ta.float(), tb.float()) + tbias.float()
        ).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_gelu_batch1() -> Builder:
    """2D GEMM + bias + GELU batch=1 inference [M,K] @ [K,N] + [1,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias_gelu(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = torch.nn.functional.gelu(
            torch.matmul(ta.float(), tb.float()) + tbias.float()
        ).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_silu() -> Builder:
    """GEMM + bias + SiLU vs F.silu(matmul + bias)."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias_silu(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = torch.nn.functional.silu(
            torch.matmul(ta.float(), tb.float()) + tbias.float()
        ).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_silu_batch1() -> Builder:
    """2D GEMM + bias + SiLU batch=1 inference [M,K] @ [K,N] + [1,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias_silu(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = torch.nn.functional.silu(
            torch.matmul(ta.float(), tb.float()) + tbias.float()
        ).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_batch2() -> Builder:
    """2D GEMM + bias batch=2 inference [M,K] @ [K,N] + [1,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = (torch.matmul(ta.float(), tb.float()) + tbias.float()).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_relu_batch2() -> Builder:
    """2D GEMM + bias + ReLU batch=2 inference [M,K] @ [K,N] + [1,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias_relu(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = torch.nn.functional.relu(
            torch.matmul(ta.float(), tb.float()) + tbias.float()
        ).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_gelu_batch2() -> Builder:
    """2D GEMM + bias + GELU batch=2 inference [M,K] @ [K,N] + [1,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias_gelu(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = torch.nn.functional.gelu(
            torch.matmul(ta.float(), tb.float()) + tbias.float()
        ).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_silu_batch2() -> Builder:
    """2D GEMM + bias + SiLU batch=2 inference [M,K] @ [K,N] + [1,N]."""

    def _build():
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(8, 32), dtype=yr.float16)
        b = g.new_input(dims=(32, 16), dtype=yr.float16)
        bias = g.new_input(dims=(1, 16), dtype=yr.float16)
        g.mark_output(g.gemm_bias_silu(a, b, bias))
        ta, tb = _f16((8, 32)), _f16((32, 16))
        tbias = _f16((1, 16))
        ref = torch.nn.functional.silu(
            torch.matmul(ta.float(), tb.float()) + tbias.float()
        ).to(torch.float16)
        return g, [ta, tb, tbias], ref

    return _build


def _gemm_bias_3d_ref(
    ta: torch.Tensor,
    tb: torch.Tensor,
    tbias: torch.Tensor,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    out = torch.matmul(ta.float(), tb.float()) + tbias.float()
    if activation == "gelu":
        out = torch.nn.functional.gelu(out)
    elif activation == "relu":
        out = torch.nn.functional.relu(out)
    elif activation == "silu":
        out = torch.nn.functional.silu(out)
    return out.to(torch.float16)


def build_gemm_bias_3d() -> Builder:
    """3D GEMM + broadcast bias [B,S,K] @ [K,N] + [1,1,N] vs PyTorch."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias(a, b, bias))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_batch1() -> Builder:
    """3D GEMM + bias batch=1 [1,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias(a, b, bias))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_batch2() -> Builder:
    """3D GEMM + bias batch=2 [2,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias(a, b, bias))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias)
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_relu() -> Builder:
    """3D GEMM + bias + ReLU [B,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias_relu(a, b, bias))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias, activation="relu")
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_relu_batch1() -> Builder:
    """3D GEMM + bias + ReLU batch=1 [1,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias_relu(a, b, bias))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias, activation="relu")
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_relu_batch2() -> Builder:
    """3D GEMM + bias + ReLU batch=2 [2,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias_relu(a, b, bias))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias, activation="relu")
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_gelu_batch1() -> Builder:
    """3D GEMM + bias + GELU batch=1 [1,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias_gelu(a, b, bias))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias, activation="gelu")
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_gelu_batch2() -> Builder:
    """3D GEMM + bias + GELU batch=2 [2,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias_gelu(a, b, bias))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias, activation="gelu")
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_silu_batch1() -> Builder:
    """3D GEMM + bias + SiLU batch=1 [1,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias_silu(a, b, bias))
        ta, tb = _f16((1, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias, activation="silu")
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_silu_batch2() -> Builder:
    """3D GEMM + bias + SiLU batch=2 [2,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias_silu(a, b, bias))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias, activation="silu")
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_gelu() -> Builder:
    """3D GEMM + bias + GELU [B,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias_gelu(a, b, bias))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias, activation="gelu")
        return g, [ta, tb, tbias], ref

    return _build


def build_gemm_bias_3d_silu() -> Builder:
    """3D GEMM + bias + SiLU [B,S,K] @ [K,N] + [1,1,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        a = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        b = g.new_input(dims=(k, n), dtype=yr.float16)
        bias = g.new_input(dims=(1, 1, n), dtype=yr.float16)
        g.mark_output(g.gemm_bias_silu(a, b, bias))
        ta, tb = _f16((batch, seq, k)), _f16((k, n))
        tbias = _f16((1, 1, n))
        ref = _gemm_bias_3d_ref(ta, tb, tbias, activation="silu")
        return g, [ta, tb, tbias], ref

    return _build


def build_gated_mlp() -> Builder:
    """Gated MLP (SiLU gate * up -> down) on 2D [S,D] vs PyTorch reference."""

    def _build():
        seq, dim, d_ff = 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="silu")
        )
        tx = _f16((seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.silu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_gelu() -> Builder:
    """Gated MLP with GELU gate activation on 2D [S,D] vs PyTorch reference."""

    def _build():
        seq, dim, d_ff = 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="gelu")
        )
        tx = _f16((seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.gelu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_batch1() -> Builder:
    """Gated MLP batch=1 inference [S,D] SiLU gate (2D single-sequence contract)."""

    def _build():
        seq, dim, d_ff = 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="silu")
        )
        tx = _f16((seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.silu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_batch2() -> Builder:
    """Gated MLP batch=2 inference [S,D] SiLU gate (2D dual-sequence contract)."""

    def _build():
        seq, dim, d_ff = 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="silu")
        )
        tx = _f16((seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.silu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_gelu_batch1() -> Builder:
    """Gated MLP batch=1 inference [S,D] GELU gate (2D single-sequence contract)."""

    def _build():
        seq, dim, d_ff = 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="gelu")
        )
        tx = _f16((seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.gelu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_gelu_batch2() -> Builder:
    """Gated MLP batch=2 inference [S,D] GELU gate (2D dual-sequence contract)."""

    def _build():
        seq, dim, d_ff = 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="gelu")
        )
        tx = _f16((seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.gelu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def _gated_mlp_batched_ref(
    tx: torch.Tensor,
    twg: torch.Tensor,
    twu: torch.Tensor,
    twd: torch.Tensor,
    *,
    activation: str,
) -> torch.Tensor:
    """PyTorch reference for 3D gated MLP with [1,D,*] weights."""
    batch = tx.shape[0]
    outs = []
    act_fn = torch.nn.functional.silu if activation == "silu" else torch.nn.functional.gelu
    for b in range(batch):
        xb = tx[b : b + 1].float()
        gate = act_fn(torch.matmul(xb, twg.float()))
        up = torch.matmul(xb, twu.float())
        inter = gate * up
        outs.append(torch.matmul(inter, twd.float()))
    return torch.cat(outs, dim=0).to(torch.float16)


def build_gated_mlp_batched() -> Builder:
    """3D gated MLP [B,S,D] with shared [1,D,D_ff] weights (SiLU gate)."""

    def _build():
        batch, seq, dim, d_ff = 2, 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(1, d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp_batched(x, w_gate, w_up, w_down, activation="silu")
        )
        tx = _f16((batch, seq, dim))
        twg = _f16((1, dim, d_ff))
        twu = _f16((1, dim, d_ff))
        twd = _f16((1, d_ff, dim))
        ref = _gated_mlp_batched_ref(tx, twg, twu, twd, activation="silu")
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_batched_gelu() -> Builder:
    """3D gated MLP [B,S,D] with shared [1,D,D_ff] weights (GELU gate)."""

    def _build():
        batch, seq, dim, d_ff = 2, 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(1, d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp_batched(x, w_gate, w_up, w_down, activation="gelu")
        )
        tx = _f16((batch, seq, dim))
        twg = _f16((1, dim, d_ff))
        twu = _f16((1, dim, d_ff))
        twd = _f16((1, d_ff, dim))
        ref = _gated_mlp_batched_ref(tx, twg, twu, twd, activation="gelu")
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_3d() -> Builder:
    """3D gated MLP [B,S,D] with shared 2D weights (KN 3D×2D matmul broadcast)."""

    def _build():
        batch, seq, dim, d_ff = 2, 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="silu")
        )
        tx = _f16((batch, seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.silu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_3d_gelu() -> Builder:
    """3D gated MLP [B,S,D] with GELU gate and shared 2D weights."""

    def _build():
        batch, seq, dim, d_ff = 2, 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="gelu")
        )
        tx = _f16((batch, seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.gelu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_3d_batch1() -> Builder:
    """3D gated MLP batch=1 [1,S,D] with shared 2D weights (SiLU gate)."""

    def _build():
        seq, dim, d_ff = 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="silu")
        )
        tx = _f16((1, seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.silu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_3d_batch2() -> Builder:
    """3D gated MLP batch=2 [2,S,D] with shared 2D weights (SiLU gate)."""

    def _build():
        batch, seq, dim, d_ff = 2, 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="silu")
        )
        tx = _f16((batch, seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.silu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_3d_gelu_batch1() -> Builder:
    """3D gated MLP batch=1 [1,S,D] with GELU gate and shared 2D weights."""

    def _build():
        seq, dim, d_ff = 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="gelu")
        )
        tx = _f16((1, seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.gelu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_3d_gelu_batch2() -> Builder:
    """3D gated MLP batch=2 [2,S,D] with GELU gate and shared 2D weights."""

    def _build():
        batch, seq, dim, d_ff = 2, 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp(x, w_gate, w_up, w_down, activation="gelu")
        )
        tx = _f16((batch, seq, dim))
        twg = _f16((dim, d_ff))
        twu = _f16((dim, d_ff))
        twd = _f16((d_ff, dim))
        gate = torch.nn.functional.gelu(torch.matmul(tx.float(), twg.float()))
        up = torch.matmul(tx.float(), twu.float())
        inter = gate * up
        ref = torch.matmul(inter, twd.float()).to(torch.float16)
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_batched_batch1() -> Builder:
    """3D gated MLP batch=1 [1,S,D] with shared [1,D,D_ff] weights (SiLU gate)."""

    def _build():
        seq, dim, d_ff = 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(1, d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp_batched(x, w_gate, w_up, w_down, activation="silu")
        )
        tx = _f16((1, seq, dim))
        twg = _f16((1, dim, d_ff))
        twu = _f16((1, dim, d_ff))
        twd = _f16((1, d_ff, dim))
        ref = _gated_mlp_batched_ref(tx, twg, twu, twd, activation="silu")
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_batched_batch2() -> Builder:
    """3D gated MLP batch=2 [2,S,D] with shared [1,D,D_ff] weights (SiLU gate)."""

    def _build():
        batch, seq, dim, d_ff = 2, 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(1, d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp_batched(x, w_gate, w_up, w_down, activation="silu")
        )
        tx = _f16((batch, seq, dim))
        twg = _f16((1, dim, d_ff))
        twu = _f16((1, dim, d_ff))
        twd = _f16((1, d_ff, dim))
        ref = _gated_mlp_batched_ref(tx, twg, twu, twd, activation="silu")
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_batched_gelu_batch1() -> Builder:
    """3D gated MLP batch=1 [1,S,D] with shared [1,D,D_ff] weights (GELU gate)."""

    def _build():
        seq, dim, d_ff = 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(1, d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp_batched(x, w_gate, w_up, w_down, activation="gelu")
        )
        tx = _f16((1, seq, dim))
        twg = _f16((1, dim, d_ff))
        twu = _f16((1, dim, d_ff))
        twd = _f16((1, d_ff, dim))
        ref = _gated_mlp_batched_ref(tx, twg, twu, twd, activation="gelu")
        return g, [tx, twg, twu, twd], ref

    return _build


def build_gated_mlp_batched_gelu_batch2() -> Builder:
    """3D gated MLP batch=2 [2,S,D] with shared [1,D,D_ff] weights (GELU gate)."""

    def _build():
        batch, seq, dim, d_ff = 2, 4, 8, 16
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        w_gate = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_up = g.new_input(dims=(1, dim, d_ff), dtype=yr.float16)
        w_down = g.new_input(dims=(1, d_ff, dim), dtype=yr.float16)
        g.mark_output(
            g.gated_mlp_batched(x, w_gate, w_up, w_down, activation="gelu")
        )
        tx = _f16((batch, seq, dim))
        twg = _f16((1, dim, d_ff))
        twu = _f16((1, dim, d_ff))
        twd = _f16((1, d_ff, dim))
        ref = _gated_mlp_batched_ref(tx, twg, twu, twd, activation="gelu")
        return g, [tx, twg, twu, twd], ref

    return _build


def build_rms_norm_linear_3d() -> Builder:
    """3D RMSNorm + linear [B,S,D] @ [D,N] with shared 2D weight."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear(x, w, normalized_shape=(k,)))
        tx, tw = _f16((batch, seq, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.matmul(tx.float() * scale, tw.float()).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_3d_batch1() -> Builder:
    """3D RMSNorm + linear batch=1 [1,S,D] @ [D,N] with shared 2D weight."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear(x, w, normalized_shape=(k,)))
        tx, tw = _f16((1, seq, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.matmul(tx.float() * scale, tw.float()).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_3d_batch2() -> Builder:
    """3D RMSNorm + linear batch=2 [2,S,D] @ [D,N] with shared 2D weight."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear(x, w, normalized_shape=(k,)))
        tx, tw = _f16((batch, seq, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.matmul(tx.float() * scale, tw.float()).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_3d_gelu_batch1() -> Builder:
    """3D RMSNorm + linear + GELU batch=1 [1,S,D] @ [D,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_gelu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((1, seq, k)), _f16((k, n))
        ref = _rms_norm_linear_3d_ref(tx, tw, activation="gelu")
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_3d_gelu_batch2() -> Builder:
    """3D RMSNorm + linear + GELU batch=2 [2,S,D] @ [D,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_gelu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((batch, seq, k)), _f16((k, n))
        ref = _rms_norm_linear_3d_ref(tx, tw, activation="gelu")
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_3d_relu_batch1() -> Builder:
    """3D RMSNorm + linear + ReLU batch=1 [1,S,D] @ [D,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_relu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((1, seq, k)), _f16((k, n))
        ref = _rms_norm_linear_3d_ref(tx, tw, activation="relu")
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_3d_relu_batch2() -> Builder:
    """3D RMSNorm + linear + ReLU batch=2 [2,S,D] @ [D,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_relu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((batch, seq, k)), _f16((k, n))
        ref = _rms_norm_linear_3d_ref(tx, tw, activation="relu")
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_3d_silu_batch1() -> Builder:
    """3D RMSNorm + linear + SiLU batch=1 [1,S,D] @ [D,N]."""

    def _build():
        seq, k, n = 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(1, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_silu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((1, seq, k)), _f16((k, n))
        ref = _rms_norm_linear_3d_ref(tx, tw, activation="silu")
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_3d_silu_batch2() -> Builder:
    """3D RMSNorm + linear + SiLU batch=2 [2,S,D] @ [D,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_silu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((batch, seq, k)), _f16((k, n))
        ref = _rms_norm_linear_3d_ref(tx, tw, activation="silu")
        return g, [tx, tw], ref

    return _build


def _rms_norm_linear_3d_ref(
    tx: torch.Tensor, tw: torch.Tensor, *, activation: str | None = None
) -> torch.Tensor:
    scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    out = torch.matmul(tx.float() * scale, tw.float())
    if activation == "gelu":
        out = torch.nn.functional.gelu(out)
    elif activation == "relu":
        out = torch.nn.functional.relu(out)
    elif activation == "silu":
        out = torch.nn.functional.silu(out)
    return out.to(torch.float16)


def build_rms_norm_linear_3d_gelu() -> Builder:
    """3D RMSNorm + linear + GELU [B,S,D] @ [D,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_gelu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((batch, seq, k)), _f16((k, n))
        ref = _rms_norm_linear_3d_ref(tx, tw, activation="gelu")
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_3d_relu() -> Builder:
    """3D RMSNorm + linear + ReLU [B,S,D] @ [D,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_relu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((batch, seq, k)), _f16((k, n))
        ref = _rms_norm_linear_3d_ref(tx, tw, activation="relu")
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_3d_silu() -> Builder:
    """3D RMSNorm + linear + SiLU [B,S,D] @ [D,N]."""

    def _build():
        batch, seq, k, n = 2, 4, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(batch, seq, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_silu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((batch, seq, k)), _f16((k, n))
        ref = _rms_norm_linear_3d_ref(tx, tw, activation="silu")
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear() -> Builder:
    """RMSNorm + linear vs rms reference matmul (QKV-style projection)."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.matmul(tx.float() * scale, tw.float()).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_batch1() -> Builder:
    """2D RMSNorm + linear batch=1 inference [M,K] @ [K,N]."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.matmul(tx.float() * scale, tw.float()).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_batch2() -> Builder:
    """2D RMSNorm + linear batch=2 inference [M,K] @ [K,N]."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.matmul(tx.float() * scale, tw.float()).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_gelu() -> Builder:
    """RMSNorm + linear + GELU vs F.gelu(rms reference matmul)."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_gelu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.nn.functional.gelu(
            torch.matmul(tx.float() * scale, tw.float())
        ).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_gelu_batch1() -> Builder:
    """2D RMSNorm + linear + GELU batch=1 inference [M,K] @ [K,N]."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_gelu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.nn.functional.gelu(
            torch.matmul(tx.float() * scale, tw.float())
        ).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_gelu_batch2() -> Builder:
    """2D RMSNorm + linear + GELU batch=2 inference [M,K] @ [K,N]."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_gelu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.nn.functional.gelu(
            torch.matmul(tx.float() * scale, tw.float())
        ).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_relu() -> Builder:
    """RMSNorm + linear + ReLU vs F.relu(rms reference matmul)."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_relu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.nn.functional.relu(
            torch.matmul(tx.float() * scale, tw.float())
        ).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_relu_batch1() -> Builder:
    """2D RMSNorm + linear + ReLU batch=1 inference [M,K] @ [K,N]."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_relu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.nn.functional.relu(
            torch.matmul(tx.float() * scale, tw.float())
        ).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_relu_batch2() -> Builder:
    """2D RMSNorm + linear + ReLU batch=2 inference [M,K] @ [K,N]."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_relu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.nn.functional.relu(
            torch.matmul(tx.float() * scale, tw.float())
        ).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_silu() -> Builder:
    """RMSNorm + linear + SiLU vs F.silu(rms reference matmul)."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_silu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.nn.functional.silu(
            torch.matmul(tx.float() * scale, tw.float())
        ).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_silu_batch1() -> Builder:
    """2D RMSNorm + linear + SiLU batch=1 inference [M,K] @ [K,N]."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_silu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.nn.functional.silu(
            torch.matmul(tx.float() * scale, tw.float())
        ).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_rms_norm_linear_silu_batch2() -> Builder:
    """2D RMSNorm + linear + SiLU batch=2 inference [M,K] @ [K,N]."""

    def _build():
        m, k, n = 8, 16, 32
        g = yr.new_kernel_graph()
        x = g.new_input(dims=(m, k), dtype=yr.float16)
        w = g.new_input(dims=(k, n), dtype=yr.float16)
        g.mark_output(g.rms_norm_linear_silu(x, w, normalized_shape=(k,)))
        tx, tw = _f16((m, k)), _f16((k, n))
        scale = torch.rsqrt(tx.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        ref = torch.nn.functional.silu(
            torch.matmul(tx.float() * scale, tw.float())
        ).to(torch.float16)
        return g, [tx, tw], ref

    return _build


def build_self_attention() -> Builder:
    """COMET self_attention: softmax(Q @ K) @ V with stable TB softmax (K transposed)."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention(q, k, v))
        tq, tk, tv = _f16((seq, dim)), _f16((dim, seq)), _f16((seq, dim))
        scores = torch.matmul(tq.float(), tk.float())
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_batch1() -> Builder:
    """2D self_attention batch=1 inference [S,D] / [D,S] / [S,D] (unscaled)."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention(q, k, v))
        tq, tk, tv = _f16((seq, dim)), _f16((dim, seq)), _f16((seq, dim))
        scores = torch.matmul(tq.float(), tk.float())
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_batch2() -> Builder:
    """2D self_attention batch=2 inference [S,D] / [D,S] / [S,D] (unscaled)."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention(q, k, v))
        tq, tk, tv = _f16((seq, dim)), _f16((dim, seq)), _f16((seq, dim))
        scores = torch.matmul(tq.float(), tk.float())
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_scaled() -> Builder:
    """Scaled self_attention: softmax(Q @ K / sqrt(d)) @ V (standard dot-product attention)."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention(q, k, v, head_dim=dim))
        tq, tk, tv = _f16((seq, dim)), _f16((dim, seq)), _f16((seq, dim))
        scale = dim ** -0.5
        scores = torch.matmul(tq.float(), tk.float()) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_scaled_batch1() -> Builder:
    """Scaled 2D self_attention batch=1 inference [S,D] / [D,S] / [S,D]."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention(q, k, v, head_dim=dim))
        tq, tk, tv = _f16((seq, dim)), _f16((dim, seq)), _f16((seq, dim))
        scale = dim ** -0.5
        scores = torch.matmul(tq.float(), tk.float()) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_scaled_batch2() -> Builder:
    """Scaled 2D self_attention batch=2 inference [S,D] / [D,S] / [S,D]."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention(q, k, v, head_dim=dim))
        tq, tk, tv = _f16((seq, dim)), _f16((dim, seq)), _f16((seq, dim))
        scale = dim ** -0.5
        scores = torch.matmul(tq.float(), tk.float()) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_online() -> Builder:
    """Online rescale self_attention (tile=4 on seq=8) vs scaled stable reference."""

    def _build():
        seq, dim, tile = 8, 32, 4
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(
            g.self_attention_online(q, k, v, head_dim=dim, tile=tile)
        )
        tq, tk, tv = _f16((seq, dim)), _f16((dim, seq)), _f16((seq, dim))
        scale = dim ** -0.5
        scores = torch.matmul(tq.float(), tk.float()) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_online_batch1() -> Builder:
    """Online rescale 2D self_attention batch=1 inference [S,D] (tile=4 on seq=8)."""

    def _build():
        seq, dim, tile = 8, 32, 4
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(
            g.self_attention_online(q, k, v, head_dim=dim, tile=tile)
        )
        tq, tk, tv = _f16((seq, dim)), _f16((dim, seq)), _f16((seq, dim))
        scale = dim ** -0.5
        scores = torch.matmul(tq.float(), tk.float()) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_online_batch2() -> Builder:
    """Online rescale 2D self_attention batch=2 inference [S,D] (tile=4 on seq=8)."""

    def _build():
        seq, dim, tile = 8, 32, 4
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(
            g.self_attention_online(q, k, v, head_dim=dim, tile=tile)
        )
        tq, tk, tv = _f16((seq, dim)), _f16((dim, seq)), _f16((seq, dim))
        scale = dim ** -0.5
        scores = torch.matmul(tq.float(), tk.float()) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_multi_head() -> Builder:
    """Multi-head self_attention on 3D [H,S,D] / [H,D,S] with scaled stable softmax."""

    def _build():
        heads, seq, dim = 4, 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(heads, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(heads, dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(heads, seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_multi_head(q, k, v, head_dim=dim))
        tq = _f16((heads, seq, dim))
        tk = _f16((heads, dim, seq))
        tv = _f16((heads, seq, dim))
        scale = dim ** -0.5
        outs = []
        for h in range(heads):
            scores = torch.matmul(tq[h].float(), tk[h].float()) * scale
            attn = torch.nn.functional.softmax(scores, dim=-1)
            outs.append(torch.matmul(attn, tv[h].float()))
        ref = torch.stack(outs, dim=0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_multi_head_batch1() -> Builder:
    """Multi-head self_attention batch=1 (H=1) [1,S,D] / [1,D,S] / [1,S,D]."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(1, dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_multi_head(q, k, v, head_dim=dim))
        tq = _f16((1, seq, dim))
        tk = _f16((1, dim, seq))
        tv = _f16((1, seq, dim))
        scale = dim ** -0.5
        scores = torch.matmul(tq[0].float(), tk[0].float()) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv[0].float()).unsqueeze(0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_multi_head_batch2() -> Builder:
    """Multi-head self_attention batch=2 (H=2) [2,S,D] / [2,D,S] / [2,S,D]."""

    def _build():
        heads, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(heads, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(heads, dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(heads, seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_multi_head(q, k, v, head_dim=dim))
        tq = _f16((heads, seq, dim))
        tk = _f16((heads, dim, seq))
        tv = _f16((heads, seq, dim))
        scale = dim ** -0.5
        outs = []
        for h in range(heads):
            scores = torch.matmul(tq[h].float(), tk[h].float()) * scale
            attn = torch.nn.functional.softmax(scores, dim=-1)
            outs.append(torch.matmul(attn, tv[h].float()))
        ref = torch.stack(outs, dim=0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_batched() -> Builder:
    """Batched scaled self_attention on 3D [B,S,D] / [B,D,S]."""

    def _build():
        batch, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(batch, dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_batched(q, k, v, head_dim=dim))
        tq = _f16((batch, seq, dim))
        tk = _f16((batch, dim, seq))
        tv = _f16((batch, seq, dim))
        scale = dim ** -0.5
        outs = []
        for b in range(batch):
            scores = torch.matmul(tq[b].float(), tk[b].float()) * scale
            attn = torch.nn.functional.softmax(scores, dim=-1)
            outs.append(torch.matmul(attn, tv[b].float()))
        ref = torch.stack(outs, dim=0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_batched_batch1() -> Builder:
    """Batched scaled self_attention batch=1 [1,S,D] / [1,D,S] / [1,S,D]."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(1, dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_batched(q, k, v, head_dim=dim))
        tq = _f16((1, seq, dim))
        tk = _f16((1, dim, seq))
        tv = _f16((1, seq, dim))
        scale = dim ** -0.5
        scores = torch.matmul(tq[0].float(), tk[0].float()) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv[0].float()).unsqueeze(0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_batched_batch2() -> Builder:
    """Batched scaled self_attention batch=2 [2,S,D] / [2,D,S] / [2,S,D]."""

    def _build():
        batch, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(batch, dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_batched(q, k, v, head_dim=dim))
        tq = _f16((batch, seq, dim))
        tk = _f16((batch, dim, seq))
        tv = _f16((batch, seq, dim))
        scale = dim ** -0.5
        outs = []
        for b in range(batch):
            scores = torch.matmul(tq[b].float(), tk[b].float()) * scale
            attn = torch.nn.functional.softmax(scores, dim=-1)
            outs.append(torch.matmul(attn, tv[b].float()))
        ref = torch.stack(outs, dim=0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_3d() -> Builder:
    """Batched self_attention [B,S,D] with shared 2D K [D,S] and V [S,D]."""

    def _build():
        batch, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_3d(q, k, v))
        tq = _f16((batch, seq, dim))
        tk = _f16((dim, seq))
        tv = _f16((seq, dim))
        outs = []
        for b in range(batch):
            scores = torch.matmul(tq[b].float(), tk.float())
            attn = torch.nn.functional.softmax(scores, dim=-1)
            outs.append(torch.matmul(attn, tv.float()))
        ref = torch.stack(outs, dim=0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_3d_batch1() -> Builder:
    """self_attention_3d batch=1 [1,S,D] with shared 2D K/V."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_3d(q, k, v))
        tq = _f16((1, seq, dim))
        tk = _f16((dim, seq))
        tv = _f16((seq, dim))
        scores = torch.matmul(tq[0].float(), tk.float())
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).unsqueeze(0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_3d_batch2() -> Builder:
    """self_attention_3d batch=2 [2,S,D] with shared 2D K/V."""

    def _build():
        batch, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_3d(q, k, v))
        tq = _f16((batch, seq, dim))
        tk = _f16((dim, seq))
        tv = _f16((seq, dim))
        outs = []
        for b in range(batch):
            scores = torch.matmul(tq[b].float(), tk.float())
            attn = torch.nn.functional.softmax(scores, dim=-1)
            outs.append(torch.matmul(attn, tv.float()))
        ref = torch.stack(outs, dim=0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_scaled_3d() -> Builder:
    """Scaled self_attention_3d: softmax(Q @ K / sqrt(d)) @ V with shared K/V."""

    def _build():
        batch, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_3d(q, k, v, head_dim=dim))
        tq = _f16((batch, seq, dim))
        tk = _f16((dim, seq))
        tv = _f16((seq, dim))
        scale = dim ** -0.5
        outs = []
        for b in range(batch):
            scores = torch.matmul(tq[b].float(), tk.float()) * scale
            attn = torch.nn.functional.softmax(scores, dim=-1)
            outs.append(torch.matmul(attn, tv.float()))
        ref = torch.stack(outs, dim=0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_scaled_3d_batch1() -> Builder:
    """Scaled self_attention_3d batch=1 [1,S,D] with shared 2D K/V."""

    def _build():
        seq, dim = 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(1, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_3d(q, k, v, head_dim=dim))
        tq = _f16((1, seq, dim))
        tk = _f16((dim, seq))
        tv = _f16((seq, dim))
        scale = dim ** -0.5
        scores = torch.matmul(tq[0].float(), tk.float()) * scale
        attn = torch.nn.functional.softmax(scores, dim=-1)
        ref = torch.matmul(attn, tv.float()).unsqueeze(0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


def build_self_attention_scaled_3d_batch2() -> Builder:
    """Scaled self_attention_3d batch=2 [2,S,D] with shared 2D K/V."""

    def _build():
        batch, seq, dim = 2, 8, 32
        g = yr.new_kernel_graph()
        q = g.new_input(dims=(batch, seq, dim), dtype=yr.float16)
        k = g.new_input(dims=(dim, seq), dtype=yr.float16)
        v = g.new_input(dims=(seq, dim), dtype=yr.float16)
        g.mark_output(g.self_attention_3d(q, k, v, head_dim=dim))
        tq = _f16((batch, seq, dim))
        tk = _f16((dim, seq))
        tv = _f16((seq, dim))
        scale = dim ** -0.5
        outs = []
        for b in range(batch):
            scores = torch.matmul(tq[b].float(), tk.float()) * scale
            attn = torch.nn.functional.softmax(scores, dim=-1)
            outs.append(torch.matmul(attn, tv.float()))
        ref = torch.stack(outs, dim=0).to(torch.float16)
        return g, [tq, tk, tv], ref

    return _build


CUSTOMIZED_OP_BUILDERS = {
    "customized_tb_matmul": build_customized_tb_matmul(),
    "customized_tb_exp": build_customized_tb_exp(),
    "customized_tb_matmul_add_bias": build_customized_tb_matmul_add_bias(),
    "kn_softmax": build_kn_softmax(),
    "kn_softmax_batch1": build_kn_softmax_batch1(),
    "kn_softmax_batch2": build_kn_softmax_batch2(),
    "kn_softmax_3d": build_kn_softmax_3d(),
    "kn_softmax_3d_batch1": build_kn_softmax_3d_batch1(),
    "kn_softmax_3d_batch2": build_kn_softmax_3d_batch2(),
    "kn_rms_norm_3d": build_kn_rms_norm_3d(),
    "kn_rms_norm_3d_batch1": build_kn_rms_norm_3d_batch1(),
    "kn_rms_norm_3d_batch2": build_kn_rms_norm_3d_batch2(),
    "kn_rms_norm_batch1": build_kn_rms_norm_batch1(),
    "kn_rms_norm_batch2": build_kn_rms_norm_batch2(),
    "kn_layer_norm": build_kn_layer_norm(),
    "kn_layer_norm_batch1": build_kn_layer_norm_batch1(),
    "kn_layer_norm_batch2": build_kn_layer_norm_batch2(),
    "kn_layer_norm_3d": build_kn_layer_norm_3d(),
    "kn_layer_norm_3d_batch1": build_kn_layer_norm_3d_batch1(),
    "kn_layer_norm_3d_batch2": build_kn_layer_norm_3d_batch2(),
    "gemm_softmax": build_gemm_softmax(),
    "gemm_softmax_batch1": build_gemm_softmax_batch1(),
    "gemm_softmax_batch2": build_gemm_softmax_batch2(),
    "gemm_softmax_scaled": build_gemm_softmax_scaled(),
    "gemm_softmax_scaled_batch1": build_gemm_softmax_scaled_batch1(),
    "gemm_softmax_scaled_batch2": build_gemm_softmax_scaled_batch2(),
    "gemm_softmax_scaled_batched": build_gemm_softmax_scaled_batched(),
    "gemm_softmax_scaled_batched_batch1": build_gemm_softmax_scaled_batched_batch1(),
    "gemm_softmax_scaled_batched_batch2": build_gemm_softmax_scaled_batched_batch2(),
    "gemm_softmax_3d": build_gemm_softmax_3d(),
    "gemm_softmax_3d_batch1": build_gemm_softmax_3d_batch1(),
    "gemm_softmax_3d_batch2": build_gemm_softmax_3d_batch2(),
    "gemm_softmax_scaled_3d": build_gemm_softmax_scaled_3d(),
    "gemm_softmax_scaled_3d_batch1": build_gemm_softmax_scaled_3d_batch1(),
    "gemm_softmax_scaled_3d_batch2": build_gemm_softmax_scaled_3d_batch2(),
    "gemm_layernorm": build_gemm_layernorm(),
    "gemm_layernorm_batch1": build_gemm_layernorm_batch1(),
    "gemm_layernorm_batch2": build_gemm_layernorm_batch2(),
    "gemm_layernorm_gelu": build_gemm_layernorm_gelu(),
    "gemm_layernorm_gelu_batch1": build_gemm_layernorm_gelu_batch1(),
    "gemm_layernorm_gelu_batch2": build_gemm_layernorm_gelu_batch2(),
    "gemm_layernorm_relu": build_gemm_layernorm_relu(),
    "gemm_layernorm_relu_batch1": build_gemm_layernorm_relu_batch1(),
    "gemm_layernorm_relu_batch2": build_gemm_layernorm_relu_batch2(),
    "gemm_layernorm_silu": build_gemm_layernorm_silu(),
    "gemm_layernorm_silu_batch1": build_gemm_layernorm_silu_batch1(),
    "gemm_layernorm_silu_batch2": build_gemm_layernorm_silu_batch2(),
    "gemm_layernorm_3d": build_gemm_layernorm_3d(),
    "gemm_layernorm_3d_batch1": build_gemm_layernorm_3d_batch1(),
    "gemm_layernorm_3d_batch2": build_gemm_layernorm_3d_batch2(),
    "gemm_layernorm_3d_gelu": build_gemm_layernorm_3d_gelu(),
    "gemm_layernorm_3d_gelu_batch1": build_gemm_layernorm_3d_gelu_batch1(),
    "gemm_layernorm_3d_gelu_batch2": build_gemm_layernorm_3d_gelu_batch2(),
    "gemm_layernorm_3d_relu": build_gemm_layernorm_3d_relu(),
    "gemm_layernorm_3d_relu_batch1": build_gemm_layernorm_3d_relu_batch1(),
    "gemm_layernorm_3d_relu_batch2": build_gemm_layernorm_3d_relu_batch2(),
    "gemm_layernorm_3d_silu": build_gemm_layernorm_3d_silu(),
    "gemm_layernorm_3d_silu_batch1": build_gemm_layernorm_3d_silu_batch1(),
    "gemm_layernorm_3d_silu_batch2": build_gemm_layernorm_3d_silu_batch2(),
    "gemm_gelu": build_gemm_gelu(),
    "gemm_gelu_batch1": build_gemm_gelu_batch1(),
    "gemm_gelu_batch2": build_gemm_gelu_batch2(),
    "gemm_gelu_3d": build_gemm_gelu_3d(),
    "gemm_gelu_3d_batch1": build_gemm_gelu_3d_batch1(),
    "gemm_gelu_3d_batch2": build_gemm_gelu_3d_batch2(),
    "gemm_silu": build_gemm_silu(),
    "gemm_silu_batch1": build_gemm_silu_batch1(),
    "gemm_silu_batch2": build_gemm_silu_batch2(),
    "gemm_silu_3d": build_gemm_silu_3d(),
    "gemm_silu_3d_batch1": build_gemm_silu_3d_batch1(),
    "gemm_silu_3d_batch2": build_gemm_silu_3d_batch2(),
    "gemm_relu": build_gemm_relu(),
    "gemm_relu_batch1": build_gemm_relu_batch1(),
    "gemm_relu_batch2": build_gemm_relu_batch2(),
    "gemm_relu_3d": build_gemm_relu_3d(),
    "gemm_relu_3d_batch1": build_gemm_relu_3d_batch1(),
    "gemm_relu_3d_batch2": build_gemm_relu_3d_batch2(),
    "gemm_bias": build_gemm_bias(),
    "gemm_bias_batch1": build_gemm_bias_batch1(),
    "gemm_bias_batch2": build_gemm_bias_batch2(),
    "gemm_bias_relu_batch1": build_gemm_bias_relu_batch1(),
    "gemm_bias_relu_batch2": build_gemm_bias_relu_batch2(),
    "gemm_bias_relu": build_gemm_bias_relu(),
    "gemm_bias_gelu": build_gemm_bias_gelu(),
    "gemm_bias_gelu_batch1": build_gemm_bias_gelu_batch1(),
    "gemm_bias_gelu_batch2": build_gemm_bias_gelu_batch2(),
    "gemm_bias_silu": build_gemm_bias_silu(),
    "gemm_bias_silu_batch1": build_gemm_bias_silu_batch1(),
    "gemm_bias_silu_batch2": build_gemm_bias_silu_batch2(),
    "gemm_bias_3d": build_gemm_bias_3d(),
    "gemm_bias_3d_batch1": build_gemm_bias_3d_batch1(),
    "gemm_bias_3d_batch2": build_gemm_bias_3d_batch2(),
    "gemm_bias_3d_relu": build_gemm_bias_3d_relu(),
    "gemm_bias_3d_relu_batch1": build_gemm_bias_3d_relu_batch1(),
    "gemm_bias_3d_relu_batch2": build_gemm_bias_3d_relu_batch2(),
    "gemm_bias_3d_gelu_batch1": build_gemm_bias_3d_gelu_batch1(),
    "gemm_bias_3d_gelu_batch2": build_gemm_bias_3d_gelu_batch2(),
    "gemm_bias_3d_silu_batch1": build_gemm_bias_3d_silu_batch1(),
    "gemm_bias_3d_silu_batch2": build_gemm_bias_3d_silu_batch2(),
    "gemm_bias_3d_gelu": build_gemm_bias_3d_gelu(),
    "gemm_bias_3d_silu": build_gemm_bias_3d_silu(),
    "gated_mlp": build_gated_mlp(),
    "gated_mlp_gelu": build_gated_mlp_gelu(),
    "gated_mlp_batch1": build_gated_mlp_batch1(),
    "gated_mlp_batch2": build_gated_mlp_batch2(),
    "gated_mlp_gelu_batch1": build_gated_mlp_gelu_batch1(),
    "gated_mlp_gelu_batch2": build_gated_mlp_gelu_batch2(),
    "gated_mlp_batched": build_gated_mlp_batched(),
    "gated_mlp_batched_gelu": build_gated_mlp_batched_gelu(),
    "gated_mlp_3d": build_gated_mlp_3d(),
    "gated_mlp_3d_gelu": build_gated_mlp_3d_gelu(),
    "gated_mlp_3d_batch1": build_gated_mlp_3d_batch1(),
    "gated_mlp_3d_batch2": build_gated_mlp_3d_batch2(),
    "gated_mlp_3d_gelu_batch1": build_gated_mlp_3d_gelu_batch1(),
    "gated_mlp_3d_gelu_batch2": build_gated_mlp_3d_gelu_batch2(),
    "gated_mlp_batched_batch1": build_gated_mlp_batched_batch1(),
    "gated_mlp_batched_batch2": build_gated_mlp_batched_batch2(),
    "gated_mlp_batched_gelu_batch1": build_gated_mlp_batched_gelu_batch1(),
    "gated_mlp_batched_gelu_batch2": build_gated_mlp_batched_gelu_batch2(),
    "rms_norm_linear": build_rms_norm_linear(),
    "rms_norm_linear_batch1": build_rms_norm_linear_batch1(),
    "rms_norm_linear_batch2": build_rms_norm_linear_batch2(),
    "rms_norm_linear_3d": build_rms_norm_linear_3d(),
    "rms_norm_linear_3d_batch1": build_rms_norm_linear_3d_batch1(),
    "rms_norm_linear_3d_batch2": build_rms_norm_linear_3d_batch2(),
    "rms_norm_linear_3d_gelu_batch1": build_rms_norm_linear_3d_gelu_batch1(),
    "rms_norm_linear_3d_gelu_batch2": build_rms_norm_linear_3d_gelu_batch2(),
    "rms_norm_linear_3d_relu_batch1": build_rms_norm_linear_3d_relu_batch1(),
    "rms_norm_linear_3d_relu_batch2": build_rms_norm_linear_3d_relu_batch2(),
    "rms_norm_linear_3d_silu_batch1": build_rms_norm_linear_3d_silu_batch1(),
    "rms_norm_linear_3d_silu_batch2": build_rms_norm_linear_3d_silu_batch2(),
    "rms_norm_linear_3d_gelu": build_rms_norm_linear_3d_gelu(),
    "rms_norm_linear_3d_relu": build_rms_norm_linear_3d_relu(),
    "rms_norm_linear_3d_silu": build_rms_norm_linear_3d_silu(),
    "rms_norm_linear_gelu": build_rms_norm_linear_gelu(),
    "rms_norm_linear_gelu_batch1": build_rms_norm_linear_gelu_batch1(),
    "rms_norm_linear_gelu_batch2": build_rms_norm_linear_gelu_batch2(),
    "rms_norm_linear_relu": build_rms_norm_linear_relu(),
    "rms_norm_linear_relu_batch1": build_rms_norm_linear_relu_batch1(),
    "rms_norm_linear_relu_batch2": build_rms_norm_linear_relu_batch2(),
    "rms_norm_linear_silu": build_rms_norm_linear_silu(),
    "rms_norm_linear_silu_batch1": build_rms_norm_linear_silu_batch1(),
    "rms_norm_linear_silu_batch2": build_rms_norm_linear_silu_batch2(),
    "self_attention": build_self_attention(),
    "self_attention_batch1": build_self_attention_batch1(),
    "self_attention_batch2": build_self_attention_batch2(),
    "self_attention_scaled": build_self_attention_scaled(),
    "self_attention_scaled_batch1": build_self_attention_scaled_batch1(),
    "self_attention_scaled_batch2": build_self_attention_scaled_batch2(),
    "self_attention_online": build_self_attention_online(),
    "self_attention_online_batch1": build_self_attention_online_batch1(),
    "self_attention_online_batch2": build_self_attention_online_batch2(),
    "self_attention_multi_head": build_self_attention_multi_head(),
    "self_attention_multi_head_batch1": build_self_attention_multi_head_batch1(),
    "self_attention_multi_head_batch2": build_self_attention_multi_head_batch2(),
    "self_attention_batched": build_self_attention_batched(),
    "self_attention_batched_batch1": build_self_attention_batched_batch1(),
    "self_attention_batched_batch2": build_self_attention_batched_batch2(),
    "self_attention_3d": build_self_attention_3d(),
    "self_attention_3d_batch1": build_self_attention_3d_batch1(),
    "self_attention_3d_batch2": build_self_attention_3d_batch2(),
    "self_attention_scaled_3d": build_self_attention_scaled_3d(),
    "self_attention_scaled_3d_batch1": build_self_attention_scaled_3d_batch1(),
    "self_attention_scaled_3d_batch2": build_self_attention_scaled_3d_batch2(),
    "conv2d_bias": build_conv2d_bias(),
    "conv2d_bias_batch1": build_conv2d_bias_batch1(),
    "conv2d_bias_batch2": build_conv2d_bias_batch2(),
    "conv2d_bias_relu": build_conv2d_bias_relu(),
    "conv2d_bias_relu_batch1": build_conv2d_bias_relu_batch1(),
    "conv2d_bias_relu_batch2": build_conv2d_bias_relu_batch2(),
    "conv2d_bias_gelu": build_conv2d_bias_gelu(),
    "conv2d_bias_gelu_batch1": build_conv2d_bias_gelu_batch1(),
    "conv2d_bias_gelu_batch2": build_conv2d_bias_gelu_batch2(),
    "conv2d_bias_silu": build_conv2d_bias_silu(),
    "conv2d_bias_silu_batch1": build_conv2d_bias_silu_batch1(),
    "conv2d_bias_silu_batch2": build_conv2d_bias_silu_batch2(),
    "conv2d_groups": build_kn_conv2d_groups(),
    "conv2d_groups_batch1": build_conv2d_groups_batch1(),
    "conv2d_groups_batch2": build_kn_conv2d_groups_batch2(),
    "conv2d_bias_groups": build_conv2d_bias_groups(),
    "conv2d_bias_groups_batch1": build_conv2d_bias_groups_batch1(),
    "conv2d_bias_groups_batch2": build_conv2d_bias_groups_batch2(),
    "conv2d_depthwise_bias": build_conv2d_depthwise_bias(),
    "conv2d_depthwise_bias_batch1": build_conv2d_depthwise_bias_batch1(),
    "conv2d_depthwise_bias_batch2": build_conv2d_depthwise_bias_batch2(),
    "conv2d_depthwise_bias_relu": build_conv2d_depthwise_bias_relu(),
    "conv2d_depthwise_bias_relu_batch1": build_conv2d_depthwise_bias_relu_batch1(),
    "conv2d_depthwise_bias_relu_batch2": build_conv2d_depthwise_bias_relu_batch2(),
    "conv2d_depthwise_bias_gelu": build_conv2d_depthwise_bias_gelu(),
    "conv2d_depthwise_bias_gelu_batch1": build_conv2d_depthwise_bias_gelu_batch1(),
    "conv2d_depthwise_bias_gelu_batch2": build_conv2d_depthwise_bias_gelu_batch2(),
    "conv2d_depthwise_bias_silu": build_conv2d_depthwise_bias_silu(),
    "conv2d_depthwise_bias_silu_batch1": build_conv2d_depthwise_bias_silu_batch1(),
    "conv2d_depthwise_bias_silu_batch2": build_conv2d_depthwise_bias_silu_batch2(),
    "conv2d_separable": build_conv2d_separable(),
    "conv2d_separable_batch1": build_conv2d_separable_batch1(),
    "conv2d_separable_batch2": build_conv2d_separable_batch2(),
    "conv2d_separable_bias": build_conv2d_separable_bias(),
    "conv2d_separable_bias_batch1": build_conv2d_separable_bias_batch1(),
    "conv2d_separable_bias_batch2": build_conv2d_separable_bias_batch2(),
    "conv2d_separable_bias_relu": build_conv2d_separable_bias_relu(),
    "conv2d_separable_bias_relu_batch1": build_conv2d_separable_bias_relu_batch1(),
    "conv2d_separable_bias_relu_batch2": build_conv2d_separable_bias_relu_batch2(),
    "conv2d_separable_bias_gelu": build_conv2d_separable_bias_gelu(),
    "conv2d_separable_bias_gelu_batch1": build_conv2d_separable_bias_gelu_batch1(),
    "conv2d_separable_bias_gelu_batch2": build_conv2d_separable_bias_gelu_batch2(),
    "conv2d_separable_bias_silu": build_conv2d_separable_bias_silu(),
    "conv2d_separable_bias_silu_batch1": build_conv2d_separable_bias_silu_batch1(),
    "conv2d_separable_bias_silu_batch2": build_conv2d_separable_bias_silu_batch2(),
    "kn_matmul_batch1": build_kn_matmul_batch1(),
    "kn_matmul_batch2": build_kn_matmul_batch2(),
    "kn_conv2d_batch1": build_kn_conv2d_batch1(),
    "kn_conv2d_batch2": build_kn_conv2d_batch2(),
    "kn_conv2d_groups_batch1": build_kn_conv2d_groups_batch1(),
    "kn_conv2d_groups_batch2": build_kn_conv2d_groups_batch2(),
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
