"""Shared MACA Qwen kernel superoptimize helpers (CUDA modeling_qwen2 aligned)."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from demo._maca_utils import maca_search_kwargs, maca_superoptimize_ray_kwargs


def _yr_dtype(name: str):
    import yirage as yr

    return yr.bfloat16 if name == "bfloat16" else yr.float16


def maca_superoptimize_search(*, quick: Optional[bool] = None) -> Dict[str, Any]:
    """Merge tractable MACA search grid + optional Ray opt-in for superoptimize."""
    if quick is None:
        quick = os.environ.get("YIRAGE_MACA_SEARCH_QUICK", "1") == "1"
    return {
        "backend": "maca",
        "config": "mlp",
        "verbose": False,
        **maca_superoptimize_ray_kwargs(),
        **maca_search_kwargs(quick=quick),
    }


def superoptimize_mlp_gate_up(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "bfloat16",
    quick: Optional[bool] = None,
):
    """RMSNorm + mul + matmul gate/up fused kernel (decode shape [1, H])."""
    import yirage as yr

    dtype = _yr_dtype(dtype_name)
    graph = yr.new_kernel_graph()
    x = graph.new_input(dims=(1, hidden_size), dtype=dtype)
    g = graph.new_input(dims=(1, hidden_size), dtype=dtype)
    w = graph.new_input(
        dims=(hidden_size, 2 * intermediate_size),
        strides=(1, hidden_size),
        dtype=dtype,
    )
    d = graph.rms_norm(x, normalized_shape=(hidden_size,))
    d = graph.mul(d, g)
    o = graph.matmul(d, w)
    graph.mark_output(o)
    return graph.superoptimize(**maca_superoptimize_search(quick=quick))


def superoptimize_mlp_down(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "bfloat16",
    quick: Optional[bool] = None,
):
    """SiLU(gate) * up + matmul down projection (decode shape)."""
    import yirage as yr

    dtype = _yr_dtype(dtype_name)
    graph = yr.new_kernel_graph()
    x = graph.new_input(dims=(1, intermediate_size), dtype=dtype)
    y = graph.new_input(dims=(1, intermediate_size), dtype=dtype)
    w = graph.new_input(
        dims=(intermediate_size, hidden_size),
        strides=(1, intermediate_size),
        dtype=dtype,
    )
    d = graph.mul(graph.silu(x), y)
    o = graph.matmul(d, w)
    graph.mark_output(o)
    return graph.superoptimize(**maca_superoptimize_search(quick=quick))


def superoptimize_attn_qkv(
    hidden_size: int,
    fused_outdim: int,
    *,
    dtype_name: str = "bfloat16",
    quick: Optional[bool] = None,
):
    """RMSNorm + mul + matmul QKV projection (decode shape)."""
    import yirage as yr

    dtype = _yr_dtype(dtype_name)
    graph = yr.new_kernel_graph()
    x = graph.new_input(dims=(1, hidden_size), dtype=dtype)
    g = graph.new_input(dims=(1, hidden_size), dtype=dtype)
    w = graph.new_input(
        dims=(hidden_size, fused_outdim),
        strides=(1, hidden_size),
        dtype=dtype,
    )
    d = graph.rms_norm(x, normalized_shape=(hidden_size,))
    d = graph.mul(d, g)
    o = graph.matmul(d, w)
    graph.mark_output(o)
    return graph.superoptimize(**maca_superoptimize_search(quick=quick))
