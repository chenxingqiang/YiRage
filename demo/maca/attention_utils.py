"""Shared helpers for MACA attention smoke (CUDA ``chameleon_maca.py`` aligned)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AttentionScaffold:
    """Subset of ``benchmark/end-to-end/maca/chameleon_maca.py`` attention shapes."""

    n_local_heads: int = 32
    n_local_kv_heads: int = 32
    head_dim: int = 128
    num_tokens: int = 4
    num_kv_tokens: int = 4096
    batch_size: int = 8


def inspect_maca_attention_scaffold(
    scaffold: Optional[AttentionScaffold] = None,
) -> Dict[str, Any]:
    """Return inspect-only attention scaffold report (no GPU / no yirage.core)."""
    scaffold = scaffold or AttentionScaffold()
    kernel_path = _REPO_ROOT / "src" / "kernel" / "maca" / "attention_kernel.maca"
    return {
        "cuda_reference": "benchmark/end-to-end/maca/chameleon_maca.py get_chameleon_attention",
        "maca_demo": "demo/maca/attention_smoke.py",
        "maca_kernel": "src/kernel/maca/attention_kernel.maca",
        "kernel_file_exists": kernel_path.is_file(),
        "warp_size": 64,
        "n_local_heads": scaffold.n_local_heads,
        "n_local_kv_heads": scaffold.n_local_kv_heads,
        "head_dim": scaffold.head_dim,
        "num_tokens": scaffold.num_tokens,
        "num_kv_tokens": scaffold.num_kv_tokens,
        "batch_size": scaffold.batch_size,
        "search_config": "attention",
        "yirage_backend": os.environ.get("YIRAGE_BACKEND", "maca"),
        "compile_note": (
            "superoptimize(backend=maca, config=attention) smoke on MetaX VM; "
            "maca_attention_native_bench_quick compares YiRage mugraph vs mcPytorch reference."
        ),
    }


def inspect_maca_attention_bench_plan(
    scaffold: Optional[AttentionScaffold] = None,
) -> Dict[str, Any]:
    """Cloud-safe bench plan: superoptimize + maca_call vs PyTorch reference."""
    scaffold = scaffold or AttentionScaffold()
    base = inspect_maca_attention_scaffold(scaffold)
    return {
        **base,
        "plan_kind": "bench",
        "bench_entry": "maca_attention_native_bench_quick",
        "baseline": "pytorch_exp_softmax_attention (graph-aligned, no flashinfer)",
        "bench_steps": [
            "build_maca_attention_graph",
            "graph.superoptimize(backend=maca, config=attention)",
            "mugraph.maca_call vs pytorch reference",
            "CUDA event timing (warmup + quick iters)",
        ],
        "bench_plan_ready": base["kernel_file_exists"],
        "requires_metax_gpu": True,
        "native_kernel_note": (
            "attention_kernel.maca launch binding remains experimental backlog; "
            "bench uses superoptimized mugraph execution path."
        ),
    }


def _maca_attention_search_kwargs(*, quick: Optional[bool] = None) -> Dict[str, Any]:
    if quick is None:
        quick = os.environ.get("YIRAGE_MACA_SEARCH_QUICK", "1") == "1"
    from yirage.maca_config import maca_superoptimize_ray_kwargs, resolve_maca_search_config

    cfg = resolve_maca_search_config(quick=quick)
    return {
        "backend": "maca",
        "config": "attention",
        "verbose": False,
        **maca_superoptimize_ray_kwargs(),
        "griddims": cfg.get("grid_dims_to_explore"),
        "blockdims": cfg.get("block_dims_to_explore"),
        "fmaps": cfg.get("fmaps_to_explore"),
        "franges": cfg.get("franges_to_explore"),
    }


def build_maca_attention_graph(scaffold: Optional[AttentionScaffold] = None):
    """Build chameleon-style attention KN graph (exp-softmax, no scaling)."""
    import yirage as yr

    scaffold = scaffold or AttentionScaffold()
    graph = yr.new_kernel_graph()
    q = graph.new_input(
        dims=(scaffold.n_local_kv_heads, scaffold.num_tokens, scaffold.head_dim),
        dtype=yr.float16,
    )
    k = graph.new_input(
        dims=(scaffold.n_local_kv_heads, scaffold.head_dim, scaffold.num_kv_tokens),
        dtype=yr.float16,
    )
    v = graph.new_input(
        dims=(scaffold.n_local_kv_heads, scaffold.num_kv_tokens, scaffold.head_dim),
        dtype=yr.float16,
    )
    scores = graph.matmul(q, k)
    exp_scores = graph.exp(scores)
    denom = graph.reduction(exp_scores, 2)
    attn = graph.div(exp_scores, denom)
    out = graph.matmul(attn, v)
    graph.mark_output(out)
    return graph


def pytorch_attention_reference(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
) -> "torch.Tensor":
    """PyTorch reference matching ``build_maca_attention_graph`` semantics."""
    import torch

    scores = torch.matmul(q, k)
    exp_scores = torch.exp(scores)
    denom = exp_scores.sum(dim=-1, keepdim=True)
    attn = exp_scores / denom
    return torch.matmul(attn, v)


def _make_attention_inputs(
    scaffold: AttentionScaffold,
    device: "torch.device",
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    import torch

    q = torch.randn(
        scaffold.n_local_kv_heads,
        scaffold.num_tokens,
        scaffold.head_dim,
        device=device,
        dtype=torch.float16,
    )
    k = torch.randn(
        scaffold.n_local_kv_heads,
        scaffold.head_dim,
        scaffold.num_kv_tokens,
        device=device,
        dtype=torch.float16,
    )
    v = torch.randn(
        scaffold.n_local_kv_heads,
        scaffold.num_kv_tokens,
        scaffold.head_dim,
        device=device,
        dtype=torch.float16,
    )
    return q, k, v


def maca_attention_superoptimize_smoke(
    scaffold: Optional[AttentionScaffold] = None,
    *,
    quick: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build chameleon-style attention graph and superoptimize on MACA (MetaX VM)."""
    scaffold = scaffold or AttentionScaffold()
    graph = build_maca_attention_graph(scaffold)
    mugraph = graph.superoptimize(**_maca_attention_search_kwargs(quick=quick))
    return {
        "superoptimized": mugraph is not None,
        "config": "attention",
        "backend": "maca",
        "num_inputs": len(mugraph.get_input_tensors()) if mugraph is not None else 0,
        "num_outputs": len(mugraph.get_output_tensors()) if mugraph is not None else 0,
    }


def maca_attention_native_bench_quick(
    scaffold: Optional[AttentionScaffold] = None,
    *,
    quick: Optional[bool] = None,
    warmup: int = 2,
    iters: int = 8,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> Dict[str, Any]:
    """Quick YiRage mugraph vs mcPytorch reference bench on MetaX MACA."""
    import torch

    scaffold = scaffold or AttentionScaffold()
    graph = build_maca_attention_graph(scaffold)
    mugraph = graph.superoptimize(**_maca_attention_search_kwargs(quick=quick))
    if mugraph is None:
        return {
            "bench_ok": False,
            "superoptimized": False,
            "reason": "superoptimize returned None",
        }

    device = torch.device("cuda:0")
    q, k, v = _make_attention_inputs(scaffold, device)
    ref = pytorch_attention_reference(q, k, v)

    yirage_out = mugraph.maca_call(inputs=[q, k, v])
    if yirage_out is None:
        return {
            "bench_ok": False,
            "superoptimized": True,
            "reason": "maca_call returned None (compile failed)",
        }

    if isinstance(yirage_out, (list, tuple)):
        yirage_tensor = yirage_out[0]
    else:
        yirage_tensor = yirage_out

    aligned = torch.allclose(yirage_tensor, ref, rtol=rtol, atol=atol)
    max_abs_diff = (yirage_tensor - ref).abs().max().item()

    for _ in range(warmup):
        mugraph.maca_call(inputs=[q, k, v])
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(iters):
        mugraph.maca_call(inputs=[q, k, v])
    ender.record()
    torch.cuda.synchronize()
    yirage_ms = starter.elapsed_time(ender) / iters

    starter.record()
    for _ in range(iters):
        pytorch_attention_reference(q, k, v)
    ender.record()
    torch.cuda.synchronize()
    pytorch_ms = starter.elapsed_time(ender) / iters

    speedup = pytorch_ms / yirage_ms if yirage_ms > 0 else None
    return {
        "bench_ok": aligned,
        "superoptimized": True,
        "runtime_verified": aligned,
        "yirage_ms": yirage_ms,
        "pytorch_ms": pytorch_ms,
        "speedup_pytorch_over_yirage": speedup,
        "max_abs_diff": max_abs_diff,
        "warmup": warmup,
        "iters": iters,
        "config": "attention",
        "backend": "maca",
    }


__all__ = [
    "AttentionScaffold",
    "build_maca_attention_graph",
    "inspect_maca_attention_bench_plan",
    "inspect_maca_attention_scaffold",
    "maca_attention_native_bench_quick",
    "maca_attention_superoptimize_smoke",
    "pytorch_attention_reference",
]
