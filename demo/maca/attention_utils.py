"""Shared helpers for MACA attention smoke (CUDA ``chameleon_maca.py`` aligned)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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
            "native attention_kernel.maca e2e bench remains backlog."
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


def maca_attention_superoptimize_smoke(
    scaffold: Optional[AttentionScaffold] = None,
    *,
    quick: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build chameleon-style attention graph and superoptimize on MACA (MetaX VM)."""
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

    mugraph = graph.superoptimize(**_maca_attention_search_kwargs(quick=quick))
    return {
        "superoptimized": mugraph is not None,
        "config": "attention",
        "backend": "maca",
        "num_inputs": len(mugraph.get_input_tensors()) if mugraph is not None else 0,
        "num_outputs": len(mugraph.get_output_tensors()) if mugraph is not None else 0,
    }


__all__ = [
    "AttentionScaffold",
    "inspect_maca_attention_scaffold",
    "maca_attention_superoptimize_smoke",
]
