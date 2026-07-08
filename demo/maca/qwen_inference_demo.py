#!/usr/bin/env python3
"""
MACA full-chain Qwen inference smoke — aligned to ``demo/qwen2.5/demo.py`` (CUDA).

CUDA reference pipeline (``demo/qwen2.5/demo.py``):
  load Qwen → ``superoptimize_kernels()`` → chat prefill → CUDA Graph decode with YiRage MLP/attn kernels.

This MACA demo mirrors the **decode-critical path** on MetaX real GPU:
  device/SDK check → build Qwen-shaped fused kernels → ``superoptimize(backend="maca")``
  → synthetic prefill (PyTorch) → multi-step decode (YiRage kernels) → argmax token sanity.

Run on MetaX VM (mcPytorch):
  export MACA_PATH=/opt/maca YIRAGE_BACKEND=maca PYTHONPATH=.
  /opt/conda/bin/python3 demo/maca/qwen_inference_demo.py --quick
  /opt/conda/bin/python3 demo/maca/qwen_inference_demo.py --quick --decode-steps 8
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from demo._maca_utils import (  # noqa: E402
    apply_maca_demo_env,
    maca_search_kwargs,
    maca_superoptimize_ray_kwargs,
    sync_device,
)


# Qwen3-8B–style shapes (matches ``demo/qwen2.5`` + ``Qwen/Qwen3-8B`` decode tensors)
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 12288
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS


@dataclass
class QwenLayerKernels:
    mlp_gate_up: Any
    mlp_down: Any
    attn_qkv: Any


def _require_maca_device() -> torch.device:
    if not torch.cuda.is_available():
        print("ERROR: mcPytorch CUDA/MACA device not available.", file=sys.stderr)
        sys.exit(1)
    name = torch.cuda.get_device_name(0)
    print(f"Device: {name}")
    if "MetaX" not in name and os.environ.get("YIRAGE_MACA_ALLOW_NON_METAX", "") != "1":
        print(
            "WARNING: expected MetaX GPU; set YIRAGE_MACA_ALLOW_NON_METAX=1 to run on other CUDA devices.",
            file=sys.stderr,
        )
    return torch.device("cuda:0")


def _superoptimize_mlp_gate_up(
    hidden_size: int,
    intermediate_size: int,
    *,
    backend: str,
    search: Dict[str, Any],
    dtype,
):
    import yirage as yr

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
    return graph.superoptimize(
        backend=backend,
        config="mlp",
        verbose=False,
        **maca_superoptimize_ray_kwargs(),
        **search,
    )


def _superoptimize_mlp_down(
    hidden_size: int,
    intermediate_size: int,
    *,
    backend: str,
    search: Dict[str, Any],
    dtype,
):
    import yirage as yr

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
    return graph.superoptimize(
        backend=backend,
        config="mlp",
        verbose=False,
        **maca_superoptimize_ray_kwargs(),
        **search,
    )


def _superoptimize_attn_qkv(
    hidden_size: int,
    fused_outdim: int,
    *,
    backend: str,
    search: Dict[str, Any],
    dtype,
):
    import yirage as yr

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
    return graph.superoptimize(
        backend=backend,
        config="mlp",
        verbose=False,
        **maca_superoptimize_ray_kwargs(),
        **search,
    )


def build_qwen_kernels(
    *,
    backend: str = "maca",
    quick: bool = True,
    dtype_name: str = "float16",
) -> QwenLayerKernels:
    import yirage as yr

    dtype = yr.float16 if dtype_name == "float16" else yr.bfloat16
    search = maca_search_kwargs(quick=quick)
    fused_out = (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM

    print("Superoptimizing Qwen MLP gate/up (RMS+matmul)...")
    mlp1 = _superoptimize_mlp_gate_up(
        HIDDEN_SIZE, INTERMEDIATE_SIZE, backend=backend, search=search, dtype=dtype
    )
    print("Superoptimizing Qwen MLP down (SiLU gate × up + matmul)...")
    mlp2 = _superoptimize_mlp_down(
        HIDDEN_SIZE, INTERMEDIATE_SIZE, backend=backend, search=search, dtype=dtype
    )
    print("Superoptimizing Qwen attention QKV (RMS+matmul)...")
    attn = _superoptimize_attn_qkv(
        HIDDEN_SIZE, fused_out, backend=backend, search=search, dtype=dtype
    )
    if mlp1 is None or mlp2 is None or attn is None:
        raise RuntimeError("superoptimize returned None — mxcc compile or search failed")
    return QwenLayerKernels(mlp_gate_up=mlp1, mlp_down=mlp2, attn_qkv=attn)


def _pytorch_mlp(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    fused_gate_up: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    variance = hidden.pow(2).mean(-1, keepdim=True)
    x = hidden * torch.rsqrt(variance)
    x = x * norm_weight
    fused = torch.matmul(x, fused_gate_up)
    gate, up = fused.chunk(2, dim=-1)
    x = torch.nn.functional.silu(gate) * up
    return torch.matmul(x, down_weight)


def _yirage_mlp(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    fused_gate_up: torch.Tensor,
    down_weight: torch.Tensor,
    kernels: QwenLayerKernels,
) -> torch.Tensor:
    out = kernels.mlp_gate_up(inputs=(hidden, norm_weight, fused_gate_up))[0]
    gate, up = out.chunk(2, dim=-1)
    return kernels.mlp_down(inputs=(gate, up, down_weight))[0]


def _pytorch_qkv(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    fused_qkv: torch.Tensor,
) -> torch.Tensor:
    variance = hidden.pow(2).mean(-1, keepdim=True)
    x = hidden * torch.rsqrt(variance)
    x = x * norm_weight
    return torch.matmul(x, fused_qkv)


def _yirage_qkv(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    fused_qkv: torch.Tensor,
    kernels: QwenLayerKernels,
) -> torch.Tensor:
    return kernels.attn_qkv(inputs=(hidden, norm_weight, fused_qkv))[0]


def _synthetic_decode_loop(
    kernels: QwenLayerKernels,
    *,
    device: torch.device,
    decode_steps: int,
    torch_dtype: torch.dtype,
    prefill_len: int,
) -> Tuple[List[int], float]:
    """Prefill (PyTorch) + decode (YiRage kernels), aligned to q_len==1 decode path in modeling_qwen2."""

    norm_w = torch.ones(HIDDEN_SIZE, dtype=torch_dtype, device=device)
    fused_gate_up = torch.randn(
        HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE, dtype=torch_dtype, device=device
    )
    down_w = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch_dtype, device=device)
    fused_qkv = torch.randn(
        HIDDEN_SIZE, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, dtype=torch_dtype, device=device
    )
    lm_head = torch.randn(HIDDEN_SIZE, 151936, dtype=torch_dtype, device=device)

    # Prefill: seq > 1 uses PyTorch (same branch as CUDA demo modeling)
    prefill = torch.randn(1, prefill_len, HIDDEN_SIZE, dtype=torch_dtype, device=device)
    _ = _pytorch_mlp(prefill[:, -1:, :], norm_w, fused_gate_up, down_w)
    _ = _pytorch_qkv(prefill[:, -1:, :], norm_w, fused_qkv)

    hidden = prefill[:, -1:, :].contiguous()
    tokens: List[int] = []

    sync_device(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(decode_steps):
        qkv = _yirage_qkv(hidden, norm_w, fused_qkv, kernels)
        # Minimal o-proj + MLP (decode path under test)
        hidden = _yirage_mlp(qkv, norm_w, fused_gate_up, down_w, kernels)
        logits = torch.matmul(hidden, lm_head)
        tok = int(logits[0, -1].argmax().item())
        tokens.append(tok)
    end.record()
    sync_device(device)
    ms_per_step = start.elapsed_time(end) / max(decode_steps, 1)
    return tokens, ms_per_step


def _parity_check_kernels(
    kernels: QwenLayerKernels,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
    atol: float = 0.05,
) -> None:
    """Runtime parity: YiRage MACA kernels vs PyTorch reference on decode shape [1,1,H]."""
    norm_w = torch.ones(HIDDEN_SIZE, dtype=torch_dtype, device=device)
    fused_gate_up = torch.randn(
        HIDDEN_SIZE, 2 * INTERMEDIATE_SIZE, dtype=torch_dtype, device=device
    )
    down_w = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE, dtype=torch_dtype, device=device)
    fused_qkv = torch.randn(
        HIDDEN_SIZE, (NUM_HEADS + 2 * NUM_KV_HEADS) * HEAD_DIM, dtype=torch_dtype, device=device
    )
    hidden = torch.randn(1, 1, HIDDEN_SIZE, dtype=torch_dtype, device=device)

    ref_mlp = _pytorch_mlp(hidden, norm_w, fused_gate_up, down_w)
    yir_mlp = _yirage_mlp(hidden, norm_w, fused_gate_up, down_w, kernels)
    if not torch.allclose(ref_mlp, yir_mlp, atol=atol, rtol=0.01):
        raise AssertionError(f"MLP parity failed: max diff {(ref_mlp - yir_mlp).abs().max().item()}")

    ref_qkv = _pytorch_qkv(hidden, norm_w, fused_qkv)
    yir_qkv = _yirage_qkv(hidden, norm_w, fused_qkv, kernels)
    if not torch.allclose(ref_qkv, yir_qkv, atol=atol, rtol=0.01):
        raise AssertionError(f"QKV parity failed: max diff {(ref_qkv - yir_qkv).abs().max().item()}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="MACA Qwen full-chain inference smoke (aligned to demo/qwen2.5/demo.py)"
    )
    parser.add_argument("--quick", action="store_true", help="Tractable MACA search (default on)")
    parser.add_argument("--full-search", action="store_true", help="Use full get_maca_search_config grid")
    parser.add_argument("--decode-steps", type=int, default=4, help="Decode iterations after prefill")
    parser.add_argument("--prefill-len", type=int, default=8, help="Synthetic prefill sequence length")
    parser.add_argument("--skip-parity", action="store_true", help="Skip YiRage vs PyTorch parity check")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="Tensor dtype (CUDA qwen2.5 demo uses bfloat16; float16 default for MACA smoke)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    apply_maca_demo_env()
    os.environ.setdefault("YIRAGE_BACKEND", "maca")

    print("=" * 60)
    print("MACA Qwen Full-Chain Inference Smoke")
    print("CUDA reference: demo/qwen2.5/demo.py")
    print("=" * 60)

    device = _require_maca_device()
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    quick = not args.full_search

    kernels = build_qwen_kernels(backend="maca", quick=quick, dtype_name=args.dtype)

    if not args.skip_parity:
        print("Checking decode-shape parity vs PyTorch reference...")
        _parity_check_kernels(kernels, device=device, torch_dtype=torch_dtype)
        print("✓ MLP + QKV parity OK")

    print(f"Running synthetic prefill (len={args.prefill_len}) + decode ({args.decode_steps} steps)...")
    tokens, ms = _synthetic_decode_loop(
        kernels,
        device=device,
        decode_steps=args.decode_steps,
        torch_dtype=torch_dtype,
        prefill_len=args.prefill_len,
    )
    print(f"✓ Generated token ids (first 8): {tokens[:8]}")
    print(f"✓ Mean decode latency: {ms:.4f} ms/step")
    print("PASS: MACA Qwen full-chain inference smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
