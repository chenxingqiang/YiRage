# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Real PyTorch execution for RuntimeFusion (no NumPy stub / mock engine).

Uses actual ``torch`` tensor ops on CPU (or CUDA when available). This is the
default execution backend for serving smoke and cert when ``--real`` is set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

TensorLike = Union["torch.Tensor", np.ndarray]


def require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "Real serving execution requires PyTorch. Install with: pip install torch"
        ) from e


def to_torch(
    x: TensorLike,
    *,
    device: Optional[str] = None,
    dtype: Optional[Any] = None,
):
    require_torch()
    import torch

    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(np.asarray(x))
    if dtype is not None:
        t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device=device)
    return t


def to_numpy(x: Any) -> np.ndarray:
    require_torch()
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def default_device() -> str:
    require_torch()
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def rms_norm_torch(hidden, weight, *, eps: float = 1e-6):
    require_torch()
    import torch

    x = hidden.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x * weight.float()


def mlp_torch(
    hidden,
    *,
    rms_weight,
    w_gate,
    w_up,
    w_down,
    eps: float = 1e-6,
):
    """Qwen-style fused MLP: rmsnorm + silu(gate)*up + down + residual."""
    require_torch()
    import torch.nn.functional as F

    if hidden.dim() != 2:
        raise ValueError(f"hidden must be rank-2 [B,H], got shape={tuple(hidden.shape)}")
    h = rms_norm_torch(hidden, rms_weight, eps=eps)
    gate = h @ w_gate
    up = h @ w_up
    mid = F.silu(gate) * up
    return hidden + mid @ w_down


@dataclass
class BenchResult:
    name: str
    mean_ms: float
    iters: int
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mean_ms": self.mean_ms,
            "iters": self.iters,
            "device": self.device,
        }


def bench_forward(
    fn: Callable[[], Any],
    *,
    name: str = "forward",
    warmup: int = 5,
    iters: int = 20,
    device: Optional[str] = None,
) -> BenchResult:
    """Wall-clock benchmark for a zero-arg forward callable."""
    require_torch()
    import torch

    dev = device or default_device()
    for _ in range(warmup):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(iters, 1)
    return BenchResult(name=name, mean_ms=elapsed_ms, iters=iters, device=dev)
