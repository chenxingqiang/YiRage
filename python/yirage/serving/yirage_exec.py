# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""YiRage core execution + CPU superoptimize helpers for RuntimeFusion serving.

Serving MLP on CPU uses a **split kernel** strategy (aligned with Qwen MACA demos):

1. **gate_up** — seed graph execute via ``yirage.core`` (rms_norm + mul + matmul)
2. **mid** — ``silu(gate) * up`` in PyTorch (tiny epilogue)
3. **down** — ``superoptimize(backend="cpu")`` on plain matmul ``(1,I) @ (I,H)``
4. **residual** — PyTorch add

Full-graph superoptimize for the entire MLP may yield 0 valid µGraphs under
tractable CPU search caps; down matmul uses ``superoptimize(backend=\"cpu\")``
with serving tractability env (``YIRAGE_SERVING_KN_MATMUL_ONLY``). No seed fallback.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .torch_exec import require_torch, to_torch


def is_yirage_core_available() -> bool:
    if os.environ.get("YIRAGE_SKIP_NATIVE") == "1":
        return False
    try:
        import yirage as yr
        import yirage.core  # noqa: F401

        return hasattr(yr, "float32")
    except ImportError:
        return False


def require_yirage_core() -> None:
    if not is_yirage_core_available():
        raise RuntimeError(
            "yirage.core is not built. Run scripts/setup_serving_yirage_core.sh "
            "or pip install -e . with YIRAGE_BACKEND=cpu."
        )


def _yr_dtype(name: str):
    import yirage as yr

    if name in ("float32", "fp32"):
        return yr.float32
    if name in ("float16", "fp16"):
        return yr.float16
    if name in ("bfloat16", "bf16"):
        return yr.bfloat16
    raise ValueError(f"unsupported yirage dtype name: {name!r}")


def apply_serving_cpu_search_tractability() -> None:
    """Cap CPU search for serving plain-matmul superoptimize smoke."""
    from scripts.cpu_cert_utils import apply_plain_matmul_search_tractability

    apply_plain_matmul_search_tractability()
    os.environ["YIRAGE_SERVING_KN_MATMUL_ONLY"] = "1"


def apply_serving_kn_down_matmul_tractability() -> None:
    """KN-only down matmul search for Qwen-scale serving (no TB explore)."""
    apply_serving_cpu_search_tractability()


def superoptimize_kwargs(*, quick: bool = True) -> Dict[str, Any]:
    return {
        "backend": "cpu",
        "griddims": [(1, 1, 1)],
        "blockdims": [(32, 1, 1)],
        "franges": [1],
        "use_ray": False,
        "use_graph_dataset": False,
        "use_cached_graphs": False,
        "use_persistent_cache": True,
        "warmup_iters": 1,
        "profile_iters": 5 if quick else 20,
        "verbose": False,
    }


def build_gate_up_seed_graph(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
):
    """RMSNorm + mul + matmul gate/up (decode shape ``[1, H]``)."""
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
    return graph


def build_mlp_down_seed_graph(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
):
    """SiLU(gate) * up + matmul down (decode shape)."""
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
    return graph


def build_down_matmul_seed_graph(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
):
    """Plain matmul ``(1,I) @ (I,H)`` for superoptimize."""
    import yirage as yr

    dtype = _yr_dtype(dtype_name)
    graph = yr.new_kernel_graph()
    mid = graph.new_input(dims=(1, intermediate_size), dtype=dtype)
    w = graph.new_input(
        dims=(intermediate_size, hidden_size),
        strides=(1, intermediate_size),
        dtype=dtype,
    )
    graph.mark_output(graph.matmul(mid, w))
    return graph


def superoptimize_down_matmul_cpu(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
    quick: bool = True,
):
    """Superoptimize down-projection matmul; raises if search finds nothing."""
    require_yirage_core()
    apply_serving_kn_down_matmul_tractability()
    graph = build_down_matmul_seed_graph(
        hidden_size,
        intermediate_size,
        dtype_name=dtype_name,
    )
    optimized = graph.superoptimize(**superoptimize_kwargs(quick=quick))
    if optimized is None:
        raise RuntimeError(
            f"CPU superoptimize found 0 valid µGraphs for down matmul "
            f"(H={hidden_size}, I={intermediate_size})"
        )
    return optimized


@dataclass
class SuperoptimizeTiming:
    elapsed_s: float
    hidden_size: int
    intermediate_size: int
    backend: str = "cpu"


def bench_superoptimize_down_matmul(
    hidden_size: int,
    intermediate_size: int,
    *,
    dtype_name: str = "float32",
    quick: bool = True,
) -> Tuple[Any, SuperoptimizeTiming]:
    """Run superoptimize once and return (optimized_graph, timing)."""
    t0 = time.perf_counter()
    opt = superoptimize_down_matmul_cpu(
        hidden_size,
        intermediate_size,
        dtype_name=dtype_name,
        quick=quick,
    )
    elapsed = time.perf_counter() - t0
    return opt, SuperoptimizeTiming(
        elapsed_s=elapsed,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )


@dataclass
class YirageMlpCompileArtifacts:
    gate_up_graph: Any
    down_optimized: Any
    superopt_elapsed_s: float


class YirageServingMlpRunner:
    """Hybrid MLP: yirage.core seed gate_up + superoptimized down matmul."""

    def __init__(
        self,
        *,
        rms_weight: Any,
        w_gate: Any,
        w_up: Any,
        w_down: Any,
        eps: float = 1e-6,
        device: Optional[str] = None,
        dtype_name: str = "float32",
        quick_superopt: bool = True,
    ):
        require_yirage_core()
        require_torch()
        import torch

        self.eps = float(eps)
        self.dtype_name = dtype_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.rms_weight = to_torch(rms_weight, device=self._device, dtype=torch.float32)
        self.w_gate = to_torch(w_gate, device=self._device, dtype=torch.float32)
        self.w_up = to_torch(w_up, device=self._device, dtype=torch.float32)
        self.w_down = to_torch(w_down, device=self._device, dtype=torch.float32)

        h = self.w_gate.shape[0]
        i = self.w_gate.shape[1]
        self.hidden_size = h
        self.intermediate_size = i

        self._w_gate_up = torch.cat([self.w_gate, self.w_up], dim=1)
        self._gate_up_graph = build_gate_up_seed_graph(h, i, dtype_name=dtype_name)
        self._down_optimized, timing = bench_superoptimize_down_matmul(
            h,
            i,
            dtype_name=dtype_name,
            quick=quick_superopt,
        )
        self.superopt_elapsed_s = timing.elapsed_s
        self._yr_dtype = _yr_dtype(dtype_name)

    @property
    def artifacts(self) -> YirageMlpCompileArtifacts:
        return YirageMlpCompileArtifacts(
            gate_up_graph=self._gate_up_graph,
            down_optimized=self._down_optimized,
            superopt_elapsed_s=self.superopt_elapsed_s,
        )

    def _torch_dtype(self):
        import torch

        if self.dtype_name in ("float16", "fp16"):
            return torch.float16
        if self.dtype_name in ("bfloat16", "bf16"):
            return torch.bfloat16
        return torch.float32

    def _gate_up_yirage(self, hidden: Any) -> Any:
        import torch

        h = to_torch(hidden, device=self._device, dtype=torch.float32)
        if h.ndim == 1:
            h = h.unsqueeze(0)
        batch = h.shape[0]
        if batch != 1:
            raise ValueError(
                f"YirageServingMlpRunner gate_up expects batch=1 decode shape, got {batch}"
            )
        rw = self.rms_weight
        if rw.ndim == 1:
            rw = rw.unsqueeze(0)
        yr_dtype = self._torch_dtype()
        yr_h = h.to(dtype=yr_dtype)
        yr_rw = rw.to(dtype=yr_dtype)
        yr_w = self._w_gate_up.to(dtype=yr_dtype)
        out = self._gate_up_graph(inputs=[yr_h, yr_rw, yr_w])
        return out[0]

    def forward(self, hidden: Any) -> Any:
        """Full MLP forward with yirage gate_up + superopt down."""
        import torch
        import torch.nn.functional as F

        residual = to_torch(hidden, device=self._device, dtype=torch.float32)
        if residual.ndim == 1:
            residual = residual.unsqueeze(0)

        gate_up = self._gate_up_yirage(residual)
        gate_up_f = gate_up.float()
        gate, up = torch.chunk(gate_up_f, 2, dim=-1)
        mid = F.silu(gate) * up

        yr_dtype = self._torch_dtype()
        mid_yr = mid.to(dtype=yr_dtype)
        w_down_yr = self.w_down.to(dtype=yr_dtype)
        down_out = self._down_optimized(inputs=[mid_yr, w_down_yr])[0].float()
        return residual + down_out

    def forward_torch_reference(self, hidden: Any) -> Any:
        from .torch_exec import mlp_torch

        return mlp_torch(
            hidden,
            rms_weight=self.rms_weight,
            w_gate=self.w_gate,
            w_up=self.w_up,
            w_down=self.w_down,
            eps=self.eps,
        )


def mlp_yirage_cpu(
    hidden: Any,
    *,
    runner: YirageServingMlpRunner,
) -> Any:
    return runner.forward(hidden)
