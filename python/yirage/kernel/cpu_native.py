# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
CPU matmul primitives for muGraph execution.

Default: ``torch.matmul`` (host BLAS — MKL/OpenBLAS on this machine).
P2: ``YIRAGE_CPU_RMS_MATMUL_NATIVE=auto|1`` uses OpenMP + cblas fused kernel when built.
Experimental: ``YIRAGE_CPU_NATIVE=1`` uses YiRage C++ SIMD GEMM for plain matmul.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

_USE_EXPERIMENTAL_NATIVE = os.environ.get("YIRAGE_CPU_NATIVE", "0") == "1"


def uses_host_blas() -> bool:
    """Plain matmul goes through PyTorch → host BLAS (MKL on typical Linux builds)."""
    return not _USE_EXPERIMENTAL_NATIVE


def native_matmul_available() -> bool:
    """Experimental C++ SIMD path is built and enabled."""
    if not _USE_EXPERIMENTAL_NATIVE:
        return False
    try:
        from yirage import core

        return hasattr(core, "cpu_gemm_f32")
    except Exception:
        return False


def native_rms_matmul_available() -> bool:
    """P2 OpenMP + cblas fused rms_matmul is built."""
    try:
        from yirage import core

        return hasattr(core, "cpu_rms_matmul_f32")
    except Exception:
        return False


def _native_rms_matmul_mode() -> str:
    # Default off: PyTorch/MKL is faster on typical Linux builds; opt-in for bare-metal.
    return os.environ.get("YIRAGE_CPU_RMS_MATMUL_NATIVE", "0").strip().lower()


def _native_rms_matmul_elem_threshold() -> int:
    raw = os.environ.get("YIRAGE_CPU_RMS_MATMUL_NATIVE_ELEMS", "1048576")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1048576


def should_use_native_rms_matmul(m: int, k: int, n: int) -> bool:
    """Whether to route rms+matmul through the C++ fused kernel."""
    if not native_rms_matmul_available():
        return False
    mode = _native_rms_matmul_mode()
    if mode in ("0", "false", "off", "no"):
        return False
    if mode in ("1", "true", "yes", "on"):
        return True
    return m * k * n >= _native_rms_matmul_elem_threshold()


def _cpu_matmul_native(
    a: torch.Tensor, b: torch.Tensor, num_threads: Optional[int] = None
) -> torch.Tensor:
    from yirage import core

    fn = core.cpu_gemm_f32
    if a.dim() != 2 or b.dim() != 2:
        return torch.matmul(a, b)

    out_dtype = torch.result_type(a, b)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        return torch.matmul(a, b)

    a_work = a if a.dtype == torch.float32 else a.float()
    b_work = b if b.dtype == torch.float32 else b.float()
    if not a_work.is_contiguous():
        a_work = a_work.contiguous()
    if not b_work.is_contiguous():
        b_work = b_work.contiguous()

    if num_threads is None:
        try:
            from yirage.backends.cpu.config import get_cpu_runtime_config

            num_threads = get_cpu_runtime_config().get("torch_num_threads", 1)
        except Exception:
            num_threads = 1

    c = torch.empty((m, n), dtype=torch.float32, device=a.device)
    rc = fn(a_work, b_work, c, m, n, k, int(num_threads))
    if rc != 0:
        return torch.matmul(a, b)

    if out_dtype != torch.float32:
        return c.to(out_dtype)
    return c


def _cpu_rms_matmul_native(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    num_threads: Optional[int] = None,
) -> torch.Tensor:
    from yirage import core

    fn = core.cpu_rms_matmul_f32
    if x.dim() != 2 or w.dim() != 2:
        return cpu_rms_matmul_torch(x, w, epsilon=epsilon)

    m, k = x.shape
    k2, n = w.shape
    if k != k2:
        return cpu_rms_matmul_torch(x, w, epsilon=epsilon)

    x_work = x if x.dtype == torch.float32 else x.float()
    w_work = w if w.dtype == torch.float32 else w.float()
    if not x_work.is_contiguous():
        x_work = x_work.contiguous()
    if not w_work.is_contiguous():
        w_work = w_work.contiguous()

    if num_threads is None:
        try:
            from yirage.backends.cpu.config import get_cpu_runtime_config

            num_threads = get_cpu_runtime_config().get("torch_num_threads", 0)
        except Exception:
            num_threads = 0

    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    rc = fn(x_work, w_work, out, m, n, k, float(epsilon), int(num_threads))
    if rc != 0:
        return cpu_rms_matmul_torch(x, w, epsilon=epsilon, num_threads=num_threads)

    return out.to(x.dtype)


def cpu_rms_matmul_torch(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    num_threads: Optional[int] = None,
) -> torch.Tensor:
    """P1 deferred-scale path via PyTorch host BLAS."""
    x32 = x.float()
    w32 = w.float()
    inv_rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    if _USE_EXPERIMENTAL_NATIVE and native_matmul_available():
        out = _cpu_matmul_native(x32, w32, num_threads=num_threads)
    else:
        out = torch.matmul(x32, w32)
    out = out * inv_rms
    return out.to(x.dtype)


def cpu_matmul(a: torch.Tensor, b: torch.Tensor, num_threads: Optional[int] = None):
    """
    Same-backend CPU GEMM used by ``cpu_call`` and CPU ``superoptimize`` profiling.

    Uses host BLAS (via ``torch.matmul``) unless ``YIRAGE_CPU_NATIVE=1``.
    """
    if _USE_EXPERIMENTAL_NATIVE and native_matmul_available():
        return _cpu_matmul_native(a, b, num_threads=num_threads)
    return torch.matmul(a, b)


def cpu_rms_norm(x: torch.Tensor, *, epsilon: float = 1e-6) -> torch.Tensor:
    """RMS norm with fp32 reduction; output dtype matches ``x``."""
    x32 = x.float()
    scale = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return (x32 * scale).to(x.dtype)


def cpu_matmul_chain(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_threads: Optional[int] = None,
) -> torch.Tensor:
    """Two GEMMs in sequence via host BLAS (``matmul(matmul(a,b), c)``)."""
    return cpu_matmul(
        cpu_matmul(a, b, num_threads=num_threads), c, num_threads=num_threads
    )


def cpu_concat_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    num_threads: Optional[int] = None,
) -> torch.Tensor:
    """LoRA concat+matmul via host BLAS (same semantics as bench MKL baseline)."""
    left = torch.cat([a, b], dim=1)
    right = torch.cat([c, d], dim=0)
    return cpu_matmul(left, right, num_threads=num_threads)


def cpu_rms_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    num_threads: Optional[int] = None,
) -> torch.Tensor:
    """Fused rms_norm + matmul.

    Supports ``[M,K] @ [K,N]`` and batched ``[B,M,K] @ [K,N] -> [B,M,N]``.

    P2 (``YIRAGE_CPU_RMS_MATMUL_NATIVE=auto``): OpenMP + cblas for large shapes.
    Otherwise P1 deferred-scale via PyTorch/MKL.
    """
    if x.dim() == 3 and w.dim() == 2:
        b, m, k = x.shape
        k2, n = w.shape
        if k != k2:
            return cpu_rms_matmul_torch(x, w, epsilon=epsilon, num_threads=num_threads)
        flat = x.reshape(b * m, k)
        if should_use_native_rms_matmul(b * m, k, n):
            out = _cpu_rms_matmul_native(
                flat, w, epsilon=epsilon, num_threads=num_threads
            )
        else:
            out = cpu_rms_matmul_torch(
                flat, w, epsilon=epsilon, num_threads=num_threads
            )
        return out.reshape(b, m, n)
    if x.dim() != 2 or w.dim() != 2:
        return cpu_rms_matmul_torch(x, w, epsilon=epsilon, num_threads=num_threads)
    m, k = x.shape
    n = w.shape[1]
    if should_use_native_rms_matmul(m, k, n):
        return _cpu_rms_matmul_native(
            x, w, epsilon=epsilon, num_threads=num_threads
        )
    return cpu_rms_matmul_torch(x, w, epsilon=epsilon, num_threads=num_threads)
