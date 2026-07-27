# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Execution backend selection for RuntimeFusion serving."""

from __future__ import annotations

BACKEND_TORCH = "torch"
BACKEND_YIRAGE_CPU = "yirage_cpu"  # yirage.core seed + CPU superoptimize
BACKEND_NUMPY_REF = "numpy_ref"  # reference only; not for --real cert


def default_serving_backend() -> str:
    """Prefer real torch execution when PyTorch is installed."""
    try:
        import torch  # noqa: F401

        return BACKEND_TORCH
    except ImportError:
        return BACKEND_NUMPY_REF


def is_real_backend(backend: str) -> bool:
    return backend in (BACKEND_TORCH, BACKEND_YIRAGE_CPU)


def is_yirage_backend(backend: str) -> bool:
    return backend == BACKEND_YIRAGE_CPU
