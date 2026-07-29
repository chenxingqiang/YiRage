# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Execution backend selection for RuntimeFusion serving."""

from __future__ import annotations

BACKEND_TORCH = "torch"
BACKEND_YIRAGE_CPU = "yirage_cpu"  # yirage.core seed + CPU superoptimize
BACKEND_YIRAGE_MACA = "yirage_maca"  # yirage.core + MACA superoptimize (MetaX VM)
BACKEND_NUMPY_REF = "numpy_ref"  # offline reference only; not used in serving cert


def default_serving_backend() -> str:
    """Prefer torch execution when PyTorch is installed."""
    try:
        import torch  # noqa: F401

        return BACKEND_TORCH
    except ImportError:
        return BACKEND_NUMPY_REF


def is_exec_backend(backend: str) -> bool:
    return backend in (BACKEND_TORCH, BACKEND_YIRAGE_CPU, BACKEND_YIRAGE_MACA)


def is_yirage_backend(backend: str) -> bool:
    return backend in (BACKEND_YIRAGE_CPU, BACKEND_YIRAGE_MACA)


def is_maca_serving_backend(backend: str) -> bool:
    return backend == BACKEND_YIRAGE_MACA
