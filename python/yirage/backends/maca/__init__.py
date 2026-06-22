# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""MetaX MACA GPU backend support."""

from .config import (
    get_maca_search_config,
    MACA_WARP_SIZE,
    MACA_MAX_THREADS_PER_BLOCK,
)

__all__ = [
    "get_maca_search_config",
    "MACA_WARP_SIZE",
    "MACA_MAX_THREADS_PER_BLOCK",
]
