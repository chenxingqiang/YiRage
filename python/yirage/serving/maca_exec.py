# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""MACA yirage.core execution helpers for RuntimeFusion serving (S15 scaffold).

Full ``backend=yirage_maca`` MLP capsules require MetaX VM build
(``YIRAGE_BACKEND=maca pip install -e .``). Cloud CPU cert uses
:mod:`yirage.serving.maca_serving_e2e` torch meta bridge instead.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from .yirage_exec import is_yirage_core_available, require_yirage_core, superoptimize_kwargs


def is_yirage_maca_available() -> bool:
    """True when ``yirage.core`` is built with MACA backend enabled."""
    if os.environ.get("YIRAGE_SKIP_NATIVE") == "1":
        return False
    if not is_yirage_core_available():
        return False
    if os.environ.get("YIRAGE_BACKEND", "").lower() == "maca":
        return True
    try:
        from yirage.backends.api import is_backend_available

        return bool(is_backend_available("maca"))
    except Exception:
        return False


def require_yirage_maca() -> None:
    if not is_yirage_maca_available():
        raise RuntimeError(
            "yirage_maca serving tier requires YIRAGE_BACKEND=maca and built yirage.core "
            "on a MetaX GPU host. Use maca_serving_e2e torch meta bridge on CPU CI."
        )


def maca_superoptimize_kwargs(*, quick: bool = True) -> Dict[str, Any]:
    """Superoptimize kwargs aligned with MACA 64-warp search (MetaX VM)."""
    require_yirage_maca()
    try:
        from yirage.backends.maca.config import resolve_maca_search_config

        cfg = resolve_maca_search_config(quick=quick)
        grid = cfg.get("grid_dims_to_explore") or [(4, 1, 1)]
        block = cfg.get("block_dims_to_explore") or [(256, 1, 1)]
        franges = cfg.get("franges_to_explore") or [8]
    except Exception:
        grid = [(4, 1, 1)]
        block = [(256, 1, 1)]
        franges = [8]
    base = superoptimize_kwargs(quick=quick)
    return {
        **base,
        "backend": "maca",
        "griddims": [grid[0]],
        "blockdims": [block[0]],
        "franges": [franges[0]],
    }


def inspect_maca_serving_yirage_tier() -> Dict[str, Any]:
    return {
        "yirage_maca_available": is_yirage_maca_available(),
        "yirage_backend_env": os.environ.get("YIRAGE_BACKEND"),
    }
