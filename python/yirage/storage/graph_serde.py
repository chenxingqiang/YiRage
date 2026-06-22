# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""Serialize / deserialize optimized KNGraph (CyKNGraph) for MuGraphStore."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Optional


def serialize_optimized_graph(optimized_graph: Any) -> Optional[str]:
    """Write an optimized graph to a JSON string suitable for MuGraphStore."""
    cygraph = getattr(optimized_graph, "cygraph", None)
    if cygraph is None:
        return None

    try:
        from yirage.core import cy_to_json
    except ImportError:
        return None

    fd, path = tempfile.mkstemp(suffix=".json", prefix="yirage_mug_save_")
    os.close(fd)
    try:
        cy_to_json(cygraph, path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def deserialize_cygraph(graph_json: Optional[Any]) -> Optional[Any]:
    """Load CyKNGraph from JSON string (or dict) written by serialize_optimized_graph."""
    if not graph_json:
        return None

    if isinstance(graph_json, dict):
        payload = json.dumps(graph_json)
    elif isinstance(graph_json, str):
        payload = graph_json
    else:
        return None

    try:
        from yirage.core import cy_from_json
    except ImportError:
        return None

    fd, path = tempfile.mkstemp(suffix=".json", prefix="yirage_mug_load_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        return cy_from_json(path)
    except Exception:
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
