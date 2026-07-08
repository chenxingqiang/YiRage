"""Re-export Qwen2Config from demo/qwen2.5 (single source)."""

from __future__ import annotations

import os
import sys

_QWEN25_MODELS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "qwen2.5", "models")
)
if _QWEN25_MODELS not in sys.path:
    sys.path.insert(0, _QWEN25_MODELS)

from configuration_qwen2 import Qwen2Config  # noqa: E402,F401
