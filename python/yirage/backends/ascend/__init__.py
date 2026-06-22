# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""Ascend NPU backend support."""

from .config import get_ascend_search_config
from .transpiler import (
    AscendDeviceType,
    AscendTranspileConfig,
    AscendTranspileResult,
    CodeGenPath,
    detect_ascend_environment,
    get_recommended_config,
)

# Align with C++ naming: compile path enum is CodeGenPath in Python.
AscendCompilePath = CodeGenPath

__all__ = [
    "get_ascend_search_config",
    "AscendCompilePath",
    "AscendDeviceType",
    "AscendTranspileConfig",
    "AscendTranspileResult",
    "CodeGenPath",
    "detect_ascend_environment",
    "get_recommended_config",
]
