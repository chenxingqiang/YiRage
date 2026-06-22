# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Utilities Module.
"""

from .common import (
    get_shared_memory_capacity,
    get_nvcc_compiler,
)

# Visualizer imports (conditional - requires graphviz)
try:
    from .visualizer import (
        visualizer,
        handle_graph_data,
        kernel_graph,
        block_graph,
    )

    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False

__all__ = [
    "get_shared_memory_capacity",
    "get_nvcc_compiler",
    "VISUALIZER_AVAILABLE",
]

if VISUALIZER_AVAILABLE:
    __all__.extend(
        [
            "visualizer",
            "handle_graph_data",
            "kernel_graph",
            "block_graph",
        ]
    )
