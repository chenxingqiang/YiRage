# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Kernel Module.

Provides kernel graph construction and optimization.
"""

import warnings

# op_registry has no torch/C++ dependency — always available.
from .op_registry import (
    CustomOpSpec,
    OpRegistry,
    global_registry,
    register_op,
    custom_op,
    list_ops,
    get_op,
)

# The remainder of the module requires torch and (optionally) C++ core bindings.
try:
    from .graph import KNGraph, get_key_paths
    from .threadblock import TBGraph
    from .multi_backend import (
        KernelBackend,
        MultiBackendKernel,
        create_kernel,
    )
    from .speculative import (
        SpecDecodeConfig,
        LookaheadConfig,
        PromptLookupConfig,
        spec_decode_class,
    )
except (ImportError, OSError) as _e:
    warnings.warn(
        f"YiRage kernel graph extensions not available ({_e}). "
        "Install torch and build C++ bindings for full functionality.",
        ImportWarning,
        stacklevel=2,
    )

__all__ = [
    # Graph (requires torch + C++ core)
    "KNGraph",
    "TBGraph",
    "get_key_paths",
    # Custom operator registry (always available)
    "CustomOpSpec",
    "OpRegistry",
    "global_registry",
    "register_op",
    "custom_op",
    "list_ops",
    "get_op",
    # Multi-backend (requires torch)
    "KernelBackend",
    "MultiBackendKernel",
    "create_kernel",
    # Speculative (requires torch)
    "SpecDecodeConfig",
    "LookaheadConfig",
    "PromptLookupConfig",
    "spec_decode_class",
]

