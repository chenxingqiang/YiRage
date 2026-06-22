# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Persistent Kernel Module.

Provides persistent kernel runtime for LLM inference.

Usage:
    from yirage.persistent_kernel import PersistentKernel, PKRuntime
    
    # Build a persistent kernel for LLM inference
    pk = PersistentKernel(num_blocks=108, threads_per_block=128)
    pk.attention_layer(...)
    pk.linear_layer(...)
    
    # Execute with runtime
    runtime = PKRuntime(backend="cuda")
    runtime.execute(pk)
"""

from .kernel import (
    PersistentKernel,
)

from .runtime import (
    PKRuntime,
    PKBackendType,
    PKMode,
    create_runtime,
    get_available_backends,
)

__all__ = [
    # Core persistent kernel
    "PersistentKernel",
    # Runtime
    "PKRuntime",
    "PKBackendType",
    "PKMode",
    "create_runtime",
    "get_available_backends",
]
