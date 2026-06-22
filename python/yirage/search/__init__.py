# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Search Module.

Provides kernel search and optimization strategies with COMET support.
"""

from .comet_search import (
    COMETSearchConfig,
    COMETCandidate,
    COMETCostModel,
    CompoundPattern,
    CompoundOpType,
    SchedulingStrategy,
    CollectiveOpType,
    MemoryLevel,
    TileConfig,
    COMETSearchStrategy,
    detect_compound_patterns,
    optimize_compound_graph,
)

from .backend_config import (
    BackendHardwareProfile,
    BACKEND_PROFILES,
    get_backend_config,
    get_auto_detected_config,
    list_supported_backends,
)

from .verifier_config import (
    VerifierConfig,
    env_truthy,
    formal_verifier_library_path,
    is_formal_verifier_built,
    resolve_verifier_config,
    runtime_verify_mugraph,
)

__all__ = [
    # COMET Search
    "COMETSearchConfig",
    "COMETCandidate",
    "COMETCostModel",
    "CompoundPattern",
    "CompoundOpType",
    "SchedulingStrategy",
    "CollectiveOpType",
    "MemoryLevel",
    "TileConfig",
    "COMETSearchStrategy",
    "detect_compound_patterns",
    "optimize_compound_graph",
    # Backend Config
    "BackendHardwareProfile",
    "BACKEND_PROFILES",
    "get_backend_config",
    "get_auto_detected_config",
    "list_supported_backends",
    # Search verification
    "VerifierConfig",
    "env_truthy",
    "formal_verifier_library_path",
    "is_formal_verifier_built",
    "resolve_verifier_config",
    "runtime_verify_mugraph",
]
