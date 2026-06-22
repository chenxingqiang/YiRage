# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Unified Compiler Module

Integrates muGraph superoptimization with MLIR compilation for end-to-end
kernel optimization across all supported backends.

Pipeline:
    PyTorch/JAX → KNGraph → muGraph Search → MLIR → Target Code → Execution

Usage:
    from yirage.compiler import UnifiedCompiler, CompileMode
    
    compiler = UnifiedCompiler(backend='cuda', mode=CompileMode.SUPEROPTIMIZE)
    optimized_fn = compiler.compile(graph)
    result = optimized_fn(inputs)
"""

from .unified import (
    UnifiedCompiler,
    CompileMode,
    CompileOptions,
    CompileResult,
    compile_graph,
    hardware_aware_compile,
    jit_compile,
)

from .search_space import (
    chip_arch_to_search_config,
    MODE_FAST,
    MODE_SUPEROPTIMIZE,
    MODE_AGGRESSIVE,
)

from .pipeline import (
    CompilePipeline,
    PipelineStage,
    SuperoptimizeStage,
    MLIRLoweringStage,
    CodeGenStage,
)

from .cache import (
    CompileCache,
    get_compile_cache,
    clear_compile_cache,
)

__all__ = [
    # Main Compiler
    "UnifiedCompiler",
    "CompileMode",
    "CompileOptions",
    "CompileResult",
    "compile_graph",
    "hardware_aware_compile",
    "jit_compile",
    # Search-space derivation (standalone, no torch/core required)
    "chip_arch_to_search_config",
    "MODE_FAST",
    "MODE_SUPEROPTIMIZE",
    "MODE_AGGRESSIVE",
    # Pipeline Stages
    "CompilePipeline",
    "PipelineStage",
    "SuperoptimizeStage",
    "MLIRLoweringStage",
    "CodeGenStage",
    # Caching
    "CompileCache",
    "get_compile_cache",
    "clear_compile_cache",
]
