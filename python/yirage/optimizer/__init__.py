# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Optimizer Module

End-to-end optimization from muGraph to MLIR with:
- COMET-based compound operation search
- Pattern detection and fusion
- Multi-backend code generation

Usage:
    from yirage.optimizer import YirageOptimizer, optimize_graph
    
    # Quick optimization
    result = optimize_graph(graph, target='cuda-h100')
    print(result.optimized_mlir)
    
    # Full control
    optimizer = YirageOptimizer(target='cuda-h100', config=custom_config)
    result = optimizer.optimize(graph)
"""

from .end_to_end import (
    # Main classes
    YirageOptimizer,
    OptimizationConfig,
    OptimizationResult,
    
    # Graph representations
    MuGraph,
    GraphOp,
    GraphTensor,
    
    # Pattern detection
    PatternDetector,
    
    # MLIR generation
    OptimizedMLIRGenerator,
    
    # Convenience functions
    optimize_graph,
    optimize_and_compile,
    get_supported_targets,
)

__all__ = [
    'YirageOptimizer',
    'OptimizationConfig',
    'OptimizationResult',
    'MuGraph',
    'GraphOp',
    'GraphTensor',
    'PatternDetector',
    'OptimizedMLIRGenerator',
    'optimize_graph',
    'optimize_and_compile',
    'get_supported_targets',
]
