# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage Storage Module.

Provides persistent storage for muGraphs and training data.

Usage:
    from yirage.storage import MuGraphStore, save_mugraph, find_best_mugraph
    
    # Save optimized kernel
    save_mugraph(graph_hash, optimized_graph, "cuda", latency_ms=1.5)
    
    # Find best cached kernel
    entry = find_best_mugraph(graph_hash, "cuda")
    
    # Export training data for ML
    store = MuGraphStore()
    store.export_training_data("./training_data")
"""

from .mugraph_store import (
    # Main store
    MuGraphStore,
    get_mugraph_store,
    # Entry types
    MuGraphEntry,
    MuGraphMetadata,
    # Convenience functions
    save_mugraph,
    find_mugraph,
    find_best_mugraph,
    # Data structures for training
    GraphStructure,
    DeviceCapabilities,
    SearchConfiguration,
    PerformanceMetrics,
    SearchTrajectory,
    CandidateEvaluation,
    TrainingFeatures,
    TrainingLabels,
    OperatorInfo,
    TensorInfo,
    OpType,
)

from .graph_dataset import (
    GraphDataset,
    DatasetEntry,
    graph_dataset,
)

from .graph_serde import deserialize_cygraph, serialize_optimized_graph

__all__ = [
    # MuGraph Store
    "MuGraphStore",
    "get_mugraph_store",
    # Entry types
    "MuGraphEntry",
    "MuGraphMetadata",
    # Convenience functions
    "save_mugraph",
    "find_mugraph",
    "find_best_mugraph",
    # Data structures
    "GraphStructure",
    "DeviceCapabilities",
    "SearchConfiguration",
    "PerformanceMetrics",
    "SearchTrajectory",
    "CandidateEvaluation",
    "TrainingFeatures",
    "TrainingLabels",
    "OperatorInfo",
    "TensorInfo",
    "OpType",
    # Graph Dataset
    "GraphDataset",
    "DatasetEntry",
    "graph_dataset",
    # Graph serialization
    "serialize_optimized_graph",
    "deserialize_cygraph",
]
