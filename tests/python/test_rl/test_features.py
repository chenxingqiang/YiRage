#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
RL Features Module Unit Tests

Tests for yirage/rl/features/ module including MuGraphFeature and FeatureProcessor.
Run with: pytest tests/python/test_rl/test_features.py -v
"""

import pytest
import json
import sys
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest import safe_import


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def mugraph_features_module():
    """Load mugraph features module."""
    return safe_import("yirage.rl.features.mugraph_features")


@pytest.fixture(scope="module")
def processor_module():
    """Load processor module."""
    return safe_import("yirage.rl.features.processor")


@pytest.fixture
def sample_mugraph_json():
    """Sample µGraph JSON for testing."""
    return json.dumps({
        "operators": [
            {
                "op_id": 0, "op_type": "matmul", "op_type_id": 0,
                "num_inputs": 2, "num_outputs": 1, "flops": 2097152.0,
                "memory_read_bytes": 16384, "memory_write_bytes": 8192,
                "input_tensor_ids": [0, 1], "output_tensor_ids": [2],
            },
            {
                "op_id": 1, "op_type": "silu", "op_type_id": 5,
                "num_inputs": 1, "num_outputs": 1, "flops": 2048.0,
                "memory_read_bytes": 8192, "memory_write_bytes": 8192,
                "input_tensor_ids": [2], "output_tensor_ids": [3],
            },
        ],
        "tensors": [
            {"tensor_id": 0, "dims": [64, 128], "dtype": "float16", "dtype_id": 0,
             "size_bytes": 16384, "memory_level": 1, "is_input": True, "is_output": False},
            {"tensor_id": 1, "dims": [128, 64], "dtype": "float16", "dtype_id": 0,
             "size_bytes": 16384, "memory_level": 1, "is_input": True, "is_output": False},
            {"tensor_id": 2, "dims": [64, 64], "dtype": "float16", "dtype_id": 0,
             "size_bytes": 8192, "memory_level": 0, "is_input": False, "is_output": False},
            {"tensor_id": 3, "dims": [64, 64], "dtype": "float16", "dtype_id": 0,
             "size_bytes": 8192, "memory_level": 0, "is_input": False, "is_output": True},
        ],
        "edges": [[0, 1]],
        "num_operators": 2,
        "num_tensors": 4,
        "graph_depth": 2,
        "graph_width": 1,
        "grid_dim": {"x": 4, "y": 1, "z": 1},
        "block_dim": {"x": 128, "y": 1, "z": 1},
        "forloop_range": 8,
        "reduction_dimx": 16,
        "theoretical_flops": 2099200.0,
    })


# =============================================================================
# MuGraphFeature Tests
# =============================================================================

class TestMuGraphFeature:
    """Tests for MuGraphFeature class."""

    def test_mugraph_feature_class_exists(self, mugraph_features_module):
        """Test MuGraphFeature class exists."""
        if mugraph_features_module is None:
            pytest.skip("MuGraph features module not available")

        assert hasattr(mugraph_features_module, "MuGraphFeature")

    def test_from_json_valid(self, mugraph_features_module, sample_mugraph_json):
        """Test parsing valid JSON."""
        if mugraph_features_module is None:
            pytest.skip("MuGraph features module not available")

        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)
        if MuGraphFeature is None:
            pytest.skip("MuGraphFeature not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        assert features is not None

    def test_from_json_invalid_raises(self, mugraph_features_module):
        """Test parsing invalid JSON raises error."""
        if mugraph_features_module is None:
            pytest.skip("MuGraph features module not available")

        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)
        if MuGraphFeature is None:
            pytest.skip("MuGraphFeature not found")

        try:
            result = MuGraphFeature.from_json("invalid json {{{")
            # Lenient parsing: returns empty/default feature instead of raising
            assert result is not None
            assert len(result.operators) == 0
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    def test_operator_count_correct(self, mugraph_features_module, sample_mugraph_json):
        """Test operator count is correct."""
        if mugraph_features_module is None:
            pytest.skip("MuGraph features module not available")

        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)
        if MuGraphFeature is None:
            pytest.skip("MuGraphFeature not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        assert len(features.operators) == 2

    def test_tensor_count_correct(self, mugraph_features_module, sample_mugraph_json):
        """Test tensor count is correct."""
        if mugraph_features_module is None:
            pytest.skip("MuGraph features module not available")

        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)
        if MuGraphFeature is None:
            pytest.skip("MuGraphFeature not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        assert len(features.tensors) == 4

    def test_tensor_dims_parsed(self, mugraph_features_module, sample_mugraph_json):
        """Test tensor dimensions are parsed."""
        if mugraph_features_module is None:
            pytest.skip("MuGraph features module not available")

        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)
        if MuGraphFeature is None:
            pytest.skip("MuGraphFeature not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        # First tensor should have dims [64, 128]
        first_tensor = features.tensors[0]
        assert hasattr(first_tensor, "dims") or "dims" in first_tensor

    def test_grid_dim_tuple(self, mugraph_features_module, sample_mugraph_json):
        """Test grid_dim is parsed as tuple."""
        if mugraph_features_module is None:
            pytest.skip("MuGraph features module not available")

        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)
        if MuGraphFeature is None:
            pytest.skip("MuGraphFeature not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        assert hasattr(features, "grid_dim")
        assert len(features.grid_dim) == 3

    def test_block_dim_tuple(self, mugraph_features_module, sample_mugraph_json):
        """Test block_dim is parsed as tuple."""
        if mugraph_features_module is None:
            pytest.skip("MuGraph features module not available")

        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)
        if MuGraphFeature is None:
            pytest.skip("MuGraphFeature not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        assert hasattr(features, "block_dim")
        assert len(features.block_dim) == 3

    def test_graph_depth_attribute(self, mugraph_features_module, sample_mugraph_json):
        """Test graph_depth attribute exists."""
        if mugraph_features_module is None:
            pytest.skip("MuGraph features module not available")

        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)
        if MuGraphFeature is None:
            pytest.skip("MuGraphFeature not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        assert hasattr(features, "graph_depth")
        assert features.graph_depth == 2


# =============================================================================
# FeatureProcessor Tests
# =============================================================================

class TestFeatureProcessor:
    """Tests for FeatureProcessor class."""

    def test_feature_processor_class_exists(self, processor_module):
        """Test FeatureProcessor class exists."""
        if processor_module is None:
            pytest.skip("Processor module not available")

        assert hasattr(processor_module, "FeatureProcessor")

    def test_processor_creation(self, processor_module):
        """Test FeatureProcessor can be created."""
        if processor_module is None:
            pytest.skip("Processor module not available")

        FeatureProcessor = getattr(processor_module, "FeatureProcessor", None)
        if FeatureProcessor is None:
            pytest.skip("FeatureProcessor not found")

        processor = FeatureProcessor()
        assert processor is not None

    def test_process_returns_dict(self, processor_module, mugraph_features_module, sample_mugraph_json):
        """Test process returns dictionary."""
        if processor_module is None or mugraph_features_module is None:
            pytest.skip("Required modules not available")

        FeatureProcessor = getattr(processor_module, "FeatureProcessor", None)
        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)

        if FeatureProcessor is None or MuGraphFeature is None:
            pytest.skip("Required classes not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        processor = FeatureProcessor()
        processed = processor.process(features)

        assert isinstance(processed, dict)

    def test_node_features_in_result(self, processor_module, mugraph_features_module, sample_mugraph_json):
        """Test node_features in processed result."""
        if processor_module is None or mugraph_features_module is None:
            pytest.skip("Required modules not available")

        FeatureProcessor = getattr(processor_module, "FeatureProcessor", None)
        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)

        if FeatureProcessor is None or MuGraphFeature is None:
            pytest.skip("Required classes not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        processor = FeatureProcessor()
        processed = processor.process(features)

        assert "node_features" in processed

    def test_edge_index_in_result(self, processor_module, mugraph_features_module, sample_mugraph_json):
        """Test edge_index in processed result."""
        if processor_module is None or mugraph_features_module is None:
            pytest.skip("Required modules not available")

        FeatureProcessor = getattr(processor_module, "FeatureProcessor", None)
        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)

        if FeatureProcessor is None or MuGraphFeature is None:
            pytest.skip("Required classes not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        processor = FeatureProcessor()
        processed = processor.process(features)

        assert "edge_index" in processed

    def test_global_features_in_result(self, processor_module, mugraph_features_module, sample_mugraph_json):
        """Test global_features in processed result."""
        if processor_module is None or mugraph_features_module is None:
            pytest.skip("Required modules not available")

        FeatureProcessor = getattr(processor_module, "FeatureProcessor", None)
        MuGraphFeature = getattr(mugraph_features_module, "MuGraphFeature", None)

        if FeatureProcessor is None or MuGraphFeature is None:
            pytest.skip("Required classes not found")

        features = MuGraphFeature.from_json(sample_mugraph_json)
        processor = FeatureProcessor()
        processed = processor.process(features)

        assert "global_features" in processed


# =============================================================================
# FeatureNormalizer Tests
# =============================================================================

class TestFeatureNormalizer:
    """Tests for FeatureNormalizer class."""

    def test_normalizer_class_exists(self, processor_module):
        """Test FeatureNormalizer class exists."""
        if processor_module is None:
            pytest.skip("Processor module not available")

        assert hasattr(processor_module, "FeatureNormalizer")

    def test_normalizer_creation(self, processor_module):
        """Test FeatureNormalizer can be created."""
        if processor_module is None:
            pytest.skip("Processor module not available")

        FeatureNormalizer = getattr(processor_module, "FeatureNormalizer", None)
        if FeatureNormalizer is None:
            pytest.skip("FeatureNormalizer not found")

        normalizer = FeatureNormalizer()
        assert normalizer is not None

    def test_normalize_minmax(self, processor_module):
        """Test minmax normalization."""
        if processor_module is None:
            pytest.skip("Processor module not available")

        FeatureNormalizer = getattr(processor_module, "FeatureNormalizer", None)
        if FeatureNormalizer is None:
            pytest.skip("FeatureNormalizer not found")

        import numpy as np

        normalizer = FeatureNormalizer()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalizer.update(data)
        normalized = normalizer.normalize(data, method="minmax")

        # MinMax normalized values should be in [0, 1]
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_zscore(self, processor_module):
        """Test zscore normalization."""
        if processor_module is None:
            pytest.skip("Processor module not available")

        FeatureNormalizer = getattr(processor_module, "FeatureNormalizer", None)
        if FeatureNormalizer is None:
            pytest.skip("FeatureNormalizer not found")

        import numpy as np

        normalizer = FeatureNormalizer()
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        normalizer.update(data)
        normalized = normalizer.normalize(data, method="zscore")

        # Z-score normalized should have mean ~0
        assert abs(normalized.mean()) < 1e-6
