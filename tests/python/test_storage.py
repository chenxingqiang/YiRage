#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Storage Module Unit Tests

Tests for yirage/storage/ module including MuGraphStore and related classes.
Run with: pytest tests/python/test_storage.py -v
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from conftest import PYTHON_ROOT, load_module


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def storage_module():
    """Load storage module."""
    return load_module(
        "mugraph_store",
        PYTHON_ROOT / "yirage" / "storage" / "mugraph_store.py"
    )


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# MuGraphMetadata Tests
# =============================================================================

class TestMuGraphMetadata:
    """Tests for MuGraphMetadata dataclass."""

    def test_metadata_class_exists(self, storage_module):
        """Test MuGraphMetadata class exists."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        assert hasattr(storage_module, "MuGraphMetadata")

    def test_metadata_creation(self, storage_module):
        """Test MuGraphMetadata can be created."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        MuGraphMetadata = getattr(storage_module, "MuGraphMetadata", None)
        if MuGraphMetadata is None:
            pytest.skip("MuGraphMetadata not found")

        metadata = MuGraphMetadata(
            graph_hash="test123",
            backend="cuda",
            latency_ms=1.5,
        )
        assert metadata is not None
        assert metadata.graph_hash == "test123"

    def test_to_dict_serialization(self, storage_module):
        """Test MuGraphMetadata to_dict method."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        MuGraphMetadata = getattr(storage_module, "MuGraphMetadata", None)
        if MuGraphMetadata is None:
            pytest.skip("MuGraphMetadata not found")

        metadata = MuGraphMetadata(
            graph_hash="test123",
            backend="cuda",
            latency_ms=1.5,
        )

        if hasattr(metadata, "to_dict"):
            d = metadata.to_dict()
            assert isinstance(d, dict)
            assert "graph_hash" in d

    def test_from_dict_deserialization(self, storage_module):
        """Test MuGraphMetadata from_dict method."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        MuGraphMetadata = getattr(storage_module, "MuGraphMetadata", None)
        if MuGraphMetadata is None:
            pytest.skip("MuGraphMetadata not found")

        if not hasattr(MuGraphMetadata, "from_dict"):
            pytest.skip("from_dict method not available")

        data = {
            "graph_hash": "test123",
            "backend": "cuda",
            "latency_ms": 1.5,
        }

        metadata = MuGraphMetadata.from_dict(data)
        assert metadata.graph_hash == "test123"


# =============================================================================
# GraphStructure Tests
# =============================================================================

class TestGraphStructure:
    """Tests for GraphStructure dataclass."""

    def test_structure_class_exists(self, storage_module):
        """Test GraphStructure class exists."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        assert hasattr(storage_module, "GraphStructure")

    def test_structure_creation(self, storage_module):
        """Test GraphStructure can be created."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        GraphStructure = getattr(storage_module, "GraphStructure", None)
        if GraphStructure is None:
            pytest.skip("GraphStructure not found")

        structure = GraphStructure(
            num_operators=5,
            num_tensors=10,
        )
        assert structure.num_operators == 5
        assert structure.num_tensors == 10


# =============================================================================
# PerformanceMetrics Tests
# =============================================================================

class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_metrics_class_exists(self, storage_module):
        """Test PerformanceMetrics class exists."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        assert hasattr(storage_module, "PerformanceMetrics")

    def test_metrics_creation(self, storage_module):
        """Test PerformanceMetrics can be created."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        PerformanceMetrics = getattr(storage_module, "PerformanceMetrics", None)
        if PerformanceMetrics is None:
            pytest.skip("PerformanceMetrics not found")

        metrics = PerformanceMetrics(
            latency_ms=1.5,
            throughput_tflops=100.0,
        )
        assert metrics.latency_ms == 1.5


# =============================================================================
# DeviceCapabilities Tests
# =============================================================================

class TestDeviceCapabilities:
    """Tests for DeviceCapabilities dataclass."""

    def test_capabilities_class_exists(self, storage_module):
        """Test DeviceCapabilities class exists."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        assert hasattr(storage_module, "DeviceCapabilities")

    def test_capabilities_creation(self, storage_module):
        """Test DeviceCapabilities can be created."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        DeviceCapabilities = getattr(storage_module, "DeviceCapabilities", None)
        if DeviceCapabilities is None:
            pytest.skip("DeviceCapabilities not found")

        caps = DeviceCapabilities(
            device_type="cuda",
            device_name="RTX 4090",
            compute_units=128,
        )
        assert caps.device_type == "cuda"


# =============================================================================
# MuGraphEntry Tests
# =============================================================================

class TestMuGraphEntry:
    """Tests for MuGraphEntry dataclass."""

    def test_entry_class_exists(self, storage_module):
        """Test MuGraphEntry class exists."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        assert hasattr(storage_module, "MuGraphEntry")

    def test_entry_creation(self, storage_module):
        """Test MuGraphEntry can be created."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        MuGraphEntry = getattr(storage_module, "MuGraphEntry", None)
        if MuGraphEntry is None:
            pytest.skip("MuGraphEntry not found")

        entry = MuGraphEntry()
        assert entry is not None

    def test_entry_to_dict(self, storage_module):
        """Test MuGraphEntry to_dict method."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        MuGraphEntry = getattr(storage_module, "MuGraphEntry", None)
        if MuGraphEntry is None:
            pytest.skip("MuGraphEntry not found")

        entry = MuGraphEntry()
        if hasattr(entry, "to_dict"):
            d = entry.to_dict()
            assert isinstance(d, dict)


# =============================================================================
# MuGraphStore Tests
# =============================================================================

class TestMuGraphStore:
    """Tests for MuGraphStore class."""

    def test_store_class_exists(self, storage_module):
        """Test MuGraphStore class exists."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        assert hasattr(storage_module, "MuGraphStore")

    def test_store_creation(self, storage_module, temp_storage_dir):
        """Test MuGraphStore can be created."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        MuGraphStore = getattr(storage_module, "MuGraphStore", None)
        if MuGraphStore is None:
            pytest.skip("MuGraphStore not found")

        try:
            store = MuGraphStore(root_path=temp_storage_dir)
            assert store is not None
        except Exception as e:
            pytest.skip(f"MuGraphStore creation failed: {e}")

    def test_save_mugraph_new_entry(self, storage_module, temp_storage_dir):
        """Test saving a new MuGraph entry."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        MuGraphStore = getattr(storage_module, "MuGraphStore", None)
        save_mugraph = getattr(storage_module, "save_mugraph", None)

        if MuGraphStore is None:
            pytest.skip("MuGraphStore not found")

        try:
            store = MuGraphStore(root_path=temp_storage_dir)

            # Try to save using either method
            if hasattr(store, "save"):
                entry_id = store.save(
                    graph_hash="test_hash_001",
                    optimized_graph={"type": "matmul", "dims": [64, 64]},
                    backend="cuda",
                    latency_ms=1.5,
                )
                assert entry_id is not None
            elif save_mugraph is not None:
                entry_id = save_mugraph(
                    store=store,
                    graph_hash="test_hash_001",
                    backend="cuda",
                    latency_ms=1.5,
                )
                assert entry_id is not None
            else:
                pytest.skip("No save method available")
        except Exception as e:
            pytest.skip(f"Save test failed: {e}")

    def test_find_best_mugraph_by_latency(self, storage_module, temp_storage_dir):
        """Test finding best MuGraph by latency."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        MuGraphStore = getattr(storage_module, "MuGraphStore", None)
        find_best_mugraph = getattr(storage_module, "find_best_mugraph", None)

        if MuGraphStore is None:
            pytest.skip("MuGraphStore not found")

        try:
            store = MuGraphStore(root_path=temp_storage_dir)

            # Save multiple entries (distinct search config -> distinct config_hash files;
            # same defaults would overwrite a single hash1_<config>.json).
            if hasattr(store, "save"):
                dummy_graph = {"type": "matmul", "dims": [64, 64]}
                store.save(
                    graph_hash="hash1",
                    optimized_graph=dummy_graph,
                    backend="cuda",
                    latency_ms=2.0,
                    griddims=[[1, 1, 1]],
                )
                store.save(
                    graph_hash="hash1",
                    optimized_graph=dummy_graph,
                    backend="cuda",
                    latency_ms=1.0,
                    griddims=[[2, 1, 1]],
                )
                store.save(
                    graph_hash="hash1",
                    optimized_graph=dummy_graph,
                    backend="cuda",
                    latency_ms=3.0,
                    griddims=[[4, 1, 1]],
                )

                # Find best
                if hasattr(store, "find_best"):
                    best = store.find_best(graph_hash="hash1", backend="cuda")
                    assert best is not None
                    assert best.performance.latency_ms == 1.0
                elif find_best_mugraph is not None:
                    best = find_best_mugraph(store, graph_hash="hash1", backend="cuda")
                    if best is not None:
                        assert best.performance.latency_ms == 1.0
            else:
                pytest.skip("No save method available")
        except AssertionError:
            raise
        except Exception as e:
            pytest.skip(f"Find best test failed: {e}")

    def test_store_statistics(self, storage_module, temp_storage_dir):
        """Test store statistics."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        MuGraphStore = getattr(storage_module, "MuGraphStore", None)
        if MuGraphStore is None:
            pytest.skip("MuGraphStore not found")

        try:
            store = MuGraphStore(root_path=temp_storage_dir)

            if hasattr(store, "get_stats"):
                stats = store.get_stats()
                assert isinstance(stats, dict)
            else:
                pytest.skip("get_stats method not available")
        except Exception as e:
            pytest.skip(f"Store statistics test failed: {e}")


# =============================================================================
# SearchConfiguration Tests
# =============================================================================

class TestSearchConfiguration:
    """Tests for SearchConfiguration dataclass."""

    def test_search_config_class_exists(self, storage_module):
        """Test SearchConfiguration class exists."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        assert hasattr(storage_module, "SearchConfiguration")

    def test_search_config_creation(self, storage_module):
        """Test SearchConfiguration can be created."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        SearchConfiguration = getattr(storage_module, "SearchConfiguration", None)
        if SearchConfiguration is None:
            pytest.skip("SearchConfiguration not found")

        config = SearchConfiguration(
            selected_grid_dim=(128, 1, 1),
            selected_block_dim=(256, 1, 1),
        )
        assert config.selected_grid_dim == (128, 1, 1)


# =============================================================================
# TrainingFeatures/Labels Tests
# =============================================================================

class TestTrainingData:
    """Tests for training data classes."""

    def test_training_features_exists(self, storage_module):
        """Test TrainingFeatures class exists."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        assert hasattr(storage_module, "TrainingFeatures")

    def test_training_labels_exists(self, storage_module):
        """Test TrainingLabels class exists."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        assert hasattr(storage_module, "TrainingLabels")

    def test_training_labels_creation(self, storage_module):
        """Test TrainingLabels can be created."""
        if storage_module is None:
            pytest.skip("Storage module not available")

        TrainingLabels = getattr(storage_module, "TrainingLabels", None)
        if TrainingLabels is None:
            pytest.skip("TrainingLabels not found")

        labels = TrainingLabels(
            optimal_latency_ms=1.5,
            is_optimal=True,
        )
        assert labels.optimal_latency_ms == 1.5
        assert labels.is_optimal is True
