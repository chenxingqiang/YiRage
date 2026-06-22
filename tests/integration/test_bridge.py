#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Python-C++ Bridge Integration Tests

Tests for the Cython bridge between Python and C++ components.
Run with: pytest tests/integration/test_bridge.py -v
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))


# =============================================================================
# Core Bridge Tests
# =============================================================================

class TestCoreBridge:
    """Tests for Python-C++ bridge functionality."""

    def test_cython_import_successful(self):
        """Test that Cython core module can be imported."""
        try:
            import yirage.core as _core

            assert _core is not None
        except ImportError as e:
            # C++ module may not be built
            pytest.skip(f"C++ core module not available: {e}")

    def test_yirage_import(self):
        """Test that yirage package can be imported."""
        try:
            import yirage
            assert hasattr(yirage, "__version__")
        except ImportError as e:
            pytest.skip(f"YiRage not available: {e}")

    def test_core_module_has_search(self):
        """Test core module exposes at least one search-related entry point."""
        try:
            from yirage import core

            has_search = (
                hasattr(core, "search")
                or hasattr(core, "cython_search")
                or hasattr(core, "KernelGraphGenerator")
                or hasattr(core, "superoptimize")
            )
            assert has_search, (
                "Expected core to expose search, cython_search, KernelGraphGenerator, or superoptimize"
            )
        except ImportError:
            pytest.skip("C++ core module not available")

    def test_error_propagation_from_cpp(self):
        """Incompatible matmul raises ValueError (Python precheck and/or C++ invalid_argument)."""
        try:
            import yirage as yr
        except ImportError:
            pytest.skip("YiRage not available")

        graph = yr.new_kernel_graph()
        A = graph.new_input(dims=(32, 64), dtype=yr.float16)
        B = graph.new_input(dims=(128, 32), dtype=yr.float16)
        with pytest.raises(ValueError, match="matmul"):
            graph.matmul(A, B)

    def test_graph_serialization_roundtrip(self, tmp_path):
        """Test graph can be serialized to JSON (KNGraph.to_json writes to a file)."""
        import json

        try:
            import yirage as yr
        except ImportError:
            pytest.skip("YiRage not available")

        graph = yr.new_kernel_graph()
        A = graph.new_input(dims=(32, 64), dtype=yr.float16)
        B = graph.new_input(dims=(64, 128), dtype=yr.float16)
        C = graph.matmul(A, B)
        graph.mark_output(C)

        if not hasattr(graph, "to_json"):
            pytest.skip("Graph has no to_json")

        out = tmp_path / "graph.json"
        try:
            graph.to_json(str(out))
        except TypeError as e:
            pytest.skip(f"Graph to_json signature unsupported in this build: {e}")

        text = out.read_text(encoding="utf-8")
        assert text.strip()
        data = json.loads(text)
        assert isinstance(data, (dict, list))
        assert len(data) > 0


# =============================================================================
# Backend Bridge Tests
# =============================================================================

class TestBackendBridge:
    """Tests for backend API bridge."""

    def test_backend_api_bridge(self):
        """Test backend API is accessible from Python."""
        try:
            import yirage as yr
            
            backends = yr.get_available_backends()
            assert isinstance(backends, list)
            
        except ImportError:
            pytest.skip("YiRage not available")

    def test_backend_selection(self):
        """Test backend selection through bridge."""
        try:
            import yirage as yr
            
            default = yr.get_default_backend()
            # Should return string or None
            assert default is None or isinstance(default, str)
            
        except ImportError:
            pytest.skip("YiRage not available")


# =============================================================================
# Kernel Graph Bridge Tests
# =============================================================================

class TestKernelGraphBridge:
    """Tests for kernel graph bridge."""

    def test_new_kernel_graph_creates_object(self):
        """Test new_kernel_graph creates valid object."""
        try:
            import yirage as yr
            
            graph = yr.new_kernel_graph()
            assert graph is not None
            
        except ImportError:
            pytest.skip("YiRage not available")

    def test_graph_input_creation(self):
        """Test creating inputs on graph."""
        try:
            import yirage as yr
            
            graph = yr.new_kernel_graph()
            tensor = graph.new_input(dims=(32, 64), dtype=yr.float16)
            assert tensor is not None
            
        except ImportError:
            pytest.skip("YiRage not available")

    def test_graph_operations(self):
        """Test graph operations work through bridge."""
        try:
            import yirage as yr
            
            graph = yr.new_kernel_graph()
            A = graph.new_input(dims=(32, 64), dtype=yr.float16)
            B = graph.new_input(dims=(64, 128), dtype=yr.float16)
            C = graph.matmul(A, B)
            
            assert C is not None
            
        except ImportError:
            pytest.skip("YiRage not available")
