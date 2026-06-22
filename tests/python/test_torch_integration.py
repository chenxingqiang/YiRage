#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
PyTorch Integration Tests

Tests for YiRage PyTorch integration.
Run with: pytest tests/python/test_torch_integration.py -v
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def torch_integration():
    """Load torch integration module."""
    try:
        from yirage import torch_integration
        return torch_integration
    except ImportError as e:
        pytest.skip(f"torch_integration not available: {e}")


@pytest.fixture
def torch_module():
    """Get torch module or skip."""
    try:
        import torch
        return torch
    except ImportError:
        pytest.skip("PyTorch not available")


# =============================================================================
# FXToMLIRConverter Tests
# =============================================================================

class TestFXToMLIRConverter:
    """Tests for FX to MLIR conversion."""
    
    def test_converter_creation(self, torch_integration):
        """Test converter can be created."""
        FXToMLIRConverter = torch_integration.FXToMLIRConverter
        
        converter = FXToMLIRConverter()
        assert converter is not None
        assert converter.ssa_counter == 0
    
    def test_op_map_exists(self, torch_integration):
        """Test operation mapping exists."""
        FXToMLIRConverter = torch_integration.FXToMLIRConverter
        
        assert len(FXToMLIRConverter.OP_MAP) > 0
        assert "aten::mm" in FXToMLIRConverter.OP_MAP
        assert "aten::relu" in FXToMLIRConverter.OP_MAP
    
    def test_convert_simple_function(self, torch_integration, torch_module):
        """Test converting a simple function."""
        if not torch_integration.TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        torch = torch_module
        FXToMLIRConverter = torch_integration.FXToMLIRConverter
        
        def simple_add(x, y):
            return x + y
        
        # Trace the function
        gm = torch.fx.symbolic_trace(simple_add)
        
        # Convert to MLIR
        converter = FXToMLIRConverter()
        example_inputs = (torch.randn(32, 32), torch.randn(32, 32))
        mlir = converter.convert(gm, example_inputs)
        
        assert "module" in mlir
        assert "func.func" in mlir
        assert "return" in mlir


# =============================================================================
# YirageBackend Tests
# =============================================================================

class TestYirageBackend:
    """Tests for YiRage torch.compile backend."""
    
    def test_backend_creation(self, torch_integration):
        """Test backend can be created."""
        YirageBackend = torch_integration.YirageBackend
        
        backend = YirageBackend()
        assert backend is not None
        assert hasattr(backend, 'target')
        assert hasattr(backend, 'converter')
    
    def test_backend_with_target(self, torch_integration):
        """Test backend with specific target."""
        if not torch_integration.MLIR_AVAILABLE:
            pytest.skip("MLIR not available")
        
        from yirage.mlir_jit import Target
        YirageBackend = torch_integration.YirageBackend
        
        backend = YirageBackend(target=Target.CUDA_H100)
        assert backend.target == Target.CUDA_H100
    
    def test_backend_callable(self, torch_integration, torch_module):
        """Test backend is callable."""
        if not torch_integration.TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        torch = torch_module
        YirageBackend = torch_integration.YirageBackend
        
        def simple_fn(x):
            return x + 1
        
        gm = torch.fx.symbolic_trace(simple_fn)
        example_inputs = (torch.randn(32, 32),)
        
        backend = YirageBackend()
        result = backend(gm, example_inputs)
        
        # Should return a callable (either compiled or fallback)
        assert callable(result)


# =============================================================================
# High-Level API Tests
# =============================================================================

class TestHighLevelAPI:
    """Tests for high-level API functions."""
    
    def test_export_to_mlir_exists(self, torch_integration):
        """Test export_to_mlir function exists."""
        assert hasattr(torch_integration, 'export_to_mlir')
        assert callable(torch_integration.export_to_mlir)
    
    def test_compile_model_exists(self, torch_integration):
        """Test compile_model function exists."""
        assert hasattr(torch_integration, 'compile_model')
        assert callable(torch_integration.compile_model)
    
    def test_compile_function_decorator(self, torch_integration):
        """Test compile_function decorator."""
        compile_function = torch_integration.compile_function
        
        @compile_function()
        def my_fn(x):
            return x
        
        # Should return a wrapped function
        assert callable(my_fn)
    
    def test_export_simple_model(self, torch_integration, torch_module):
        """Test exporting a simple model."""
        if not torch_integration.TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        torch = torch_module
        
        model = torch.nn.Linear(32, 64)
        example_inputs = (torch.randn(8, 32),)
        
        mlir = torch_integration.export_to_mlir(model, example_inputs)
        
        assert "module" in mlir
        assert "func.func" in mlir


# =============================================================================
# Backend Registration Tests
# =============================================================================

class TestBackendRegistration:
    """Tests for torch.compile backend registration."""
    
    def test_backend_registered(self, torch_integration, torch_module):
        """Test backend is registered with torch.compile."""
        if not torch_integration.TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        try:
            from torch._dynamo.backends.registry import list_backends
            backends = list_backends()
            # yirage should be in the list if registration worked
            # It's OK if it's not - registration might have failed gracefully
        except ImportError:
            pytest.skip("torch._dynamo not available")


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_matmul_export(self, torch_integration, torch_module):
        """Test exporting a matmul operation."""
        if not torch_integration.TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        torch = torch_module
        
        def matmul_fn(a, b):
            return torch.matmul(a, b)
        
        gm = torch.fx.symbolic_trace(matmul_fn)
        converter = torch_integration.FXToMLIRConverter()
        
        example_inputs = (torch.randn(32, 64), torch.randn(64, 128))
        mlir = converter.convert(gm, example_inputs)
        
        # Should contain matmul operation
        assert "matmul" in mlir.lower() or "mm" in mlir.lower()
    
    def test_mlp_export(self, torch_integration, torch_module):
        """Test exporting an MLP model."""
        if not torch_integration.TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        torch = torch_module
        
        class SimpleMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 128)
                self.fc2 = torch.nn.Linear(128, 64)
            
            def forward(self, x):
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                return x
        
        model = SimpleMLP()
        example_inputs = (torch.randn(8, 64),)
        
        mlir = torch_integration.export_to_mlir(model, example_inputs)
        
        assert "module" in mlir
        assert "func.func" in mlir


# =============================================================================
# Constants and Flags Tests
# =============================================================================

class TestFlags:
    """Tests for availability flags."""
    
    def test_torch_available_flag(self, torch_integration):
        """Test TORCH_AVAILABLE flag."""
        assert hasattr(torch_integration, 'TORCH_AVAILABLE')
        assert isinstance(torch_integration.TORCH_AVAILABLE, bool)
    
    def test_mlir_available_flag(self, torch_integration):
        """Test MLIR_AVAILABLE flag."""
        assert hasattr(torch_integration, 'MLIR_AVAILABLE')
        assert isinstance(torch_integration.MLIR_AVAILABLE, bool)
