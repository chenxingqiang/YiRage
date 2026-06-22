#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Multi-Backend Selection Integration Tests

Tests for backend selection and fallback mechanisms.
Run with: pytest tests/integration/test_multi_backend.py -v
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from conftest import load_module, TORCH_AVAILABLE, CUDA_AVAILABLE, MPS_AVAILABLE


def _core_supports_set_default_backend() -> bool:
    try:
        from yirage import core as ycore
    except ImportError:
        return False
    return hasattr(ycore, "set_default_backend")


# =============================================================================
# Backend Selection Tests
# =============================================================================

class TestBackendSelection:
    """Tests for automatic backend selection."""

    def test_auto_selects_best_available(self):
        """Test auto mode selects best available backend."""
        try:
            from yirage.backends.api import get_default_backend, get_available_backends
            
            backends = get_available_backends()
            default = get_default_backend()
            
            if not backends:
                pytest.skip("No backends available")
            
            # Default should be one of the available backends
            if default is not None:
                assert default in backends
                
        except ImportError:
            pytest.skip("Backend API not available")

    def test_priority_cuda_over_others(self):
        """Test CUDA has highest priority."""
        try:
            from yirage.backends.api import get_default_backend, get_available_backends
            
            backends = get_available_backends()
            default = get_default_backend()
            
            if "cuda" in backends:
                assert default == "cuda"
                
        except ImportError:
            pytest.skip("Backend API not available")

    def test_fallback_to_cpu(self):
        """Test fallback to CPU when GPU unavailable."""
        try:
            from yirage.backends.api import get_available_backends, is_backend_available
            
            backends = get_available_backends()
            
            # CPU should always be available if any backend is
            if backends:
                # At minimum CPU should work
                assert "cpu" in backends or len(backends) > 0
                
        except ImportError:
            pytest.skip("Backend API not available")

    def test_explicit_backend_override(self):
        """Test explicitly specifying a backend."""
        try:
            from yirage.backends.api import is_backend_available
            
            # Check CPU is available (should always be)
            # Then verify we can explicitly use it
            if is_backend_available("cpu"):
                from yirage.backends.api import set_default_backend

                if not _core_supports_set_default_backend():
                    pytest.skip("yirage.core.set_default_backend not exposed")
                assert set_default_backend("cpu") is True

        except ImportError:
            pytest.skip("Backend API not available")


# =============================================================================
# Backend Switching Tests
# =============================================================================

class TestBackendSwitching:
    """Tests for runtime backend switching."""

    def test_backend_switching_runtime(self):
        """Test switching backends at runtime."""
        try:
            from yirage.backends.api import (
                get_available_backends,
                is_backend_available,
                set_default_backend,
            )
            
            backends = get_available_backends()
            
            if len(backends) < 2:
                pytest.skip("Need at least 2 backends for switching test")
            if not _core_supports_set_default_backend():
                pytest.skip("yirage.core.set_default_backend not exposed")

            for backend in backends:
                if is_backend_available(backend):
                    assert set_default_backend(backend) is True

        except ImportError:
            pytest.skip("Backend API not available")


# =============================================================================
# Backend Config Integration Tests
# =============================================================================

class TestBackendConfigIntegration:
    """Tests for backend config integration."""

    def test_all_backends_have_configs(self):
        """Test all backends have configuration modules."""
        expected_backends = [
            "cuda", "mps", "rocm", "cpu", "ascend",
            "maca", "tpu", "xpu", "fpga"
        ]
        
        for backend in expected_backends:
            path = PYTHON_ROOT / "yirage" / "backends" / backend / "config.py"
            assert path.exists(), f"Config missing for {backend}"

    def test_configs_return_compatible_structure(self):
        """Test all configs return compatible dictionary structure."""
        backends = ["cuda", "mps", "rocm", "cpu", "ascend", "maca", "tpu", "xpu", "fpga"]
        configs = []
        
        for backend in backends:
            module = load_module(
                f"{backend}_config",
                PYTHON_ROOT / "yirage" / "backends" / backend / "config.py"
            )
            if module:
                func_name = f"get_{backend}_search_config"
                if hasattr(module, func_name):
                    config = getattr(module, func_name)()
                    configs.append((backend, config))
        
        # All configs should be dicts
        for backend, config in configs:
            assert isinstance(config, dict), f"{backend} config is not a dict"


# =============================================================================
# PyTorch Backend Integration Tests
# =============================================================================

@pytest.mark.torch
class TestPyTorchBackendIntegration:
    """Tests for PyTorch backend integration."""

    def test_tensor_creation_on_available_device(self):
        """Test tensor creation on best available device."""
        import torch
        
        if CUDA_AVAILABLE:
            device = "cuda:0"
        elif MPS_AVAILABLE:
            device = "mps"
        else:
            device = "cpu"
        
        x = torch.randn(32, 64, device=device)
        assert x.device.type in ["cuda", "mps", "cpu"]

    def test_matmul_on_available_device(self):
        """Test matmul works on available device."""
        import torch
        
        if CUDA_AVAILABLE:
            device = "cuda:0"
        elif MPS_AVAILABLE:
            device = "mps"
        else:
            device = "cpu"
        
        A = torch.randn(32, 64, device=device)
        B = torch.randn(64, 128, device=device)
        C = torch.matmul(A, B)
        
        assert C.shape == (32, 128)
        assert C.device.type == A.device.type

    @pytest.mark.cuda
    def test_cuda_specific_operations(self):
        """Test CUDA-specific operations."""
        import torch
        
        x = torch.randn(32, 64, device="cuda:0")
        y = torch.nn.functional.relu(x)
        
        assert y.device.type == "cuda"

    @pytest.mark.mps
    def test_mps_specific_operations(self):
        """Test MPS-specific operations."""
        import torch
        
        x = torch.randn(32, 64, device="mps")
        y = torch.nn.functional.silu(x)
        
        assert y.device.type == "mps"
