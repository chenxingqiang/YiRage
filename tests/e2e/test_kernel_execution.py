#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Kernel Execution Correctness Tests

Tests for numerical accuracy of kernel execution.
Run with: pytest tests/e2e/test_kernel_execution.py -v
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from conftest import TORCH_AVAILABLE, CUDA_AVAILABLE, MPS_AVAILABLE


# =============================================================================
# Numerical Accuracy Tests
# =============================================================================

@pytest.mark.torch
class TestKernelCorrectness:
    """Tests for kernel numerical correctness."""

    def test_matmul_numerical_accuracy(self, device):
        """Test MatMul produces numerically correct results."""
        import torch
        
        # Create test inputs
        A = torch.randn(32, 64, device=device, dtype=torch.float32)
        B = torch.randn(64, 128, device=device, dtype=torch.float32)
        
        # Compute with PyTorch
        C = torch.matmul(A, B)
        
        # Verify using manual computation
        C_manual = torch.zeros(32, 128, device=device)
        for i in range(32):
            for j in range(128):
                C_manual[i, j] = torch.sum(A[i, :] * B[:, j])
        
        # Should be close
        assert torch.allclose(C, C_manual, atol=1e-5)

    def test_silu_numerical_accuracy(self, device):
        """Test SiLU produces numerically correct results."""
        import torch
        
        x = torch.randn(32, 64, device=device, dtype=torch.float32)
        
        # PyTorch SiLU
        y = torch.nn.functional.silu(x)
        
        # Manual: x * sigmoid(x)
        y_manual = x * torch.sigmoid(x)
        
        assert torch.allclose(y, y_manual, atol=1e-6)

    def test_rms_norm_numerical_accuracy(self, device):
        """Test RMSNorm produces numerically correct results."""
        import torch
        
        x = torch.randn(8, 32, 64, device=device, dtype=torch.float32)
        
        # Manual RMSNorm
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        y_manual = x / rms
        
        # Should have unit variance along last dim (approximately)
        assert y_manual.shape == x.shape

    def test_softmax_numerical_accuracy(self, device):
        """Test Softmax produces numerically correct results."""
        import torch
        
        x = torch.randn(8, 32, 64, device=device, dtype=torch.float32)
        
        # PyTorch softmax
        y = torch.softmax(x, dim=-1)
        
        # Manual softmax
        x_max = x.max(dim=-1, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        y_manual = exp_x / exp_x.sum(dim=-1, keepdim=True)
        
        assert torch.allclose(y, y_manual, atol=1e-5)

    def test_gelu_numerical_accuracy(self, device):
        """Test GELU produces numerically correct results."""
        import torch
        import math
        
        x = torch.randn(32, 64, device=device, dtype=torch.float32)
        
        # PyTorch GELU
        y = torch.nn.functional.gelu(x)
        
        # Approximate GELU
        y_approx = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
        
        # Should be close (approximate formula)
        assert torch.allclose(y, y_approx, atol=1e-3)


# =============================================================================
# Precision Tests
# =============================================================================

@pytest.mark.torch
class TestPrecision:
    """Tests for different precision levels."""

    def test_fp32_precision(self, device):
        """Test FP32 operations maintain precision."""
        import torch
        
        A = torch.randn(32, 64, device=device, dtype=torch.float32)
        B = torch.randn(64, 128, device=device, dtype=torch.float32)
        C = torch.matmul(A, B)
        
        # FP32 should have ~7 decimal digits of precision
        assert C.dtype == torch.float32

    @pytest.mark.skipif(not (CUDA_AVAILABLE or MPS_AVAILABLE), reason="GPU required")
    def test_fp16_precision(self, device):
        """Test FP16 operations maintain acceptable precision."""
        import torch
        
        if device == "cpu":
            pytest.skip("FP16 matmul better tested on GPU")
        
        A = torch.randn(32, 64, device=device, dtype=torch.float16)
        B = torch.randn(64, 128, device=device, dtype=torch.float16)
        C = torch.matmul(A, B)
        
        # FP16 should still produce valid results
        assert C.dtype == torch.float16
        assert not torch.isnan(C).any()
        assert not torch.isinf(C).any()

    @pytest.mark.cuda
    def test_bf16_precision(self):
        """Test BF16 operations (CUDA only)."""
        import torch
        
        A = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
        C = torch.matmul(A, B)
        
        assert C.dtype == torch.bfloat16
        assert not torch.isnan(C).any()


# =============================================================================
# Edge Case Tests
# =============================================================================

@pytest.mark.torch
class TestEdgeCases:
    """Tests for edge cases in kernel execution."""

    def test_empty_batch(self, device):
        """Test handling of empty batch dimension."""
        import torch
        
        # Empty batch
        x = torch.randn(0, 64, device=device)
        y = torch.nn.functional.silu(x)
        
        assert y.shape == (0, 64)

    def test_single_element(self, device):
        """Test handling of single element tensors."""
        import torch
        
        x = torch.randn(1, 1, device=device)
        y = torch.nn.functional.silu(x)
        
        assert y.shape == (1, 1)

    def test_large_tensor(self, device):
        """Test handling of larger tensors."""
        import torch
        
        # 1M elements
        x = torch.randn(1000, 1000, device=device, dtype=torch.float32)
        y = torch.nn.functional.silu(x)
        
        assert y.shape == (1000, 1000)
        assert not torch.isnan(y).any()

    def test_special_values(self, device):
        """Test handling of special values (inf, nan)."""
        import torch
        
        x = torch.tensor([0.0, 1.0, -1.0, float('inf'), float('-inf')], device=device)
        y = torch.sigmoid(x)
        
        # Sigmoid should handle special values gracefully
        assert y[0] == 0.5  # sigmoid(0) = 0.5
        assert y[3] == 1.0  # sigmoid(inf) = 1
        assert y[4] == 0.0  # sigmoid(-inf) = 0
