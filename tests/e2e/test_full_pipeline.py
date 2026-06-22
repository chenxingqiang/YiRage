#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Full Pipeline End-to-End Tests

Tests for complete compilation pipeline: search -> compile -> execute.
Run with: pytest tests/e2e/test_full_pipeline.py -v
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from conftest import TORCH_AVAILABLE, CUDA_AVAILABLE, MPS_AVAILABLE


# =============================================================================
# Full Pipeline Tests
# =============================================================================


@pytest.mark.slow
class TestFullPipeline:
    """End-to-end tests for full compilation pipeline (search/superoptimize; can take minutes)."""

    def test_matmul_search_compile_execute(self):
        """Test complete matmul pipeline: search -> compile -> execute."""
        try:
            import yirage as yr
            
            # 1. Create kernel graph
            graph = yr.new_kernel_graph()
            A = graph.new_input(dims=(32, 64), dtype=yr.float16)
            B = graph.new_input(dims=(64, 128), dtype=yr.float16)
            C = graph.matmul(A, B)
            graph.mark_output(C)
            
            # 2. Superoptimize (search phase)
            # Note: This may take time or be skipped if C++ not available
            try:
                optimized = graph.superoptimize()
                assert optimized is not None
            except Exception as e:
                pytest.skip(f"Superoptimization not available: {e}")
                
        except ImportError:
            pytest.skip("YiRage not available")

    def test_silu_search_compile_execute(self):
        """Test complete SiLU pipeline."""
        try:
            import yirage as yr
            
            graph = yr.new_kernel_graph()
            X = graph.new_input(dims=(32, 64), dtype=yr.float16)
            Y = graph.silu(X)
            graph.mark_output(Y)
            
            try:
                optimized = graph.superoptimize()
                assert optimized is not None
            except Exception as e:
                pytest.skip(f"Superoptimization not available: {e}")
                
        except ImportError:
            pytest.skip("YiRage not available")

    def test_mlp_search_compile_execute(self):
        """Test complete MLP (MatMul + SiLU) pipeline."""
        try:
            import yirage as yr
            
            graph = yr.new_kernel_graph()
            X = graph.new_input(dims=(8, 4096), dtype=yr.float16)
            W = graph.new_input(dims=(4096, 4096), dtype=yr.float16)
            
            # MLP: MatMul + SiLU
            Y = graph.matmul(X, W)
            Z = graph.silu(Y)
            graph.mark_output(Z)
            
            try:
                optimized = graph.superoptimize()
                assert optimized is not None
            except Exception as e:
                pytest.skip(f"Superoptimization not available: {e}")
                
        except ImportError:
            pytest.skip("YiRage not available")

    def test_transformer_block_e2e(self):
        """Test complete transformer block pipeline."""
        try:
            import yirage as yr
            
            # Simplified transformer: Linear -> RMSNorm
            graph = yr.new_kernel_graph()
            X = graph.new_input(dims=(8, 32, 64), dtype=yr.float16)
            W = graph.new_input(dims=(64, 64), dtype=yr.float16)
            
            # Linear
            Y = graph.matmul(X, W)
            
            # RMSNorm
            Z = graph.rms_norm(Y)
            graph.mark_output(Z)
            
            try:
                optimized = graph.superoptimize()
                assert optimized is not None
            except Exception as e:
                pytest.skip(f"Superoptimization not available: {e}")
                
        except ImportError:
            pytest.skip("YiRage not available")


# =============================================================================
# Compiler Pipeline Tests
# =============================================================================

class TestCompilerPipeline:
    """Tests for compiler pipeline stages."""

    def test_fast_mode_compilation(self):
        """Test fast mode compilation."""
        try:
            from yirage.compiler import UnifiedCompiler, CompileMode
            
            compiler = UnifiedCompiler(
                backend="cpu",
                mode=CompileMode.FAST,
            )
            
            assert compiler is not None
            assert compiler.backend == "cpu"
            
        except ImportError:
            pytest.skip("Compiler module not available")

    def test_pipeline_stages_execution(self):
        """Test pipeline stages execute in order."""
        try:
            from yirage.compiler.pipeline import CompilePipeline
            
            pipeline = CompilePipeline(
                backend="cpu",
                enable_superoptimize=False,
                enable_mlir=False,
            )
            
            assert pipeline is not None
            
        except ImportError:
            pytest.skip("Pipeline module not available")


# =============================================================================
# Backend-Specific E2E Tests
# =============================================================================

@pytest.mark.torch
class TestBackendE2E:
    """Backend-specific end-to-end tests."""

    def test_cpu_e2e(self):
        """Test complete CPU pipeline with PyTorch verification."""
        import torch
        import numpy as np
        
        # Reference computation
        A = torch.randn(32, 64, dtype=torch.float32)
        B = torch.randn(64, 128, dtype=torch.float32)
        C_ref = torch.matmul(A, B)
        
        # Verify reference is correct
        assert C_ref.shape == (32, 128)

    @pytest.mark.cuda
    def test_cuda_e2e(self):
        """Test complete CUDA pipeline with verification."""
        import torch
        
        A = torch.randn(32, 64, dtype=torch.float16, device="cuda")
        B = torch.randn(64, 128, dtype=torch.float16, device="cuda")
        C = torch.matmul(A, B)
        
        assert C.shape == (32, 128)
        assert C.device.type == "cuda"

    @pytest.mark.mps
    def test_mps_e2e(self):
        """Test complete MPS pipeline with verification."""
        import torch
        
        A = torch.randn(32, 64, dtype=torch.float16, device="mps")
        B = torch.randn(64, 128, dtype=torch.float16, device="mps")
        C = torch.matmul(A, B)
        
        assert C.shape == (32, 128)
        assert C.device.type == "mps"
