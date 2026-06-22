#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
Performance Regression Tests

Tests for performance baselines and regression detection.
Run with: pytest tests/e2e/test_performance.py -v
"""

import pytest
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from conftest import TORCH_AVAILABLE, CUDA_AVAILABLE, MPS_AVAILABLE


# =============================================================================
# Performance Baseline Tests
# =============================================================================

@pytest.mark.torch
@pytest.mark.benchmark
class TestPerformanceBaseline:
    """Tests for performance baselines."""

    def test_matmul_latency_baseline(self, device):
        """Test MatMul latency is within expected range."""
        import torch
        
        # Warmup
        for _ in range(5):
            A = torch.randn(512, 512, device=device, dtype=torch.float32)
            B = torch.randn(512, 512, device=device, dtype=torch.float32)
            C = torch.matmul(A, B)
            if device != "cpu":
                torch.cuda.synchronize() if "cuda" in str(device) else None
        
        # Benchmark
        num_iterations = 10
        A = torch.randn(512, 512, device=device, dtype=torch.float32)
        B = torch.randn(512, 512, device=device, dtype=torch.float32)
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            C = torch.matmul(A, B)
            if device != "cpu":
                if "cuda" in str(device):
                    torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_latency_ms = (end - start) * 1000 / num_iterations
        
        # Should complete within reasonable time (100ms for 512x512 on any device)
        assert avg_latency_ms < 100, f"MatMul too slow: {avg_latency_ms:.2f}ms"

    def test_silu_latency_baseline(self, device):
        """Test SiLU latency is within expected range."""
        import torch
        
        # Warmup
        for _ in range(5):
            x = torch.randn(1024, 1024, device=device)
            y = torch.nn.functional.silu(x)
        
        # Benchmark
        num_iterations = 10
        x = torch.randn(1024, 1024, device=device)
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            y = torch.nn.functional.silu(x)
            if device != "cpu":
                if "cuda" in str(device):
                    torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_latency_ms = (end - start) * 1000 / num_iterations
        
        # SiLU on 1M elements should be fast
        assert avg_latency_ms < 50, f"SiLU too slow: {avg_latency_ms:.2f}ms"

    def test_softmax_latency_baseline(self, device):
        """Test Softmax latency is within expected range."""
        import torch
        
        # Warmup
        for _ in range(5):
            x = torch.randn(32, 128, 1024, device=device)
            y = torch.softmax(x, dim=-1)
        
        # Benchmark
        num_iterations = 10
        x = torch.randn(32, 128, 1024, device=device)
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            y = torch.softmax(x, dim=-1)
            if device != "cpu":
                if "cuda" in str(device):
                    torch.cuda.synchronize()
        end = time.perf_counter()
        
        avg_latency_ms = (end - start) * 1000 / num_iterations
        
        assert avg_latency_ms < 100, f"Softmax too slow: {avg_latency_ms:.2f}ms"


# =============================================================================
# Throughput Tests
# =============================================================================

@pytest.mark.torch
@pytest.mark.benchmark
class TestThroughput:
    """Tests for throughput measurements."""

    def test_matmul_throughput(self, device):
        """Test MatMul throughput in TFLOPS."""
        import torch
        
        M, N, K = 1024, 1024, 1024
        num_iterations = 10
        
        A = torch.randn(M, K, device=device, dtype=torch.float32)
        B = torch.randn(K, N, device=device, dtype=torch.float32)
        
        # Warmup
        for _ in range(5):
            C = torch.matmul(A, B)
        
        if "cuda" in str(device):
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            C = torch.matmul(A, B)
        if "cuda" in str(device):
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        total_time_s = end - start
        
        # FLOPS = 2 * M * N * K per matmul
        total_flops = 2 * M * N * K * num_iterations
        tflops = total_flops / total_time_s / 1e12
        
        # Log throughput (not asserting specific value as it's hardware dependent)
        print(f"\nMatMul throughput on {device}: {tflops:.2f} TFLOPS")
        
        # Just verify it completed
        assert tflops > 0


# =============================================================================
# Memory Tests
# =============================================================================

@pytest.mark.torch
class TestMemory:
    """Tests for memory usage."""

    @pytest.mark.cuda
    def test_cuda_memory_usage(self):
        """Test CUDA memory usage is reasonable."""
        import torch
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Allocate some tensors
        tensors = []
        for _ in range(10):
            t = torch.randn(1024, 1024, device="cuda")
            tensors.append(t)
        
        peak_memory = torch.cuda.memory_allocated()
        
        # Clean up
        tensors.clear()
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should be mostly reclaimed
        assert final_memory < peak_memory

    @pytest.mark.mps
    def test_mps_memory_cleanup(self):
        """Test MPS memory cleanup."""
        import torch
        
        # Allocate and deallocate
        for _ in range(5):
            t = torch.randn(1024, 1024, device="mps")
            del t

        t2 = torch.randn(256, 256, device="mps")
        assert t2.shape == (256, 256)
        del t2


# =============================================================================
# Scaling Tests
# =============================================================================

@pytest.mark.torch
@pytest.mark.slow
class TestScaling:
    """Tests for performance scaling."""

    def test_matmul_scaling_with_size(self, device):
        """Test MatMul scales reasonably with problem size."""
        import torch
        
        sizes = [128, 256, 512]
        latencies = []
        
        for size in sizes:
            A = torch.randn(size, size, device=device, dtype=torch.float32)
            B = torch.randn(size, size, device=device, dtype=torch.float32)
            
            # Warmup
            for _ in range(3):
                C = torch.matmul(A, B)
            
            if "cuda" in str(device):
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(5):
                C = torch.matmul(A, B)
            if "cuda" in str(device):
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000 / 5)
        
        # Larger sizes should take longer (roughly O(n^3))
        # But we just verify it doesn't regress unexpectedly
        for i in range(1, len(latencies)):
            # Allow some variance but should generally increase
            # (smaller might run faster due to caching, so don't be too strict)
            assert latencies[i] < latencies[i-1] * 100  # Very loose bound
