"""
Multi-Backend Test Framework for YiRage

This module provides a comprehensive test framework for validating kernel
correctness across multiple hardware backends (CUDA, CPU, MPS, Ascend, MACA).

Design Principles:
1. Backend-agnostic test definitions using PyTorch as reference
2. Automatic skip for unavailable backends
3. Tolerance-based comparison with backend-specific thresholds
4. TDD approach: tests define expected behavior before implementation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import functools


# ============================================================================
# Backend Detection and Configuration
# ============================================================================

class BackendType(Enum):
    """Supported backend types"""
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
    ASCEND = "ascend"
    MACA = "maca"
    TRITON = "triton"


@dataclass
class BackendConfig:
    """Configuration for a specific backend"""
    name: str
    device: str
    is_available: bool
    rtol: float = 1e-2  # Relative tolerance
    atol: float = 1e-2  # Absolute tolerance
    supports_fp16: bool = True
    supports_bf16: bool = False


def detect_available_backends() -> Dict[BackendType, BackendConfig]:
    """Detect which backends are available on the current system."""
    backends = {}
    
    # CUDA backend
    cuda_available = torch.cuda.is_available()
    backends[BackendType.CUDA] = BackendConfig(
        name="cuda",
        device="cuda:0" if cuda_available else "cpu",
        is_available=cuda_available,
        rtol=1e-2,
        atol=1e-2,
        supports_fp16=True,
        supports_bf16=cuda_available and torch.cuda.get_device_capability()[0] >= 8,
    )
    
    # CPU backend (always available)
    backends[BackendType.CPU] = BackendConfig(
        name="cpu",
        device="cpu",
        is_available=True,
        rtol=1e-3,
        atol=1e-3,
        supports_fp16=True,
        supports_bf16=True,
    )
    
    # MPS backend (Apple Silicon)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    backends[BackendType.MPS] = BackendConfig(
        name="mps",
        device="mps" if mps_available else "cpu",
        is_available=mps_available,
        rtol=1e-2,
        atol=1e-2,
        supports_fp16=True,
        supports_bf16=False,
    )
    
    # Ascend backend (Huawei NPU)
    ascend_available = False
    try:
        import torch_npu
        ascend_available = torch.npu.is_available()
    except ImportError:
        pass
    backends[BackendType.ASCEND] = BackendConfig(
        name="ascend",
        device="npu:0" if ascend_available else "cpu",
        is_available=ascend_available,
        rtol=1e-2,
        atol=1e-2,
        supports_fp16=True,
        supports_bf16=ascend_available,
    )
    
    # MACA backend (MetaX GPU)
    maca_available = False
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            if any(x in device_name for x in ["MetaX", "C500", "MACA"]):
                maca_available = True
        except:
            pass
    backends[BackendType.MACA] = BackendConfig(
        name="maca",
        device="cuda:0" if maca_available else "cpu",
        is_available=maca_available,
        rtol=1e-2,
        atol=1e-2,
        supports_fp16=True,
        supports_bf16=maca_available,
    )
    
    return backends


AVAILABLE_BACKENDS = detect_available_backends()


def get_backend_config(backend: BackendType) -> BackendConfig:
    """Get configuration for a specific backend."""
    return AVAILABLE_BACKENDS.get(backend, AVAILABLE_BACKENDS[BackendType.CPU])


def skip_if_unavailable(backend: BackendType):
    """Decorator to skip test if backend is not available."""
    config = get_backend_config(backend)
    return pytest.mark.skipif(
        not config.is_available,
        reason=f"{backend.value} backend not available"
    )


# ============================================================================
# Tensor Comparison Utilities
# ============================================================================

def tensors_are_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    verbose: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Compare two tensors with tolerance and return detailed statistics.
    
    Args:
        actual: The tensor to validate
        expected: The reference tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: Whether to print detailed comparison
    
    Returns:
        Tuple of (is_close, statistics_dict)
    """
    if actual.shape != expected.shape:
        return False, {"error": f"Shape mismatch: {actual.shape} vs {expected.shape}"}
    
    # Convert to float32 for comparison
    actual_f32 = actual.float().cpu()
    expected_f32 = expected.float().cpu()
    
    # Compute differences
    abs_diff = torch.abs(actual_f32 - expected_f32)
    rel_diff = abs_diff / (torch.abs(expected_f32) + 1e-8)
    
    # Check tolerance
    within_tol = (abs_diff <= atol) | (rel_diff <= rtol)
    num_violations = (~within_tol).sum().item()
    total_elements = actual_f32.numel()
    
    stats = {
        "max_abs_diff": abs_diff.max().item(),
        "max_rel_diff": rel_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "mean_rel_diff": rel_diff.mean().item(),
        "num_violations": num_violations,
        "total_elements": total_elements,
        "violation_ratio": num_violations / total_elements if total_elements > 0 else 0,
    }
    
    if verbose:
        print(f"\nTensor Comparison Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Allow up to 1% violations for numerical stability
    is_close = stats["violation_ratio"] < 0.01
    return is_close, stats


def compare_with_pytorch_reference(
    yirage_output: torch.Tensor,
    pytorch_fn: Callable,
    inputs: List[torch.Tensor],
    backend_config: BackendConfig,
    verbose: bool = False,
) -> bool:
    """
    Compare YiRage output with PyTorch reference implementation.
    
    Args:
        yirage_output: Output from YiRage kernel
        pytorch_fn: PyTorch function to compute reference
        inputs: Input tensors
        backend_config: Backend configuration with tolerances
        verbose: Whether to print comparison details
    
    Returns:
        True if outputs match within tolerance
    """
    # Compute reference on CPU with float32 for accuracy
    cpu_inputs = [x.float().cpu() for x in inputs]
    reference = pytorch_fn(*cpu_inputs)
    
    is_close, stats = tensors_are_close(
        yirage_output,
        reference.to(yirage_output.dtype),
        rtol=backend_config.rtol,
        atol=backend_config.atol,
        verbose=verbose,
    )
    
    return is_close


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

@pytest.fixture
def backend_configs():
    """Provide all backend configurations."""
    return AVAILABLE_BACKENDS


@pytest.fixture
def available_backends():
    """Provide list of available backend types."""
    return [bt for bt, cfg in AVAILABLE_BACKENDS.items() if cfg.is_available]


def create_test_tensors(
    shapes: List[Tuple[int, ...]],
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
    seed: int = 42,
) -> List[torch.Tensor]:
    """Create reproducible test tensors."""
    torch.manual_seed(seed)
    tensors = []
    for shape in shapes:
        t = torch.randn(shape, dtype=dtype, device=device)
        tensors.append(t)
    return tensors


# ============================================================================
# Backend-Parametrized Test Base Classes
# ============================================================================

def parametrize_backends(*backend_types: BackendType):
    """
    Decorator to run a test across multiple backends.
    
    Usage:
        @parametrize_backends(BackendType.CUDA, BackendType.CPU, BackendType.MPS)
        def test_matmul(backend_config):
            ...
    """
    configs = []
    ids = []
    for bt in backend_types:
        cfg = get_backend_config(bt)
        if cfg.is_available:
            configs.append(cfg)
            ids.append(bt.value)
    
    return pytest.mark.parametrize("backend_config", configs, ids=ids)


# ============================================================================
# Core Operation Tests
# ============================================================================

class TestMatMul:
    """Test matrix multiplication across backends."""
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU, BackendType.MPS)
    def test_matmul_square(self, backend_config: BackendConfig):
        """Test square matrix multiplication."""
        M, N, K = 256, 256, 256
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        
        # PyTorch reference
        expected = torch.matmul(A, B)
        
        # For now, just verify PyTorch works on this backend
        # YiRage kernel will be tested when available
        assert expected.shape == (M, N)
        assert expected.device.type == device.split(":")[0]
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU)
    def test_matmul_rectangular(self, backend_config: BackendConfig):
        """Test rectangular matrix multiplication."""
        M, N, K = 128, 512, 256
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        A = torch.randn(M, K, dtype=dtype, device=device)
        B = torch.randn(K, N, dtype=dtype, device=device)
        
        expected = torch.matmul(A, B)
        
        assert expected.shape == (M, N)
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU)
    def test_matmul_batch(self, backend_config: BackendConfig):
        """Test batched matrix multiplication."""
        B, M, N, K = 4, 128, 128, 128
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        A = torch.randn(B, M, K, dtype=dtype, device=device)
        B_mat = torch.randn(B, K, N, dtype=dtype, device=device)
        
        expected = torch.matmul(A, B_mat)
        
        assert expected.shape == (B, M, N)


class TestElementwiseOps:
    """Test elementwise operations across backends."""
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU, BackendType.MPS)
    def test_silu(self, backend_config: BackendConfig):
        """Test SiLU activation."""
        shape = (8, 4096)
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        x = torch.randn(shape, dtype=dtype, device=device)
        expected = nn.functional.silu(x)
        
        assert expected.shape == shape
        # SiLU should produce bounded outputs for bounded inputs
        assert not torch.isnan(expected).any()
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU, BackendType.MPS)
    def test_gelu(self, backend_config: BackendConfig):
        """Test GELU activation."""
        shape = (8, 4096)
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        x = torch.randn(shape, dtype=dtype, device=device)
        expected = nn.functional.gelu(x)
        
        assert expected.shape == shape
        assert not torch.isnan(expected).any()
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU)
    def test_add_mul_div(self, backend_config: BackendConfig):
        """Test basic arithmetic operations."""
        shape = (8, 4096)
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        a = torch.randn(shape, dtype=dtype, device=device)
        b = torch.randn(shape, dtype=dtype, device=device) + 1.0  # Avoid div by zero
        
        add_result = a + b
        mul_result = a * b
        div_result = a / b
        
        assert add_result.shape == shape
        assert mul_result.shape == shape
        assert div_result.shape == shape


class TestNormalization:
    """Test normalization operations across backends."""
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU)
    def test_rms_norm(self, backend_config: BackendConfig):
        """Test RMS normalization."""
        batch_size, hidden_size = 8, 4096
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        
        # PyTorch reference implementation
        rms_norm = nn.RMSNorm(hidden_size, dtype=dtype, device=device)
        expected = rms_norm(x)
        
        assert expected.shape == (batch_size, hidden_size)
        assert not torch.isnan(expected).any()
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU, BackendType.MPS)
    def test_layer_norm(self, backend_config: BackendConfig):
        """Test layer normalization."""
        batch_size, hidden_size = 8, 4096
        device = backend_config.device
        dtype = torch.float32  # LayerNorm often needs float32
        
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        
        layer_norm = nn.LayerNorm(hidden_size, dtype=dtype, device=device)
        expected = layer_norm(x)
        
        assert expected.shape == (batch_size, hidden_size)
        assert not torch.isnan(expected).any()


class TestAttention:
    """Test attention mechanisms across backends."""
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU)
    def test_scaled_dot_product_attention(self, backend_config: BackendConfig):
        """Test scaled dot-product attention."""
        batch_size, num_heads, seq_len, head_dim = 2, 8, 256, 64
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        Q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        K = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype, device=device)
        
        # PyTorch reference
        scale = 1.0 / (head_dim ** 0.5)
        attn_weights = torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1)
        expected = torch.matmul(attn_weights, V)
        
        assert expected.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not torch.isnan(expected).any()
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU)
    def test_group_query_attention(self, backend_config: BackendConfig):
        """Test group query attention (GQA)."""
        batch_size = 2
        num_q_heads = 32
        num_kv_heads = 8  # GQA: fewer KV heads than Q heads
        seq_len = 256
        head_dim = 64
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        Q = torch.randn(batch_size, num_q_heads, seq_len, head_dim, dtype=dtype, device=device)
        K = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
        V = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
        
        # Expand KV heads to match Q heads
        repeat_factor = num_q_heads // num_kv_heads
        K_expanded = K.repeat_interleave(repeat_factor, dim=1)
        V_expanded = V.repeat_interleave(repeat_factor, dim=1)
        
        # Standard attention
        scale = 1.0 / (head_dim ** 0.5)
        attn_weights = torch.softmax(torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale, dim=-1)
        expected = torch.matmul(attn_weights, V_expanded)
        
        assert expected.shape == (batch_size, num_q_heads, seq_len, head_dim)
        assert not torch.isnan(expected).any()


class TestMLP:
    """Test MLP operations across backends."""
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU)
    def test_gated_mlp(self, backend_config: BackendConfig):
        """Test gated MLP (SiLU-gated as in LLaMA)."""
        batch_size, hidden_size, intermediate_size = 8, 4096, 11008
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        gate_proj = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
        up_proj = torch.randn(intermediate_size, hidden_size, dtype=dtype, device=device)
        
        # PyTorch reference: SiLU(x @ gate_proj.T) * (x @ up_proj.T)
        gate_out = nn.functional.silu(torch.matmul(x, gate_proj.T))
        up_out = torch.matmul(x, up_proj.T)
        expected = gate_out * up_out
        
        assert expected.shape == (batch_size, intermediate_size)
        assert not torch.isnan(expected).any()


class TestLoRA:
    """Test LoRA (Low-Rank Adaptation) operations."""
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU)
    def test_lora_forward(self, backend_config: BackendConfig):
        """Test LoRA forward pass."""
        batch_size = 8
        in_features, out_features = 4096, 4096
        lora_rank = 16
        device = backend_config.device
        dtype = torch.float16 if backend_config.supports_fp16 else torch.float32
        
        x = torch.randn(batch_size, in_features, dtype=dtype, device=device)
        W = torch.randn(out_features, in_features, dtype=dtype, device=device)
        A = torch.randn(lora_rank, in_features, dtype=dtype, device=device)
        B = torch.randn(out_features, lora_rank, dtype=dtype, device=device)
        
        # PyTorch reference: y = x @ W.T + x @ A.T @ B.T
        base_out = torch.matmul(x, W.T)
        lora_out = torch.matmul(torch.matmul(x, A.T), B.T)
        expected = base_out + lora_out
        
        assert expected.shape == (batch_size, out_features)
        assert not torch.isnan(expected).any()


# ============================================================================
# Integration Tests
# ============================================================================

class TestYiRageIntegration:
    """Integration tests with YiRage kernels."""
    
    @pytest.fixture
    def yirage_available(self):
        """Check if YiRage module is available."""
        try:
            import yirage as yr
            return True
        except ImportError:
            return False
    
    @skip_if_unavailable(BackendType.CUDA)
    def test_yirage_import(self, yirage_available):
        """Test YiRage can be imported."""
        if not yirage_available:
            pytest.skip("YiRage not available")
        
        import yirage as yr
        assert hasattr(yr, 'new_kernel_graph')
        assert hasattr(yr, 'get_available_backends')
    
    @skip_if_unavailable(BackendType.CUDA)
    def test_yirage_kernel_graph(self, yirage_available):
        """Test creating a YiRage kernel graph."""
        if not yirage_available:
            pytest.skip("YiRage not available")
        
        import yirage as yr
        
        graph = yr.new_kernel_graph()
        assert graph is not None


# ============================================================================
# Backend Capability Tests
# ============================================================================

class TestBackendCapabilities:
    """Test backend-specific capabilities."""
    
    def test_backend_detection(self):
        """Verify backend detection works."""
        backends = detect_available_backends()
        
        # CPU should always be available
        assert BackendType.CPU in backends
        assert backends[BackendType.CPU].is_available
        
        # All backends should have proper configuration
        for bt, cfg in backends.items():
            assert cfg.name == bt.value
            assert cfg.rtol > 0
            assert cfg.atol > 0
    
    def test_available_backends_list(self, available_backends):
        """Verify available backends list."""
        assert BackendType.CPU in available_backends
        assert len(available_backends) >= 1
    
    @parametrize_backends(BackendType.CUDA, BackendType.CPU, BackendType.MPS)
    def test_tensor_creation(self, backend_config: BackendConfig):
        """Test tensor creation on each backend."""
        device = backend_config.device
        x = torch.randn(128, 128, device=device)
        
        assert x.device.type == device.split(":")[0]
        assert x.shape == (128, 128)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
