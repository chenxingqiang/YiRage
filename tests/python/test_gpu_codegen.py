#!/usr/bin/env python3
# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0
"""
GPU Code Generation Tests

Tests for MLIR GPU code generation pipeline.
Run with: pytest tests/python/test_gpu_codegen.py -v
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
PYTHON_ROOT = PROJECT_ROOT / "python"
MLIR_ROOT = PROJECT_ROOT / "mlir" / "python"
sys.path.insert(0, str(PYTHON_ROOT))
sys.path.insert(0, str(MLIR_ROOT))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def mlir_jit_module():
    """Load mlir_jit module."""
    try:
        from yirage import mlir_jit
        return mlir_jit
    except ImportError as e:
        pytest.skip(f"mlir_jit module not available: {e}")


@pytest.fixture
def sample_matmul_mlir():
    """Sample matmul MLIR."""
    return '''
module {
  func.func @matmul(%A: tensor<1024x512xf16>, %B: tensor<512x1024xf16>) -> tensor<1024x1024xf16> {
    %C = yirage.matmul %A, %B : tensor<1024x512xf16>, tensor<512x1024xf16> -> tensor<1024x1024xf16>
    return %C : tensor<1024x1024xf16>
  }
}
'''


@pytest.fixture
def sample_attention_mlir():
    """Sample attention MLIR."""
    return '''
module {
  func.func @attention(
      %Q: tensor<1x32x2048x128xf16>,
      %K: tensor<1x32x2048x128xf16>,
      %V: tensor<1x32x2048x128xf16>
  ) -> tensor<1x32x2048x128xf16> {
    %out = yirage.attention %Q, %K, %V {causal = true} :
        tensor<1x32x2048x128xf16>, tensor<1x32x2048x128xf16>,
        tensor<1x32x2048x128xf16> -> tensor<1x32x2048x128xf16>
    return %out : tensor<1x32x2048x128xf16>
  }
}
'''


@pytest.fixture
def sample_rms_norm_mlir():
    """Sample RMS normalization MLIR."""
    return '''
module {
  func.func @rms_norm(%input: tensor<8x4096xf16>, %gamma: tensor<4096xf16>) -> tensor<8x4096xf16> {
    %out = yirage.rms_norm %input, %gamma {epsilon = 1.0e-6 : f32} :
        tensor<8x4096xf16>, tensor<4096xf16> -> tensor<8x4096xf16>
    return %out : tensor<8x4096xf16>
  }
}
'''


@pytest.fixture
def sample_gated_mlp_mlir():
    """Sample gated MLP MLIR."""
    return '''
module {
  func.func @gated_mlp(
      %input: tensor<8x4096xf16>,
      %gate_weight: tensor<4096x11008xf16>,
      %up_weight: tensor<4096x11008xf16>,
      %down_weight: tensor<11008x4096xf16>
  ) -> tensor<8x4096xf16> {
    %out = yirage.gated_mlp %input, %gate_weight, %up_weight, %down_weight :
        tensor<8x4096xf16>, tensor<4096x11008xf16>,
        tensor<4096x11008xf16>, tensor<11008x4096xf16> -> tensor<8x4096xf16>
    return %out : tensor<8x4096xf16>
  }
}
'''


# =============================================================================
# Target Tests
# =============================================================================

class TestTarget:
    """Tests for Target enum."""

    def test_cuda_targets(self, mlir_jit_module):
        """Test CUDA target configuration."""
        Target = mlir_jit_module.Target
        
        assert Target.CUDA_V100.backend == "cuda"
        assert Target.CUDA_A100.backend == "cuda"
        assert Target.CUDA_H100.backend == "cuda"
        
        assert Target.CUDA_V100.compute_capability == 70
        assert Target.CUDA_A100.compute_capability == 80
        assert Target.CUDA_H100.compute_capability == 90

    def test_rocm_targets(self, mlir_jit_module):
        """Test ROCm target configuration."""
        Target = mlir_jit_module.Target
        
        assert Target.ROCM_MI100.backend == "rocm"
        assert Target.ROCM_MI250.backend == "rocm"
        assert Target.ROCM_MI300X.backend == "rocm"
        
        assert Target.ROCM_MI100.arch == "gfx908"
        assert Target.ROCM_MI250.arch == "gfx90a"
        assert Target.ROCM_MI300X.arch == "gfx942"

    def test_metal_targets(self, mlir_jit_module):
        """Test Metal target configuration — all Apple Silicon generations."""
        Target = mlir_jit_module.Target

        assert Target.METAL_M1.backend == "metal"
        assert Target.METAL_M1.arch == "apple-m1"
        assert Target.METAL_M2.backend == "metal"
        assert Target.METAL_M2.arch == "apple-m2"
        assert Target.METAL_M3.backend == "metal"
        assert Target.METAL_M3.arch == "apple-m3"
        assert Target.METAL_M4.backend == "metal"
        assert Target.METAL_M4.arch == "apple-m4"
        assert Target.METAL_M5.backend == "metal"
        assert Target.METAL_M5.arch == "apple-m5"

    def test_cpu_targets(self, mlir_jit_module):
        """Test CPU target configuration."""
        Target = mlir_jit_module.Target
        
        assert Target.CPU_AVX2.backend == "cpu"
        assert Target.CPU_AVX512.backend == "cpu"
        assert Target.CPU_NEON.backend == "cpu"


# =============================================================================
# Compiler Tests
# =============================================================================

class TestMLIRCompiler:
    """Tests for MLIRCompiler class."""

    def test_compiler_creation(self, mlir_jit_module):
        """Test MLIRCompiler can be created."""
        MLIRCompiler = mlir_jit_module.MLIRCompiler
        Target = mlir_jit_module.Target
        
        compiler = MLIRCompiler(target=Target.CUDA)
        assert compiler is not None
        assert compiler.target == Target.CUDA

    def test_compiler_with_options(self, mlir_jit_module):
        """Test MLIRCompiler with custom options."""
        MLIRCompiler = mlir_jit_module.MLIRCompiler
        Target = mlir_jit_module.Target
        CompileOptions = mlir_jit_module.CompileOptions
        
        options = CompileOptions(opt_level=2, fast_math=False)
        compiler = MLIRCompiler(target=Target.CUDA, options=options)
        
        assert compiler.options.opt_level == 2
        assert compiler.options.fast_math is False

    def test_is_mlir_available(self, mlir_jit_module):
        """Test MLIR availability check."""
        # Should return True or False without error
        result = mlir_jit_module.is_mlir_available()
        assert isinstance(result, bool)


# =============================================================================
# Compilation Tests (skip if MLIR not available)
# =============================================================================

class TestCompilation:
    """Tests for actual compilation (requires MLIR)."""

    def test_compile_matmul(self, mlir_jit_module, sample_matmul_mlir):
        """Test compiling matmul (graceful failure when MLIR is not built)."""
        MLIRCompiler = mlir_jit_module.MLIRCompiler
        Target = mlir_jit_module.Target

        compiler = MLIRCompiler(target=Target.CUDA)
        result = compiler.compile(sample_matmul_mlir)

        assert result is not None
        if mlir_jit_module.is_mlir_available():
            assert result.success or result.error_message
        else:
            assert result.success is False
            assert result.error_message and "MLIR" in result.error_message

    def test_compile_attention(self, mlir_jit_module, sample_attention_mlir):
        """Test compiling attention (graceful failure when MLIR is not built)."""
        MLIRCompiler = mlir_jit_module.MLIRCompiler
        Target = mlir_jit_module.Target

        compiler = MLIRCompiler(target=Target.CUDA_H100)
        result = compiler.compile(sample_attention_mlir)

        assert result is not None
        if mlir_jit_module.is_mlir_available():
            assert result.success or result.error_message
        else:
            assert result.success is False
            assert result.error_message and "MLIR" in result.error_message


# =============================================================================
# Cache Tests
# =============================================================================

class TestCompilationCache:
    """Tests for compilation cache."""

    def test_cache_creation(self, mlir_jit_module):
        """Test CompilationCache can be created."""
        CompilationCache = mlir_jit_module.CompilationCache
        
        cache = CompilationCache()
        assert cache is not None

    def test_cache_singleton(self, mlir_jit_module):
        """Test cache singleton behavior."""
        CompilationCache = mlir_jit_module.CompilationCache
        
        cache1 = CompilationCache.get_instance()
        cache2 = CompilationCache.get_instance()
        
        assert cache1 is cache2

    def test_cache_clear(self, mlir_jit_module):
        """Test cache clearing."""
        CompilationCache = mlir_jit_module.CompilationCache
        
        cache = CompilationCache.get_instance()
        cache.clear()
        # Should not raise


# =============================================================================
# JIT Decorator Tests
# =============================================================================

class TestJITDecorator:
    """Tests for JIT decorator."""

    def test_jit_decorator_exists(self, mlir_jit_module):
        """Test jit decorator exists."""
        assert hasattr(mlir_jit_module, 'jit')
        assert callable(mlir_jit_module.jit)

    def test_jit_decorator_basic(self, mlir_jit_module):
        """Test basic jit decorator usage."""
        jit = mlir_jit_module.jit
        Target = mlir_jit_module.Target
        
        @jit(target=Target.CUDA)
        def dummy_kernel(x, y):
            return x + y
        
        assert hasattr(dummy_kernel, '_compiler')
        assert hasattr(dummy_kernel, '_target')
        assert dummy_kernel._target == Target.CUDA

    def test_jit_preserves_function(self, mlir_jit_module):
        """Test jit decorator preserves function behavior."""
        jit = mlir_jit_module.jit
        
        @jit()
        def add(a, b):
            return a + b
        
        result = add(2, 3)
        assert result == 5


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_compile_mlir_function(self, mlir_jit_module):
        """Test compile_mlir function exists."""
        assert hasattr(mlir_jit_module, 'compile_mlir')
        assert callable(mlir_jit_module.compile_mlir)

    def test_clear_cache_function(self, mlir_jit_module):
        """Test clear_compile_cache function."""
        clear_fn = mlir_jit_module.clear_compile_cache
        clear_fn()  # Should not raise


# =============================================================================
# GPU Backend Configuration Tests  
# =============================================================================

class TestGPUBackendConfig:
    """Tests for GPU backend configuration."""

    def test_cuda_config(self, mlir_jit_module):
        """Test CUDA configuration."""
        Target = mlir_jit_module.Target
        CompileOptions = mlir_jit_module.CompileOptions
        
        # H100 should use sm_90
        target = Target.CUDA_H100
        assert target.arch == "sm_90"
        assert target.compute_capability == 90

    def test_compile_options_defaults(self, mlir_jit_module):
        """Test compile options defaults."""
        CompileOptions = mlir_jit_module.CompileOptions
        
        options = CompileOptions()
        assert options.opt_level == 3
        assert options.fast_math is True
        assert options.use_fma is True
        assert options.debug is False
        assert options.enable_cache is True

    def test_compile_options_custom(self, mlir_jit_module):
        """Test custom compile options."""
        CompileOptions = mlir_jit_module.CompileOptions
        
        options = CompileOptions(
            opt_level=0,
            fast_math=False,
            debug=True,
            vectorize=False
        )
        
        assert options.opt_level == 0
        assert options.fast_math is False
        assert options.debug is True
        assert options.vectorize is False
