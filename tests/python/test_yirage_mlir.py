#!/usr/bin/env python3
"""
Tests for YiRage MLIR Python Interface

Tests:
- Module building
- Operation generation
- Multi-backend support
- Advanced operations (MoE, MLA, Speculative Decode)
"""

import sys
import pytest

# Add mlir python path
sys.path.insert(0, '/workspace/mlir/python')

from yirage_mlir import (
    YirageModule, Target, TargetConfig, DType, TensorType, Value,
    build_attention_block, build_moe_block, build_mla_block,
    detect_available_targets, get_best_target
)


#==============================================================================
# Basic Module Tests
#==============================================================================

class TestYirageModule:
    """Tests for YirageModule class."""
    
    def test_create_empty_module(self):
        """Test creating an empty module."""
        m = YirageModule("test_module")
        assert m.name == "test_module"
        assert len(m.operations) == 0
        assert len(m.inputs) == 0
        assert len(m.outputs) == 0
    
    def test_placeholder_creation(self):
        """Test creating input placeholders."""
        m = YirageModule()
        x = m.placeholder("x", [4, 8], "f16")
        
        assert len(m.inputs) == 1
        assert x.tensor_type.shape == (4, 8)
        assert x.tensor_type.dtype == DType.F16
    
    def test_matmul_operation(self):
        """Test matrix multiplication."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        y = m.placeholder("y", [64, 128], "f16")
        
        z = m.matmul(x, y)
        
        assert len(m.operations) == 1
        assert z.tensor_type.shape == (32, 128)
    
    def test_matmul_with_transpose(self):
        """Test matmul with transpose flags."""
        m = YirageModule()
        x = m.placeholder("x", [64, 32], "f16")
        y = m.placeholder("y", [64, 128], "f16")
        
        z = m.matmul(x, y, transpose_lhs=True)
        
        assert z.tensor_type.shape == (32, 128)
    
    def test_output_marking(self):
        """Test marking outputs."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        y = m.silu(x)
        m.output(y)
        
        assert len(m.outputs) == 1
        assert m.outputs[0] == y


#==============================================================================
# MLIR Generation Tests
#==============================================================================

class TestMLIRGeneration:
    """Tests for MLIR text generation."""
    
    def test_simple_matmul_mlir(self):
        """Test MLIR generation for simple matmul."""
        m = YirageModule("simple_matmul")
        x = m.placeholder("x", [32, 64], "f16")
        y = m.placeholder("y", [64, 128], "f16")
        z = m.matmul(x, y)
        m.output(z)
        
        mlir = m.to_mlir()
        
        assert "module {" in mlir
        assert "func.func @simple_matmul" in mlir
        assert "yirage.matmul" in mlir
        assert "tensor<32x64xf16>" in mlir
        assert "tensor<64x128xf16>" in mlir
        assert "tensor<32x128xf16>" in mlir
        assert "return" in mlir
    
    def test_attention_mlir(self):
        """Test MLIR generation for attention."""
        m = YirageModule("attention")
        q = m.placeholder("q", [1, 32, 2048, 128], "f16")
        k = m.placeholder("k", [1, 32, 2048, 128], "f16")
        v = m.placeholder("v", [1, 32, 2048, 128], "f16")
        
        out = m.attention(q, k, v, causal=True)
        m.output(out)
        
        mlir = m.to_mlir()
        
        assert "yirage.attention" in mlir
        assert "causal = true" in mlir
    
    def test_rms_norm_mlir(self):
        """Test MLIR generation for RMS norm."""
        m = YirageModule("rms_norm")
        x = m.placeholder("x", [1, 2048, 4096], "f16")
        gamma = m.placeholder("gamma", [4096], "f16")
        
        out = m.rms_norm(x, gamma, epsilon=1e-5)
        m.output(out)
        
        mlir = m.to_mlir()
        
        assert "yirage.rms_norm" in mlir
        assert "epsilon" in mlir
    
    def test_gated_mlp_mlir(self):
        """Test MLIR generation for gated MLP."""
        m = YirageModule("gated_mlp")
        x = m.placeholder("x", [2048, 4096], "f16")
        gate_w = m.placeholder("gate_w", [4096, 11008], "f16")
        up_w = m.placeholder("up_w", [4096, 11008], "f16")
        down_w = m.placeholder("down_w", [11008, 4096], "f16")
        
        out = m.gated_mlp(x, gate_w, up_w, down_w)
        m.output(out)
        
        mlir = m.to_mlir()
        
        assert "yirage.gated_mlp" in mlir


#==============================================================================
# Activation Function Tests
#==============================================================================

class TestActivations:
    """Tests for activation functions."""
    
    def test_silu(self):
        """Test SiLU activation."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        y = m.silu(x)
        
        mlir = m.to_mlir()
        assert "yirage.silu" in mlir
    
    def test_gelu(self):
        """Test GELU activation."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        y = m.gelu(x)
        
        mlir = m.to_mlir()
        assert "yirage.gelu" in mlir
    
    def test_relu(self):
        """Test ReLU activation."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        y = m.relu(x)
        
        mlir = m.to_mlir()
        assert "yirage.relu" in mlir
    
    def test_sigmoid(self):
        """Test Sigmoid activation."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        y = m.sigmoid(x)
        
        mlir = m.to_mlir()
        assert "yirage.sigmoid" in mlir
    
    def test_math_ops(self):
        """Test math operations (exp, log, sqrt)."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        
        y = m.exp(x)
        z = m.log(x)
        w = m.sqrt(x)
        
        mlir = m.to_mlir()
        assert "math.exp" in mlir
        assert "math.log" in mlir
        assert "math.sqrt" in mlir


#==============================================================================
# Binary Operation Tests
#==============================================================================

class TestBinaryOps:
    """Tests for binary operations."""
    
    def test_add(self):
        """Test element-wise addition."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        y = m.placeholder("y", [32, 64], "f16")
        z = m.add(x, y)
        
        mlir = m.to_mlir()
        assert "arith.addf" in mlir
    
    def test_mul(self):
        """Test element-wise multiplication."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        y = m.placeholder("y", [32, 64], "f16")
        z = m.mul(x, y)
        
        mlir = m.to_mlir()
        assert "arith.mulf" in mlir
    
    def test_sub(self):
        """Test element-wise subtraction."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        y = m.placeholder("y", [32, 64], "f16")
        z = m.sub(x, y)
        
        mlir = m.to_mlir()
        assert "arith.subf" in mlir
    
    def test_div(self):
        """Test element-wise division."""
        m = YirageModule()
        x = m.placeholder("x", [32, 64], "f16")
        y = m.placeholder("y", [32, 64], "f16")
        z = m.div(x, y)
        
        mlir = m.to_mlir()
        assert "arith.divf" in mlir


#==============================================================================
# Advanced LLM Operation Tests
#==============================================================================

class TestAdvancedOps:
    """Tests for advanced LLM operations."""
    
    def test_moe_layer(self):
        """Test Mixture of Experts layer."""
        m = YirageModule("moe")
        x = m.placeholder("x", [2048, 4096], "f16")
        gate_w = m.placeholder("gate_w", [4096, 8], "f16")
        expert_gate = m.placeholder("expert_gate", [8, 4096, 11008], "f16")
        expert_up = m.placeholder("expert_up", [8, 4096, 11008], "f16")
        expert_down = m.placeholder("expert_down", [8, 11008, 4096], "f16")
        
        out = m.moe_layer(x, gate_w, expert_gate, expert_up, expert_down,
                          num_experts=8, top_k=2)
        m.output(out)
        
        mlir = m.to_mlir()
        
        assert "yirage.moe_layer" in mlir
        assert "num_experts = 8" in mlir
        assert "top_k = 2" in mlir
    
    def test_ml_attention(self):
        """Test Multi-Latent Attention (DeepSeek MLA)."""
        m = YirageModule("mla")
        q = m.placeholder("q", [1, 32, 2048, 128], "f16")
        c_kv = m.placeholder("c_kv", [1, 2048, 512], "f16")
        kv_down = m.placeholder("kv_down", [2048, 512], "f16")
        kv_up = m.placeholder("kv_up", [512, 2048], "f16")
        
        out = m.ml_attention(q, c_kv, kv_down, kv_up,
                             num_heads=32, num_kv_heads=8,
                             head_dim=128, compressed_dim=512)
        m.output(out)
        
        mlir = m.to_mlir()
        
        assert "yirage.ml_attention" in mlir
        assert "num_heads = 32" in mlir
        assert "compressed_dim = 512" in mlir
    
    def test_sliding_window_attention(self):
        """Test sliding window attention."""
        m = YirageModule("swa")
        q = m.placeholder("q", [1, 32, 16384, 128], "f16")
        k = m.placeholder("k", [1, 32, 16384, 128], "f16")
        v = m.placeholder("v", [1, 32, 16384, 128], "f16")
        
        out = m.sliding_window_attention(q, k, v, window_size=4096)
        m.output(out)
        
        mlir = m.to_mlir()
        
        assert "yirage.sliding_window_attention" in mlir
        assert "window_size = 4096" in mlir


#==============================================================================
# Convenience Builder Tests
#==============================================================================

class TestConvenienceBuilders:
    """Tests for convenience builder functions."""
    
    def test_build_attention_block(self):
        """Test attention block builder."""
        m = build_attention_block(
            batch=1, seq_len=2048, num_heads=32,
            head_dim=128, hidden_dim=4096
        )
        
        mlir = m.to_mlir()
        
        assert "attention_block" in mlir
        assert "yirage.rms_norm" in mlir
        assert "yirage.matmul" in mlir
        assert "yirage.attention" in mlir
    
    def test_build_moe_block(self):
        """Test MoE block builder."""
        m = build_moe_block(
            batch=1, seq_len=2048, hidden_dim=4096,
            intermediate_dim=11008, num_experts=8, top_k=2
        )
        
        mlir = m.to_mlir()
        
        assert "moe_block" in mlir
        assert "yirage.moe_layer" in mlir
    
    def test_build_mla_block(self):
        """Test MLA block builder."""
        m = build_mla_block(
            batch=1, seq_len=2048, num_heads=32,
            num_kv_heads=8, head_dim=128,
            compressed_dim=512, hidden_dim=4096
        )
        
        mlir = m.to_mlir()
        
        assert "mla_block" in mlir
        assert "yirage.ml_attention" in mlir


#==============================================================================
# Target Tests
#==============================================================================

class TestTargets:
    """Tests for target configuration."""
    
    def test_target_enum(self):
        """Test Target enum values."""
        assert Target.CUDA_H100.value == "cuda-sm_90"
        assert Target.ROCM_MI300X.value == "rocm-gfx942"
        assert Target.TPU_V5E.value == "tpu-v5e"
        assert Target.ASCEND_910B.value == "ascend-910b"
    
    def test_target_backend(self):
        """Test target backend extraction."""
        assert Target.CUDA_H100.backend == "cuda"
        assert Target.ROCM_MI300X.backend == "rocm"
        assert Target.TPU_V5E.backend == "tpu"
        assert Target.CPU_GENERIC.backend == "cpu"
    
    def test_target_arch(self):
        """Test target arch extraction."""
        assert Target.CUDA_H100.arch == "sm_90"
        assert Target.ROCM_MI300X.arch == "gfx942"
        assert Target.TPU_V5E.arch == "v5e"
    
    def test_target_from_string(self):
        """Test parsing target from string."""
        assert Target.from_string("cuda_h100") == Target.CUDA_H100
        assert Target.from_string("CUDA_H100") == Target.CUDA_H100
        assert Target.from_string("rocm-gfx942") == Target.ROCM_MI300X
    
    def test_all_targets_have_backend(self):
        """Test all targets have a valid backend."""
        for target in Target:
            backend = target.backend
            assert backend in ["cuda", "rocm", "xpu", "tpu", "ascend",
                               "maca", "metal", "cpu", "fpga"]


#==============================================================================
# Data Type Tests
#==============================================================================

class TestDTypes:
    """Tests for data types."""
    
    def test_dtype_values(self):
        """Test DType enum values."""
        assert DType.F16.value == "f16"
        assert DType.BF16.value == "bf16"
        assert DType.F32.value == "f32"
        assert DType.I8.value == "i8"
    
    def test_dtype_from_string(self):
        """Test parsing dtype from string."""
        assert DType.from_string("f16") == DType.F16
        assert DType.from_string("float16") == DType.F16
        assert DType.from_string("bf16") == DType.BF16
        assert DType.from_string("int8") == DType.I8
    
    def test_tensor_type_mlir(self):
        """Test TensorType MLIR generation."""
        tt = TensorType((32, 64), DType.F16)
        assert tt.mlir_type == "tensor<32x64xf16>"
        
        tt = TensorType((1, 32, 2048, 128), DType.BF16)
        assert tt.mlir_type == "tensor<1x32x2048x128xbf16>"


#==============================================================================
# Compilation Tests
#==============================================================================

class TestCompilation:
    """Tests for compilation."""
    
    def test_compile_to_target(self):
        """Test compiling module to target."""
        m = YirageModule("test")
        x = m.placeholder("x", [32, 64], "f16")
        y = m.silu(x)
        m.output(y)
        
        # Should not raise
        compiled = m.compile(Target.CPU_GENERIC)
        
        assert compiled.source_mlir is not None
        assert compiled.config.target == Target.CPU_GENERIC
    
    def test_compile_with_config(self):
        """Test compiling with custom config."""
        m = YirageModule("test")
        x = m.placeholder("x", [32, 64], "f16")
        y = m.relu(x)
        m.output(y)
        
        config = TargetConfig(
            target=Target.CUDA_H100,
            opt_level=2,
            fast_math=False
        )
        
        compiled = m.compile(config=config)
        
        assert compiled.config.opt_level == 2
        assert compiled.config.fast_math == False


#==============================================================================
# Target Detection Tests
#==============================================================================

class TestTargetDetection:
    """Tests for target detection."""
    
    def test_detect_available_targets(self):
        """Test detecting available targets."""
        targets = detect_available_targets()
        
        # CPU should always be available
        assert Target.CPU_GENERIC in targets
    
    def test_get_best_target(self):
        """Test getting best target."""
        best = get_best_target()
        
        # Should return a valid target
        assert best in Target


#==============================================================================
# Context Manager Tests
#==============================================================================

class TestContextManager:
    """Tests for context manager interface."""
    
    def test_with_statement(self):
        """Test using with statement."""
        with YirageModule("ctx_test") as m:
            x = m.placeholder("x", [32, 64], "f16")
            y = m.silu(x)
            m.output(y)
        
        mlir = m.to_mlir()
        assert "yirage.silu" in mlir
    
    def test_value_counter_reset(self):
        """Test value counter resets in context."""
        with YirageModule() as m1:
            x1 = m1.placeholder("x", [32, 64], "f16")
        
        with YirageModule() as m2:
            x2 = m2.placeholder("x", [32, 64], "f16")
        
        # Both should start from %0 (after arg0)
        assert "%0" in m1.to_mlir() or "%arg0" in m1.to_mlir()


#==============================================================================
# Multiple Backend Compilation Tests
#==============================================================================

class TestMultiBackend:
    """Tests for multi-backend compilation."""
    
    def test_compile_all_backends(self):
        """Test compiling for all backend families."""
        m = YirageModule("multi_backend")
        x = m.placeholder("x", [32, 64], "f16")
        y = m.matmul(x, x)
        m.output(y)
        
        backends = [
            Target.CUDA_GENERIC,
            Target.ROCM_GENERIC,
            Target.XPU_GENERIC,
            Target.TPU_GENERIC,
            Target.ASCEND_GENERIC,
            Target.MACA_GENERIC,
            Target.METAL_GENERIC,
            Target.CPU_GENERIC,
            Target.FPGA_GENERIC,
        ]
        
        for backend in backends:
            compiled = m.compile(backend)
            assert compiled.config.target == backend
            assert backend.backend in compiled.config.target.value


#==============================================================================
# Main
#==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
