#!/usr/bin/env python3
"""
YiRage MLIR Python Interface

Comprehensive Python API for building and compiling MLIR for all backends.

Features:
- Direct operation building API (similar to PyTorch)
- Multi-backend support (CUDA, ROCm, TPU, Ascend, XPU, MACA, Metal, CPU, FPGA)
- JIT compilation with caching
- Integration with advanced ops (MoE, MLA, Speculative Decode)

Usage:
    from mlir.python.yirage_mlir import YirageModule, Target
    
    # Build a module
    with YirageModule() as m:
        x = m.placeholder("x", [4096, 4096], dtype="f16")
        y = m.matmul(x, x)
        out = m.rms_norm(y)
        m.output(out)
    
    # Compile for target
    kernel = m.compile(Target.CUDA_H100)
    
    # Or export MLIR
    print(m.to_mlir())
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import io
import hashlib
import os
import tempfile
import subprocess


#==============================================================================
# Backend Target Configuration
#==============================================================================

class Target(Enum):
    """Supported compilation targets."""
    # NVIDIA CUDA
    CUDA_V100 = "cuda-sm_70"
    CUDA_A100 = "cuda-sm_80"
    CUDA_H100 = "cuda-sm_90"
    CUDA_GENERIC = "cuda"
    
    # AMD ROCm
    ROCM_MI100 = "rocm-gfx908"
    ROCM_MI250 = "rocm-gfx90a"
    ROCM_MI300X = "rocm-gfx942"
    ROCM_GENERIC = "rocm"
    
    # Intel XPU
    XPU_MAX_1550 = "xpu-pvc"
    XPU_ARC = "xpu-dg2"
    XPU_GENERIC = "xpu"
    
    # Google TPU
    TPU_V3 = "tpu-v3"
    TPU_V4 = "tpu-v4"
    TPU_V5E = "tpu-v5e"
    TPU_GENERIC = "tpu"
    
    # Huawei Ascend
    ASCEND_910A = "ascend-910a"
    ASCEND_910B = "ascend-910b"
    ASCEND_GENERIC = "ascend"
    
    # MetaX MACA
    MACA_MXC500 = "maca-mxc500"
    MACA_GENERIC = "maca"
    
    # Apple Metal
    METAL_M1 = "metal-m1"
    METAL_M2 = "metal-m2"
    METAL_M3 = "metal-m3"
    METAL_GENERIC = "metal"
    
    # CPU
    CPU_X86_AVX2 = "cpu-avx2"
    CPU_X86_AVX512 = "cpu-avx512"
    CPU_ARM_NEON = "cpu-neon"
    CPU_ARM_SVE = "cpu-sve"
    CPU_GENERIC = "cpu"
    
    # FPGA
    FPGA_XILINX = "fpga-xilinx"
    FPGA_INTEL = "fpga-intel"
    FPGA_GENERIC = "fpga"
    
    @classmethod
    def from_string(cls, s: str) -> 'Target':
        """Parse target from string."""
        s_normalized = s.lower().replace("-", "_")
        for target in cls:
            name_normalized = target.name.lower()
            value_normalized = target.value.lower().replace("-", "_")
            if name_normalized == s_normalized or value_normalized == s_normalized:
                return target
        raise ValueError(f"Unknown target: {s}")
    
    @property
    def backend(self) -> str:
        """Get the backend family (cuda, rocm, tpu, etc.)."""
        return self.value.split("-")[0]
    
    @property
    def arch(self) -> Optional[str]:
        """Get the specific architecture if any."""
        parts = self.value.split("-")
        return parts[1] if len(parts) > 1 else None


@dataclass
class TargetConfig:
    """Detailed target configuration."""
    target: Target
    compute_capability: Optional[str] = None
    opt_level: int = 3
    fast_math: bool = True
    fma: bool = True
    
    # Memory configuration
    max_shared_memory: int = 0  # 0 = auto-detect
    max_registers: int = 0
    
    # Parallelism
    num_warps: int = 0
    block_size: Tuple[int, int, int] = (128, 1, 1)
    
    # Backend-specific
    extra_flags: List[str] = field(default_factory=list)


#==============================================================================
# Tensor Types
#==============================================================================

class DType(Enum):
    """Data types."""
    F16 = "f16"
    BF16 = "bf16"
    F32 = "f32"
    F64 = "f64"
    I8 = "i8"
    I16 = "i16"
    I32 = "i32"
    I64 = "i64"
    UI8 = "ui8"
    UI32 = "ui32"
    
    @classmethod
    def from_string(cls, s: str) -> 'DType':
        s = s.lower().replace("float", "f").replace("int", "i")
        for dt in cls:
            if dt.name.lower() == s or dt.value == s:
                return dt
        raise ValueError(f"Unknown dtype: {s}")


@dataclass
class TensorType:
    """Tensor type specification."""
    shape: Tuple[int, ...]
    dtype: DType = DType.F16
    
    @property
    def mlir_type(self) -> str:
        shape_str = "x".join(str(d) for d in self.shape)
        return f"tensor<{shape_str}x{self.dtype.value}>"


#==============================================================================
# SSA Values
#==============================================================================

class Value:
    """Represents an SSA value in the MLIR graph."""
    _counter = 0
    
    def __init__(self, name: Optional[str] = None, 
                 tensor_type: Optional[TensorType] = None):
        if name is None:
            name = f"%{Value._counter}"
            Value._counter += 1
        self.name = name
        self.tensor_type = tensor_type
        self.defining_op: Optional['Operation'] = None
    
    @property
    def mlir_type(self) -> str:
        if self.tensor_type:
            return self.tensor_type.mlir_type
        return "tensor<*xf16>"
    
    def __repr__(self):
        return f"Value({self.name}, {self.mlir_type})"


#==============================================================================
# Operations
#==============================================================================

@dataclass
class Operation:
    """Base class for MLIR operations."""
    op_name: str
    inputs: List[Value]
    outputs: List[Value]
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_mlir(self) -> str:
        """Generate MLIR text for this operation."""
        raise NotImplementedError


class UnaryOp(Operation):
    """Unary operation (one input, one output)."""
    
    def to_mlir(self) -> str:
        inp = self.inputs[0]
        out = self.outputs[0]
        return f"{out.name} = {self.op_name} {inp.name} : {out.mlir_type}"


class BinaryOp(Operation):
    """Binary operation (two inputs, one output)."""
    
    def to_mlir(self) -> str:
        lhs, rhs = self.inputs[0], self.inputs[1]
        out = self.outputs[0]
        return f"{out.name} = {self.op_name} {lhs.name}, {rhs.name} : {out.mlir_type}"


class MatmulOp(Operation):
    """Matrix multiplication."""
    
    def to_mlir(self) -> str:
        lhs, rhs = self.inputs[0], self.inputs[1]
        out = self.outputs[0]
        
        attrs = []
        if self.attributes.get("transpose_lhs"):
            attrs.append("transpose_lhs = true")
        if self.attributes.get("transpose_rhs"):
            attrs.append("transpose_rhs = true")
        
        attr_str = " {" + ", ".join(attrs) + "}" if attrs else ""
        return (f"{out.name} = yirage.matmul {lhs.name}, {rhs.name}{attr_str} "
                f": {lhs.mlir_type}, {rhs.mlir_type} -> {out.mlir_type}")


class AttentionOp(Operation):
    """Attention operation."""
    
    def to_mlir(self) -> str:
        q, k, v = self.inputs[0], self.inputs[1], self.inputs[2]
        out = self.outputs[0]
        
        attrs = []
        if self.attributes.get("causal"):
            attrs.append("causal = true")
        if "scale" in self.attributes:
            attrs.append(f"scale = {self.attributes['scale']} : f32")
        
        attr_str = " {" + ", ".join(attrs) + "}" if attrs else ""
        return (f"{out.name} = yirage.attention {q.name}, {k.name}, {v.name}{attr_str} "
                f": {q.mlir_type}, {k.mlir_type}, {v.mlir_type} -> {out.mlir_type}")


class RMSNormOp(Operation):
    """RMS Normalization."""
    
    def to_mlir(self) -> str:
        inp = self.inputs[0]
        gamma = self.inputs[1] if len(self.inputs) > 1 else None
        out = self.outputs[0]
        
        eps = self.attributes.get("epsilon", 1e-6)
        
        if gamma:
            return (f"{out.name} = yirage.rms_norm {inp.name}, {gamma.name} "
                    f"{{epsilon = {eps} : f32}} : {inp.mlir_type}, {gamma.mlir_type} -> {out.mlir_type}")
        return (f"{out.name} = yirage.rms_norm {inp.name} {{epsilon = {eps} : f32}} "
                f": {inp.mlir_type} -> {out.mlir_type}")


class GatedMLPOp(Operation):
    """Gated MLP (SwiGLU/GeGLU)."""
    
    def to_mlir(self) -> str:
        inp = self.inputs[0]
        gate_w, up_w, down_w = self.inputs[1], self.inputs[2], self.inputs[3]
        out = self.outputs[0]
        
        return (f"{out.name} = yirage.gated_mlp {inp.name}, {gate_w.name}, "
                f"{up_w.name}, {down_w.name} : {inp.mlir_type}, {gate_w.mlir_type}, "
                f"{up_w.mlir_type}, {down_w.mlir_type} -> {out.mlir_type}")


class MoELayerOp(Operation):
    """Mixture of Experts layer."""
    
    def to_mlir(self) -> str:
        inp = self.inputs[0]
        gate_w = self.inputs[1]
        expert_gate = self.inputs[2]
        expert_up = self.inputs[3]
        expert_down = self.inputs[4]
        out = self.outputs[0]
        
        num_experts = self.attributes.get("num_experts", 8)
        top_k = self.attributes.get("top_k", 2)
        
        return (f"{out.name} = yirage.moe_layer {inp.name}, {gate_w.name}, "
                f"{expert_gate.name}, {expert_up.name}, {expert_down.name} "
                f"{{num_experts = {num_experts} : i64, top_k = {top_k} : i64}} "
                f": {inp.mlir_type}, {gate_w.mlir_type}, {expert_gate.mlir_type}, "
                f"{expert_up.mlir_type}, {expert_down.mlir_type} -> {out.mlir_type}")


class MLAttentionOp(Operation):
    """Multi-Latent Attention (DeepSeek MLA)."""
    
    def to_mlir(self) -> str:
        q = self.inputs[0]
        compressed_kv = self.inputs[1]
        kv_down = self.inputs[2]
        kv_up = self.inputs[3]
        out = self.outputs[0]
        
        attrs = []
        attrs.append(f"num_heads = {self.attributes.get('num_heads', 32)} : i64")
        attrs.append(f"num_kv_heads = {self.attributes.get('num_kv_heads', 8)} : i64")
        attrs.append(f"head_dim = {self.attributes.get('head_dim', 128)} : i64")
        attrs.append(f"compressed_dim = {self.attributes.get('compressed_dim', 512)} : i64")
        if self.attributes.get("causal", True):
            attrs.append("causal = true")
        
        attr_str = " {" + ", ".join(attrs) + "}"
        return (f"{out.name} = yirage.ml_attention {q.name}, {compressed_kv.name}, "
                f"{kv_down.name}, {kv_up.name}{attr_str} : ... -> {out.mlir_type}")


#==============================================================================
# Module Builder
#==============================================================================

class YirageModule:
    """Builder for YiRage MLIR modules."""
    
    def __init__(self, name: str = "yirage_module"):
        self.name = name
        self.operations: List[Operation] = []
        self.inputs: List[Value] = []
        self.outputs: List[Value] = []
        self._arg_counter = 0
    
    def __enter__(self):
        Value._counter = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    #==========================================================================
    # Input/Output
    #==========================================================================
    
    def placeholder(self, name: str, shape: List[int], 
                    dtype: Union[str, DType] = "f16") -> Value:
        """Create an input placeholder."""
        if isinstance(dtype, str):
            dtype = DType.from_string(dtype)
        
        tensor_type = TensorType(tuple(shape), dtype)
        val = Value(f"%arg{self._arg_counter}", tensor_type)
        self._arg_counter += 1
        self.inputs.append(val)
        return val
    
    def output(self, *values: Value):
        """Mark values as outputs."""
        self.outputs.extend(values)
    
    #==========================================================================
    # Basic Operations
    #==========================================================================
    
    def _infer_output_type(self, *inputs: Value) -> TensorType:
        """Infer output type from inputs (basic broadcast)."""
        # For simplicity, use first input's type
        if inputs and inputs[0].tensor_type:
            return inputs[0].tensor_type
        return TensorType((1,), DType.F16)
    
    def _make_output(self, tensor_type: Optional[TensorType] = None) -> Value:
        """Create a new output value."""
        return Value(tensor_type=tensor_type)
    
    def matmul(self, lhs: Value, rhs: Value, 
               transpose_lhs: bool = False, 
               transpose_rhs: bool = False) -> Value:
        """Matrix multiplication."""
        # Compute output shape
        lhs_shape = list(lhs.tensor_type.shape) if lhs.tensor_type else [1, 1]
        rhs_shape = list(rhs.tensor_type.shape) if rhs.tensor_type else [1, 1]
        
        if transpose_lhs:
            lhs_shape[-2], lhs_shape[-1] = lhs_shape[-1], lhs_shape[-2]
        if transpose_rhs:
            rhs_shape[-2], rhs_shape[-1] = rhs_shape[-1], rhs_shape[-2]
        
        # Output shape: batch dims + M + N
        out_shape = lhs_shape[:-1] + [rhs_shape[-1]]
        dtype = lhs.tensor_type.dtype if lhs.tensor_type else DType.F16
        
        out = self._make_output(TensorType(tuple(out_shape), dtype))
        op = MatmulOp(
            op_name="yirage.matmul",
            inputs=[lhs, rhs],
            outputs=[out],
            attributes={"transpose_lhs": transpose_lhs, "transpose_rhs": transpose_rhs}
        )
        self.operations.append(op)
        out.defining_op = op
        return out
    
    def attention(self, query: Value, key: Value, value: Value,
                  causal: bool = True, scale: Optional[float] = None) -> Value:
        """Multi-head attention."""
        out = self._make_output(query.tensor_type)
        attrs = {"causal": causal}
        if scale is not None:
            attrs["scale"] = scale
        
        op = AttentionOp(
            op_name="yirage.attention",
            inputs=[query, key, value],
            outputs=[out],
            attributes=attrs
        )
        self.operations.append(op)
        out.defining_op = op
        return out
    
    def rms_norm(self, input: Value, gamma: Optional[Value] = None,
                 epsilon: float = 1e-6) -> Value:
        """RMS normalization."""
        out = self._make_output(input.tensor_type)
        inputs = [input]
        if gamma:
            inputs.append(gamma)
        
        op = RMSNormOp(
            op_name="yirage.rms_norm",
            inputs=inputs,
            outputs=[out],
            attributes={"epsilon": epsilon}
        )
        self.operations.append(op)
        out.defining_op = op
        return out
    
    def gated_mlp(self, input: Value, gate_weight: Value,
                  up_weight: Value, down_weight: Value) -> Value:
        """Gated MLP (SwiGLU)."""
        out = self._make_output(input.tensor_type)
        op = GatedMLPOp(
            op_name="yirage.gated_mlp",
            inputs=[input, gate_weight, up_weight, down_weight],
            outputs=[out]
        )
        self.operations.append(op)
        out.defining_op = op
        return out
    
    #==========================================================================
    # Activation Functions
    #==========================================================================
    
    def _unary_op(self, name: str, input: Value) -> Value:
        out = self._make_output(input.tensor_type)
        op = UnaryOp(op_name=name, inputs=[input], outputs=[out])
        self.operations.append(op)
        out.defining_op = op
        return out
    
    def silu(self, input: Value) -> Value:
        return self._unary_op("yirage.silu", input)
    
    def gelu(self, input: Value) -> Value:
        return self._unary_op("yirage.gelu", input)
    
    def relu(self, input: Value) -> Value:
        return self._unary_op("yirage.relu", input)
    
    def sigmoid(self, input: Value) -> Value:
        return self._unary_op("yirage.sigmoid", input)
    
    def tanh(self, input: Value) -> Value:
        return self._unary_op("yirage.tanh", input)
    
    def exp(self, input: Value) -> Value:
        return self._unary_op("math.exp", input)
    
    def log(self, input: Value) -> Value:
        return self._unary_op("math.log", input)
    
    def sqrt(self, input: Value) -> Value:
        return self._unary_op("math.sqrt", input)
    
    def rsqrt(self, input: Value) -> Value:
        return self._unary_op("math.rsqrt", input)
    
    #==========================================================================
    # Binary Operations
    #==========================================================================
    
    def _binary_op(self, name: str, lhs: Value, rhs: Value) -> Value:
        out = self._make_output(lhs.tensor_type)
        op = BinaryOp(op_name=name, inputs=[lhs, rhs], outputs=[out])
        self.operations.append(op)
        out.defining_op = op
        return out
    
    def add(self, lhs: Value, rhs: Value) -> Value:
        return self._binary_op("arith.addf", lhs, rhs)
    
    def mul(self, lhs: Value, rhs: Value) -> Value:
        return self._binary_op("arith.mulf", lhs, rhs)
    
    def sub(self, lhs: Value, rhs: Value) -> Value:
        return self._binary_op("arith.subf", lhs, rhs)
    
    def div(self, lhs: Value, rhs: Value) -> Value:
        return self._binary_op("arith.divf", lhs, rhs)
    
    #==========================================================================
    # Advanced LLM Operations
    #==========================================================================
    
    def moe_layer(self, input: Value, gate_weight: Value,
                  expert_gate_weights: Value, expert_up_weights: Value,
                  expert_down_weights: Value,
                  num_experts: int = 8, top_k: int = 2) -> Value:
        """Complete Mixture of Experts layer."""
        out = self._make_output(input.tensor_type)
        op = MoELayerOp(
            op_name="yirage.moe_layer",
            inputs=[input, gate_weight, expert_gate_weights, 
                    expert_up_weights, expert_down_weights],
            outputs=[out],
            attributes={"num_experts": num_experts, "top_k": top_k}
        )
        self.operations.append(op)
        out.defining_op = op
        return out
    
    def ml_attention(self, query: Value, compressed_kv: Value,
                     kv_down_proj: Value, kv_up_proj: Value,
                     num_heads: int = 32, num_kv_heads: int = 8,
                     head_dim: int = 128, compressed_dim: int = 512,
                     causal: bool = True) -> Value:
        """Multi-Latent Attention (DeepSeek MLA)."""
        out = self._make_output(query.tensor_type)
        op = MLAttentionOp(
            op_name="yirage.ml_attention",
            inputs=[query, compressed_kv, kv_down_proj, kv_up_proj],
            outputs=[out],
            attributes={
                "num_heads": num_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "compressed_dim": compressed_dim,
                "causal": causal
            }
        )
        self.operations.append(op)
        out.defining_op = op
        return out
    
    def sliding_window_attention(self, query: Value, key: Value, value: Value,
                                  window_size: int = 4096,
                                  causal: bool = True) -> Value:
        """Sliding window attention (Mistral style)."""
        out = self._make_output(query.tensor_type)
        
        # Build operation manually
        op = Operation(
            op_name="yirage.sliding_window_attention",
            inputs=[query, key, value],
            outputs=[out],
            attributes={"window_size": window_size, "causal": causal}
        )
        
        # Override to_mlir
        q, k, v = query, key, value
        op.to_mlir = lambda: (
            f"{out.name} = yirage.sliding_window_attention {q.name}, {k.name}, {v.name} "
            f"{{window_size = {window_size} : i64, causal = {str(causal).lower()}}} "
            f": {q.mlir_type}, {k.mlir_type}, {v.mlir_type} -> {out.mlir_type}"
        )
        
        self.operations.append(op)
        out.defining_op = op
        return out
    
    #==========================================================================
    # MLIR Generation
    #==========================================================================
    
    def to_mlir(self) -> str:
        """Generate MLIR text representation."""
        output = io.StringIO()
        
        # Module header
        output.write("module {\n")
        
        # Function signature
        input_sig = ", ".join(
            f"{v.name}: {v.mlir_type}" for v in self.inputs
        )
        output_types = ", ".join(v.mlir_type for v in self.outputs) or "tensor<*xf16>"
        output.write(f"  func.func @{self.name}({input_sig}) -> ({output_types}) {{\n")
        
        # Operations
        for op in self.operations:
            output.write(f"    {op.to_mlir()}\n")
        
        # Return
        if self.outputs:
            return_vals = ", ".join(v.name for v in self.outputs)
            return_types = ", ".join(v.mlir_type for v in self.outputs)
            output.write(f"    return {return_vals} : {return_types}\n")
        else:
            output.write("    return\n")
        
        output.write("  }\n")
        output.write("}\n")
        
        return output.getvalue()
    
    #==========================================================================
    # Compilation
    #==========================================================================
    
    def compile(self, target: Union[Target, str] = Target.CUDA_H100,
                config: Optional[TargetConfig] = None) -> 'CompiledModule':
        """Compile the module for a target backend.
        
        Args:
            target: Target backend
            config: Optional detailed configuration
            
        Returns:
            CompiledModule ready for execution
        """
        if isinstance(target, str):
            target = Target.from_string(target)
        
        if config is None:
            config = TargetConfig(target=target)
        
        mlir_text = self.to_mlir()
        return CompiledModule.compile(mlir_text, config)
    
    def save(self, path: str):
        """Save MLIR to file."""
        with open(path, 'w') as f:
            f.write(self.to_mlir())


#==============================================================================
# Compiled Module
#==============================================================================

class CompiledModule:
    """A compiled MLIR module ready for execution."""
    
    # Cache for compiled modules
    _cache: Dict[str, 'CompiledModule'] = {}
    
    def __init__(self, source_mlir: str, lowered_mlir: str,
                 binary: bytes, config: TargetConfig):
        self.source_mlir = source_mlir
        self.lowered_mlir = lowered_mlir
        self.binary = binary
        self.config = config
        
    @classmethod
    def compile(cls, mlir_text: str, config: TargetConfig) -> 'CompiledModule':
        """Compile MLIR text to a module."""
        # Check cache
        cache_key = hashlib.md5(
            (mlir_text + str(config.target.value)).encode()
        ).hexdigest()
        
        if cache_key in cls._cache:
            return cls._cache[cache_key]
        
        # Determine pipeline based on target
        backend = config.target.backend
        pipelines = {
            "cuda": "yirage-cuda-pipeline",
            "rocm": "yirage-rocm-pipeline",
            "tpu": "yirage-tpu-pipeline",
            "ascend": "yirage-ascend-pipeline",
            "xpu": "yirage-xpu-pipeline",
            "maca": "yirage-maca-pipeline",
            "metal": "yirage-metal-pipeline",
            "cpu": "yirage-cpu-pipeline",
            "fpga": "yirage-fpga-pipeline",
        }
        
        pipeline = pipelines.get(backend, "yirage-generic-pipeline")
        
        # For now, return module with source
        # Full compilation would invoke yirage-opt
        module = cls(
            source_mlir=mlir_text,
            lowered_mlir=mlir_text,  # Would be lowered by yirage-opt
            binary=b"",  # Would be generated binary
            config=config
        )
        
        cls._cache[cache_key] = module
        return module
    
    def __call__(self, *args, **kwargs):
        """Execute the compiled module.
        
        Note: Requires native runtime integration.
        """
        raise NotImplementedError(
            "Native execution requires C++ runtime. "
            "Export to MLIR and use yirage-opt + runtime."
        )
    
    def get_ptx(self) -> str:
        """Get PTX code (CUDA only)."""
        if self.config.target.backend != "cuda":
            raise ValueError("PTX only available for CUDA targets")
        # Would generate PTX via BinaryGen
        return "// PTX generation requires C++ compilation"
    
    def get_hsaco(self) -> bytes:
        """Get HSACO binary (ROCm only)."""
        if self.config.target.backend != "rocm":
            raise ValueError("HSACO only available for ROCm targets")
        return self.binary
    
    def get_stablehlo(self) -> str:
        """Get StableHLO representation (TPU)."""
        if self.config.target.backend != "tpu":
            raise ValueError("StableHLO for TPU targets")
        # Would lower to StableHLO
        return self.lowered_mlir


#==============================================================================
# Convenience Functions
#==============================================================================

def build_attention_block(batch: int, seq_len: int, num_heads: int,
                          head_dim: int, hidden_dim: int,
                          dtype: str = "f16") -> YirageModule:
    """Build a standard transformer attention block."""
    m = YirageModule("attention_block")
    
    # Inputs
    x = m.placeholder("x", [batch, seq_len, hidden_dim], dtype)
    wq = m.placeholder("wq", [hidden_dim, num_heads * head_dim], dtype)
    wk = m.placeholder("wk", [hidden_dim, num_heads * head_dim], dtype)
    wv = m.placeholder("wv", [hidden_dim, num_heads * head_dim], dtype)
    wo = m.placeholder("wo", [num_heads * head_dim, hidden_dim], dtype)
    norm_weight = m.placeholder("norm_weight", [hidden_dim], dtype)
    
    # RMS Norm
    x_norm = m.rms_norm(x, norm_weight)
    
    # QKV projections
    q = m.matmul(x_norm, wq)  # [batch, seq, num_heads * head_dim]
    k = m.matmul(x_norm, wk)
    v = m.matmul(x_norm, wv)
    
    # Attention (simplified - would need reshape for multi-head)
    attn_out = m.attention(q, k, v, causal=True)
    
    # Output projection
    out = m.matmul(attn_out, wo)
    
    # Residual
    out = m.add(x, out)
    
    m.output(out)
    return m


def build_moe_block(batch: int, seq_len: int, hidden_dim: int,
                    intermediate_dim: int, num_experts: int = 8,
                    top_k: int = 2, dtype: str = "f16") -> YirageModule:
    """Build a Mixture of Experts block."""
    m = YirageModule("moe_block")
    
    # Inputs
    x = m.placeholder("x", [batch * seq_len, hidden_dim], dtype)
    gate_w = m.placeholder("gate_weight", [hidden_dim, num_experts], dtype)
    expert_gate = m.placeholder("expert_gate", [num_experts, hidden_dim, intermediate_dim], dtype)
    expert_up = m.placeholder("expert_up", [num_experts, hidden_dim, intermediate_dim], dtype)
    expert_down = m.placeholder("expert_down", [num_experts, intermediate_dim, hidden_dim], dtype)
    
    # MoE layer
    out = m.moe_layer(x, gate_w, expert_gate, expert_up, expert_down,
                      num_experts=num_experts, top_k=top_k)
    
    m.output(out)
    return m


def build_mla_block(batch: int, seq_len: int, num_heads: int,
                    num_kv_heads: int, head_dim: int, 
                    compressed_dim: int, hidden_dim: int,
                    dtype: str = "f16") -> YirageModule:
    """Build a Multi-Latent Attention block (DeepSeek style)."""
    m = YirageModule("mla_block")
    
    # Inputs
    q = m.placeholder("query", [batch, num_heads, seq_len, head_dim], dtype)
    c_kv = m.placeholder("compressed_kv", [batch, seq_len, compressed_dim], dtype)
    kv_down = m.placeholder("kv_down_proj", [num_kv_heads * head_dim * 2, compressed_dim], dtype)
    kv_up = m.placeholder("kv_up_proj", [compressed_dim, num_kv_heads * head_dim * 2], dtype)
    
    # MLA
    out = m.ml_attention(q, c_kv, kv_down, kv_up,
                         num_heads=num_heads,
                         num_kv_heads=num_kv_heads,
                         head_dim=head_dim,
                         compressed_dim=compressed_dim,
                         causal=True)
    
    m.output(out)
    return m


#==============================================================================
# Target Detection
#==============================================================================

def detect_available_targets() -> List[Target]:
    """Detect available compilation targets on this system."""
    available = []
    
    # Check CUDA
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            output = result.stdout.decode()
            if "H100" in output:
                available.append(Target.CUDA_H100)
            elif "A100" in output:
                available.append(Target.CUDA_A100)
            elif "V100" in output:
                available.append(Target.CUDA_V100)
            else:
                available.append(Target.CUDA_GENERIC)
    except:
        pass
    
    # Check ROCm
    try:
        result = subprocess.run(["rocm-smi"], capture_output=True)
        if result.returncode == 0:
            available.append(Target.ROCM_GENERIC)
    except:
        pass
    
    # CPU is always available
    available.append(Target.CPU_GENERIC)
    
    return available


def get_best_target() -> Target:
    """Get the best available target for this system."""
    available = detect_available_targets()
    
    # Priority order
    priority = [
        Target.CUDA_H100, Target.CUDA_A100, Target.CUDA_V100, Target.CUDA_GENERIC,
        Target.ROCM_MI300X, Target.ROCM_MI250, Target.ROCM_MI100, Target.ROCM_GENERIC,
        Target.TPU_V5E, Target.TPU_V4, Target.TPU_GENERIC,
        Target.ASCEND_910B, Target.ASCEND_GENERIC,
        Target.XPU_MAX_1550, Target.XPU_GENERIC,
        Target.MACA_MXC500, Target.MACA_GENERIC,
        Target.METAL_M3, Target.METAL_M2, Target.METAL_GENERIC,
        Target.CPU_X86_AVX512, Target.CPU_X86_AVX2, Target.CPU_GENERIC,
    ]
    
    for target in priority:
        if target in available:
            return target
    
    return Target.CPU_GENERIC


#==============================================================================
# Main
#==============================================================================

if __name__ == "__main__":
    # Example: Build and print MLIR for attention block
    print("=== Attention Block ===")
    attn = build_attention_block(
        batch=1, seq_len=2048, num_heads=32, 
        head_dim=128, hidden_dim=4096
    )
    print(attn.to_mlir())
    
    print("\n=== MoE Block ===")
    moe = build_moe_block(
        batch=1, seq_len=2048, hidden_dim=4096,
        intermediate_dim=11008, num_experts=8, top_k=2
    )
    print(moe.to_mlir())
    
    print("\n=== MLA Block ===")
    mla = build_mla_block(
        batch=1, seq_len=2048, num_heads=32,
        num_kv_heads=8, head_dim=128, 
        compressed_dim=512, hidden_dim=4096
    )
    print(mla.to_mlir())
    
    print("\n=== Available Targets ===")
    targets = detect_available_targets()
    for t in targets:
        print(f"  - {t.name}: {t.value}")
    
    best = get_best_target()
    print(f"\nBest target: {best.name}")
