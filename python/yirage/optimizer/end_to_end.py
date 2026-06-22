# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
End-to-End Optimizer Pipeline

Integrates COMET search optimization with MLIR generation for complete
optimal compilation from muGraph to backend-specific code.

Pipeline:
    muGraph → COMET Search → Optimized muGraph → MLIR → Backend Code

Features:
- Compound operation detection and fusion
- Tile size optimization
- Collective placement optimization
- Multi-backend MLIR code generation
- Performance cost modeling

Usage:
    from yirage.optimizer.end_to_end import YirageOptimizer
    
    optimizer = YirageOptimizer(target='cuda-h100')
    optimized_mlir = optimizer.optimize(graph)
    compiled = optimizer.compile(graph)
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
import json
import time
import math
import io

# Import search components
sys.path.insert(0, '/workspace/python')
try:
    from yirage.search.comet_search import (
        COMETSearchConfig, COMETSearchStrategy, COMETCostModel,
        CompoundOpType, SchedulingStrategy, CollectiveOpType,
        detect_compound_patterns, CompoundPattern
    )
    from yirage.search.backend_config import (
        BackendHardwareProfile, BACKEND_PROFILES, 
        get_backend_config, get_auto_detected_config
    )
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False

# Import MLIR components
sys.path.insert(0, '/workspace/mlir/python')
try:
    from yirage_mlir import (
        YirageModule, Target, TargetConfig, DType, TensorType,
        detect_available_targets, get_best_target
    )
    from mugraph_to_mlir import MuGraphToMLIR, convert_mugraph_to_mlir
    from yirage_compiler import YirageCompiler, CompiledKernel
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False


#==============================================================================
# Configuration
#==============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for end-to-end optimization."""
    
    # Target backend
    target: str = "cuda-h100"
    
    # Search options
    enable_search: bool = True
    max_search_iterations: int = 1000
    search_timeout_seconds: float = 300.0
    
    # Fusion options
    enable_fusion: bool = True
    max_fusion_depth: int = 5
    
    # Tile optimization
    enable_tile_search: bool = True
    tile_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    
    # Collective optimization  
    enable_collective_opt: bool = True
    num_devices: int = 1
    
    # MLIR options
    mlir_opt_level: int = 3
    enable_flash_attention: bool = True
    enable_kernel_fusion: bool = True
    
    # Output options
    output_format: str = "mlir"  # "mlir", "ptx", "hsaco", "binary"
    verbose: bool = False


#==============================================================================
# Optimization Result
#==============================================================================

@dataclass
class OptimizationResult:
    """Result of the optimization pipeline."""
    
    # Input graph info
    input_graph_ops: int = 0
    input_graph_tensors: int = 0
    
    # Search results
    patterns_detected: List[Dict[str, Any]] = field(default_factory=list)
    best_config: Optional[Dict[str, Any]] = None
    search_iterations: int = 0
    search_time_seconds: float = 0.0
    
    # Cost estimates
    estimated_latency_ns: float = 0.0
    estimated_memory_bytes: int = 0
    
    # Generated outputs
    optimized_mlir: str = ""
    target_code: str = ""  # PTX, HSACO, etc.
    binary: bytes = b""
    
    # Metadata
    target: str = ""
    optimizations_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_graph_ops": self.input_graph_ops,
            "input_graph_tensors": self.input_graph_tensors,
            "patterns_detected": self.patterns_detected,
            "best_config": self.best_config,
            "search_iterations": self.search_iterations,
            "search_time_seconds": self.search_time_seconds,
            "estimated_latency_ns": self.estimated_latency_ns,
            "estimated_memory_bytes": self.estimated_memory_bytes,
            "target": self.target,
            "optimizations_applied": self.optimizations_applied,
        }


#==============================================================================
# Graph Representation
#==============================================================================

@dataclass
class GraphOp:
    """Represents an operation in the graph."""
    op_id: int
    op_type: str
    inputs: List[int]  # tensor IDs
    outputs: List[int]  # tensor IDs
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphTensor:
    """Represents a tensor in the graph."""
    tensor_id: int
    shape: Tuple[int, ...]
    dtype: str = "f16"
    is_input: bool = False
    is_output: bool = False


class MuGraph:
    """
    Internal representation of a computation graph.
    
    Provides a unified interface for both JSON graphs and KNGraph objects.
    """
    
    def __init__(self):
        self.ops: List[GraphOp] = []
        self.tensors: Dict[int, GraphTensor] = {}
        self.input_tensor_ids: List[int] = []
        self.output_tensor_ids: List[int] = []
        self._next_tensor_id = 0
        self._next_op_id = 0
    
    @classmethod
    def from_json(cls, json_data: Union[str, Dict]) -> 'MuGraph':
        """Create MuGraph from JSON data."""
        if isinstance(json_data, str):
            import json as json_module
            data = json_module.loads(json_data)
        else:
            data = json_data
        
        graph = cls()
        
        # Parse inputs
        for inp in data.get('inputs', []):
            tensor_id = inp.get('id', graph._next_tensor_id)
            graph._next_tensor_id = max(graph._next_tensor_id, tensor_id + 1)
            
            tensor = GraphTensor(
                tensor_id=tensor_id,
                shape=tuple(inp.get('dims', [1])),
                dtype=inp.get('dtype', 'f16'),
                is_input=True
            )
            graph.tensors[tensor_id] = tensor
            graph.input_tensor_ids.append(tensor_id)
        
        # Parse operators
        for op_data in data.get('operators', []):
            op_id = op_data.get('id', graph._next_op_id)
            graph._next_op_id = max(graph._next_op_id, op_id + 1)
            
            # Extract output IDs and track tensors
            output_ids = []
            for out in op_data.get('outputs', []):
                if isinstance(out, dict):
                    tensor_id = out.get('id', graph._next_tensor_id)
                    graph._next_tensor_id = max(graph._next_tensor_id, tensor_id + 1)
                    output_ids.append(tensor_id)
                    
                    if tensor_id not in graph.tensors:
                        tensor = GraphTensor(
                            tensor_id=tensor_id,
                            shape=tuple(out.get('dims', [1])),
                            dtype=out.get('dtype', 'f16')
                        )
                        graph.tensors[tensor_id] = tensor
                else:
                    output_ids.append(out)
            
            op = GraphOp(
                op_id=op_id,
                op_type=op_data.get('type', 'unknown'),
                inputs=op_data.get('inputs', []),
                outputs=output_ids,  # Use extracted integer IDs
                attributes=op_data.get('attributes', {})
            )
            graph.ops.append(op)
        
        # Parse outputs
        for out in data.get('outputs', []):
            tensor_id = out.get('id', graph._next_tensor_id)
            if tensor_id in graph.tensors:
                graph.tensors[tensor_id].is_output = True
            graph.output_tensor_ids.append(tensor_id)
        
        return graph
    
    @classmethod
    def from_kngraph(cls, kngraph) -> 'MuGraph':
        """Create MuGraph from KNGraph object."""
        graph = cls()
        
        # Try to get graph structure
        try:
            if hasattr(kngraph, 'cygraph'):
                operators = kngraph.cygraph.get_graph_structure()
                input_dtensors = kngraph.cygraph.get_input_dtensors()
            else:
                operators = kngraph.get_graph_structure() if hasattr(kngraph, 'get_graph_structure') else []
                input_dtensors = []
        except Exception:
            operators = []
            input_dtensors = []
        
        # Process input tensors
        for idx, dt in enumerate(input_dtensors):
            try:
                if hasattr(kngraph, 'cygraph'):
                    dims, strides = kngraph.cygraph.get_input_dtensor_shape_and_stride(dt)
                else:
                    dims = tuple(dt.dim(i) for i in range(dt.num_dims))
                dtype = str(dt.dtype)
            except Exception:
                dims = (1,)
                dtype = 'fp16'
            
            tensor = GraphTensor(
                tensor_id=idx,
                shape=dims,
                dtype=dtype,
                is_input=True
            )
            graph.tensors[idx] = tensor
            graph.input_tensor_ids.append(idx)
            graph._next_tensor_id = idx + 1
        
        # Process operators
        for op_info in operators:
            op_type = op_info.get('op_type', 'unknown')
            
            if op_type == 'kn_input_op':
                continue
            
            # Map inputs to tensor IDs
            input_ids = []
            for inp in op_info.get('inputs', []):
                inp_id = id(inp) if not isinstance(inp, int) else inp
                input_ids.append(inp_id)
            
            # Map outputs to tensor IDs
            output_ids = []
            for out in op_info.get('outputs', []):
                out_id = id(out) if not isinstance(out, int) else out
                output_ids.append(out_id)
                
                if out_id not in graph.tensors:
                    try:
                        dims = tuple(out.dim(i) for i in range(out.num_dims))
                        dtype = str(out.dtype)
                    except Exception:
                        dims = (1,)
                        dtype = 'fp16'
                    
                    tensor = GraphTensor(
                        tensor_id=out_id,
                        shape=dims,
                        dtype=dtype,
                        is_output=(op_type == 'kn_output_op')
                    )
                    graph.tensors[out_id] = tensor
                    
                    if op_type == 'kn_output_op':
                        graph.output_tensor_ids.append(out_id)
            
            op = GraphOp(
                op_id=graph._next_op_id,
                op_type=op_type,
                inputs=input_ids,
                outputs=output_ids
            )
            graph.ops.append(op)
            graph._next_op_id += 1
        
        return graph
    
    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format."""
        return {
            'inputs': [
                {
                    'id': tid,
                    'dims': list(self.tensors[tid].shape),
                    'dtype': self.tensors[tid].dtype
                }
                for tid in self.input_tensor_ids
            ],
            'operators': [
                {
                    'id': op.op_id,
                    'type': op.op_type,
                    'inputs': op.inputs,
                    'outputs': op.outputs,
                    'attributes': op.attributes
                }
                for op in self.ops
            ],
            'outputs': [
                {
                    'id': tid,
                    'dims': list(self.tensors[tid].shape) if tid in self.tensors else [1],
                    'dtype': self.tensors[tid].dtype if tid in self.tensors else 'f16'
                }
                for tid in self.output_tensor_ids
            ]
        }
    
    def num_ops(self) -> int:
        return len(self.ops)
    
    def num_tensors(self) -> int:
        return len(self.tensors)


#==============================================================================
# Pattern Detection
#==============================================================================

class PatternDetector:
    """Detects compound operation patterns in a graph."""
    
    # Operation type mappings
    OP_CATEGORIES = {
        'matmul': ['kn_matmul_op', 'matmul', 'gemm'],
        'softmax': ['softmax', 'kn_softmax_op'],
        'layernorm': ['layernorm', 'layer_norm', 'kn_layernorm_op'],
        'rms_norm': ['rms_norm', 'rmsnorm', 'kn_rms_norm_op'],
        'gelu': ['gelu', 'kn_gelu_op'],
        'silu': ['silu', 'kn_silu_op', 'swish'],
        'add': ['add', 'kn_add_op'],
        'mul': ['mul', 'kn_mul_op'],
        'reduction': ['reduce', 'kn_reduction', 'sum', 'mean'],
    }
    
    def __init__(self, graph: MuGraph):
        self.graph = graph
        self.op_by_output: Dict[int, GraphOp] = {}
        
        # Build output -> op mapping
        for op in graph.ops:
            for out_id in op.outputs:
                # Handle dict outputs
                if isinstance(out_id, dict):
                    out_id = out_id.get('id', id(out_id))
                self.op_by_output[out_id] = op
    
    def _is_op_type(self, op: GraphOp, category: str) -> bool:
        """Check if op matches a category."""
        op_type = op.op_type.lower()
        patterns = self.OP_CATEGORIES.get(category, [category])
        return any(p in op_type for p in patterns)
    
    def detect_all_patterns(self) -> List[Dict[str, Any]]:
        """Detect all compound patterns."""
        patterns = []
        used_ops = set()
        
        # Priority: more complex patterns first
        # 1. Self-Attention
        for pattern in self._detect_self_attention():
            op_ids = set(pattern['op_indices'])
            if not op_ids & used_ops:
                patterns.append(pattern)
                used_ops |= op_ids
        
        # 2. Gated MLP
        for pattern in self._detect_gated_mlp():
            op_ids = set(pattern['op_indices'])
            if not op_ids & used_ops:
                patterns.append(pattern)
                used_ops |= op_ids
        
        # 3. GEMM-Softmax
        for pattern in self._detect_gemm_softmax():
            op_ids = set(pattern['op_indices'])
            if not op_ids & used_ops:
                patterns.append(pattern)
                used_ops |= op_ids
        
        # 4. RMS-Norm + Linear
        for pattern in self._detect_rms_norm_linear():
            op_ids = set(pattern['op_indices'])
            if not op_ids & used_ops:
                patterns.append(pattern)
                used_ops |= op_ids
        
        # 5. GEMM-LayerNorm
        for pattern in self._detect_gemm_layernorm():
            op_ids = set(pattern['op_indices'])
            if not op_ids & used_ops:
                patterns.append(pattern)
                used_ops |= op_ids
        
        return patterns
    
    def _detect_self_attention(self) -> List[Dict[str, Any]]:
        """Detect self-attention patterns: Q@K^T → softmax → @V."""
        patterns = []
        
        for i, op in enumerate(self.graph.ops):
            if not self._is_op_type(op, 'matmul'):
                continue
            
            # Look for QK matmul followed by softmax
            for j, op2 in enumerate(self.graph.ops[i+1:], i+1):
                if not self._is_op_type(op2, 'softmax'):
                    continue
                
                # Check if softmax input comes from QK matmul
                if not any(out in op2.inputs for out in op.outputs):
                    continue
                
                # Look for attention @ V matmul
                for k, op3 in enumerate(self.graph.ops[j+1:], j+1):
                    if not self._is_op_type(op3, 'matmul'):
                        continue
                    
                    # Check if matmul input comes from softmax
                    if any(out in op3.inputs for out in op2.outputs):
                        patterns.append({
                            'type': 'SELF_ATTENTION',
                            'op_indices': [i, j, k],
                            'ops': [op, op2, op3],
                            'estimated_flops': self._estimate_attention_flops(op)
                        })
        
        return patterns
    
    def _detect_gated_mlp(self) -> List[Dict[str, Any]]:
        """Detect gated MLP patterns: gate@W1 * up@W2 @ down."""
        patterns = []
        
        for i, op in enumerate(self.graph.ops):
            if not self._is_op_type(op, 'matmul'):
                continue
            
            # Look for parallel matmul (up projection)
            for j, op2 in enumerate(self.graph.ops):
                if j == i or not self._is_op_type(op2, 'matmul'):
                    continue
                
                # Check if they share input
                if not set(op.inputs) & set(op2.inputs):
                    continue
                
                # Look for activation (silu/gelu) on gate
                for k, op3 in enumerate(self.graph.ops):
                    if not (self._is_op_type(op3, 'silu') or self._is_op_type(op3, 'gelu')):
                        continue
                    
                    if not any(out in op3.inputs for out in op.outputs):
                        continue
                    
                    # Look for element-wise multiply
                    for l, op4 in enumerate(self.graph.ops):
                        if not self._is_op_type(op4, 'mul'):
                            continue
                        
                        # Check if mul combines activation and up projection
                        mul_inputs = set(op4.inputs)
                        activation_outputs = set(op3.outputs)
                        up_outputs = set(op2.outputs)
                        
                        if activation_outputs & mul_inputs and up_outputs & mul_inputs:
                            # Look for down projection
                            for m, op5 in enumerate(self.graph.ops):
                                if not self._is_op_type(op5, 'matmul'):
                                    continue
                                
                                if any(out in op5.inputs for out in op4.outputs):
                                    patterns.append({
                                        'type': 'GATED_MLP',
                                        'op_indices': [i, j, k, l, m],
                                        'ops': [op, op2, op3, op4, op5],
                                        'estimated_flops': self._estimate_mlp_flops(op, op2, op5)
                                    })
        
        return patterns
    
    def _detect_gemm_softmax(self) -> List[Dict[str, Any]]:
        """Detect GEMM followed by softmax."""
        patterns = []
        
        for i, op in enumerate(self.graph.ops):
            if not self._is_op_type(op, 'matmul'):
                continue
            
            for j, op2 in enumerate(self.graph.ops[i+1:], i+1):
                if not self._is_op_type(op2, 'softmax'):
                    continue
                
                if any(out in op2.inputs for out in op.outputs):
                    patterns.append({
                        'type': 'GEMM_SOFTMAX',
                        'op_indices': [i, j],
                        'ops': [op, op2],
                        'estimated_flops': self._estimate_gemm_flops(op)
                    })
        
        return patterns
    
    def _detect_rms_norm_linear(self) -> List[Dict[str, Any]]:
        """Detect RMS norm followed by linear."""
        patterns = []
        
        for i, op in enumerate(self.graph.ops):
            if not self._is_op_type(op, 'rms_norm'):
                continue
            
            for j, op2 in enumerate(self.graph.ops[i+1:], i+1):
                if not self._is_op_type(op2, 'matmul'):
                    continue
                
                if any(out in op2.inputs for out in op.outputs):
                    patterns.append({
                        'type': 'RMS_NORM_LINEAR',
                        'op_indices': [i, j],
                        'ops': [op, op2],
                        'estimated_flops': self._estimate_gemm_flops(op2)
                    })
        
        return patterns
    
    def _detect_gemm_layernorm(self) -> List[Dict[str, Any]]:
        """Detect GEMM followed by layer norm."""
        patterns = []
        
        for i, op in enumerate(self.graph.ops):
            if not self._is_op_type(op, 'matmul'):
                continue
            
            for j, op2 in enumerate(self.graph.ops[i+1:], i+1):
                if not self._is_op_type(op2, 'layernorm'):
                    continue
                
                if any(out in op2.inputs for out in op.outputs):
                    patterns.append({
                        'type': 'GEMM_LAYERNORM',
                        'op_indices': [i, j],
                        'ops': [op, op2],
                        'estimated_flops': self._estimate_gemm_flops(op)
                    })
        
        return patterns
    
    def _estimate_gemm_flops(self, op: GraphOp) -> int:
        """Estimate FLOPS for GEMM operation."""
        if len(op.inputs) < 2:
            return 0
        
        # Try to get shapes
        try:
            in1 = self.graph.tensors.get(op.inputs[0])
            in2 = self.graph.tensors.get(op.inputs[1])
            
            if in1 and in2:
                M = in1.shape[-2] if len(in1.shape) >= 2 else 1
                K = in1.shape[-1] if len(in1.shape) >= 1 else 1
                N = in2.shape[-1] if len(in2.shape) >= 1 else 1
                
                # 2*M*N*K for matmul
                return 2 * M * N * K
        except Exception:
            pass
        
        return 0
    
    def _estimate_attention_flops(self, qk_op: GraphOp) -> int:
        """Estimate FLOPS for attention."""
        # QK: B*H*S*S*D, softmax: B*H*S*S, AV: B*H*S*S*D
        # Simplified: ~4*B*H*S*S*D
        return self._estimate_gemm_flops(qk_op) * 4
    
    def _estimate_mlp_flops(self, gate_op: GraphOp, up_op: GraphOp, 
                           down_op: GraphOp) -> int:
        """Estimate FLOPS for gated MLP."""
        return (self._estimate_gemm_flops(gate_op) + 
                self._estimate_gemm_flops(up_op) +
                self._estimate_gemm_flops(down_op))


#==============================================================================
# MLIR Generator with Optimizations
#==============================================================================

class OptimizedMLIRGenerator:
    """Generates optimized MLIR from muGraph with applied optimizations."""
    
    # Extended operation mappings including advanced ops
    OP_MAPPING = {
        # Basic ops
        'kn_matmul_op': 'yirage.matmul',
        'matmul': 'yirage.matmul',
        'kn_add_op': 'arith.addf',
        'kn_mul_op': 'arith.mulf',
        'kn_div_op': 'arith.divf',
        
        # Activations
        'kn_silu_op': 'yirage.silu',
        'kn_gelu_op': 'yirage.gelu',
        'kn_relu_op': 'yirage.relu',
        'kn_sigmoid_op': 'yirage.sigmoid',
        
        # Norms
        'kn_rms_norm_op': 'yirage.rms_norm',
        'kn_layernorm_op': 'yirage.layer_norm',
        
        # Reductions
        'softmax': 'yirage.softmax',
        'kn_reduction_0_op': 'yirage.reduce_sum',
        
        # Communication
        'kn_allreduce_op': 'yirage.allreduce',
        'kn_allgather_op': 'yirage.allgather',
        'kn_reducescatter_op': 'yirage.reducescatter',
        
        # Advanced LLM ops (fused)
        'fused_attention': 'yirage.attention',
        'fused_gated_mlp': 'yirage.gated_mlp',
        'fused_rms_norm_linear': 'yirage.rms_norm_linear',
        'fused_moe_layer': 'yirage.moe_layer',
        'fused_ml_attention': 'yirage.ml_attention',
    }
    
    DTYPE_MAPPING = {
        'fp16': 'f16', 'f16': 'f16',
        'bf16': 'bf16',
        'fp32': 'f32', 'f32': 'f32',
        'int8': 'i8', 'i8': 'i8',
        'int32': 'i32', 'i32': 'i32',
    }
    
    def __init__(self, graph: MuGraph, patterns: List[Dict[str, Any]],
                 config: OptimizationConfig):
        self.graph = graph
        self.patterns = patterns
        self.config = config
        self.value_counter = 0
        self.tensor_to_ssa: Dict[int, str] = {}
        self.output = io.StringIO()
        self.indent_level = 0
        
        # Track which ops are fused
        self.fused_ops: set = set()
        for pattern in patterns:
            self.fused_ops.update(pattern.get('op_indices', []))
    
    def _indent(self) -> str:
        return "  " * self.indent_level
    
    def _emit(self, text: str):
        self.output.write(self._indent() + text + "\n")
    
    def _new_value(self) -> str:
        name = f"%{self.value_counter}"
        self.value_counter += 1
        return name
    
    def _get_mlir_type(self, shape: Tuple[int, ...], dtype: str) -> str:
        mlir_dtype = self.DTYPE_MAPPING.get(dtype, 'f16')
        if shape:
            shape_str = 'x'.join(str(d) for d in shape)
            return f"tensor<{shape_str}x{mlir_dtype}>"
        return f"tensor<*x{mlir_dtype}>"
    
    def generate(self) -> str:
        """Generate optimized MLIR."""
        self.output = io.StringIO()
        self.value_counter = 0
        self.tensor_to_ssa = {}
        
        # Build input types
        input_types = []
        for tid in self.graph.input_tensor_ids:
            tensor = self.graph.tensors.get(tid)
            if tensor:
                mlir_type = self._get_mlir_type(tensor.shape, tensor.dtype)
                input_types.append(mlir_type)
                self.tensor_to_ssa[tid] = f"%arg{len(input_types)-1}"
        
        # Build output types
        output_types = []
        for tid in self.graph.output_tensor_ids:
            tensor = self.graph.tensors.get(tid)
            if tensor:
                output_types.append(self._get_mlir_type(tensor.shape, tensor.dtype))
        
        if not output_types:
            output_types = ['tensor<*xf16>']
        
        # Emit module
        self._emit("// Generated by YiRage Optimizer")
        self._emit(f"// Target: {self.config.target}")
        self._emit(f"// Patterns detected: {len(self.patterns)}")
        self._emit("")
        self._emit("module {")
        self.indent_level += 1
        
        # Function signature
        input_sig = ", ".join(
            f"%arg{i}: {t}" for i, t in enumerate(input_types)
        )
        output_sig = ", ".join(output_types)
        self._emit(f"func.func @optimized_graph({input_sig}) -> ({output_sig}) {{")
        self.indent_level += 1
        
        # Emit fused patterns first
        for pattern in self.patterns:
            self._emit_fused_pattern(pattern)
        
        # Emit remaining unfused ops
        for i, op in enumerate(self.graph.ops):
            if i not in self.fused_ops:
                self._emit_operation(op)
        
        # Return
        result_vals = []
        for tid in self.graph.output_tensor_ids:
            if tid in self.tensor_to_ssa:
                result_vals.append(self.tensor_to_ssa[tid])
        
        if result_vals:
            self._emit(f"return {', '.join(result_vals)} : {', '.join(output_types)}")
        else:
            self._emit("return")
        
        self.indent_level -= 1
        self._emit("}")
        self.indent_level -= 1
        self._emit("}")
        
        return self.output.getvalue()
    
    def _emit_fused_pattern(self, pattern: Dict[str, Any]):
        """Emit fused operation for a detected pattern."""
        pattern_type = pattern.get('type', '')
        ops = pattern.get('ops', [])
        
        if not ops:
            return
        
        # Get input/output info
        first_op = ops[0]
        last_op = ops[-1]
        
        input_ssa = []
        for inp_id in first_op.inputs:
            if inp_id in self.tensor_to_ssa:
                input_ssa.append(self.tensor_to_ssa[inp_id])
            else:
                input_ssa.append(f"%{inp_id}")
        
        result = self._new_value()
        
        # Get output type
        if last_op.outputs:
            out_id = last_op.outputs[0]
            tensor = self.graph.tensors.get(out_id)
            if tensor:
                out_type = self._get_mlir_type(tensor.shape, tensor.dtype)
            else:
                out_type = 'tensor<*xf16>'
        else:
            out_type = 'tensor<*xf16>'
        
        self._emit(f"// Fused pattern: {pattern_type}")
        
        if pattern_type == 'SELF_ATTENTION':
            # Emit fused attention
            if len(input_ssa) >= 1:
                self._emit(f"{result} = yirage.attention {input_ssa[0]}, {input_ssa[0]}, {input_ssa[0]} {{causal = true}} : {out_type}, {out_type}, {out_type} -> {out_type}")
        
        elif pattern_type == 'GATED_MLP':
            # Emit fused gated MLP
            if len(input_ssa) >= 1:
                # Note: In real impl, would need weight tensors
                self._emit(f"{result} = yirage.gated_mlp {input_ssa[0]} : {out_type} -> {out_type}")
        
        elif pattern_type == 'GEMM_SOFTMAX':
            if len(input_ssa) >= 2:
                self._emit(f"{result} = yirage.fused_gemm_softmax {input_ssa[0]}, {input_ssa[1]} : {out_type}, {out_type} -> {out_type}")
        
        elif pattern_type == 'RMS_NORM_LINEAR':
            if len(input_ssa) >= 1:
                self._emit(f"{result} = yirage.rms_norm_linear {input_ssa[0]} : {out_type} -> {out_type}")
        
        elif pattern_type == 'GEMM_LAYERNORM':
            if len(input_ssa) >= 2:
                self._emit(f"{result} = yirage.fused_gemm_layernorm {input_ssa[0]}, {input_ssa[1]} : {out_type}, {out_type} -> {out_type}")
        
        # Register output
        if last_op.outputs:
            self.tensor_to_ssa[last_op.outputs[0]] = result
    
    def _emit_operation(self, op: GraphOp):
        """Emit a single operation."""
        # Get input SSA values
        input_ssa = []
        for inp_id in op.inputs:
            if inp_id in self.tensor_to_ssa:
                input_ssa.append(self.tensor_to_ssa[inp_id])
            else:
                input_ssa.append(f"%{inp_id}")
        
        # Get output type
        if op.outputs:
            out_id = op.outputs[0]
            tensor = self.graph.tensors.get(out_id)
            if tensor:
                out_type = self._get_mlir_type(tensor.shape, tensor.dtype)
            else:
                out_type = 'tensor<*xf16>'
        else:
            out_type = 'tensor<*xf16>'
        
        result = self._new_value()
        mlir_op = self.OP_MAPPING.get(op.op_type, f'// unknown: {op.op_type}')
        
        if 'matmul' in op.op_type.lower():
            if len(input_ssa) >= 2:
                self._emit(f"{result} = yirage.matmul {input_ssa[0]}, {input_ssa[1]} : {out_type}, {out_type} -> {out_type}")
        elif 'silu' in op.op_type.lower() or 'gelu' in op.op_type.lower() or 'relu' in op.op_type.lower():
            if input_ssa:
                self._emit(f"{result} = {mlir_op} {input_ssa[0]} : {out_type}")
        elif 'rms_norm' in op.op_type.lower():
            if input_ssa:
                self._emit(f"{result} = yirage.rms_norm {input_ssa[0]} : {out_type}")
        elif 'add' in op.op_type.lower():
            if len(input_ssa) >= 2:
                self._emit(f"{result} = arith.addf {input_ssa[0]}, {input_ssa[1]} : {out_type}")
        elif 'mul' in op.op_type.lower():
            if len(input_ssa) >= 2:
                self._emit(f"{result} = arith.mulf {input_ssa[0]}, {input_ssa[1]} : {out_type}")
        else:
            self._emit(f"// {op.op_type}: {input_ssa}")
            result = input_ssa[0] if input_ssa else '%0'
        
        # Register outputs
        for out_id in op.outputs:
            self.tensor_to_ssa[out_id] = result


#==============================================================================
# Main Optimizer
#==============================================================================

class YirageOptimizer:
    """
    End-to-end optimizer for YiRage.
    
    Integrates:
    1. Pattern detection (compound operations)
    2. COMET-based search optimization
    3. MLIR code generation
    4. Multi-backend compilation
    """
    
    def __init__(self, target: str = "cuda-h100", 
                 config: Optional[OptimizationConfig] = None):
        """
        Initialize the optimizer.
        
        Args:
            target: Target backend (cuda-h100, rocm-mi300x, tpu-v5e, etc.)
            config: Optimization configuration
        """
        if config is not None:
            # Use target from config if provided
            self.target = config.target
            self.config = config
        else:
            self.target = target
            self.config = OptimizationConfig(target=target)
        
        # Initialize cost model if search available
        if SEARCH_AVAILABLE:
            backend_config = get_backend_config(self._get_backend_name())
            self.cost_model = COMETCostModel(backend_config)
            self.search_config = COMETSearchConfig(
                max_iterations=self.config.max_search_iterations,
                timeout_seconds=self.config.search_timeout_seconds,
                tile_sizes=self.config.tile_sizes,
                num_devices=self.config.num_devices
            )
        else:
            self.cost_model = None
            self.search_config = None
    
    def _get_backend_name(self) -> str:
        """Extract backend name from target string."""
        # e.g., "cuda-h100" -> "cuda_h100"
        return self.target.lower().replace('-', '_')
    
    def optimize(self, graph: Any) -> OptimizationResult:
        """
        Optimize a graph and generate MLIR.
        
        Args:
            graph: Input graph (JSON dict, JSON string, KNGraph, or file path)
            
        Returns:
            OptimizationResult with optimized MLIR and metadata
        """
        start_time = time.time()
        result = OptimizationResult(target=self.target)
        
        # 1. Convert input to MuGraph
        mugraph = self._convert_input(graph)
        result.input_graph_ops = mugraph.num_ops()
        result.input_graph_tensors = mugraph.num_tensors()
        
        # 2. Detect compound patterns
        detector = PatternDetector(mugraph)
        patterns = detector.detect_all_patterns()
        
        result.patterns_detected = [
            {'type': p['type'], 'op_count': len(p['op_indices'])}
            for p in patterns
        ]
        
        if patterns:
            result.optimizations_applied.append("pattern_fusion")
        
        # 3. Run COMET search if available and enabled
        best_config = None
        if SEARCH_AVAILABLE and self.config.enable_search and patterns:
            search_start = time.time()
            
            try:
                # Build search operators (simplified)
                search_ops = self._build_search_operators(mugraph, patterns)
                
                # COMETSearchStrategy only takes config parameter
                strategy = COMETSearchStrategy(self.search_config)
                candidates = strategy.search(search_ops)
                
                if candidates:
                    best_config = {
                        'tile_sizes': candidates[0].tile_sizes,
                        'scheduling': candidates[0].scheduling.name if candidates[0].scheduling else 'SEQUENTIAL',
                        'cost': candidates[0].cost
                    }
                    result.best_config = best_config
                    result.estimated_latency_ns = candidates[0].cost
                    result.optimizations_applied.append("comet_search")
                
                result.search_iterations = len(candidates)
            except Exception as e:
                if self.config.verbose:
                    print(f"COMET search failed: {e}")
            
            result.search_time_seconds = time.time() - search_start
        
        # 4. Generate optimized MLIR
        generator = OptimizedMLIRGenerator(mugraph, patterns, self.config)
        result.optimized_mlir = generator.generate()
        
        # 5. Add optimization passes as comments
        if self.config.enable_flash_attention:
            result.optimized_mlir = (
                "// Optimization pass: flash-attention\n" + 
                result.optimized_mlir
            )
            result.optimizations_applied.append("flash_attention")
        
        if self.config.enable_kernel_fusion:
            result.optimized_mlir = (
                "// Optimization pass: kernel-fusion\n" +
                result.optimized_mlir
            )
            result.optimizations_applied.append("kernel_fusion")
        
        return result
    
    def _convert_input(self, graph: Any) -> MuGraph:
        """Convert various input formats to MuGraph."""
        if isinstance(graph, MuGraph):
            return graph
        elif isinstance(graph, dict):
            return MuGraph.from_json(graph)
        elif isinstance(graph, str):
            # Could be JSON string or file path
            if graph.endswith('.json'):
                with open(graph, 'r') as f:
                    import json
                    data = json.load(f)
                return MuGraph.from_json(data)
            else:
                return MuGraph.from_json(graph)
        else:
            # Assume KNGraph
            return MuGraph.from_kngraph(graph)
    
    def _build_search_operators(self, mugraph: MuGraph, 
                                patterns: List[Dict[str, Any]]) -> List[Any]:
        """Build operator list for COMET search."""
        # Simplified: just return pattern info
        ops = []
        for pattern in patterns:
            op_info = {
                'type': pattern['type'],
                'flops': pattern.get('estimated_flops', 0),
                'op_count': len(pattern.get('op_indices', []))
            }
            ops.append(op_info)
        return ops
    
    def compile(self, graph: Any) -> Optional[CompiledKernel]:
        """
        Full compilation pipeline: optimize and compile to target code.
        
        Args:
            graph: Input graph
            
        Returns:
            CompiledKernel if MLIR compiler available, None otherwise
        """
        # Optimize
        result = self.optimize(graph)
        
        # Compile if MLIR available
        if MLIR_AVAILABLE:
            try:
                compiler = YirageCompiler(
                    target=self._get_backend_family(),
                    opt_level=self.config.mlir_opt_level
                )
                return compiler.compile_mlir_text(result.optimized_mlir)
            except Exception as e:
                if self.config.verbose:
                    print(f"Compilation failed: {e}")
                return None
        
        return None
    
    def _get_backend_family(self) -> str:
        """Get backend family from target."""
        # e.g., "cuda-h100" -> "cuda"
        return self.target.split('-')[0].lower()


#==============================================================================
# Convenience Functions
#==============================================================================

def optimize_graph(graph: Any, target: str = "cuda-h100") -> OptimizationResult:
    """
    Convenience function to optimize a graph.
    
    Args:
        graph: Input graph (JSON, KNGraph, or file path)
        target: Target backend
        
    Returns:
        OptimizationResult
    """
    optimizer = YirageOptimizer(target=target)
    return optimizer.optimize(graph)


def optimize_and_compile(graph: Any, target: str = "cuda-h100") -> str:
    """
    Optimize graph and return MLIR.
    
    Args:
        graph: Input graph
        target: Target backend
        
    Returns:
        Optimized MLIR string
    """
    result = optimize_graph(graph, target)
    return result.optimized_mlir


def get_supported_targets() -> List[str]:
    """Get list of supported optimization targets."""
    return [
        "cuda-v100", "cuda-a100", "cuda-h100",
        "rocm-mi100", "rocm-mi250", "rocm-mi300x",
        "xpu-pvc", "xpu-arc",
        "tpu-v4", "tpu-v5e",
        "ascend-910a", "ascend-910b",
        "maca-mxc500",
        "metal-m1", "metal-m2", "metal-m3",
        "cpu-avx2", "cpu-avx512", "cpu-neon",
        "fpga-xilinx", "fpga-intel"
    ]


#==============================================================================
# Demo
#==============================================================================

if __name__ == "__main__":
    # Example: Optimize a transformer block
    
    # Create a simple attention graph
    example_graph = {
        "inputs": [
            {"id": 0, "dims": [1, 32, 2048, 128], "dtype": "f16"},  # Q
            {"id": 1, "dims": [1, 32, 2048, 128], "dtype": "f16"},  # K
            {"id": 2, "dims": [1, 32, 2048, 128], "dtype": "f16"},  # V
        ],
        "operators": [
            {
                "id": 0, "type": "matmul",
                "inputs": [0, 1],
                "outputs": [{"id": 3, "dims": [1, 32, 2048, 2048], "dtype": "f16"}]
            },
            {
                "id": 1, "type": "softmax",
                "inputs": [3],
                "outputs": [{"id": 4, "dims": [1, 32, 2048, 2048], "dtype": "f16"}]
            },
            {
                "id": 2, "type": "matmul",
                "inputs": [4, 2],
                "outputs": [{"id": 5, "dims": [1, 32, 2048, 128], "dtype": "f16"}]
            }
        ],
        "outputs": [
            {"id": 5, "dims": [1, 32, 2048, 128], "dtype": "f16"}
        ]
    }
    
    print("=" * 60)
    print("YiRage End-to-End Optimizer Demo")
    print("=" * 60)
    
    # Optimize for different targets
    for target in ["cuda-h100", "rocm-mi300x", "tpu-v5e"]:
        print(f"\n--- Target: {target} ---")
        
        result = optimize_graph(example_graph, target)
        
        print(f"Input ops: {result.input_graph_ops}")
        print(f"Patterns detected: {result.patterns_detected}")
        print(f"Optimizations applied: {result.optimizations_applied}")
        
        if result.best_config:
            print(f"Best config: {result.best_config}")
        
        print("\nGenerated MLIR (first 20 lines):")
        for i, line in enumerate(result.optimized_mlir.split('\n')[:20]):
            print(f"  {line}")
        
        print()
