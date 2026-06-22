# Copyright 2025 Chen Xingqiang (YiRage Project)
# SPDX-License-Identifier: Apache-2.0

"""
YiRage PyTorch Integration

This module provides seamless integration between YiRage MLIR compilation
and PyTorch models, enabling:

1. torch.compile() backend for YiRage
2. Custom operator registration
3. Module export to MLIR
4. AOT compilation of PyTorch models

Example:
    import torch
    import yirage
    
    # Using as torch.compile backend
    @torch.compile(backend="yirage")
    def my_model(x, y):
        return torch.matmul(x, y)
    
    # Or explicit compilation
    compiled = yirage.torch.compile_model(model, target="cuda-sm_90")
"""

from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import functools
import logging
import sys
from pathlib import Path

# Check for PyTorch
try:
    import torch
    import torch.fx
    from torch._dynamo.backends.common import aot_autograd
    from torch._dynamo.backends.registry import register_backend
    from torch._inductor.decomposition import select_decomp_table
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import MLIR components
try:
    from .mlir_jit import MLIRCompiler, Target, CompileOptions, CompilationResult
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False

logger = logging.getLogger(__name__)


#==============================================================================
# PyTorch to MLIR Converter
#==============================================================================

class FXToMLIRConverter:
    """
    Convert PyTorch FX graph to YiRage MLIR.
    
    This converter handles:
    - Tensor operations (matmul, attention, etc.)
    - Control flow (limited)
    - Memory management
    - Type propagation
    """
    
    # Mapping from PyTorch ops to YiRage ops
    OP_MAP = {
        # Matrix operations
        "aten::mm": "yirage.matmul",
        "aten::bmm": "yirage.batch_matmul",
        "aten::matmul": "yirage.matmul",
        "aten::linear": "yirage.linear",
        
        # Attention
        "aten::scaled_dot_product_attention": "yirage.attention",
        
        # Activations
        "aten::relu": "yirage.relu",
        "aten::silu": "yirage.silu",
        "aten::gelu": "yirage.gelu",
        
        # Normalization
        "aten::layer_norm": "yirage.layer_norm",
        "aten::rms_norm": "yirage.rms_norm",
        
        # Elementwise
        "aten::add": "arith.addf",
        "aten::sub": "arith.subf",
        "aten::mul": "arith.mulf",
        "aten::div": "arith.divf",
        
        # Reduction
        "aten::sum": "yirage.reduce_sum",
        "aten::mean": "yirage.reduce_mean",
        "aten::softmax": "yirage.softmax",
    }
    
    def __init__(self):
        self.ssa_counter = 0
        self.value_map: Dict[str, str] = {}
        self.mlir_lines: List[str] = []
        
    def convert(self, gm: 'torch.fx.GraphModule', 
                example_inputs: Tuple) -> str:
        """
        Convert FX GraphModule to MLIR text.
        """
        self.ssa_counter = 0
        self.value_map = {}
        self.mlir_lines = []
        
        # Start module
        self.mlir_lines.append("module {")
        
        # Convert the graph to a function
        func_mlir = self._convert_graph(gm.graph, example_inputs)
        self.mlir_lines.append(func_mlir)
        
        self.mlir_lines.append("}")
        
        return "\n".join(self.mlir_lines)
    
    def _convert_graph(self, graph: 'torch.fx.Graph', 
                        example_inputs: Tuple) -> str:
        """Convert FX graph to MLIR function."""
        lines = []
        
        # Collect placeholders for function signature
        placeholders = []
        for node in graph.nodes:
            if node.op == "placeholder":
                placeholders.append(node)
        
        # Build function signature
        arg_types = []
        for i, ph in enumerate(placeholders):
            if i < len(example_inputs):
                tensor = example_inputs[i]
                arg_types.append(self._tensor_to_mlir_type(tensor))
            else:
                arg_types.append("tensor<*xf32>")  # Unknown type
        
        func_args = ", ".join(
            f"%{ph.name}: {t}" for ph, t in zip(placeholders, arg_types)
        )
        
        # Get return type (from output node)
        return_type = "tensor<*xf32>"  # Default
        for node in graph.nodes:
            if node.op == "output":
                if node.args and len(node.args) > 0:
                    ret_node = node.args[0]
                    if isinstance(ret_node, torch.fx.Node):
                        return_type = self._get_node_type(ret_node, example_inputs)
                        
        lines.append(f"  func.func @forward({func_args}) -> {return_type} {{")
        
        # Map placeholders
        for ph in placeholders:
            self.value_map[ph.name] = f"%{ph.name}"
        
        # Convert operations
        for node in graph.nodes:
            if node.op == "placeholder":
                continue
            elif node.op == "call_function":
                line = self._convert_call_function(node)
                if line:
                    lines.append(f"    {line}")
            elif node.op == "call_method":
                line = self._convert_call_method(node)
                if line:
                    lines.append(f"    {line}")
            elif node.op == "output":
                ret_val = self._get_value(node.args[0])
                lines.append(f"    return {ret_val} : {return_type}")
        
        lines.append("  }")
        return "\n".join(lines)
    
    def _convert_call_function(self, node: 'torch.fx.Node') -> Optional[str]:
        """Convert a function call to MLIR."""
        target = str(node.target)
        
        # Get MLIR op name
        mlir_op = None
        for torch_op, yirage_op in self.OP_MAP.items():
            if torch_op in target:
                mlir_op = yirage_op
                break
        
        if not mlir_op:
            # Fallback - just comment it
            return f"// Unsupported: {target}"
        
        # Get operand values
        operands = [self._get_value(arg) for arg in node.args 
                    if isinstance(arg, torch.fx.Node)]
        
        # Generate result SSA value
        result = self._new_ssa()
        self.value_map[node.name] = result
        
        # Build the operation
        operand_str = ", ".join(operands)
        type_str = self._get_node_type(node, ())
        
        return f"{result} = {mlir_op} {operand_str} : {type_str}"
    
    def _convert_call_method(self, node: 'torch.fx.Node') -> Optional[str]:
        """Convert a method call to MLIR."""
        method = node.target
        
        # Common tensor methods
        if method in ("view", "reshape"):
            operand = self._get_value(node.args[0])
            result = self._new_ssa()
            self.value_map[node.name] = result
            # Get target shape
            shape = node.args[1:] if len(node.args) > 1 else []
            shape_str = "x".join(str(s) for s in shape if isinstance(s, int))
            return f"{result} = tensor.reshape {operand} : tensor<*xf32> -> tensor<{shape_str}xf32>"
        
        if method == "transpose":
            operand = self._get_value(node.args[0])
            result = self._new_ssa()
            self.value_map[node.name] = result
            return f"{result} = yirage.transpose {operand} : tensor<*xf32>"
        
        return f"// Unsupported method: {method}"
    
    def _get_value(self, arg) -> str:
        """Get the SSA value for an argument."""
        if isinstance(arg, torch.fx.Node):
            return self.value_map.get(arg.name, f"%{arg.name}")
        elif isinstance(arg, (int, float)):
            return f"{arg}"
        else:
            return "???"
    
    def _new_ssa(self) -> str:
        """Generate a new SSA value name."""
        name = f"%v{self.ssa_counter}"
        self.ssa_counter += 1
        return name
    
    def _tensor_to_mlir_type(self, tensor: 'torch.Tensor') -> str:
        """Convert PyTorch tensor to MLIR type string."""
        shape = "x".join(str(d) for d in tensor.shape)
        dtype = self._dtype_to_mlir(tensor.dtype)
        return f"tensor<{shape}x{dtype}>"
    
    def _get_node_type(self, node: 'torch.fx.Node', 
                       example_inputs: Tuple) -> str:
        """Get MLIR type for a node."""
        # Try to get from metadata
        if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
            meta = node.meta['tensor_meta']
            if hasattr(meta, 'shape') and hasattr(meta, 'dtype'):
                shape = "x".join(str(d) for d in meta.shape)
                dtype = self._dtype_to_mlir(meta.dtype)
                return f"tensor<{shape}x{dtype}>"
        
        return "tensor<*xf32>"
    
    def _dtype_to_mlir(self, dtype) -> str:
        """Convert PyTorch dtype to MLIR element type."""
        if not TORCH_AVAILABLE:
            return "f32"
            
        dtype_map = {
            torch.float16: "f16",
            torch.bfloat16: "bf16",
            torch.float32: "f32",
            torch.float64: "f64",
            torch.int8: "i8",
            torch.int16: "i16",
            torch.int32: "i32",
            torch.int64: "i64",
        }
        return dtype_map.get(dtype, "f32")


#==============================================================================
# YiRage Backend for torch.compile
#==============================================================================

class YirageBackend:
    """
    torch.compile backend using YiRage MLIR compilation.
    
    Example:
        @torch.compile(backend=YirageBackend(target=Target.CUDA_H100))
        def my_model(x, y):
            return torch.matmul(x, y)
    """
    
    def __init__(self, 
                 target: 'Target' = None,
                 options: 'CompileOptions' = None):
        if target is None and MLIR_AVAILABLE:
            from .mlir_jit import Target
            target = Target.CUDA
        self.target = target
        self.options = options
        self.converter = FXToMLIRConverter()
        self._cache: Dict[int, Callable] = {}
        
    def __call__(self, gm: 'torch.fx.GraphModule', 
                 example_inputs: Tuple) -> Callable:
        """
        Compile the graph module.
        """
        # Check cache
        cache_key = id(gm.graph)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Convert to MLIR
            mlir_code = self.converter.convert(gm, example_inputs)
            logger.debug(f"Generated MLIR:\n{mlir_code}")
            
            # Compile with YiRage
            if MLIR_AVAILABLE:
                compiler = MLIRCompiler(target=self.target, options=self.options)
                result = compiler.compile(mlir_code)
                
                if result.success:
                    # Create a compiled function wrapper
                    compiled_fn = self._create_wrapper(result, gm, example_inputs)
                    self._cache[cache_key] = compiled_fn
                    return compiled_fn
                else:
                    logger.warning(f"YiRage compilation failed: {result.error_message}")
            
        except Exception as e:
            logger.warning(f"YiRage backend failed: {e}, falling back to eager")
        
        # Fall back to original graph
        return gm
    
    def _create_wrapper(self, result: 'CompilationResult',
                        gm: 'torch.fx.GraphModule',
                        example_inputs: Tuple) -> Callable:
        """Create a callable wrapper for the compiled kernel."""
        # For now, just return the original graph
        # Full implementation would load and call the compiled binary
        return gm


def yirage_backend(gm: 'torch.fx.GraphModule', 
                   example_inputs: Tuple) -> Callable:
    """
    YiRage backend function for torch.compile.
    
    Usage:
        @torch.compile(backend="yirage")
        def my_model(x, y):
            return torch.matmul(x, y)
    """
    backend = YirageBackend()
    return backend(gm, example_inputs)


# Register with torch.compile if available
if TORCH_AVAILABLE:
    try:
        from torch._dynamo.backends.registry import register_backend
        register_backend(name="yirage", compiler_fn=yirage_backend)
    except ImportError:
        pass


#==============================================================================
# High-Level API
#==============================================================================

def compile_model(model: 'torch.nn.Module',
                  target: Union[str, 'Target'] = None,
                  example_inputs: Tuple = None,
                  options: 'CompileOptions' = None) -> 'torch.nn.Module':
    """
    Compile a PyTorch model using YiRage.
    
    Args:
        model: PyTorch module to compile
        target: Target device (e.g., "cuda-sm_90", "rocm-gfx942")
        example_inputs: Example inputs for tracing
        options: Compilation options
        
    Returns:
        Compiled module
        
    Example:
        model = torch.nn.Linear(1024, 1024)
        compiled = yirage.torch.compile_model(model, target="cuda-sm_90")
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for compile_model")
    
    # Parse target
    if isinstance(target, str) and MLIR_AVAILABLE:
        from .mlir_jit import Target
        target = Target.CUDA  # Default
        
    # Use torch.compile with yirage backend
    backend = YirageBackend(target=target, options=options)
    
    return torch.compile(model, backend=backend)


def export_to_mlir(model: 'torch.nn.Module',
                   example_inputs: Tuple) -> str:
    """
    Export a PyTorch model to MLIR text.
    
    Args:
        model: PyTorch module
        example_inputs: Example inputs for tracing
        
    Returns:
        MLIR text representation
        
    Example:
        mlir = yirage.torch.export_to_mlir(model, (torch.randn(32, 1024),))
        print(mlir)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for export_to_mlir")
    
    # Trace the model
    gm = torch.fx.symbolic_trace(model)
    
    # Convert to MLIR
    converter = FXToMLIRConverter()
    return converter.convert(gm, example_inputs)


def compile_function(target: Union[str, 'Target'] = None):
    """
    Decorator to compile a function with YiRage.
    
    Example:
        @yirage.torch.compile_function(target="cuda-sm_90")
        def my_kernel(x, y):
            return torch.matmul(x, y)
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if TORCH_AVAILABLE:
                # Use torch.compile
                backend = YirageBackend(target=target)
                compiled = torch.compile(fn, backend=backend)
                return compiled(*args, **kwargs)
            else:
                return fn(*args, **kwargs)
        return wrapper
    return decorator


#==============================================================================
# Module Exports
#==============================================================================

__all__ = [
    'FXToMLIRConverter',
    'YirageBackend',
    'yirage_backend',
    'compile_model',
    'export_to_mlir',
    'compile_function',
    'TORCH_AVAILABLE',
    'MLIR_AVAILABLE',
]
