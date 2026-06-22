"""
Universal Task Representation

Hardware-agnostic representation of any compute task that can be
automatically analyzed, decomposed, and optimized.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Union
import numpy as np
import json


class DataType(Enum):
    """Supported data types."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT32 = "int32"
    INT64 = "int64"

    def bytes_per_element(self) -> int:
        """Get bytes per element."""
        sizes = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1, "int32": 4, "int64": 8}
        return sizes.get(self.value, 4)


@dataclass
class TensorSpec:
    """Specification of a tensor."""

    name: str
    shape: Tuple[int, ...]
    dtype: DataType = DataType.FP16

    # Memory layout
    layout: str = "row_major"  # row_major, col_major, custom

    # Partitioning hints
    partition_dims: List[int] = field(default_factory=list)

    def num_elements(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def size_bytes(self) -> int:
        """Total size in bytes."""
        return self.num_elements() * self.dtype.bytes_per_element()

    def size_gb(self) -> float:
        """Size in GB."""
        return self.size_bytes() / (1024**3)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype.value,
            "layout": self.layout,
            "partition_dims": self.partition_dims,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "TensorSpec":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            shape=tuple(d["shape"]),
            dtype=DataType(d.get("dtype", "fp16")),
            layout=d.get("layout", "row_major"),
            partition_dims=d.get("partition_dims", []),
        )


class OperatorType(Enum):
    """Standard operator types."""

    # Linear algebra
    MATMUL = "matmul"
    BATCH_MATMUL = "batch_matmul"
    GEMM = "gemm"

    # Element-wise
    ADD = "add"
    MUL = "mul"
    DIV = "div"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    GELU = "gelu"
    SILU = "silu"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"

    # Reduction
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    SOFTMAX = "softmax"
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"

    # Data movement
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"
    CONCAT = "concat"
    SPLIT = "split"
    GATHER = "gather"
    SCATTER = "scatter"

    # Attention
    ATTENTION = "attention"
    FLASH_ATTENTION = "flash_attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"

    # Convolution
    CONV2D = "conv2d"
    CONV3D = "conv3d"
    DEPTHWISE_CONV = "depthwise_conv"

    # Custom
    CUSTOM = "custom"


@dataclass
class TorchOp:
    """
    Wrapper for any PyTorch operator.
    Preserves full PyTorch semantics without manual mapping.
    """

    # PyTorch operator identity
    target: str = ""  # Full qualified name: "torch.matmul"
    op_schema: Optional[str] = None  # PyTorch schema string

    # For aten ops (core primitives)
    aten_op: Optional[str] = None  # e.g., "aten::mm", "aten::add"

    # Decomposition level
    is_composite: bool = False  # True if can be decomposed further
    decomposed_ops: List[str] = field(default_factory=list)

    # Parallelization characteristics (auto-detected)
    parallel_dims: List[int] = field(default_factory=list)
    reduction_dims: List[int] = field(default_factory=list)
    is_elementwise: bool = False
    is_pointwise: bool = False

    # Elementwise op names for detection
    _ELEMENTWISE_OPS = frozenset(
        {
            "exp",
            "log",
            "sqrt",
            "rsqrt",
            "abs",
            "neg",
            "sign",
            "sin",
            "cos",
            "tan",
            "sinh",
            "cosh",
            "tanh",
            "asin",
            "acos",
            "atan",
            "asinh",
            "acosh",
            "atanh",
            "ceil",
            "floor",
            "round",
            "trunc",
            "frac",
            "erf",
            "erfc",
            "erfinv",
            "relu",
            "gelu",
            "silu",
            "sigmoid",
            "hardsigmoid",
            "hardswish",
            "leaky_relu",
            "elu",
            "selu",
            "celu",
            "mish",
            "add",
            "sub",
            "mul",
            "div",
            "pow",
            "fmod",
            "remainder",
            "maximum",
            "minimum",
            "clamp",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "logical_and",
            "logical_or",
            "logical_not",
            "logical_xor",
            "bitwise_and",
            "bitwise_or",
            "bitwise_not",
            "bitwise_xor",
            "where",
            "masked_fill",
        }
    )

    # Reduction op names for detection
    _REDUCTION_OPS = frozenset(
        {
            "sum",
            "mean",
            "prod",
            "max",
            "min",
            "std",
            "var",
            "argmax",
            "argmin",
            "all",
            "any",
            "logsumexp",
            "norm",
            "cumsum",
            "cumprod",
            "softmax",
            "log_softmax",
        }
    )

    @classmethod
    def from_fx_node(cls, node) -> "TorchOp":
        """Create TorchOp from a torch.fx Node."""
        target = node.target

        # Get target string
        if hasattr(target, "__module__") and hasattr(target, "__name__"):
            target_str = f"{target.__module__}.{target.__name__}"
        elif hasattr(target, "__name__"):
            target_str = target.__name__
        else:
            target_str = str(target)

        # Determine aten op if available
        aten_op = None
        if "aten::" in target_str:
            aten_op = target_str
        elif hasattr(target, "_schema"):
            aten_op = str(target._schema).split("(")[0]

        # Get schema if available
        op_schema = None
        if hasattr(target, "_schema"):
            op_schema = str(target._schema)

        torch_op = cls(
            target=target_str,
            aten_op=aten_op,
            op_schema=op_schema,
        )

        # Auto-detect characteristics
        torch_op.detect_characteristics()

        return torch_op

    def detect_characteristics(self) -> None:
        """Auto-detect parallelization characteristics from op name."""
        # Extract base op name
        op_name = self.target.split(".")[-1].lower()
        op_name = op_name.replace("_", "")

        # Check if elementwise
        for ew_op in self._ELEMENTWISE_OPS:
            if ew_op.replace("_", "") in op_name:
                self.is_elementwise = True
                self.is_pointwise = True
                break

        # Check if reduction
        for red_op in self._REDUCTION_OPS:
            if red_op.replace("_", "") in op_name:
                # Reduction ops typically reduce along some dimension
                self.is_elementwise = False
                break

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target": self.target,
            "op_schema": self.op_schema,
            "aten_op": self.aten_op,
            "is_composite": self.is_composite,
            "decomposed_ops": self.decomposed_ops,
            "parallel_dims": self.parallel_dims,
            "reduction_dims": self.reduction_dims,
            "is_elementwise": self.is_elementwise,
            "is_pointwise": self.is_pointwise,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TorchOp":
        """Create from dictionary."""
        return cls(
            target=d.get("target", ""),
            op_schema=d.get("op_schema"),
            aten_op=d.get("aten_op"),
            is_composite=d.get("is_composite", False),
            decomposed_ops=d.get("decomposed_ops", []),
            parallel_dims=d.get("parallel_dims", []),
            reduction_dims=d.get("reduction_dims", []),
            is_elementwise=d.get("is_elementwise", False),
            is_pointwise=d.get("is_pointwise", False),
        )


@dataclass
class Operator:
    """A single operator in the computation graph."""

    op_id: str
    op_type: OperatorType

    # I/O
    inputs: List[str]  # Tensor names
    outputs: List[str]  # Tensor names

    # PyTorch-native operator info (optional, for full semantics)
    torch_op: Optional[TorchOp] = None

    # Attributes
    attrs: Dict[str, Any] = field(default_factory=dict)

    # Performance hints
    is_compute_bound: bool = True
    arithmetic_intensity: float = 0.0  # FLOPs per byte

    def estimate_flops(self, tensor_specs: Dict[str, TensorSpec]) -> int:
        """Estimate FLOPs for this operator."""

        if self.op_type == OperatorType.MATMUL:
            # [M, K] x [K, N] = 2*M*K*N
            a = tensor_specs.get(self.inputs[0])
            b = tensor_specs.get(self.inputs[1])
            if a and b:
                M, K = a.shape[-2], a.shape[-1]
                N = b.shape[-1]
                batch = np.prod(a.shape[:-2]) if len(a.shape) > 2 else 1
                return int(2 * batch * M * K * N)

        elif self.op_type == OperatorType.ATTENTION:
            # Attention: 4*B*H*S^2*D for Q@K^T and attn@V
            q = tensor_specs.get(self.inputs[0])
            if q:
                if len(q.shape) == 4:  # [B, H, S, D]
                    B, H, S, D = q.shape
                else:  # [B, S, H*D]
                    B, S, HD = q.shape
                    H = self.attrs.get("num_heads", 8)
                    D = HD // H
                return int(4 * B * H * S * S * D)

        elif self.op_type in (OperatorType.LAYER_NORM, OperatorType.RMS_NORM):
            x = tensor_specs.get(self.inputs[0])
            if x:
                return int(5 * x.num_elements())  # mean, var, normalize

        elif self.op_type == OperatorType.SOFTMAX:
            x = tensor_specs.get(self.inputs[0])
            if x:
                return int(5 * x.num_elements())  # max, sub, exp, sum, div

        elif self.op_type in (OperatorType.GELU, OperatorType.SILU):
            x = tensor_specs.get(self.inputs[0])
            if x:
                return int(10 * x.num_elements())  # Complex activation

        # Default: 1 FLOP per element for element-wise ops
        out = tensor_specs.get(self.outputs[0]) if self.outputs else None
        if out:
            return out.num_elements()

        return 0

    def estimate_memory_bytes(self, tensor_specs: Dict[str, TensorSpec]) -> int:
        """Estimate memory traffic in bytes."""
        total = 0
        for inp in self.inputs:
            if inp in tensor_specs:
                total += tensor_specs[inp].size_bytes()
        for out in self.outputs:
            if out in tensor_specs:
                total += tensor_specs[out].size_bytes()
        return total

    def compute_arithmetic_intensity(self, tensor_specs: Dict[str, TensorSpec]) -> float:
        """Compute arithmetic intensity (FLOPs/byte)."""
        flops = self.estimate_flops(tensor_specs)
        memory = self.estimate_memory_bytes(tensor_specs)
        if memory > 0:
            return flops / memory
        return 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "op_id": self.op_id,
            "op_type": self.op_type.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "attrs": self.attrs,
        }
        if self.torch_op is not None:
            result["torch_op"] = self.torch_op.to_dict()
        return result

    @classmethod
    def from_dict(cls, d: Dict) -> "Operator":
        """Create from dictionary."""
        torch_op = None
        if "torch_op" in d and d["torch_op"] is not None:
            torch_op = TorchOp.from_dict(d["torch_op"])

        return cls(
            op_id=d["op_id"],
            op_type=OperatorType(d["op_type"]),
            inputs=d["inputs"],
            outputs=d["outputs"],
            torch_op=torch_op,
            attrs=d.get("attrs", {}),
        )


@dataclass
class DataDependency:
    """Data dependency between operators."""

    producer_op: str
    consumer_op: str
    tensor_name: str

    # Dependency type
    is_inplace: bool = False
    requires_sync: bool = True


@dataclass
class ComputeTask:
    """
    Universal representation of any compute task.

    Can be specified via:
    1. High-level operation name (e.g., "llm_forward", "attention")
    2. Explicit operator graph
    3. PyTorch module (traced)
    """

    name: str

    # Tensor specifications
    tensors: Dict[str, TensorSpec] = field(default_factory=dict)

    # Input/output tensor names
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # Operator graph
    operators: List[Operator] = field(default_factory=list)

    # Constraints
    latency_budget_ms: Optional[float] = None
    memory_budget_gb: Optional[float] = None
    throughput_target_tps: Optional[float] = None

    # Batch configuration
    batch_sizes: List[int] = field(default_factory=lambda: [1, 8, 32])

    # Precision
    precision: str = "auto"  # auto, fp16, bf16, fp32, mixed

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_flops(self) -> int:
        """Total FLOPs for the task."""
        return sum(op.estimate_flops(self.tensors) for op in self.operators)

    def total_memory_bytes(self) -> int:
        """Total tensor memory."""
        return sum(t.size_bytes() for t in self.tensors.values())

    def input_memory_bytes(self) -> int:
        """Input tensor memory."""
        return sum(self.tensors[n].size_bytes() for n in self.inputs if n in self.tensors)

    def output_memory_bytes(self) -> int:
        """Output tensor memory."""
        return sum(self.tensors[n].size_bytes() for n in self.outputs if n in self.tensors)

    def get_dependencies(self) -> List[DataDependency]:
        """Extract data dependencies from operator graph."""
        deps = []
        tensor_producers = {}

        for op in self.operators:
            for out in op.outputs:
                tensor_producers[out] = op.op_id

        for op in self.operators:
            for inp in op.inputs:
                if inp in tensor_producers:
                    deps.append(
                        DataDependency(
                            producer_op=tensor_producers[inp],
                            consumer_op=op.op_id,
                            tensor_name=inp,
                        )
                    )

        return deps

    def topological_order(self) -> List[str]:
        """Get operators in topological order."""
        deps = self.get_dependencies()

        # Build adjacency list
        adj = {op.op_id: [] for op in self.operators}
        in_degree = {op.op_id: 0 for op in self.operators}

        for dep in deps:
            adj[dep.producer_op].append(dep.consumer_op)
            in_degree[dep.consumer_op] += 1

        # Kahn's algorithm
        queue = [op_id for op_id, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            op_id = queue.pop(0)
            order.append(op_id)
            for consumer in adj[op_id]:
                in_degree[consumer] -= 1
                if in_degree[consumer] == 0:
                    queue.append(consumer)

        return order

    def get_aten_ops(self) -> List[str]:
        """
        Get list of aten operator names from the task.

        Returns:
            List of aten op names (e.g., ["aten::mm", "aten::add"])
        """
        aten_ops = []
        for op in self.operators:
            if op.torch_op and op.torch_op.aten_op:
                aten_ops.append(op.torch_op.aten_op)
            elif op.torch_op and op.torch_op.target:
                # Try to extract aten op from target
                target = op.torch_op.target
                if "aten" in target.lower():
                    aten_ops.append(target)
                else:
                    # Convert torch function to aten format
                    op_name = target.split(".")[-1]
                    aten_ops.append(f"aten::{op_name}")
            elif "torch_target" in op.attrs:
                target = op.attrs["torch_target"]
                op_name = str(target).split(".")[-1].split("'")[0]
                aten_ops.append(f"aten::{op_name}")
            elif "original_op" in op.attrs:
                aten_ops.append(f"aten::{op.attrs['original_op']}")

        return aten_ops

    def decompose(self) -> "ComputeTask":
        """
        Decompose high-level operators to aten primitives.

        Uses torch._decompositions to break down composite operations
        into more fundamental operations.

        Returns:
            New ComputeTask with decomposed operators
        """
        try:
            import torch
        except ImportError:
            # If torch not available, return a copy with minimal changes
            return ComputeTask(
                name=f"{self.name}_decomposed",
                tensors=dict(self.tensors),
                inputs=list(self.inputs),
                outputs=list(self.outputs),
                operators=list(self.operators),
                latency_budget_ms=self.latency_budget_ms,
                memory_budget_gb=self.memory_budget_gb,
                batch_sizes=list(self.batch_sizes),
                precision=self.precision,
                metadata={**self.metadata, "decomposed": True},
            )

        # Create decomposed operators
        decomposed_ops = []

        for op in self.operators:
            # Try to decompose each operator
            decomposed = self._decompose_operator(op)
            decomposed_ops.extend(decomposed)

        return ComputeTask(
            name=f"{self.name}_decomposed",
            tensors=dict(self.tensors),
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            operators=decomposed_ops,
            latency_budget_ms=self.latency_budget_ms,
            memory_budget_gb=self.memory_budget_gb,
            batch_sizes=list(self.batch_sizes),
            precision=self.precision,
            metadata={**self.metadata, "decomposed": True},
        )

    def _decompose_operator(self, op: Operator) -> List[Operator]:
        """
        Decompose a single operator to primitives.

        Returns a list of operators (may be just the original if not decomposable).
        """
        # Define decomposition rules for common ops
        decompositions = {
            OperatorType.GELU: self._decompose_gelu,
            OperatorType.SILU: self._decompose_silu,
            OperatorType.LAYER_NORM: self._decompose_layer_norm,
            OperatorType.RMS_NORM: self._decompose_rms_norm,
            OperatorType.SOFTMAX: self._decompose_softmax,
        }

        if op.op_type in decompositions:
            try:
                return decompositions[op.op_type](op)
            except Exception:
                # If decomposition fails, return original
                return [op]

        # Check if torch_op indicates it's composite
        if op.torch_op and op.torch_op.is_composite:
            # Could use torch._decompositions here if available
            pass

        # Return original if no decomposition available
        return [op]

    def _decompose_gelu(self, op: Operator) -> List[Operator]:
        """Decompose GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
        base_id = op.op_id
        inp = op.inputs[0]
        out = op.outputs[0]

        return [
            Operator(
                f"{base_id}_pow",
                OperatorType.MUL,
                [inp],
                [f"{base_id}_x3"],
                attrs={"decomposed_from": "gelu", "operation": "x^3"},
            ),
            Operator(
                f"{base_id}_scale1",
                OperatorType.MUL,
                [f"{base_id}_x3"],
                [f"{base_id}_scaled"],
                attrs={"decomposed_from": "gelu", "operation": "0.044715*x^3"},
            ),
            Operator(
                f"{base_id}_add1",
                OperatorType.ADD,
                [inp, f"{base_id}_scaled"],
                [f"{base_id}_sum"],
                attrs={"decomposed_from": "gelu"},
            ),
            Operator(
                f"{base_id}_scale2",
                OperatorType.MUL,
                [f"{base_id}_sum"],
                [f"{base_id}_inner"],
                attrs={"decomposed_from": "gelu", "operation": "sqrt(2/pi)*..."},
            ),
            Operator(
                f"{base_id}_tanh",
                OperatorType.TANH,
                [f"{base_id}_inner"],
                [f"{base_id}_tanh_out"],
                attrs={"decomposed_from": "gelu"},
            ),
            Operator(
                f"{base_id}_add2",
                OperatorType.ADD,
                [f"{base_id}_tanh_out"],
                [f"{base_id}_one_plus"],
                attrs={"decomposed_from": "gelu", "operation": "1+tanh(...)"},
            ),
            Operator(
                f"{base_id}_mul1",
                OperatorType.MUL,
                [inp, f"{base_id}_one_plus"],
                [f"{base_id}_pre_scale"],
                attrs={"decomposed_from": "gelu"},
            ),
            Operator(
                f"{base_id}_mul2",
                OperatorType.MUL,
                [f"{base_id}_pre_scale"],
                [out],
                attrs={"decomposed_from": "gelu", "operation": "0.5*..."},
            ),
        ]

    def _decompose_silu(self, op: Operator) -> List[Operator]:
        """Decompose SiLU: x * sigmoid(x)"""
        base_id = op.op_id
        inp = op.inputs[0]
        out = op.outputs[0]

        return [
            Operator(
                f"{base_id}_sigmoid",
                OperatorType.SIGMOID,
                [inp],
                [f"{base_id}_sig"],
                attrs={"decomposed_from": "silu"},
            ),
            Operator(
                f"{base_id}_mul",
                OperatorType.MUL,
                [inp, f"{base_id}_sig"],
                [out],
                attrs={"decomposed_from": "silu"},
            ),
        ]

    def _decompose_layer_norm(self, op: Operator) -> List[Operator]:
        """Decompose LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta"""
        base_id = op.op_id
        inp = op.inputs[0]
        out = op.outputs[0]

        return [
            Operator(
                f"{base_id}_mean",
                OperatorType.MEAN,
                [inp],
                [f"{base_id}_mu"],
                attrs={"decomposed_from": "layer_norm"},
            ),
            Operator(
                f"{base_id}_sub",
                OperatorType.ADD,
                [inp, f"{base_id}_mu"],
                [f"{base_id}_centered"],
                attrs={"decomposed_from": "layer_norm", "operation": "x - mean"},
            ),
            Operator(
                f"{base_id}_var",
                OperatorType.MEAN,
                [f"{base_id}_centered"],
                [f"{base_id}_var"],
                attrs={"decomposed_from": "layer_norm", "operation": "variance"},
            ),
            Operator(
                f"{base_id}_sqrt",
                OperatorType.SQRT,
                [f"{base_id}_var"],
                [f"{base_id}_std"],
                attrs={"decomposed_from": "layer_norm"},
            ),
            Operator(
                f"{base_id}_div",
                OperatorType.DIV,
                [f"{base_id}_centered", f"{base_id}_std"],
                [out],
                attrs={"decomposed_from": "layer_norm"},
            ),
        ]

    def _decompose_rms_norm(self, op: Operator) -> List[Operator]:
        """Decompose RMSNorm: x / sqrt(mean(x^2) + eps) * gamma"""
        base_id = op.op_id
        inp = op.inputs[0]
        out = op.outputs[0]

        return [
            Operator(
                f"{base_id}_sq",
                OperatorType.MUL,
                [inp, inp],
                [f"{base_id}_sq"],
                attrs={"decomposed_from": "rms_norm", "operation": "x^2"},
            ),
            Operator(
                f"{base_id}_mean",
                OperatorType.MEAN,
                [f"{base_id}_sq"],
                [f"{base_id}_ms"],
                attrs={"decomposed_from": "rms_norm"},
            ),
            Operator(
                f"{base_id}_rsqrt",
                OperatorType.SQRT,
                [f"{base_id}_ms"],
                [f"{base_id}_rms"],
                attrs={"decomposed_from": "rms_norm", "operation": "rsqrt"},
            ),
            Operator(
                f"{base_id}_div",
                OperatorType.DIV,
                [inp, f"{base_id}_rms"],
                [out],
                attrs={"decomposed_from": "rms_norm"},
            ),
        ]

    def _decompose_softmax(self, op: Operator) -> List[Operator]:
        """Decompose Softmax: exp(x - max(x)) / sum(exp(x - max(x)))"""
        base_id = op.op_id
        inp = op.inputs[0]
        out = op.outputs[0]

        return [
            Operator(
                f"{base_id}_max",
                OperatorType.MAX,
                [inp],
                [f"{base_id}_max"],
                attrs={"decomposed_from": "softmax"},
            ),
            Operator(
                f"{base_id}_sub",
                OperatorType.ADD,
                [inp, f"{base_id}_max"],
                [f"{base_id}_shifted"],
                attrs={"decomposed_from": "softmax", "operation": "x - max"},
            ),
            Operator(
                f"{base_id}_exp",
                OperatorType.EXP,
                [f"{base_id}_shifted"],
                [f"{base_id}_exp"],
                attrs={"decomposed_from": "softmax"},
            ),
            Operator(
                f"{base_id}_sum",
                OperatorType.SUM,
                [f"{base_id}_exp"],
                [f"{base_id}_sum"],
                attrs={"decomposed_from": "softmax"},
            ),
            Operator(
                f"{base_id}_div",
                OperatorType.DIV,
                [f"{base_id}_exp", f"{base_id}_sum"],
                [out],
                attrs={"decomposed_from": "softmax"},
            ),
        ]

    def detect_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect common computation patterns for optimization.

        Scans the operator graph to identify patterns like:
        - Attention (Q@K^T -> softmax -> @V)
        - MLP (Linear -> Activation -> Linear)
        - LayerNorm + Linear
        - Residual connections

        Returns:
            List of detected patterns with type and operator ids
        """
        patterns = []

        # Build operator lookup
        op_map = {op.op_id: op for op in self.operators}
        op_types = {op.op_id: op.op_type for op in self.operators}

        # Track which operators are part of patterns
        in_pattern = set()

        # Detect attention pattern: matmul -> softmax -> matmul
        patterns.extend(self._detect_attention_pattern(op_map, op_types, in_pattern))

        # Detect MLP pattern: linear -> activation -> linear
        patterns.extend(self._detect_mlp_pattern(op_map, op_types, in_pattern))

        # Detect GEMM patterns
        patterns.extend(self._detect_gemm_patterns(op_map, op_types, in_pattern))

        # Detect elementwise fusion opportunities
        patterns.extend(self._detect_elementwise_chain(op_map, op_types, in_pattern))

        return patterns

    def _detect_attention_pattern(
        self,
        op_map: Dict[str, Operator],
        op_types: Dict[str, OperatorType],
        in_pattern: set,
    ) -> List[Dict[str, Any]]:
        """Detect attention patterns (Q@K^T -> softmax -> @V)."""
        patterns = []
        deps = self.get_dependencies()

        # Build consumer map
        consumers = {}
        for dep in deps:
            if dep.producer_op not in consumers:
                consumers[dep.producer_op] = []
            consumers[dep.producer_op].append(dep.consumer_op)

        # Look for matmul followed by softmax followed by matmul
        for op in self.operators:
            if op.op_type in (OperatorType.MATMUL, OperatorType.BATCH_MATMUL):
                if op.op_id in in_pattern:
                    continue

                # Check if followed by softmax
                next_ops = consumers.get(op.op_id, [])
                for next_id in next_ops:
                    if next_id in op_map and op_types.get(next_id) == OperatorType.SOFTMAX:
                        # Check if softmax is followed by matmul
                        softmax_next = consumers.get(next_id, [])
                        for final_id in softmax_next:
                            if final_id in op_map and op_types.get(final_id) in (
                                OperatorType.MATMUL,
                                OperatorType.BATCH_MATMUL,
                            ):
                                patterns.append(
                                    {
                                        "pattern_type": "attention",
                                        "type": "attention",
                                        "operators": [op.op_id, next_id, final_id],
                                        "description": "Q@K^T -> softmax -> @V",
                                        "optimization_hint": "fuse_attention",
                                    }
                                )
                                in_pattern.update([op.op_id, next_id, final_id])

        return patterns

    def _detect_mlp_pattern(
        self,
        op_map: Dict[str, Operator],
        op_types: Dict[str, OperatorType],
        in_pattern: set,
    ) -> List[Dict[str, Any]]:
        """Detect MLP patterns (Linear -> Activation -> Linear)."""
        patterns = []
        deps = self.get_dependencies()

        # Build consumer map
        consumers = {}
        for dep in deps:
            if dep.producer_op not in consumers:
                consumers[dep.producer_op] = []
            consumers[dep.producer_op].append(dep.consumer_op)

        activation_types = {
            OperatorType.GELU,
            OperatorType.RELU,
            OperatorType.SILU,
            OperatorType.SIGMOID,
            OperatorType.TANH,
        }
        linear_types = {OperatorType.MATMUL, OperatorType.BATCH_MATMUL, OperatorType.GEMM}

        for op in self.operators:
            if op.op_type in linear_types:
                if op.op_id in in_pattern:
                    continue

                # Check if followed by activation
                next_ops = consumers.get(op.op_id, [])
                for next_id in next_ops:
                    if next_id in op_map and op_types.get(next_id) in activation_types:
                        # Check if activation is followed by linear
                        act_next = consumers.get(next_id, [])
                        for final_id in act_next:
                            if final_id in op_map and op_types.get(final_id) in linear_types:
                                patterns.append(
                                    {
                                        "pattern_type": "mlp",
                                        "type": "mlp",
                                        "operators": [op.op_id, next_id, final_id],
                                        "description": "Linear -> Activation -> Linear",
                                        "optimization_hint": "fuse_mlp",
                                    }
                                )
                                in_pattern.update([op.op_id, next_id, final_id])

        return patterns

    def _detect_gemm_patterns(
        self,
        op_map: Dict[str, Operator],
        op_types: Dict[str, OperatorType],
        in_pattern: set,
    ) -> List[Dict[str, Any]]:
        """Detect GEMM patterns (matmul possibly with bias add)."""
        patterns = []
        deps = self.get_dependencies()

        consumers = {}
        for dep in deps:
            if dep.producer_op not in consumers:
                consumers[dep.producer_op] = []
            consumers[dep.producer_op].append(dep.consumer_op)

        for op in self.operators:
            if op.op_type in (OperatorType.MATMUL, OperatorType.BATCH_MATMUL):
                if op.op_id in in_pattern:
                    continue

                # Check if followed by add (bias)
                next_ops = consumers.get(op.op_id, [])
                for next_id in next_ops:
                    if next_id in op_map and op_types.get(next_id) == OperatorType.ADD:
                        patterns.append(
                            {
                                "pattern_type": "gemm_bias",
                                "type": "gemm",
                                "operators": [op.op_id, next_id],
                                "description": "MatMul + Bias",
                                "optimization_hint": "fuse_gemm_bias",
                            }
                        )
                        in_pattern.update([op.op_id, next_id])
                        break
                else:
                    # Standalone matmul
                    patterns.append(
                        {
                            "pattern_type": "matmul",
                            "type": "linear",
                            "operators": [op.op_id],
                            "description": "MatMul",
                            "optimization_hint": "optimize_gemm",
                        }
                    )

        return patterns

    def _detect_elementwise_chain(
        self,
        op_map: Dict[str, Operator],
        op_types: Dict[str, OperatorType],
        in_pattern: set,
    ) -> List[Dict[str, Any]]:
        """Detect chains of elementwise operations that can be fused."""
        patterns = []
        deps = self.get_dependencies()

        consumers = {}
        for dep in deps:
            if dep.producer_op not in consumers:
                consumers[dep.producer_op] = []
            consumers[dep.producer_op].append(dep.consumer_op)

        elementwise_types = {
            OperatorType.ADD,
            OperatorType.MUL,
            OperatorType.DIV,
            OperatorType.EXP,
            OperatorType.LOG,
            OperatorType.SQRT,
            OperatorType.RELU,
            OperatorType.SIGMOID,
            OperatorType.TANH,
        }

        visited = set()

        for op in self.operators:
            if op.op_id in visited or op.op_id in in_pattern:
                continue

            if op.op_type in elementwise_types:
                # Start a chain
                chain = [op.op_id]
                visited.add(op.op_id)

                current = op.op_id
                while True:
                    next_ops = consumers.get(current, [])
                    extended = False
                    for next_id in next_ops:
                        if (
                            next_id in op_map
                            and op_types.get(next_id) in elementwise_types
                            and next_id not in visited
                        ):
                            chain.append(next_id)
                            visited.add(next_id)
                            current = next_id
                            extended = True
                            break
                    if not extended:
                        break

                if len(chain) >= 2:
                    patterns.append(
                        {
                            "pattern_type": "elementwise_chain",
                            "type": "elementwise",
                            "operators": chain,
                            "description": f"Chain of {len(chain)} elementwise ops",
                            "optimization_hint": "fuse_elementwise",
                        }
                    )

        return patterns

    def get_optimization_hints(self) -> Dict[str, Any]:
        """
        Get optimization hints for the task.

        Analyzes the task to provide hints about:
        - Compute intensity (compute-bound vs memory-bound)
        - Parallelism potential
        - Recommended optimizations

        Returns:
            Dict with optimization hints
        """
        hints = {
            "compute_intensity": self._compute_intensity(),
            "parallelism_potential": self._parallelism_potential(),
            "memory_bound_ops": [],
            "compute_bound_ops": [],
            "recommended_fusions": [],
            "patterns": self.detect_patterns(),
        }

        # Classify operators
        for op in self.operators:
            intensity = op.compute_arithmetic_intensity(self.tensors)
            if intensity > 10:  # Threshold for compute-bound
                hints["compute_bound_ops"].append(op.op_id)
            else:
                hints["memory_bound_ops"].append(op.op_id)

        # Recommend fusions based on patterns
        for pattern in hints["patterns"]:
            if pattern.get("optimization_hint"):
                hints["recommended_fusions"].append(
                    {
                        "type": pattern.get("optimization_hint"),
                        "operators": pattern.get("operators", []),
                    }
                )

        return hints

    def _compute_intensity(self) -> float:
        """Compute overall arithmetic intensity."""
        total_flops = self.total_flops()
        total_memory = self.total_memory_bytes()

        if total_memory > 0:
            return total_flops / total_memory
        return 0.0

    def _parallelism_potential(self) -> float:
        """Estimate parallelism potential (0-1 scale)."""
        if not self.operators:
            return 0.0

        deps = self.get_dependencies()
        num_ops = len(self.operators)
        num_deps = len(deps)

        # More dependencies = less parallelism
        # Ratio of independent ops
        if num_ops > 1:
            return max(0.0, 1.0 - (num_deps / (num_ops * (num_ops - 1) / 2)))
        return 1.0

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
            "inputs": self.inputs,
            "outputs": self.outputs,
            "operators": [op.to_dict() for op in self.operators],
            "latency_budget_ms": self.latency_budget_ms,
            "memory_budget_gb": self.memory_budget_gb,
            "batch_sizes": self.batch_sizes,
            "precision": self.precision,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "ComputeTask":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            tensors={k: TensorSpec.from_dict(v) for k, v in d.get("tensors", {}).items()},
            inputs=d.get("inputs", []),
            outputs=d.get("outputs", []),
            operators=[Operator.from_dict(op) for op in d.get("operators", [])],
            latency_budget_ms=d.get("latency_budget_ms"),
            memory_budget_gb=d.get("memory_budget_gb"),
            batch_sizes=d.get("batch_sizes", [1, 8, 32]),
            precision=d.get("precision", "auto"),
            metadata=d.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ComputeTask":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ==========================================================================
    # Factory methods for common compute patterns
    # ==========================================================================

    @classmethod
    def create_matmul(
        cls,
        M: int,
        K: int,
        N: int,
        batch: int = 1,
        dtype: DataType = DataType.FP16,
    ) -> "ComputeTask":
        """Create a matrix multiplication task."""

        if batch == 1:
            a_shape = (M, K)
            b_shape = (K, N)
            c_shape = (M, N)
        else:
            a_shape = (batch, M, K)
            b_shape = (batch, K, N)
            c_shape = (batch, M, N)

        tensors = {
            "A": TensorSpec("A", a_shape, dtype),
            "B": TensorSpec("B", b_shape, dtype),
            "C": TensorSpec("C", c_shape, dtype),
        }

        operators = [
            Operator(
                op_id="matmul_0",
                op_type=OperatorType.MATMUL,
                inputs=["A", "B"],
                outputs=["C"],
                is_compute_bound=True,
            )
        ]

        return cls(
            name=f"matmul_{M}x{K}x{N}",
            tensors=tensors,
            inputs=["A", "B"],
            outputs=["C"],
            operators=operators,
        )

    @classmethod
    def create_attention(
        cls,
        batch: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: DataType = DataType.FP16,
        causal: bool = True,
    ) -> "ComputeTask":
        """Create an attention task."""

        hidden_dim = num_heads * head_dim

        tensors = {
            "Q": TensorSpec("Q", (batch, seq_len, hidden_dim), dtype),
            "K": TensorSpec("K", (batch, seq_len, hidden_dim), dtype),
            "V": TensorSpec("V", (batch, seq_len, hidden_dim), dtype),
            "output": TensorSpec("output", (batch, seq_len, hidden_dim), dtype),
        }

        operators = [
            Operator(
                op_id="attention_0",
                op_type=OperatorType.ATTENTION,
                inputs=["Q", "K", "V"],
                outputs=["output"],
                attrs={
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "causal": causal,
                },
                is_compute_bound=True,
            )
        ]

        return cls(
            name=f"attention_b{batch}_s{seq_len}_h{num_heads}_d{head_dim}",
            tensors=tensors,
            inputs=["Q", "K", "V"],
            outputs=["output"],
            operators=operators,
            metadata={"causal": causal},
        )

    @classmethod
    def create_mlp(
        cls,
        batch: int,
        seq_len: int,
        hidden_dim: int,
        intermediate_dim: int,
        dtype: DataType = DataType.FP16,
        activation: str = "gelu",
    ) -> "ComputeTask":
        """Create an MLP task (FFN layer)."""

        tensors = {
            "input": TensorSpec("input", (batch, seq_len, hidden_dim), dtype),
            "W1": TensorSpec("W1", (hidden_dim, intermediate_dim), dtype),
            "W2": TensorSpec("W2", (intermediate_dim, hidden_dim), dtype),
            "hidden": TensorSpec("hidden", (batch, seq_len, intermediate_dim), dtype),
            "activated": TensorSpec("activated", (batch, seq_len, intermediate_dim), dtype),
            "output": TensorSpec("output", (batch, seq_len, hidden_dim), dtype),
        }

        activation_type = {
            "gelu": OperatorType.GELU,
            "relu": OperatorType.RELU,
            "silu": OperatorType.SILU,
        }.get(activation, OperatorType.GELU)

        operators = [
            Operator("fc1", OperatorType.MATMUL, ["input", "W1"], ["hidden"]),
            Operator("act", activation_type, ["hidden"], ["activated"]),
            Operator("fc2", OperatorType.MATMUL, ["activated", "W2"], ["output"]),
        ]

        return cls(
            name=f"mlp_b{batch}_s{seq_len}_h{hidden_dim}_i{intermediate_dim}",
            tensors=tensors,
            inputs=["input", "W1", "W2"],
            outputs=["output"],
            operators=operators,
        )

    @classmethod
    def create_transformer_block(
        cls,
        batch: int,
        seq_len: int,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        dtype: DataType = DataType.FP16,
    ) -> "ComputeTask":
        """Create a full transformer block."""

        head_dim = hidden_dim // num_heads

        tensors = {
            # Input
            "input": TensorSpec("input", (batch, seq_len, hidden_dim), dtype),
            # Attention
            "Wq": TensorSpec("Wq", (hidden_dim, hidden_dim), dtype),
            "Wk": TensorSpec("Wk", (hidden_dim, hidden_dim), dtype),
            "Wv": TensorSpec("Wv", (hidden_dim, hidden_dim), dtype),
            "Wo": TensorSpec("Wo", (hidden_dim, hidden_dim), dtype),
            "Q": TensorSpec("Q", (batch, seq_len, hidden_dim), dtype),
            "K": TensorSpec("K", (batch, seq_len, hidden_dim), dtype),
            "V": TensorSpec("V", (batch, seq_len, hidden_dim), dtype),
            "attn_out": TensorSpec("attn_out", (batch, seq_len, hidden_dim), dtype),
            "attn_proj": TensorSpec("attn_proj", (batch, seq_len, hidden_dim), dtype),
            # Layer norm
            "ln1_out": TensorSpec("ln1_out", (batch, seq_len, hidden_dim), dtype),
            "residual1": TensorSpec("residual1", (batch, seq_len, hidden_dim), dtype),
            # MLP
            "W1": TensorSpec("W1", (hidden_dim, intermediate_dim), dtype),
            "W2": TensorSpec("W2", (intermediate_dim, hidden_dim), dtype),
            "mlp_hidden": TensorSpec("mlp_hidden", (batch, seq_len, intermediate_dim), dtype),
            "mlp_act": TensorSpec("mlp_act", (batch, seq_len, intermediate_dim), dtype),
            "mlp_out": TensorSpec("mlp_out", (batch, seq_len, hidden_dim), dtype),
            "ln2_out": TensorSpec("ln2_out", (batch, seq_len, hidden_dim), dtype),
            "output": TensorSpec("output", (batch, seq_len, hidden_dim), dtype),
        }

        operators = [
            # Layer norm 1
            Operator("ln1", OperatorType.RMS_NORM, ["input"], ["ln1_out"]),
            # QKV projections
            Operator("q_proj", OperatorType.MATMUL, ["ln1_out", "Wq"], ["Q"]),
            Operator("k_proj", OperatorType.MATMUL, ["ln1_out", "Wk"], ["K"]),
            Operator("v_proj", OperatorType.MATMUL, ["ln1_out", "Wv"], ["V"]),
            # Attention
            Operator(
                "attention",
                OperatorType.ATTENTION,
                ["Q", "K", "V"],
                ["attn_out"],
                attrs={"num_heads": num_heads, "head_dim": head_dim},
            ),
            # Output projection
            Operator("o_proj", OperatorType.MATMUL, ["attn_out", "Wo"], ["attn_proj"]),
            # Residual
            Operator("res1", OperatorType.ADD, ["input", "attn_proj"], ["residual1"]),
            # Layer norm 2
            Operator("ln2", OperatorType.RMS_NORM, ["residual1"], ["ln2_out"]),
            # MLP
            Operator("mlp_fc1", OperatorType.MATMUL, ["ln2_out", "W1"], ["mlp_hidden"]),
            Operator("mlp_act", OperatorType.GELU, ["mlp_hidden"], ["mlp_act"]),
            Operator("mlp_fc2", OperatorType.MATMUL, ["mlp_act", "W2"], ["mlp_out"]),
            # Residual
            Operator("res2", OperatorType.ADD, ["residual1", "mlp_out"], ["output"]),
        ]

        return cls(
            name=f"transformer_b{batch}_s{seq_len}_h{hidden_dim}",
            tensors=tensors,
            inputs=["input", "Wq", "Wk", "Wv", "Wo", "W1", "W2"],
            outputs=["output"],
            operators=operators,
            metadata={
                "num_heads": num_heads,
                "head_dim": head_dim,
                "intermediate_dim": intermediate_dim,
            },
        )

    @classmethod
    def from_pytorch(
        cls,
        module,  # torch.nn.Module, function, or callable
        example_inputs: Union[List, Dict, Tuple],
        name: str = "pytorch_module",
        capture_method: str = "auto",  # "fx", "dynamo", "auto"
        decompose: bool = False,  # Decompose to aten primitives
    ) -> "ComputeTask":
        """
        Create ComputeTask from ANY PyTorch computation.

        Supports:
        - nn.Module
        - Plain Python functions with torch ops
        - torch.fx GraphModule

        Capture methods:
        - "fx": torch.fx.symbolic_trace (static, most compatible)
        - "dynamo": torch._dynamo.export (dynamic shapes, more flexible)
        - "auto": Try dynamo first, fall back to fx

        Args:
            module: PyTorch module, function, or callable
            example_inputs: Example inputs for tracing (list, dict, or tuple)
            name: Name for the compute task
            capture_method: Graph capture method ("fx", "dynamo", "auto")
            decompose: Whether to decompose to aten primitives

        Returns:
            ComputeTask with captured computation graph
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for from_pytorch")

        # Normalize example_inputs to list
        if isinstance(example_inputs, dict):
            example_inputs_list = list(example_inputs.values())
        elif isinstance(example_inputs, tuple):
            example_inputs_list = list(example_inputs)
        else:
            example_inputs_list = example_inputs

        # Try capture methods based on selection
        traced = None
        capture_used = None

        if capture_method == "auto":
            # Try dynamo first, then fx
            traced, capture_used = cls._try_dynamo_capture(module, example_inputs_list)
            if traced is None:
                traced, capture_used = cls._try_fx_capture(module)
        elif capture_method == "dynamo":
            traced, capture_used = cls._try_dynamo_capture(module, example_inputs_list)
            if traced is None:
                raise RuntimeError("TorchDynamo capture failed")
        elif capture_method == "fx":
            traced, capture_used = cls._try_fx_capture(module)
            if traced is None:
                raise RuntimeError("torch.fx capture failed")
        else:
            raise ValueError(f"Unknown capture_method: {capture_method}")

        if traced is None:
            raise RuntimeError(f"Failed to capture graph with method: {capture_method}")

        # Parse the traced graph
        return cls._parse_fx_graph(
            traced,
            example_inputs_list,
            name,
            capture_used,
            decompose,
        )

    @classmethod
    def _try_fx_capture(cls, module) -> Tuple[Optional[Any], Optional[str]]:
        """Try to capture graph using torch.fx.symbolic_trace."""
        try:
            import torch
            from torch.fx import symbolic_trace

            traced = symbolic_trace(module)
            return traced, "fx"
        except Exception:
            return None, None

    @classmethod
    def _try_dynamo_capture(
        cls, module, example_inputs: List
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Try to capture graph using torch._dynamo.export."""
        try:
            import torch
            import torch._dynamo as dynamo

            # Export using dynamo
            exported = dynamo.export(module, aten_graph=False)
            graph_module, guards = exported(*example_inputs)
            return graph_module, "dynamo"
        except Exception:
            return None, None

    @classmethod
    def _parse_fx_graph(
        cls,
        traced,
        example_inputs: List,
        name: str,
        capture_method: str,
        decompose: bool,
    ) -> "ComputeTask":
        """Parse a torch.fx GraphModule into ComputeTask."""
        import torch

        tensors = {}
        operators = []
        inputs = []
        outputs = []

        # Extended PyTorch op mapping
        op_mapping = cls._get_pytorch_op_mapping()

        # Dtype mapping
        dtype_map = {
            torch.float32: DataType.FP32,
            torch.float16: DataType.FP16,
            torch.bfloat16: DataType.BF16,
            torch.float8_e4m3fn: DataType.FP8 if hasattr(torch, "float8_e4m3fn") else DataType.FP16,
            torch.int8: DataType.INT8,
            torch.int32: DataType.INT32,
            torch.int64: DataType.INT64,
        }

        input_idx = 0

        # Parse the traced graph
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                # Input placeholder
                inputs.append(node.name)

                # Get shape and dtype from example inputs
                if input_idx < len(example_inputs):
                    inp = example_inputs[input_idx]
                    if hasattr(inp, "shape"):
                        tensors[node.name] = TensorSpec(
                            node.name,
                            tuple(inp.shape),
                            dtype_map.get(inp.dtype, DataType.FP16),
                        )
                    input_idx += 1

            elif node.op == "call_function":
                # Function call (e.g., torch.matmul, F.relu)
                op = cls._parse_call_function(node, op_mapping)
                if op:
                    operators.append(op)

            elif node.op == "call_method":
                # Method call (e.g., x.view(), t.transpose())
                op = cls._parse_call_method(node, op_mapping)
                if op:
                    operators.append(op)

            elif node.op == "call_module":
                # Module call (e.g., self.fc1, self.attention)
                ops = cls._parse_call_module(node, traced, op_mapping)
                operators.extend(ops)

            elif node.op == "get_attr":
                # Attribute access (weights, buffers)
                # These are constants, not operators
                pass

            elif node.op == "output":
                # Output
                for arg in node.args:
                    if hasattr(arg, "__iter__") and not isinstance(arg, str):
                        outputs.extend([str(a.name) for a in arg if hasattr(a, "name")])
                    elif hasattr(arg, "name"):
                        outputs.append(str(arg.name))

        return cls(
            name=name,
            tensors=tensors,
            inputs=inputs,
            outputs=outputs,
            operators=operators,
            metadata={
                "capture_method": capture_method,
                "decomposed": decompose,
            },
        )

    @classmethod
    def _get_pytorch_op_mapping(cls) -> Dict[str, OperatorType]:
        """Get extended mapping from PyTorch ops to OperatorType."""
        return {
            # Matrix operations
            "matmul": OperatorType.MATMUL,
            "bmm": OperatorType.BATCH_MATMUL,
            "mm": OperatorType.MATMUL,
            "linear": OperatorType.MATMUL,
            "addmm": OperatorType.GEMM,
            "baddbmm": OperatorType.BATCH_MATMUL,
            # Element-wise operations
            "add": OperatorType.ADD,
            "sub": OperatorType.ADD,  # Treat as add with negation
            "mul": OperatorType.MUL,
            "div": OperatorType.DIV,
            "exp": OperatorType.EXP,
            "log": OperatorType.LOG,
            "sqrt": OperatorType.SQRT,
            "rsqrt": OperatorType.SQRT,
            "pow": OperatorType.MUL,
            "neg": OperatorType.MUL,
            # Activation functions
            "gelu": OperatorType.GELU,
            "relu": OperatorType.RELU,
            "silu": OperatorType.SILU,
            "sigmoid": OperatorType.SIGMOID,
            "tanh": OperatorType.TANH,
            "leaky_relu": OperatorType.RELU,
            "elu": OperatorType.RELU,
            "hardswish": OperatorType.SILU,
            "mish": OperatorType.SILU,
            # Reduction operations
            "sum": OperatorType.SUM,
            "mean": OperatorType.MEAN,
            "max": OperatorType.MAX,
            "min": OperatorType.MIN,
            "softmax": OperatorType.SOFTMAX,
            "log_softmax": OperatorType.SOFTMAX,
            # Normalization
            "layer_norm": OperatorType.LAYER_NORM,
            "rms_norm": OperatorType.RMS_NORM,
            "batch_norm": OperatorType.LAYER_NORM,
            "group_norm": OperatorType.LAYER_NORM,
            "instance_norm": OperatorType.LAYER_NORM,
            # Data movement
            "transpose": OperatorType.TRANSPOSE,
            "permute": OperatorType.TRANSPOSE,
            "reshape": OperatorType.RESHAPE,
            "view": OperatorType.RESHAPE,
            "flatten": OperatorType.RESHAPE,
            "squeeze": OperatorType.RESHAPE,
            "unsqueeze": OperatorType.RESHAPE,
            "cat": OperatorType.CONCAT,
            "concat": OperatorType.CONCAT,
            "stack": OperatorType.CONCAT,
            "split": OperatorType.SPLIT,
            "chunk": OperatorType.SPLIT,
            "gather": OperatorType.GATHER,
            "scatter": OperatorType.SCATTER,
            "index_select": OperatorType.GATHER,
            "embedding": OperatorType.GATHER,
            # Attention
            "scaled_dot_product_attention": OperatorType.ATTENTION,
            "multi_head_attention_forward": OperatorType.MULTI_HEAD_ATTENTION,
            # Convolution
            "conv1d": OperatorType.CONV2D,
            "conv2d": OperatorType.CONV2D,
            "conv3d": OperatorType.CONV3D,
            "conv_transpose2d": OperatorType.CONV2D,
            "depthwise_conv2d": OperatorType.DEPTHWISE_CONV,
        }

    @classmethod
    def _parse_call_function(cls, node, op_mapping: Dict) -> Optional[Operator]:
        """Parse a call_function node."""
        # Get function name
        target = node.target
        if hasattr(target, "__name__"):
            func_name = target.__name__.lower()
        elif hasattr(target, "__str__"):
            func_name = str(target).split(".")[-1].lower()
        else:
            func_name = "unknown"

        # Remove common suffixes/prefixes
        func_name_clean = func_name.replace("_", "").replace("aten::", "")

        # Map to operator type
        op_type = op_mapping.get(func_name_clean, OperatorType.CUSTOM)

        # Also try original func_name
        if op_type == OperatorType.CUSTOM:
            op_type = op_mapping.get(func_name, OperatorType.CUSTOM)

        # Extract input names
        input_names = []
        for arg in node.args:
            if hasattr(arg, "name"):
                input_names.append(arg.name)

        # Also check kwargs
        for key, val in node.kwargs.items():
            if hasattr(val, "name"):
                input_names.append(val.name)

        # Create TorchOp for full PyTorch semantics
        torch_op = TorchOp.from_fx_node(node)

        return Operator(
            op_id=node.name,
            op_type=op_type,
            inputs=input_names,
            outputs=[node.name],
            torch_op=torch_op,
            attrs={
                "original_op": func_name,
                "torch_target": str(target),
            },
        )

    @classmethod
    def _parse_call_method(cls, node, op_mapping: Dict) -> Optional[Operator]:
        """Parse a call_method node."""
        method_name = str(node.target).lower()

        # Map to operator type
        op_type = op_mapping.get(method_name, OperatorType.CUSTOM)

        # First arg is the object the method is called on
        input_names = []
        for arg in node.args:
            if hasattr(arg, "name"):
                input_names.append(arg.name)

        return Operator(
            op_id=node.name,
            op_type=op_type,
            inputs=input_names,
            outputs=[node.name],
            attrs={
                "original_op": method_name,
                "is_method": True,
            },
        )

    @classmethod
    def _parse_call_module(cls, node, traced, op_mapping: Dict) -> List[Operator]:
        """Parse a call_module node (e.g., self.fc1, self.attention)."""
        import torch.nn as nn

        operators = []
        module_name = node.target

        # Get the actual module
        try:
            submodule = traced.get_submodule(module_name)
        except AttributeError:
            submodule = None

        # Determine operator type based on module class
        if submodule is not None:
            module_class = type(submodule).__name__.lower()
        else:
            module_class = module_name.lower()

        # Map module types to operators
        module_op_mapping = {
            "linear": OperatorType.MATMUL,
            "conv1d": OperatorType.CONV2D,
            "conv2d": OperatorType.CONV2D,
            "conv3d": OperatorType.CONV3D,
            "batchnorm1d": OperatorType.LAYER_NORM,
            "batchnorm2d": OperatorType.LAYER_NORM,
            "layernorm": OperatorType.LAYER_NORM,
            "groupnorm": OperatorType.LAYER_NORM,
            "rmsnorm": OperatorType.RMS_NORM,
            "dropout": OperatorType.CUSTOM,
            "embedding": OperatorType.GATHER,
            "multiheadattention": OperatorType.MULTI_HEAD_ATTENTION,
            "gelu": OperatorType.GELU,
            "relu": OperatorType.RELU,
            "silu": OperatorType.SILU,
            "softmax": OperatorType.SOFTMAX,
        }

        op_type = module_op_mapping.get(module_class, OperatorType.CUSTOM)

        # Extract input names
        input_names = []
        for arg in node.args:
            if hasattr(arg, "name"):
                input_names.append(arg.name)

        operators.append(
            Operator(
                op_id=node.name,
                op_type=op_type,
                inputs=input_names,
                outputs=[node.name],
                attrs={
                    "original_op": module_class,
                    "module_name": module_name,
                    "is_module": True,
                },
            )
        )

        return operators

    # ==========================================================================
    # Additional Factory Methods for Universal Task Creation
    # ==========================================================================

    @classmethod
    def from_torch_function(
        cls,
        fn,  # Callable with torch ops
        input_specs: Dict[str, Tuple[int, ...]],
        name: str = "torch_function",
        dtype: DataType = DataType.FP32,
    ) -> "ComputeTask":
        """
        Create ComputeTask from a plain Python function using torch ops.

        Args:
            fn: Python function that uses torch operations
            input_specs: Dict mapping input names to shapes
            name: Name for the compute task
            dtype: Data type for tensors

        Returns:
            ComputeTask with captured computation graph

        Example:
            def my_gelu(x):
                return x * 0.5 * (1 + torch.tanh(...))

            task = ComputeTask.from_torch_function(
                my_gelu,
                input_specs={"x": (1024, 4096)},
            )
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for from_torch_function")

        # Create example inputs from specs
        torch_dtype = {
            DataType.FP32: torch.float32,
            DataType.FP16: torch.float16,
            DataType.BF16: torch.bfloat16,
        }.get(dtype, torch.float32)

        example_inputs = []
        for input_name, shape in input_specs.items():
            example_inputs.append(torch.randn(*shape, dtype=torch_dtype))

        # Wrap function in a module for tracing
        class FunctionWrapper(nn.Module):
            def __init__(self, func, param_names):
                super().__init__()
                self.func = func
                self.param_names = param_names

            def forward(self, *args):
                # Map positional args to named kwargs
                kwargs = dict(zip(self.param_names, args))
                return self.func(**kwargs)

        wrapper = FunctionWrapper(fn, list(input_specs.keys()))

        return cls.from_pytorch(
            wrapper,
            example_inputs=example_inputs,
            name=name,
            capture_method="auto",
        )

    @classmethod
    def from_einsum(
        cls,
        equation: str,
        shapes: Dict[str, Tuple[int, ...]],
        name: str = "einsum",
        dtype: DataType = DataType.FP32,
    ) -> "ComputeTask":
        """
        Create ComputeTask from Einstein summation notation.

        This provides a concise way to express tensor contractions
        including matmul, batched matmul, attention patterns, etc.

        Args:
            equation: Einstein notation (e.g., "ij,jk->ik" for matmul)
            shapes: Dict mapping tensor names to shapes
            name: Name for the compute task
            dtype: Data type for tensors

        Returns:
            ComputeTask with einsum operation

        Example:
            # Matrix multiplication
            task = ComputeTask.from_einsum(
                "ij,jk->ik",
                shapes={"A": (128, 256), "B": (256, 512)},
            )

            # Batched matmul
            task = ComputeTask.from_einsum(
                "bij,bjk->bik",
                shapes={"A": (32, 128, 256), "B": (32, 256, 512)},
            )
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for from_einsum")

        # Parse equation
        if "->" not in equation:
            raise ValueError("Equation must contain '->' to specify output")

        input_part, output_part = equation.split("->")
        input_parts = input_part.split(",")

        # Validate shape count matches
        tensor_names = list(shapes.keys())
        if len(input_parts) != len(tensor_names):
            raise ValueError(
                f"Number of inputs in equation ({len(input_parts)}) "
                f"doesn't match number of shapes ({len(tensor_names)})"
            )

        # Create tensor specs
        tensors = {}
        inputs = []
        for i, (tname, shape) in enumerate(shapes.items()):
            tensors[tname] = TensorSpec(tname, shape, dtype)
            inputs.append(tname)

        # Infer output shape from equation
        output_shape = cls._infer_einsum_output_shape(equation, shapes)
        output_name = "output"
        tensors[output_name] = TensorSpec(output_name, output_shape, dtype)

        # Create einsum operator
        operators = [
            Operator(
                op_id="einsum_0",
                op_type=OperatorType.CUSTOM,
                inputs=inputs,
                outputs=[output_name],
                torch_op=TorchOp(
                    target="torch.einsum",
                    is_composite=False,
                ),
                attrs={
                    "equation": equation,
                    "original_op": "einsum",
                },
            )
        ]

        return cls(
            name=name,
            tensors=tensors,
            inputs=inputs,
            outputs=[output_name],
            operators=operators,
            metadata={
                "einsum_equation": equation,
            },
        )

    @classmethod
    def _infer_einsum_output_shape(
        cls,
        equation: str,
        shapes: Dict[str, Tuple[int, ...]],
    ) -> Tuple[int, ...]:
        """Infer output shape from einsum equation."""
        input_part, output_part = equation.split("->")
        input_parts = input_part.split(",")

        # Build index to dimension mapping
        index_to_dim = {}
        for i, (tname, shape) in enumerate(shapes.items()):
            indices = input_parts[i].strip()
            for j, idx in enumerate(indices):
                if idx not in index_to_dim:
                    index_to_dim[idx] = shape[j]

        # Build output shape
        output_shape = []
        for idx in output_part.strip():
            if idx in index_to_dim:
                output_shape.append(index_to_dim[idx])
            else:
                output_shape.append(1)

        return tuple(output_shape) if output_shape else (1,)

    @classmethod
    def from_expression(
        cls,
        expression: str,
        shapes: Dict[str, Tuple[int, ...]],
        name: str = "expression",
        dtype: DataType = DataType.FP32,
    ) -> "ComputeTask":
        """
        Create ComputeTask from a math expression string.

        The expression can use standard math operators and torch functions.
        Variable names in the expression must match keys in shapes dict.

        Args:
            expression: Math expression (e.g., "a + b * c", "torch.relu(x)")
            shapes: Dict mapping variable names to shapes
            name: Name for the compute task
            dtype: Data type for tensors

        Returns:
            ComputeTask with the expression as computation

        Example:
            task = ComputeTask.from_expression(
                "torch.sqrt(x ** 2 + y ** 2)",
                shapes={"x": (128, 128), "y": (128, 128)},
                name="euclidean_norm",
            )
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch is required for from_expression")

        # Create a function from the expression
        param_names = list(shapes.keys())
        func_params = ", ".join(param_names)

        # Create function code
        func_code = f"""
def _expr_func({func_params}):
    return {expression}
"""

        # Execute to get the function
        local_ns = {"torch": torch}
        exec(func_code, local_ns)
        expr_func = local_ns["_expr_func"]

        # Use from_torch_function
        return cls.from_torch_function(
            expr_func,
            input_specs=shapes,
            name=name,
            dtype=dtype,
        )


@dataclass
class SubTask:
    """A sub-task after decomposition, assigned to a specific device."""

    subtask_id: str
    original_task: str

    # Assigned operators
    operators: List[str]

    # Device assignment
    device_id: Optional[str] = None

    # Input/output from other subtasks
    external_inputs: List[str] = field(default_factory=list)
    external_outputs: List[str] = field(default_factory=list)

    # Estimated costs
    estimated_flops: int = 0
    estimated_memory_bytes: int = 0
    estimated_time_ms: float = 0.0


@dataclass
class TaskGraph:
    """
    A decomposed task graph with sub-tasks and their dependencies.
    Ready for cluster-level scheduling.
    """

    original_task: ComputeTask
    subtasks: List[SubTask] = field(default_factory=list)

    # Dependencies between subtasks
    dependencies: List[Tuple[str, str, int]] = field(default_factory=list)  # (src, dst, bytes)

    def get_critical_path(self) -> List[str]:
        """Find the critical path through the task graph."""
        if not self.subtasks:
            return []

        # Build time estimates including communication
        times = {st.subtask_id: st.estimated_time_ms for st in self.subtasks}

        # Topological sort and compute longest path
        # ... simplified implementation
        return [st.subtask_id for st in self.subtasks]

    def estimate_total_time_ms(self) -> float:
        """Estimate total execution time considering parallelism."""
        # Critical path analysis
        critical_path = self.get_critical_path()
        return sum(st.estimated_time_ms for st in self.subtasks if st.subtask_id in critical_path)
