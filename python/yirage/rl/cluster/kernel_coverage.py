"""
Kernel Coverage Matrix

Documents which kernel operations are supported on which hardware backends.
This helps identify gaps in hardware support and guides optimization decisions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional
from .topology import DeviceType


class KernelOpType(Enum):
    """Kernel operation types supported by YiRage."""

    # Basic I/O
    INPUT = "input"
    OUTPUT = "output"

    # Matrix operations
    MATMUL = "matmul"
    BATCHED_MATMUL = "batched_matmul"
    GEMM = "gemm"

    # Elementwise unary
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    SQUARE = "square"
    SILU = "silu"
    SIGMOID = "sigmoid"
    GELU = "gelu"
    RELU = "relu"
    TANH = "tanh"
    CLAMP = "clamp"
    SOFTMAX = "softmax"

    # Elementwise binary
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"

    # Reductions
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"

    # Normalization
    RMS_NORM = "rms_norm"
    LAYER_NORM = "layer_norm"
    BATCH_NORM = "batch_norm"

    # Attention
    ATTENTION = "attention"
    FLASH_ATTENTION = "flash_attention"
    MULTI_HEAD_ATTENTION = "multi_head_attention"
    GROUPED_QUERY_ATTENTION = "grouped_query_attention"

    # Memory operations
    CONCAT = "concat"
    SPLIT = "split"
    TRANSPOSE = "transpose"
    RESHAPE = "reshape"

    # Communication
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"

    # Quantization
    QUANTIZE = "quantize"
    DEQUANTIZE = "dequantize"

    # Custom
    CUSTOM = "custom"


class SupportLevel(Enum):
    """Support level for a kernel on a hardware backend."""

    NATIVE = "native"  # Native optimized implementation
    FALLBACK = "fallback"  # Fallback implementation (slower)
    TRITON = "triton"  # Via Triton compilation
    EMULATED = "emulated"  # Emulated via other ops
    UNSUPPORTED = "unsupported"  # Not supported


@dataclass
class KernelSupport:
    """Support information for a kernel on a backend."""

    level: SupportLevel
    notes: str = ""
    estimated_efficiency: float = 1.0  # 1.0 = optimal, lower = slower

    def __repr__(self):
        if self.level == SupportLevel.NATIVE:
            return "✅"
        elif self.level == SupportLevel.TRITON:
            return "🔷"
        elif self.level == SupportLevel.FALLBACK:
            return "⚠️"
        elif self.level == SupportLevel.EMULATED:
            return "🔄"
        else:
            return "❌"


# ============================================================================
# Kernel Coverage Matrix
# ============================================================================

KERNEL_COVERAGE: Dict[DeviceType, Dict[KernelOpType, KernelSupport]] = {
    # =========================================================================
    # NVIDIA CUDA - Full native support
    # =========================================================================
    DeviceType.CUDA: {
        # Basic
        KernelOpType.INPUT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.OUTPUT: KernelSupport(SupportLevel.NATIVE),
        # Matrix
        KernelOpType.MATMUL: KernelSupport(SupportLevel.NATIVE, "cuBLAS + Tensor Cores"),
        KernelOpType.BATCHED_MATMUL: KernelSupport(SupportLevel.NATIVE, "cuBLAS batched GEMM"),
        KernelOpType.GEMM: KernelSupport(SupportLevel.NATIVE, "cuBLAS GEMM"),
        # Elementwise unary
        KernelOpType.EXP: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.LOG: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQRT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQUARE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SILU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SIGMOID: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TANH: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.CLAMP: KernelSupport(SupportLevel.NATIVE),
        # Elementwise binary
        KernelOpType.ADD: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SUB: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.DIV: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.POW: KernelSupport(SupportLevel.NATIVE),
        # Reductions
        KernelOpType.SUM: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MEAN: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MAX: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MIN: KernelSupport(SupportLevel.NATIVE),
        # Normalization
        KernelOpType.RMS_NORM: KernelSupport(SupportLevel.NATIVE, "Fused kernel"),
        KernelOpType.LAYER_NORM: KernelSupport(SupportLevel.NATIVE, "cuDNN"),
        KernelOpType.BATCH_NORM: KernelSupport(SupportLevel.NATIVE, "cuDNN"),
        # Attention
        KernelOpType.ATTENTION: KernelSupport(SupportLevel.NATIVE, "FlashAttention"),
        KernelOpType.FLASH_ATTENTION: KernelSupport(SupportLevel.NATIVE, "FlashAttention-2"),
        KernelOpType.MULTI_HEAD_ATTENTION: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GROUPED_QUERY_ATTENTION: KernelSupport(SupportLevel.NATIVE),
        # Memory
        KernelOpType.CONCAT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SPLIT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TRANSPOSE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RESHAPE: KernelSupport(SupportLevel.NATIVE),
        # Communication
        KernelOpType.ALL_REDUCE: KernelSupport(SupportLevel.NATIVE, "NCCL"),
        KernelOpType.ALL_GATHER: KernelSupport(SupportLevel.NATIVE, "NCCL"),
        KernelOpType.REDUCE_SCATTER: KernelSupport(SupportLevel.NATIVE, "NCCL"),
        KernelOpType.ALL_TO_ALL: KernelSupport(SupportLevel.NATIVE, "NCCL"),
        # Quantization
        KernelOpType.QUANTIZE: KernelSupport(SupportLevel.NATIVE, "INT8/FP8"),
        KernelOpType.DEQUANTIZE: KernelSupport(SupportLevel.NATIVE),
        # Custom
        KernelOpType.CUSTOM: KernelSupport(SupportLevel.NATIVE, "µGraph search"),
    },
    # =========================================================================
    # AMD ROCm - Good support via HIP
    # =========================================================================
    DeviceType.ROCM: {
        # Basic
        KernelOpType.INPUT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.OUTPUT: KernelSupport(SupportLevel.NATIVE),
        # Matrix
        KernelOpType.MATMUL: KernelSupport(SupportLevel.NATIVE, "rocBLAS + Matrix Cores"),
        KernelOpType.BATCHED_MATMUL: KernelSupport(SupportLevel.NATIVE, "rocBLAS"),
        KernelOpType.GEMM: KernelSupport(SupportLevel.NATIVE, "rocBLAS"),
        # Elementwise unary
        KernelOpType.EXP: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.LOG: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQRT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQUARE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SILU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SIGMOID: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TANH: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.CLAMP: KernelSupport(SupportLevel.NATIVE),
        # Elementwise binary
        KernelOpType.ADD: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SUB: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.DIV: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.POW: KernelSupport(SupportLevel.NATIVE),
        # Reductions
        KernelOpType.SUM: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MEAN: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MAX: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MIN: KernelSupport(SupportLevel.NATIVE),
        # Normalization
        KernelOpType.RMS_NORM: KernelSupport(SupportLevel.TRITON, "Via Triton"),
        KernelOpType.LAYER_NORM: KernelSupport(SupportLevel.NATIVE, "MIOpen"),
        KernelOpType.BATCH_NORM: KernelSupport(SupportLevel.NATIVE, "MIOpen"),
        # Attention
        KernelOpType.ATTENTION: KernelSupport(SupportLevel.TRITON, "Via Triton"),
        KernelOpType.FLASH_ATTENTION: KernelSupport(SupportLevel.TRITON, "FlashAttention-Triton"),
        KernelOpType.MULTI_HEAD_ATTENTION: KernelSupport(SupportLevel.TRITON),
        KernelOpType.GROUPED_QUERY_ATTENTION: KernelSupport(SupportLevel.TRITON),
        # Memory
        KernelOpType.CONCAT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SPLIT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TRANSPOSE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RESHAPE: KernelSupport(SupportLevel.NATIVE),
        # Communication
        KernelOpType.ALL_REDUCE: KernelSupport(SupportLevel.NATIVE, "RCCL"),
        KernelOpType.ALL_GATHER: KernelSupport(SupportLevel.NATIVE, "RCCL"),
        KernelOpType.REDUCE_SCATTER: KernelSupport(SupportLevel.NATIVE, "RCCL"),
        KernelOpType.ALL_TO_ALL: KernelSupport(SupportLevel.NATIVE, "RCCL"),
        # Quantization
        KernelOpType.QUANTIZE: KernelSupport(SupportLevel.NATIVE, "INT8"),
        KernelOpType.DEQUANTIZE: KernelSupport(SupportLevel.NATIVE),
        # Custom
        KernelOpType.CUSTOM: KernelSupport(SupportLevel.TRITON, "Via Triton µGraph"),
    },
    # =========================================================================
    # Huawei Ascend - Via BiSheng Triton compiler
    # =========================================================================
    DeviceType.ASCEND: {
        # Basic
        KernelOpType.INPUT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.OUTPUT: KernelSupport(SupportLevel.NATIVE),
        # Matrix
        KernelOpType.MATMUL: KernelSupport(SupportLevel.NATIVE, "CANN + Cube Unit"),
        KernelOpType.BATCHED_MATMUL: KernelSupport(SupportLevel.NATIVE, "CANN"),
        KernelOpType.GEMM: KernelSupport(SupportLevel.NATIVE, "CANN"),
        # Elementwise unary
        KernelOpType.EXP: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.LOG: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQRT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQUARE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SILU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SIGMOID: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TANH: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.CLAMP: KernelSupport(SupportLevel.NATIVE),
        # Elementwise binary
        KernelOpType.ADD: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SUB: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.DIV: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.POW: KernelSupport(SupportLevel.NATIVE),
        # Reductions
        KernelOpType.SUM: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MEAN: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MAX: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MIN: KernelSupport(SupportLevel.NATIVE),
        # Normalization
        KernelOpType.RMS_NORM: KernelSupport(SupportLevel.TRITON, "Via BiSheng Triton"),
        KernelOpType.LAYER_NORM: KernelSupport(SupportLevel.NATIVE, "CANN"),
        KernelOpType.BATCH_NORM: KernelSupport(SupportLevel.NATIVE, "CANN"),
        # Attention
        KernelOpType.ATTENTION: KernelSupport(SupportLevel.TRITON, "Via BiSheng"),
        KernelOpType.FLASH_ATTENTION: KernelSupport(SupportLevel.TRITON, "BiSheng FlashAttention"),
        KernelOpType.MULTI_HEAD_ATTENTION: KernelSupport(SupportLevel.NATIVE, "CANN"),
        KernelOpType.GROUPED_QUERY_ATTENTION: KernelSupport(SupportLevel.TRITON),
        # Memory
        KernelOpType.CONCAT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SPLIT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TRANSPOSE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RESHAPE: KernelSupport(SupportLevel.NATIVE),
        # Communication
        KernelOpType.ALL_REDUCE: KernelSupport(SupportLevel.NATIVE, "HCCL"),
        KernelOpType.ALL_GATHER: KernelSupport(SupportLevel.NATIVE, "HCCL"),
        KernelOpType.REDUCE_SCATTER: KernelSupport(SupportLevel.NATIVE, "HCCL"),
        KernelOpType.ALL_TO_ALL: KernelSupport(SupportLevel.NATIVE, "HCCL"),
        # Quantization
        KernelOpType.QUANTIZE: KernelSupport(SupportLevel.NATIVE, "INT8"),
        KernelOpType.DEQUANTIZE: KernelSupport(SupportLevel.NATIVE),
        # Custom
        KernelOpType.CUSTOM: KernelSupport(SupportLevel.TRITON, "Via BiSheng µGraph"),
    },
    # =========================================================================
    # MetaX MACA - CUDA compatible
    # =========================================================================
    DeviceType.MACA: {
        # Basic
        KernelOpType.INPUT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.OUTPUT: KernelSupport(SupportLevel.NATIVE),
        # Matrix
        KernelOpType.MATMUL: KernelSupport(SupportLevel.NATIVE, "mcBLAS"),
        KernelOpType.BATCHED_MATMUL: KernelSupport(SupportLevel.NATIVE, "mcBLAS"),
        KernelOpType.GEMM: KernelSupport(SupportLevel.NATIVE, "mcBLAS"),
        # Elementwise unary
        KernelOpType.EXP: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.LOG: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQRT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQUARE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SILU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SIGMOID: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TANH: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.CLAMP: KernelSupport(SupportLevel.NATIVE),
        # Elementwise binary
        KernelOpType.ADD: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SUB: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.DIV: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.POW: KernelSupport(SupportLevel.NATIVE),
        # Reductions
        KernelOpType.SUM: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MEAN: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MAX: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MIN: KernelSupport(SupportLevel.NATIVE),
        # Normalization
        KernelOpType.RMS_NORM: KernelSupport(SupportLevel.NATIVE, "mcDNN"),
        KernelOpType.LAYER_NORM: KernelSupport(SupportLevel.NATIVE, "mcDNN"),
        KernelOpType.BATCH_NORM: KernelSupport(SupportLevel.NATIVE, "mcDNN"),
        # Attention
        KernelOpType.ATTENTION: KernelSupport(SupportLevel.NATIVE, "Optimized kernel"),
        KernelOpType.FLASH_ATTENTION: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MULTI_HEAD_ATTENTION: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GROUPED_QUERY_ATTENTION: KernelSupport(SupportLevel.NATIVE),
        # Memory
        KernelOpType.CONCAT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SPLIT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TRANSPOSE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RESHAPE: KernelSupport(SupportLevel.NATIVE),
        # Communication
        KernelOpType.ALL_REDUCE: KernelSupport(SupportLevel.NATIVE, "mcCCL"),
        KernelOpType.ALL_GATHER: KernelSupport(SupportLevel.NATIVE, "mcCCL"),
        KernelOpType.REDUCE_SCATTER: KernelSupport(SupportLevel.NATIVE, "mcCCL"),
        KernelOpType.ALL_TO_ALL: KernelSupport(SupportLevel.NATIVE, "mcCCL"),
        # Quantization
        KernelOpType.QUANTIZE: KernelSupport(SupportLevel.NATIVE, "INT8"),
        KernelOpType.DEQUANTIZE: KernelSupport(SupportLevel.NATIVE),
        # Custom
        KernelOpType.CUSTOM: KernelSupport(SupportLevel.NATIVE, "µGraph search"),
    },
    # =========================================================================
    # Google TPU - Via XLA
    # =========================================================================
    DeviceType.TPU: {
        # Basic
        KernelOpType.INPUT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.OUTPUT: KernelSupport(SupportLevel.NATIVE),
        # Matrix
        KernelOpType.MATMUL: KernelSupport(SupportLevel.NATIVE, "XLA + MXU"),
        KernelOpType.BATCHED_MATMUL: KernelSupport(SupportLevel.NATIVE, "XLA"),
        KernelOpType.GEMM: KernelSupport(SupportLevel.NATIVE, "XLA"),
        # Elementwise unary
        KernelOpType.EXP: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.LOG: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQRT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQUARE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SILU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SIGMOID: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TANH: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.CLAMP: KernelSupport(SupportLevel.NATIVE),
        # Elementwise binary
        KernelOpType.ADD: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SUB: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.DIV: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.POW: KernelSupport(SupportLevel.NATIVE),
        # Reductions
        KernelOpType.SUM: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MEAN: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MAX: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MIN: KernelSupport(SupportLevel.NATIVE),
        # Normalization
        KernelOpType.RMS_NORM: KernelSupport(SupportLevel.NATIVE, "Pallas"),
        KernelOpType.LAYER_NORM: KernelSupport(SupportLevel.NATIVE, "XLA"),
        KernelOpType.BATCH_NORM: KernelSupport(SupportLevel.NATIVE, "XLA"),
        # Attention
        KernelOpType.ATTENTION: KernelSupport(SupportLevel.NATIVE, "Pallas FlashAttention"),
        KernelOpType.FLASH_ATTENTION: KernelSupport(SupportLevel.NATIVE, "Pallas"),
        KernelOpType.MULTI_HEAD_ATTENTION: KernelSupport(SupportLevel.NATIVE, "XLA"),
        KernelOpType.GROUPED_QUERY_ATTENTION: KernelSupport(SupportLevel.NATIVE),
        # Memory
        KernelOpType.CONCAT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SPLIT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TRANSPOSE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RESHAPE: KernelSupport(SupportLevel.NATIVE),
        # Communication
        KernelOpType.ALL_REDUCE: KernelSupport(SupportLevel.NATIVE, "ICI"),
        KernelOpType.ALL_GATHER: KernelSupport(SupportLevel.NATIVE, "ICI"),
        KernelOpType.REDUCE_SCATTER: KernelSupport(SupportLevel.NATIVE, "ICI"),
        KernelOpType.ALL_TO_ALL: KernelSupport(SupportLevel.NATIVE, "ICI"),
        # Quantization
        KernelOpType.QUANTIZE: KernelSupport(SupportLevel.NATIVE, "INT8/INT4"),
        KernelOpType.DEQUANTIZE: KernelSupport(SupportLevel.NATIVE),
        # Custom
        KernelOpType.CUSTOM: KernelSupport(SupportLevel.FALLBACK, "Via Pallas"),
    },
    # =========================================================================
    # Intel XPU - oneAPI/SYCL
    # =========================================================================
    DeviceType.XPU: {
        # Basic
        KernelOpType.INPUT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.OUTPUT: KernelSupport(SupportLevel.NATIVE),
        # Matrix
        KernelOpType.MATMUL: KernelSupport(SupportLevel.NATIVE, "oneMKL + XMX"),
        KernelOpType.BATCHED_MATMUL: KernelSupport(SupportLevel.NATIVE, "oneMKL"),
        KernelOpType.GEMM: KernelSupport(SupportLevel.NATIVE, "oneMKL"),
        # Elementwise unary
        KernelOpType.EXP: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.LOG: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQRT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQUARE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SILU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SIGMOID: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TANH: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.CLAMP: KernelSupport(SupportLevel.NATIVE),
        # Elementwise binary
        KernelOpType.ADD: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SUB: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.DIV: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.POW: KernelSupport(SupportLevel.NATIVE),
        # Reductions
        KernelOpType.SUM: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MEAN: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MAX: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MIN: KernelSupport(SupportLevel.NATIVE),
        # Normalization
        KernelOpType.RMS_NORM: KernelSupport(SupportLevel.TRITON, "Via Triton"),
        KernelOpType.LAYER_NORM: KernelSupport(SupportLevel.NATIVE, "oneDNN"),
        KernelOpType.BATCH_NORM: KernelSupport(SupportLevel.NATIVE, "oneDNN"),
        # Attention
        KernelOpType.ATTENTION: KernelSupport(SupportLevel.TRITON, "Via Triton"),
        KernelOpType.FLASH_ATTENTION: KernelSupport(SupportLevel.TRITON),
        KernelOpType.MULTI_HEAD_ATTENTION: KernelSupport(SupportLevel.NATIVE, "oneDNN"),
        KernelOpType.GROUPED_QUERY_ATTENTION: KernelSupport(SupportLevel.TRITON),
        # Memory
        KernelOpType.CONCAT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SPLIT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TRANSPOSE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RESHAPE: KernelSupport(SupportLevel.NATIVE),
        # Communication
        KernelOpType.ALL_REDUCE: KernelSupport(SupportLevel.NATIVE, "oneCCL"),
        KernelOpType.ALL_GATHER: KernelSupport(SupportLevel.NATIVE, "oneCCL"),
        KernelOpType.REDUCE_SCATTER: KernelSupport(SupportLevel.NATIVE, "oneCCL"),
        KernelOpType.ALL_TO_ALL: KernelSupport(SupportLevel.NATIVE, "oneCCL"),
        # Quantization
        KernelOpType.QUANTIZE: KernelSupport(SupportLevel.NATIVE, "INT8"),
        KernelOpType.DEQUANTIZE: KernelSupport(SupportLevel.NATIVE),
        # Custom
        KernelOpType.CUSTOM: KernelSupport(SupportLevel.TRITON, "Via Triton µGraph"),
    },
    # =========================================================================
    # Apple MPS - Metal Performance Shaders
    # =========================================================================
    DeviceType.MPS: {
        # Basic
        KernelOpType.INPUT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.OUTPUT: KernelSupport(SupportLevel.NATIVE),
        # Matrix
        KernelOpType.MATMUL: KernelSupport(SupportLevel.NATIVE, "MPSGraph"),
        KernelOpType.BATCHED_MATMUL: KernelSupport(SupportLevel.NATIVE, "MPSGraph"),
        KernelOpType.GEMM: KernelSupport(SupportLevel.NATIVE, "MPSGraph"),
        # Elementwise unary
        KernelOpType.EXP: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.LOG: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQRT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQUARE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SILU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SIGMOID: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TANH: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.CLAMP: KernelSupport(SupportLevel.NATIVE),
        # Elementwise binary
        KernelOpType.ADD: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SUB: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.DIV: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.POW: KernelSupport(SupportLevel.NATIVE),
        # Reductions
        KernelOpType.SUM: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MEAN: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MAX: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MIN: KernelSupport(SupportLevel.NATIVE),
        # Normalization
        KernelOpType.RMS_NORM: KernelSupport(SupportLevel.EMULATED, "Via elementwise"),
        KernelOpType.LAYER_NORM: KernelSupport(SupportLevel.NATIVE, "MPSGraph"),
        KernelOpType.BATCH_NORM: KernelSupport(SupportLevel.NATIVE, "MPSGraph"),
        # Attention
        KernelOpType.ATTENTION: KernelSupport(SupportLevel.EMULATED, "Via decomposition"),
        KernelOpType.FLASH_ATTENTION: KernelSupport(SupportLevel.UNSUPPORTED),
        KernelOpType.MULTI_HEAD_ATTENTION: KernelSupport(SupportLevel.NATIVE, "MPSGraph"),
        KernelOpType.GROUPED_QUERY_ATTENTION: KernelSupport(SupportLevel.EMULATED),
        # Memory
        KernelOpType.CONCAT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SPLIT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TRANSPOSE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RESHAPE: KernelSupport(SupportLevel.NATIVE),
        # Communication (single device)
        KernelOpType.ALL_REDUCE: KernelSupport(SupportLevel.UNSUPPORTED, "Single device"),
        KernelOpType.ALL_GATHER: KernelSupport(SupportLevel.UNSUPPORTED, "Single device"),
        KernelOpType.REDUCE_SCATTER: KernelSupport(SupportLevel.UNSUPPORTED, "Single device"),
        KernelOpType.ALL_TO_ALL: KernelSupport(SupportLevel.UNSUPPORTED, "Single device"),
        # Quantization
        KernelOpType.QUANTIZE: KernelSupport(SupportLevel.NATIVE, "INT8"),
        KernelOpType.DEQUANTIZE: KernelSupport(SupportLevel.NATIVE),
        # Custom
        KernelOpType.CUSTOM: KernelSupport(SupportLevel.FALLBACK, "Limited"),
    },
    # =========================================================================
    # CPU - Via MKL/OpenBLAS
    # =========================================================================
    DeviceType.CPU: {
        # Basic
        KernelOpType.INPUT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.OUTPUT: KernelSupport(SupportLevel.NATIVE),
        # Matrix
        KernelOpType.MATMUL: KernelSupport(SupportLevel.NATIVE, "MKL/OpenBLAS"),
        KernelOpType.BATCHED_MATMUL: KernelSupport(SupportLevel.NATIVE, "MKL"),
        KernelOpType.GEMM: KernelSupport(SupportLevel.NATIVE, "MKL/OpenBLAS"),
        # Elementwise unary
        KernelOpType.EXP: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        KernelOpType.LOG: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        KernelOpType.SQRT: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        KernelOpType.SQUARE: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        KernelOpType.SILU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SIGMOID: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TANH: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.CLAMP: KernelSupport(SupportLevel.NATIVE),
        # Elementwise binary
        KernelOpType.ADD: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        KernelOpType.SUB: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        KernelOpType.MUL: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        KernelOpType.DIV: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        KernelOpType.POW: KernelSupport(SupportLevel.NATIVE),
        # Reductions
        KernelOpType.SUM: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        KernelOpType.MEAN: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MAX: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        KernelOpType.MIN: KernelSupport(SupportLevel.NATIVE, "SIMD"),
        # Normalization
        KernelOpType.RMS_NORM: KernelSupport(SupportLevel.NATIVE, "oneDNN"),
        KernelOpType.LAYER_NORM: KernelSupport(SupportLevel.NATIVE, "oneDNN"),
        KernelOpType.BATCH_NORM: KernelSupport(SupportLevel.NATIVE, "oneDNN"),
        # Attention
        KernelOpType.ATTENTION: KernelSupport(SupportLevel.NATIVE, "xFormers CPU"),
        KernelOpType.FLASH_ATTENTION: KernelSupport(SupportLevel.FALLBACK, "Slower"),
        KernelOpType.MULTI_HEAD_ATTENTION: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GROUPED_QUERY_ATTENTION: KernelSupport(SupportLevel.NATIVE),
        # Memory
        KernelOpType.CONCAT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SPLIT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TRANSPOSE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RESHAPE: KernelSupport(SupportLevel.NATIVE),
        # Communication
        KernelOpType.ALL_REDUCE: KernelSupport(SupportLevel.NATIVE, "MPI/Gloo"),
        KernelOpType.ALL_GATHER: KernelSupport(SupportLevel.NATIVE, "MPI/Gloo"),
        KernelOpType.REDUCE_SCATTER: KernelSupport(SupportLevel.NATIVE, "MPI/Gloo"),
        KernelOpType.ALL_TO_ALL: KernelSupport(SupportLevel.NATIVE, "MPI/Gloo"),
        # Quantization
        KernelOpType.QUANTIZE: KernelSupport(SupportLevel.NATIVE, "INT8 + AMX"),
        KernelOpType.DEQUANTIZE: KernelSupport(SupportLevel.NATIVE),
        # Custom
        KernelOpType.CUSTOM: KernelSupport(SupportLevel.FALLBACK, "Limited"),
    },
    # =========================================================================
    # AWS Neuron
    # =========================================================================
    DeviceType.NEURON: {
        # Basic
        KernelOpType.INPUT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.OUTPUT: KernelSupport(SupportLevel.NATIVE),
        # Matrix
        KernelOpType.MATMUL: KernelSupport(SupportLevel.NATIVE, "NeuronCore"),
        KernelOpType.BATCHED_MATMUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GEMM: KernelSupport(SupportLevel.NATIVE),
        # Elementwise unary
        KernelOpType.EXP: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.LOG: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQRT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQUARE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SILU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SIGMOID: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TANH: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.CLAMP: KernelSupport(SupportLevel.NATIVE),
        # Elementwise binary
        KernelOpType.ADD: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SUB: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.DIV: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.POW: KernelSupport(SupportLevel.NATIVE),
        # Reductions
        KernelOpType.SUM: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MEAN: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MAX: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MIN: KernelSupport(SupportLevel.NATIVE),
        # Normalization
        KernelOpType.RMS_NORM: KernelSupport(SupportLevel.NATIVE, "NKI"),
        KernelOpType.LAYER_NORM: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.BATCH_NORM: KernelSupport(SupportLevel.NATIVE),
        # Attention
        KernelOpType.ATTENTION: KernelSupport(SupportLevel.NATIVE, "NKI FlashAttention"),
        KernelOpType.FLASH_ATTENTION: KernelSupport(SupportLevel.NATIVE, "NKI"),
        KernelOpType.MULTI_HEAD_ATTENTION: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GROUPED_QUERY_ATTENTION: KernelSupport(SupportLevel.NATIVE),
        # Memory
        KernelOpType.CONCAT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SPLIT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TRANSPOSE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RESHAPE: KernelSupport(SupportLevel.NATIVE),
        # Communication
        KernelOpType.ALL_REDUCE: KernelSupport(SupportLevel.NATIVE, "NeuronLink"),
        KernelOpType.ALL_GATHER: KernelSupport(SupportLevel.NATIVE, "NeuronLink"),
        KernelOpType.REDUCE_SCATTER: KernelSupport(SupportLevel.NATIVE, "NeuronLink"),
        KernelOpType.ALL_TO_ALL: KernelSupport(SupportLevel.NATIVE, "NeuronLink"),
        # Quantization
        KernelOpType.QUANTIZE: KernelSupport(SupportLevel.NATIVE, "INT8/FP8"),
        KernelOpType.DEQUANTIZE: KernelSupport(SupportLevel.NATIVE),
        # Custom
        KernelOpType.CUSTOM: KernelSupport(SupportLevel.FALLBACK, "Via NKI"),
    },
    # =========================================================================
    # FPGA - Limited support
    # =========================================================================
    DeviceType.FPGA: {
        # Basic
        KernelOpType.INPUT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.OUTPUT: KernelSupport(SupportLevel.NATIVE),
        # Matrix
        KernelOpType.MATMUL: KernelSupport(SupportLevel.NATIVE, "Custom IP"),
        KernelOpType.BATCHED_MATMUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GEMM: KernelSupport(SupportLevel.NATIVE),
        # Elementwise unary
        KernelOpType.EXP: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.LOG: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQRT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SQUARE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SILU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SIGMOID: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.GELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RELU: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TANH: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.CLAMP: KernelSupport(SupportLevel.NATIVE),
        # Elementwise binary
        KernelOpType.ADD: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SUB: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MUL: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.DIV: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.POW: KernelSupport(SupportLevel.FALLBACK),
        # Reductions
        KernelOpType.SUM: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MEAN: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MAX: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.MIN: KernelSupport(SupportLevel.NATIVE),
        # Normalization
        KernelOpType.RMS_NORM: KernelSupport(SupportLevel.FALLBACK, "Custom IP needed"),
        KernelOpType.LAYER_NORM: KernelSupport(SupportLevel.FALLBACK),
        KernelOpType.BATCH_NORM: KernelSupport(SupportLevel.FALLBACK),
        # Attention
        KernelOpType.ATTENTION: KernelSupport(SupportLevel.FALLBACK, "Custom IP"),
        KernelOpType.FLASH_ATTENTION: KernelSupport(SupportLevel.UNSUPPORTED),
        KernelOpType.MULTI_HEAD_ATTENTION: KernelSupport(SupportLevel.FALLBACK),
        KernelOpType.GROUPED_QUERY_ATTENTION: KernelSupport(SupportLevel.UNSUPPORTED),
        # Memory
        KernelOpType.CONCAT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.SPLIT: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.TRANSPOSE: KernelSupport(SupportLevel.NATIVE),
        KernelOpType.RESHAPE: KernelSupport(SupportLevel.NATIVE),
        # Communication
        KernelOpType.ALL_REDUCE: KernelSupport(SupportLevel.FALLBACK, "PCIe"),
        KernelOpType.ALL_GATHER: KernelSupport(SupportLevel.FALLBACK, "PCIe"),
        KernelOpType.REDUCE_SCATTER: KernelSupport(SupportLevel.FALLBACK, "PCIe"),
        KernelOpType.ALL_TO_ALL: KernelSupport(SupportLevel.FALLBACK, "PCIe"),
        # Quantization
        KernelOpType.QUANTIZE: KernelSupport(SupportLevel.NATIVE, "INT4/INT8"),
        KernelOpType.DEQUANTIZE: KernelSupport(SupportLevel.NATIVE),
        # Custom
        KernelOpType.CUSTOM: KernelSupport(SupportLevel.NATIVE, "HLS/RTL"),
    },
}

# Add NPU, Hexagon with similar to ASCEND coverage
KERNEL_COVERAGE[DeviceType.NPU] = KERNEL_COVERAGE[DeviceType.ASCEND].copy()
KERNEL_COVERAGE[DeviceType.HEXAGON] = {
    op: KernelSupport(
        (
            SupportLevel.NATIVE
            if support.level in [SupportLevel.NATIVE, SupportLevel.TRITON]
            else support.level
        ),
        f"Hexagon DSP: {support.notes}",
    )
    for op, support in KERNEL_COVERAGE[DeviceType.CPU].items()
}


class KernelCoverageAnalyzer:
    """Analyzes kernel coverage across hardware backends."""

    @classmethod
    def get_support(cls, device_type: DeviceType, op: KernelOpType) -> KernelSupport:
        """Get support level for an operation on a device type."""
        if device_type not in KERNEL_COVERAGE:
            return KernelSupport(SupportLevel.UNSUPPORTED, "Device not supported")

        coverage = KERNEL_COVERAGE[device_type]
        if op not in coverage:
            return KernelSupport(SupportLevel.UNSUPPORTED, "Operation not supported")

        return coverage[op]

    @classmethod
    def get_coverage_matrix(cls) -> Dict[str, Dict[str, str]]:
        """Generate a coverage matrix for all devices and operations."""
        matrix = {}

        for device_type in DeviceType:
            if device_type not in KERNEL_COVERAGE:
                continue

            device_coverage = {}
            for op in KernelOpType:
                support = cls.get_support(device_type, op)
                device_coverage[op.value] = str(support)

            matrix[device_type.value] = device_coverage

        return matrix

    @classmethod
    def get_gaps(cls, device_type: DeviceType) -> List[KernelOpType]:
        """Get operations that are not fully supported on a device."""
        gaps = []

        if device_type not in KERNEL_COVERAGE:
            return list(KernelOpType)

        coverage = KERNEL_COVERAGE[device_type]
        for op in KernelOpType:
            if op not in coverage:
                gaps.append(op)
            elif coverage[op].level in [SupportLevel.UNSUPPORTED, SupportLevel.FALLBACK]:
                gaps.append(op)

        return gaps

    @classmethod
    def get_native_ops(cls, device_type: DeviceType) -> List[KernelOpType]:
        """Get operations with native support on a device."""
        native = []

        if device_type not in KERNEL_COVERAGE:
            return []

        coverage = KERNEL_COVERAGE[device_type]
        for op, support in coverage.items():
            if support.level == SupportLevel.NATIVE:
                native.append(op)

        return native

    @classmethod
    def print_coverage_table(cls):
        """Print a formatted coverage table."""
        print("\n" + "=" * 100)
        print("KERNEL COVERAGE MATRIX")
        print("=" * 100)
        print("\nLegend: ✅=Native  🔷=Triton  ⚠️=Fallback  🔄=Emulated  ❌=Unsupported\n")

        # Header
        devices = [d for d in DeviceType if d in KERNEL_COVERAGE]
        header = f"{'Operation':<25}"
        for d in devices[:8]:  # Limit to first 8 for readability
            header += f"{d.value:^8}"
        print(header)
        print("-" * len(header))

        # Rows
        for op in KernelOpType:
            row = f"{op.value:<25}"
            for d in devices[:8]:
                support = cls.get_support(d, op)
                row += f"{str(support):^8}"
            print(row)

        print("\n")

    @classmethod
    def get_summary(cls) -> Dict[str, Dict[str, int]]:
        """Get summary statistics for each device."""
        summary = {}

        for device_type in DeviceType:
            if device_type not in KERNEL_COVERAGE:
                continue

            stats = {
                "native": 0,
                "triton": 0,
                "fallback": 0,
                "emulated": 0,
                "unsupported": 0,
                "total": len(KernelOpType),
            }

            coverage = KERNEL_COVERAGE[device_type]
            for op in KernelOpType:
                if op not in coverage:
                    stats["unsupported"] += 1
                else:
                    level = coverage[op].level
                    if level == SupportLevel.NATIVE:
                        stats["native"] += 1
                    elif level == SupportLevel.TRITON:
                        stats["triton"] += 1
                    elif level == SupportLevel.FALLBACK:
                        stats["fallback"] += 1
                    elif level == SupportLevel.EMULATED:
                        stats["emulated"] += 1
                    else:
                        stats["unsupported"] += 1

            summary[device_type.value] = stats

        return summary


def check_kernel_support(
    device_type: DeviceType, ops: List[KernelOpType]
) -> Dict[KernelOpType, KernelSupport]:
    """Check support for a list of operations on a device."""
    return {op: KernelCoverageAnalyzer.get_support(device_type, op) for op in ops}


def get_best_device_for_ops(
    ops: List[KernelOpType], available_devices: List[DeviceType]
) -> DeviceType:
    """Find the best device for a set of operations."""
    best_device = None
    best_score = -1

    for device in available_devices:
        score = 0
        for op in ops:
            support = KernelCoverageAnalyzer.get_support(device, op)
            if support.level == SupportLevel.NATIVE:
                score += 3
            elif support.level == SupportLevel.TRITON:
                score += 2
            elif support.level == SupportLevel.FALLBACK:
                score += 1
            elif support.level == SupportLevel.EMULATED:
                score += 0.5

        if score > best_score:
            best_score = score
            best_device = device

    return best_device
