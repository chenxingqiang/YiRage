"""
Universal Kernel Generator

Python interface for generating hardware-specific kernels.
Supports all major accelerator platforms.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .topology import DeviceType
from .kernel_coverage import KernelOpType, SupportLevel, KernelCoverageAnalyzer
from . import kernel_templates


class KernelDataType(Enum):
    """Data types for kernel operations."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    INT4 = "int4"
    UINT8 = "uint8"
    UINT4 = "uint4"


@dataclass
class TensorSpec:
    """Tensor specification for kernel operations."""

    shape: List[int]
    dtype: KernelDataType = KernelDataType.FP16
    is_contiguous: bool = True

    def numel(self) -> int:
        """Get total number of elements."""
        result = 1
        for d in self.shape:
            result *= d
        return result

    def size_bytes(self) -> int:
        """Get size in bytes."""
        dtype_sizes = {
            KernelDataType.FP32: 4,
            KernelDataType.FP16: 2,
            KernelDataType.BF16: 2,
            KernelDataType.FP8: 1,
            KernelDataType.INT8: 1,
            KernelDataType.INT4: 1,  # Packed
            KernelDataType.UINT8: 1,
            KernelDataType.UINT4: 1,
        }
        return self.numel() * dtype_sizes.get(self.dtype, 2)


@dataclass
class KernelSpec:
    """Specification for a kernel operation."""

    op: KernelOpType
    inputs: List[TensorSpec]
    outputs: List[TensorSpec] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    # Constraints
    require_tensor_core: bool = False
    require_flash_attention: bool = False
    max_shared_memory: int = 0


@dataclass
class GeneratedKernel:
    """Generated kernel code and configuration."""

    target: DeviceType
    source_code: str
    kernel_name: str
    compile_flags: List[str] = field(default_factory=list)

    # Performance estimates
    estimated_latency_us: float = 0.0
    estimated_throughput_tflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0


class KernelGenerator:
    """
    Universal kernel generator for all hardware platforms.

    Generates optimized kernel code for:
    - NVIDIA CUDA (Tensor Cores, cuBLAS)
    - AMD ROCm (Matrix Cores, rocBLAS)
    - Intel XPU (XMX, oneMKL)
    - Google TPU (MXU, XLA/Pallas)
    - Huawei Ascend (Cube Unit, CANN)
    - MetaX MACA (mcBLAS)
    - Apple MPS (Metal)
    - AWS Neuron (NKI)
    - FPGA (HLS)
    - CPU (AVX-512, MKL)
    """

    # Template registry
    _templates: Dict[Tuple[KernelOpType, DeviceType], callable] = {}

    @classmethod
    def register_template(cls, op: KernelOpType, target: DeviceType):
        """Decorator to register a kernel template."""

        def decorator(func):
            cls._templates[(op, target)] = func
            return func

        return decorator

    @classmethod
    def generate(cls, spec: KernelSpec, target: DeviceType) -> GeneratedKernel:
        """Generate kernel for specified target."""
        # Check if we have a registered template
        if (spec.op, target) in cls._templates:
            return cls._templates[(spec.op, target)](spec)

        # Check for optimized templates from kernel_templates module
        template = kernel_templates.get_template(spec.op, target)
        if template:
            return cls._generate_from_template(spec, target, template)

        # Fall back to Triton if available
        if (
            spec.op in kernel_templates.TRITON_TEMPLATES
            and target in kernel_templates.TRITON_TARGETS
        ):
            return cls._generate_from_template(
                spec, target, kernel_templates.TRITON_TEMPLATES[spec.op]
            )

        # Generate backend-specific kernel using basic templates
        generators = {
            DeviceType.CUDA: cls._generate_cuda,
            DeviceType.ROCM: cls._generate_rocm,
            DeviceType.MPS: cls._generate_mps,
            DeviceType.ASCEND: cls._generate_ascend,
            DeviceType.MACA: cls._generate_maca,
            DeviceType.XPU: cls._generate_xpu,
            DeviceType.TPU: cls._generate_tpu,
            DeviceType.NEURON: cls._generate_neuron,
            DeviceType.FPGA: cls._generate_fpga,
            DeviceType.CPU: cls._generate_cpu,
        }

        generator = generators.get(target, cls._generate_cpu)
        return generator(spec)

    @classmethod
    def _generate_from_template(
        cls, spec: KernelSpec, target: DeviceType, template: str
    ) -> GeneratedKernel:
        """Generate kernel from a template."""
        kernel_name = f"kernel_{target.value}_{spec.op.value}"

        # Get compile flags based on target
        compile_flags = {
            DeviceType.CUDA: ["-std=c++17", "-O3", "--use_fast_math", "-arch=sm_80"],
            DeviceType.ROCM: ["-std=c++17", "-O3", "-ffast-math", "--offload-arch=gfx90a"],
            DeviceType.CPU: ["-O3", "-mavx512f", "-mavx512bw", "-fopenmp", "-ffast-math"],
            DeviceType.XPU: ["-fsycl", "-O3", "-fp-model=fast"],
        }.get(target, [])

        perf = cls.estimate_performance(spec, target)

        return GeneratedKernel(
            target=target,
            source_code=template,
            kernel_name=kernel_name,
            compile_flags=compile_flags,
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )

    @classmethod
    def generate_all(cls, spec: KernelSpec) -> List[GeneratedKernel]:
        """Generate kernels for all supported targets."""
        results = []
        for target in DeviceType:
            if (
                KernelCoverageAnalyzer.get_support(target, spec.op).level
                != SupportLevel.UNSUPPORTED
            ):
                try:
                    results.append(cls.generate(spec, target))
                except Exception:
                    pass  # Skip if generation fails
        return results

    @classmethod
    def get_best_target(cls, spec: KernelSpec, available: List[DeviceType]) -> DeviceType:
        """Get best target for given operation."""
        # Priority based on operation type
        if spec.op in [KernelOpType.FLASH_ATTENTION, KernelOpType.GROUPED_QUERY_ATTENTION]:
            priority = [
                DeviceType.CUDA,
                DeviceType.ROCM,
                DeviceType.TPU,
                DeviceType.NEURON,
                DeviceType.ASCEND,
            ]
        elif spec.op in [KernelOpType.MATMUL, KernelOpType.GEMM]:
            priority = [
                DeviceType.CUDA,
                DeviceType.TPU,
                DeviceType.ROCM,
                DeviceType.XPU,
                DeviceType.ASCEND,
                DeviceType.MACA,
            ]
        elif spec.op in [KernelOpType.QUANTIZE, KernelOpType.DEQUANTIZE]:
            priority = [DeviceType.FPGA, DeviceType.CUDA, DeviceType.TPU]
        else:
            priority = [
                DeviceType.CUDA,
                DeviceType.ROCM,
                DeviceType.XPU,
                DeviceType.TPU,
                DeviceType.ASCEND,
                DeviceType.MACA,
            ]

        for target in priority:
            if target in available:
                support = KernelCoverageAnalyzer.get_support(target, spec.op)
                if support.level != SupportLevel.UNSUPPORTED:
                    return target

        return DeviceType.CPU

    @classmethod
    def estimate_performance(cls, spec: KernelSpec, target: DeviceType) -> Dict[str, float]:
        """Estimate kernel performance on target."""
        # Calculate FLOPs
        flops = cls._calculate_flops(spec)
        bytes_accessed = cls._calculate_bytes(spec)

        # Hardware peak performance
        hw_specs = {
            DeviceType.CUDA: (312.0, 2039.0),  # A100: TFLOPS, GB/s
            DeviceType.ROCM: (383.0, 3276.0),  # MI250X
            DeviceType.TPU: (275.0, 1200.0),  # v4
            DeviceType.ASCEND: (320.0, 1200.0),  # 910B
            DeviceType.XPU: (839.0, 3276.0),  # Max 1550
            DeviceType.MACA: (200.0, 1600.0),  # C500
            DeviceType.MPS: (27.0, 800.0),  # M2 Ultra
            DeviceType.NEURON: (190.0, 820.0),  # Inferentia2
            DeviceType.FPGA: (2.0, 460.0),  # Alveo U280
            DeviceType.CPU: (2.0, 300.0),  # EPYC
        }

        peak_tflops, peak_bw_gbps = hw_specs.get(target, (10.0, 500.0))

        # Roofline model
        ai = flops / bytes_accessed if bytes_accessed > 0 else 0
        compute_bound = peak_tflops * 1e12
        memory_bound = peak_bw_gbps * 1e9 * ai

        effective_perf = min(compute_bound, memory_bound)
        efficiency = 0.7  # Assume 70% efficiency

        throughput_tflops = (effective_perf * efficiency) / 1e12
        latency_us = (
            flops / (throughput_tflops * 1e12) * 1e6 if throughput_tflops > 0 else float("inf")
        )
        memory_bw = bytes_accessed / (latency_us * 1e-6) / 1e9 if latency_us > 0 else 0

        return {
            "latency_us": latency_us,
            "throughput_tflops": throughput_tflops,
            "memory_bandwidth_gbps": memory_bw,
            "arithmetic_intensity": ai,
            "efficiency": efficiency,
        }

    @classmethod
    def _calculate_flops(cls, spec: KernelSpec) -> int:
        """Calculate FLOPs for operation."""
        if spec.op in [KernelOpType.MATMUL, KernelOpType.GEMM]:
            if len(spec.inputs) >= 2:
                M = spec.inputs[0].shape[0]
                K = spec.inputs[0].shape[1] if len(spec.inputs[0].shape) > 1 else 1
                N = spec.inputs[1].shape[1] if len(spec.inputs[1].shape) > 1 else 1
                return 2 * M * N * K

        elif spec.op in [KernelOpType.RMS_NORM, KernelOpType.LAYER_NORM]:
            if spec.inputs:
                return spec.inputs[0].numel() * 5

        elif spec.op == KernelOpType.ATTENTION:
            if len(spec.inputs) >= 3:
                batch = spec.inputs[0].shape[0]
                seq_len = spec.inputs[0].shape[1] if len(spec.inputs[0].shape) > 1 else 1
                heads = spec.inputs[0].shape[2] if len(spec.inputs[0].shape) > 2 else 1
                head_dim = spec.inputs[0].shape[3] if len(spec.inputs[0].shape) > 3 else 64
                # QK^T + softmax + V
                return batch * heads * (2 * seq_len * seq_len * head_dim + 2 * seq_len * seq_len)

        # Default: assume 2 FLOPs per element
        if spec.inputs:
            return spec.inputs[0].numel() * 2
        return 0

    @classmethod
    def _calculate_bytes(cls, spec: KernelSpec) -> int:
        """Calculate bytes accessed."""
        total = 0
        for inp in spec.inputs:
            total += inp.size_bytes()
        for out in spec.outputs:
            total += out.size_bytes()
        return total if total > 0 else (spec.inputs[0].size_bytes() * 2 if spec.inputs else 0)

    # =========================================================================
    # Backend-specific generators
    # =========================================================================

    @classmethod
    def _generate_cuda(cls, spec: KernelSpec) -> GeneratedKernel:
        """Generate CUDA kernel."""
        kernel_name = f"kernel_cuda_{spec.op.value}"

        code = f"""// CUDA kernel for {spec.op.value}
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void {kernel_name}(
    const half* __restrict__ input,
    half* __restrict__ output,
    int N) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {{
        // Operation-specific implementation
        output[idx] = input[idx];
    }}
}}
"""

        perf = cls.estimate_performance(spec, DeviceType.CUDA)

        return GeneratedKernel(
            target=DeviceType.CUDA,
            source_code=code,
            kernel_name=kernel_name,
            compile_flags=["-std=c++17", "-O3", "--use_fast_math"],
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )

    @classmethod
    def _generate_rocm(cls, spec: KernelSpec) -> GeneratedKernel:
        """Generate ROCm/HIP kernel."""
        kernel_name = f"kernel_rocm_{spec.op.value}"

        code = f"""// ROCm/HIP kernel for {spec.op.value}
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void {kernel_name}(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {{
        output[idx] = input[idx];
    }}
}}
"""

        perf = cls.estimate_performance(spec, DeviceType.ROCM)

        return GeneratedKernel(
            target=DeviceType.ROCM,
            source_code=code,
            kernel_name=kernel_name,
            compile_flags=["-std=c++17", "-O3", "-ffast-math"],
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )

    @classmethod
    def _generate_mps(cls, spec: KernelSpec) -> GeneratedKernel:
        """Generate Metal kernel for Apple MPS."""
        kernel_name = f"kernel_mps_{spec.op.value}"

        code = f"""// Metal kernel for {spec.op.value}
#include <metal_stdlib>
using namespace metal;

kernel void {kernel_name}(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant int& N [[buffer(2)]],
    uint idx [[thread_position_in_grid]]) {{
    if (idx < uint(N)) {{
        output[idx] = input[idx];
    }}
}}
"""

        perf = cls.estimate_performance(spec, DeviceType.MPS)

        return GeneratedKernel(
            target=DeviceType.MPS,
            source_code=code,
            kernel_name=kernel_name,
            compile_flags=[],
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )

    @classmethod
    def _generate_ascend(cls, spec: KernelSpec) -> GeneratedKernel:
        """Generate Ascend C kernel."""
        kernel_name = f"kernel_ascend_{spec.op.value}"

        code = f"""// Ascend C kernel for {spec.op.value}
#include "kernel_operator.h"
using namespace AscendC;

__aicore__ void {kernel_name}(
    __gm__ half* input,
    __gm__ half* output,
    int N) {{
    // Vector unit implementation
    int block_idx = GetBlockIdx();
    int block_num = GetBlockNum();
    int block_size = N / block_num;
    
    LocalTensor<half> local_input;
    LocalTensor<half> local_output;
    
    // Copy from global to local
    DataCopy(local_input, input + block_idx * block_size, block_size);
    
    // Process
    local_output = local_input;
    
    // Copy back to global
    DataCopy(output + block_idx * block_size, local_output, block_size);
}}
"""

        perf = cls.estimate_performance(spec, DeviceType.ASCEND)

        return GeneratedKernel(
            target=DeviceType.ASCEND,
            source_code=code,
            kernel_name=kernel_name,
            compile_flags=[],
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )

    @classmethod
    def _generate_maca(cls, spec: KernelSpec) -> GeneratedKernel:
        """Generate MACA kernel (CUDA-compatible)."""
        kernel = cls._generate_cuda(spec)
        kernel.target = DeviceType.MACA
        kernel.kernel_name = f"kernel_maca_{spec.op.value}"
        kernel.source_code = kernel.source_code.replace("cuda_runtime.h", "maca_runtime.h")

        perf = cls.estimate_performance(spec, DeviceType.MACA)
        kernel.estimated_latency_us = perf["latency_us"]
        kernel.estimated_throughput_tflops = perf["throughput_tflops"]
        kernel.memory_bandwidth_gbps = perf["memory_bandwidth_gbps"]

        return kernel

    @classmethod
    def _generate_xpu(cls, spec: KernelSpec) -> GeneratedKernel:
        """Generate Intel XPU/SYCL kernel."""
        kernel_name = f"kernel_xpu_{spec.op.value}"

        code = f"""// SYCL kernel for Intel XPU {spec.op.value}
#include <sycl/sycl.hpp>

void {kernel_name}(sycl::queue& q,
    sycl::half* input, sycl::half* output, int N) {{
    q.parallel_for(sycl::range<1>(N), [=](sycl::item<1> item) {{
        int idx = item.get_id(0);
        output[idx] = input[idx];
    }}).wait();
}}
"""

        perf = cls.estimate_performance(spec, DeviceType.XPU)

        return GeneratedKernel(
            target=DeviceType.XPU,
            source_code=code,
            kernel_name=kernel_name,
            compile_flags=["-fsycl", "-O3"],
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )

    @classmethod
    def _generate_tpu(cls, spec: KernelSpec) -> GeneratedKernel:
        """Generate TPU kernel (Pallas/JAX)."""
        kernel_name = f"kernel_tpu_{spec.op.value}"

        code = f"""# Pallas kernel for TPU {spec.op.value}
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def {kernel_name}(input_ref, output_ref):
    output_ref[...] = input_ref[...]

@jax.jit
def run_{kernel_name}(input):
    return pl.pallas_call(
        {kernel_name},
        out_shape=jax.ShapeDtypeStruct(input.shape, input.dtype),
        grid=(1,)
    )(input)
"""

        perf = cls.estimate_performance(spec, DeviceType.TPU)

        return GeneratedKernel(
            target=DeviceType.TPU,
            source_code=code,
            kernel_name=kernel_name,
            compile_flags=[],
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )

    @classmethod
    def _generate_neuron(cls, spec: KernelSpec) -> GeneratedKernel:
        """Generate AWS Neuron NKI kernel."""
        kernel_name = f"kernel_neuron_{spec.op.value}"

        code = f"""# NKI kernel for AWS Neuron {spec.op.value}
import neuronxcc.nki as nki
import neuronxcc.nki.language as nl

@nki.jit
def {kernel_name}(input):
    return input
"""

        perf = cls.estimate_performance(spec, DeviceType.NEURON)

        return GeneratedKernel(
            target=DeviceType.NEURON,
            source_code=code,
            kernel_name=kernel_name,
            compile_flags=[],
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )

    @classmethod
    def _generate_fpga(cls, spec: KernelSpec) -> GeneratedKernel:
        """Generate FPGA HLS kernel."""
        kernel_name = f"kernel_fpga_{spec.op.value}"

        code = f"""// Vitis HLS kernel for FPGA {spec.op.value}
#include "ap_int.h"
#include "hls_stream.h"
#include "hls_half.h"

void {kernel_name}(half* input, half* output, int N) {{
    #pragma HLS INTERFACE m_axi port=input bundle=gmem0
    #pragma HLS INTERFACE m_axi port=output bundle=gmem1
    #pragma HLS INTERFACE s_axilite port=N
    #pragma HLS INTERFACE s_axilite port=return
    
    for (int i = 0; i < N; i++) {{
        #pragma HLS PIPELINE II=1
        output[i] = input[i];
    }}
}}
"""

        perf = cls.estimate_performance(spec, DeviceType.FPGA)

        return GeneratedKernel(
            target=DeviceType.FPGA,
            source_code=code,
            kernel_name=kernel_name,
            compile_flags=[],
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )

    @classmethod
    def _generate_cpu(cls, spec: KernelSpec) -> GeneratedKernel:
        """Generate CPU kernel with SIMD."""
        kernel_name = f"kernel_cpu_{spec.op.value}"

        code = f"""// CPU kernel with AVX-512 for {spec.op.value}
#include <immintrin.h>
#include <omp.h>

void {kernel_name}(const float* input, float* output, int N) {{
    #pragma omp parallel for
    for (int i = 0; i < N; i += 16) {{
        __m512 in = _mm512_loadu_ps(&input[i]);
        _mm512_storeu_ps(&output[i], in);
    }}
}}
"""

        perf = cls.estimate_performance(spec, DeviceType.CPU)

        return GeneratedKernel(
            target=DeviceType.CPU,
            source_code=code,
            kernel_name=kernel_name,
            compile_flags=["-O3", "-mavx512f", "-fopenmp"],
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )

    @classmethod
    def _generate_triton(cls, spec: KernelSpec, target: DeviceType) -> GeneratedKernel:
        """Generate Triton kernel (cross-platform)."""
        kernel_name = f"kernel_triton_{spec.op.value}"

        code = f"""# Triton kernel for {spec.op.value}
import triton
import triton.language as tl

@triton.jit
def {kernel_name}(
    input_ptr, output_ptr, N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    x = tl.load(input_ptr + offs, mask=mask)
    tl.store(output_ptr + offs, x, mask=mask)
"""

        perf = cls.estimate_performance(spec, target)

        return GeneratedKernel(
            target=target,
            source_code=code,
            kernel_name=kernel_name,
            compile_flags=[],
            estimated_latency_us=perf["latency_us"],
            estimated_throughput_tflops=perf["throughput_tflops"],
            memory_bandwidth_gbps=perf["memory_bandwidth_gbps"],
        )


def generate_kernel(
    op: KernelOpType, inputs: List[TensorSpec], target: DeviceType
) -> GeneratedKernel:
    """Convenience function to generate a kernel."""
    spec = KernelSpec(op=op, inputs=inputs)
    return KernelGenerator.generate(spec, target)


def generate_kernels_for_all_targets(
    op: KernelOpType, inputs: List[TensorSpec]
) -> List[GeneratedKernel]:
    """Generate kernels for all supported targets."""
    spec = KernelSpec(op=op, inputs=inputs)
    return KernelGenerator.generate_all(spec)
