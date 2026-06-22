/* Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Universal Kernel Generator Implementation
 */

#include "kernel/universal/universal_kernel.h"
#include <cmath>
#include <sstream>

namespace yirage {
namespace kernel {
namespace universal {

// ============================================================================
// Kernel Template Registry
// ============================================================================

std::unordered_map<int, KernelTemplateRegistry::TemplateFunc>&
KernelTemplateRegistry::get_registry() {
  static std::unordered_map<int, TemplateFunc> registry;
  return registry;
}

int KernelTemplateRegistry::make_key(KernelOp op, HardwareTarget target) {
  return static_cast<int>(op) * 100 + static_cast<int>(target);
}

void KernelTemplateRegistry::register_template(KernelOp op,
                                               HardwareTarget target,
                                               TemplateFunc func) {
  get_registry()[make_key(op, target)] = func;
}

KernelTemplateRegistry::TemplateFunc
KernelTemplateRegistry::get_template(KernelOp op, HardwareTarget target) {
  auto& registry = get_registry();
  int key = make_key(op, target);
  auto it = registry.find(key);
  if (it != registry.end()) {
    return it->second;
  }
  return nullptr;
}

bool KernelTemplateRegistry::has_template(KernelOp op, HardwareTarget target) {
  return get_template(op, target) != nullptr;
}

std::vector<std::pair<KernelOp, HardwareTarget>>
KernelTemplateRegistry::list_templates() {
  std::vector<std::pair<KernelOp, HardwareTarget>> result;
  auto& registry = get_registry();
  for (auto const& [key, _] : registry) {
    KernelOp op = static_cast<KernelOp>(key / 100);
    HardwareTarget target = static_cast<HardwareTarget>(key % 100);
    result.push_back({op, target});
  }
  return result;
}

// ============================================================================
// Universal Kernel Generator
// ============================================================================

bool UniversalKernelGenerator::is_supported(KernelOp op, HardwareTarget target) {
  // Check if we have a native template
  if (KernelTemplateRegistry::has_template(op, target)) {
    return true;
  }
  
  // Check if Triton can handle it
  if (target != HardwareTarget::TRITON &&
      KernelTemplateRegistry::has_template(op, HardwareTarget::TRITON)) {
    // Triton supports CUDA, ROCm, XPU, and can be extended
    if (target == HardwareTarget::CUDA ||
        target == HardwareTarget::ROCM ||
        target == HardwareTarget::XPU ||
        target == HardwareTarget::ASCEND) {  // via BiSheng
      return true;
    }
  }
  
  return false;
}

std::string UniversalKernelGenerator::get_support_level(KernelOp op,
                                                        HardwareTarget target) {
  if (KernelTemplateRegistry::has_template(op, target)) {
    return "native";
  }
  if (KernelTemplateRegistry::has_template(op, HardwareTarget::TRITON)) {
    if (target == HardwareTarget::CUDA ||
        target == HardwareTarget::ROCM ||
        target == HardwareTarget::XPU) {
      return "triton";
    }
    if (target == HardwareTarget::ASCEND) {
      return "triton_bisheng";
    }
  }
  
  // Check for fallback implementations
  switch (target) {
    case HardwareTarget::CPU:
      return "fallback";  // CPU always has fallback
    case HardwareTarget::MPS:
      // MPS has limited support
      if (op == KernelOp::FLASH_ATTENTION ||
          op == KernelOp::ALL_REDUCE ||
          op == KernelOp::ALL_GATHER) {
        return "unsupported";
      }
      return "fallback";
    case HardwareTarget::FPGA:
      // FPGA has limited support for complex ops
      if (op == KernelOp::FLASH_ATTENTION ||
          op == KernelOp::GROUPED_QUERY_ATTENTION) {
        return "unsupported";
      }
      return "fallback";
    default:
      return "unsupported";
  }
}

HardwareTarget UniversalKernelGenerator::get_best_target(
    KernelSpec const& spec,
    std::vector<HardwareTarget> available) {
  
  // Priority order for different operations
  std::vector<HardwareTarget> priority;
  
  switch (spec.op) {
    case KernelOp::FLASH_ATTENTION:
    case KernelOp::GROUPED_QUERY_ATTENTION:
      priority = {HardwareTarget::CUDA, HardwareTarget::ROCM,
                  HardwareTarget::TPU, HardwareTarget::NEURON,
                  HardwareTarget::ASCEND};
      break;
    
    case KernelOp::MATMUL:
    case KernelOp::GEMM:
      // All accelerators are good at GEMM
      priority = {HardwareTarget::CUDA, HardwareTarget::TPU,
                  HardwareTarget::ROCM, HardwareTarget::XPU,
                  HardwareTarget::ASCEND, HardwareTarget::MACA,
                  HardwareTarget::NEURON, HardwareTarget::FPGA};
      break;
    
    case KernelOp::QUANTIZE:
    case KernelOp::DEQUANTIZE:
      // FPGA excels at quantization
      priority = {HardwareTarget::FPGA, HardwareTarget::CUDA,
                  HardwareTarget::TPU, HardwareTarget::NEURON};
      break;
    
    default:
      priority = {HardwareTarget::CUDA, HardwareTarget::ROCM,
                  HardwareTarget::XPU, HardwareTarget::TPU,
                  HardwareTarget::ASCEND, HardwareTarget::MACA,
                  HardwareTarget::NEURON, HardwareTarget::MPS,
                  HardwareTarget::FPGA, HardwareTarget::CPU};
  }
  
  // Find first available target from priority list
  for (auto target : priority) {
    for (auto avail : available) {
      if (target == avail && is_supported(spec.op, target)) {
        return target;
      }
    }
  }
  
  // Fallback to CPU
  return HardwareTarget::CPU;
}

GeneratedKernel UniversalKernelGenerator::generate(KernelSpec const& spec,
                                                   HardwareTarget target) {
  switch (target) {
    case HardwareTarget::CUDA:
      return generate_cuda(spec);
    case HardwareTarget::ROCM:
      return generate_rocm(spec);
    case HardwareTarget::MPS:
      return generate_mps(spec);
    case HardwareTarget::ASCEND:
      return generate_ascend(spec);
    case HardwareTarget::MACA:
      return generate_maca(spec);
    case HardwareTarget::XPU:
      return generate_xpu(spec);
    case HardwareTarget::TPU:
      return generate_tpu(spec);
    case HardwareTarget::NEURON:
      return generate_neuron(spec);
    case HardwareTarget::FPGA:
      return generate_fpga(spec);
    case HardwareTarget::CPU:
      return generate_cpu(spec);
    case HardwareTarget::TRITON:
      return generate_triton(spec);
    case HardwareTarget::AUTO:
      // Auto-detect best target
      return generate(spec, get_best_target(spec, {
        HardwareTarget::CUDA, HardwareTarget::ROCM,
        HardwareTarget::CPU
      }));
    default:
      return generate_cpu(spec);  // Fallback to CPU
  }
}

std::vector<GeneratedKernel>
UniversalKernelGenerator::generate_all(KernelSpec const& spec) {
  std::vector<GeneratedKernel> results;
  
  std::vector<HardwareTarget> targets = {
    HardwareTarget::CUDA, HardwareTarget::ROCM, HardwareTarget::MPS,
    HardwareTarget::ASCEND, HardwareTarget::MACA, HardwareTarget::XPU,
    HardwareTarget::TPU, HardwareTarget::NEURON, HardwareTarget::FPGA,
    HardwareTarget::CPU, HardwareTarget::TRITON
  };
  
  for (auto target : targets) {
    if (is_supported(spec.op, target)) {
      results.push_back(generate(spec, target));
    }
  }
  
  return results;
}

// ============================================================================
// Backend-specific generators (stubs - to be implemented)
// ============================================================================

GeneratedKernel UniversalKernelGenerator::generate_cuda(KernelSpec const& spec) {
  GeneratedKernel kernel;
  kernel.target = HardwareTarget::CUDA;
  kernel.kernel_name = "kernel_cuda_" + std::to_string(static_cast<int>(spec.op));
  kernel.compile_flags = {"-std=c++17", "-O3", "--use_fast_math"};
  
  // Generate CUDA kernel based on operation type
  std::ostringstream ss;
  ss << "// CUDA kernel for " << static_cast<int>(spec.op) << "\n";
  ss << "#include <cuda_runtime.h>\n";
  ss << "#include <cuda_fp16.h>\n\n";
  
  switch (spec.op) {
    case KernelOp::MATMUL:
      ss << "// Use cuBLAS or custom Tensor Core kernel\n";
      ss << "__global__ void " << kernel.kernel_name << "(\n";
      ss << "    const half* A, const half* B, half* C,\n";
      ss << "    int M, int N, int K) {\n";
      ss << "    // Tensor Core WMMA implementation\n";
      ss << "}\n";
      break;
    
    case KernelOp::RMS_NORM:
      ss << "__global__ void " << kernel.kernel_name << "(\n";
      ss << "    const half* input, half* output, const half* weight,\n";
      ss << "    int hidden_size, float eps) {\n";
      ss << "    // Fused RMSNorm kernel\n";
      ss << "}\n";
      break;
    
    default:
      ss << "// Generic kernel placeholder\n";
      ss << "__global__ void " << kernel.kernel_name << "() {}\n";
  }
  
  kernel.source_code = ss.str();
  return kernel;
}

GeneratedKernel UniversalKernelGenerator::generate_rocm(KernelSpec const& spec) {
  GeneratedKernel kernel;
  kernel.target = HardwareTarget::ROCM;
  kernel.kernel_name = "kernel_rocm_" + std::to_string(static_cast<int>(spec.op));
  kernel.compile_flags = {"-std=c++17", "-O3", "-ffast-math"};
  
  std::ostringstream ss;
  ss << "// ROCm/HIP kernel for " << static_cast<int>(spec.op) << "\n";
  ss << "#include <hip/hip_runtime.h>\n";
  ss << "#include <hip/hip_fp16.h>\n\n";
  
  switch (spec.op) {
    case KernelOp::MATMUL:
      ss << "// Use rocBLAS or custom Matrix Core kernel\n";
      ss << "__global__ void " << kernel.kernel_name << "(\n";
      ss << "    const __half* A, const __half* B, __half* C,\n";
      ss << "    int M, int N, int K) {\n";
      ss << "    // MFMA (Matrix FMA) implementation\n";
      ss << "}\n";
      break;
    
    default:
      ss << "// Generic HIP kernel placeholder\n";
      ss << "__global__ void " << kernel.kernel_name << "() {}\n";
  }
  
  kernel.source_code = ss.str();
  return kernel;
}

GeneratedKernel UniversalKernelGenerator::generate_mps(KernelSpec const& spec) {
  GeneratedKernel kernel;
  kernel.target = HardwareTarget::MPS;
  kernel.kernel_name = "kernel_mps_" + std::to_string(static_cast<int>(spec.op));
  
  std::ostringstream ss;
  ss << "// Metal kernel for " << static_cast<int>(spec.op) << "\n";
  ss << "#include <metal_stdlib>\n";
  ss << "using namespace metal;\n\n";
  
  switch (spec.op) {
    case KernelOp::MATMUL:
      ss << "kernel void " << kernel.kernel_name << "(\n";
      ss << "    device const half* A [[buffer(0)]],\n";
      ss << "    device const half* B [[buffer(1)]],\n";
      ss << "    device half* C [[buffer(2)]],\n";
      ss << "    constant int& M [[buffer(3)]],\n";
      ss << "    constant int& N [[buffer(4)]],\n";
      ss << "    constant int& K [[buffer(5)]],\n";
      ss << "    uint2 gid [[thread_position_in_grid]]) {\n";
      ss << "    // Tiled matmul using threadgroup memory\n";
      ss << "}\n";
      break;
    
    default:
      ss << "kernel void " << kernel.kernel_name << "() {}\n";
  }
  
  kernel.source_code = ss.str();
  return kernel;
}

GeneratedKernel UniversalKernelGenerator::generate_ascend(KernelSpec const& spec) {
  GeneratedKernel kernel;
  kernel.target = HardwareTarget::ASCEND;
  kernel.kernel_name = "kernel_ascend_" + std::to_string(static_cast<int>(spec.op));
  
  std::ostringstream ss;
  ss << "// Ascend C kernel for " << static_cast<int>(spec.op) << "\n";
  ss << "#include \"kernel_operator.h\"\n\n";
  ss << "using namespace AscendC;\n\n";
  
  switch (spec.op) {
    case KernelOp::MATMUL:
      ss << "__aicore__ void " << kernel.kernel_name << "(\n";
      ss << "    __gm__ half* A, __gm__ half* B, __gm__ half* C,\n";
      ss << "    int M, int N, int K) {\n";
      ss << "    // Cube Unit implementation\n";
      ss << "}\n";
      break;
    
    default:
      ss << "__aicore__ void " << kernel.kernel_name << "() {}\n";
  }
  
  kernel.source_code = ss.str();
  return kernel;
}

GeneratedKernel UniversalKernelGenerator::generate_maca(KernelSpec const& spec) {
  // MACA is CUDA-compatible, reuse CUDA kernel with minor modifications
  GeneratedKernel kernel = generate_cuda(spec);
  kernel.target = HardwareTarget::MACA;
  kernel.kernel_name = "kernel_maca_" + std::to_string(static_cast<int>(spec.op));
  kernel.compile_flags = {"-std=c++17", "-O3"};
  
  // Replace CUDA includes with MACA equivalents
  std::string& code = kernel.source_code;
  size_t pos = code.find("cuda_runtime.h");
  if (pos != std::string::npos) {
    code.replace(pos, 14, "maca_runtime.h");
  }
  
  return kernel;
}

GeneratedKernel UniversalKernelGenerator::generate_xpu(KernelSpec const& spec) {
  GeneratedKernel kernel;
  kernel.target = HardwareTarget::XPU;
  kernel.kernel_name = "kernel_xpu_" + std::to_string(static_cast<int>(spec.op));
  
  std::ostringstream ss;
  ss << "// SYCL kernel for Intel XPU\n";
  ss << "#include <sycl/sycl.hpp>\n";
  ss << "#include <sycl/ext/intel/esimd.hpp>\n\n";
  
  switch (spec.op) {
    case KernelOp::MATMUL:
      ss << "void " << kernel.kernel_name << "(sycl::queue& q,\n";
      ss << "    sycl::half* A, sycl::half* B, sycl::half* C,\n";
      ss << "    int M, int N, int K) {\n";
      ss << "    // XMX (Xe Matrix eXtensions) implementation\n";
      ss << "    q.submit([&](sycl::handler& h) {\n";
      ss << "        h.parallel_for(sycl::range<2>(M, N), [=](sycl::item<2> item) {\n";
      ss << "            // DPAS (Dot Product Accumulate Systolic)\n";
      ss << "        });\n";
      ss << "    });\n";
      ss << "}\n";
      break;
    
    default:
      ss << "void " << kernel.kernel_name << "(sycl::queue& q) {}\n";
  }
  
  kernel.source_code = ss.str();
  return kernel;
}

GeneratedKernel UniversalKernelGenerator::generate_tpu(KernelSpec const& spec) {
  GeneratedKernel kernel;
  kernel.target = HardwareTarget::TPU;
  kernel.kernel_name = "kernel_tpu_" + std::to_string(static_cast<int>(spec.op));
  
  std::ostringstream ss;
  ss << "# Pallas/JAX kernel for TPU\n";
  ss << "import jax\n";
  ss << "import jax.numpy as jnp\n";
  ss << "from jax.experimental import pallas as pl\n\n";
  
  switch (spec.op) {
    case KernelOp::MATMUL:
      ss << "def " << kernel.kernel_name << "(a_ref, b_ref, c_ref):\n";
      ss << "    # MXU (Matrix Multiply Unit) kernel\n";
      ss << "    c_ref[...] = pl.dot(a_ref[...], b_ref[...])\n\n";
      ss << "@jax.jit\n";
      ss << "def matmul_tpu(A, B):\n";
      ss << "    return pl.pallas_call(\n";
      ss << "        " << kernel.kernel_name << ",\n";
      ss << "        out_shape=jax.ShapeDtypeStruct(A.shape[:-1] + B.shape[-1:], A.dtype),\n";
      ss << "        grid=(1,)\n";
      ss << "    )(A, B)\n";
      break;
    
    default:
      ss << "def " << kernel.kernel_name << "():\n";
      ss << "    pass\n";
  }
  
  kernel.source_code = ss.str();
  return kernel;
}

GeneratedKernel UniversalKernelGenerator::generate_neuron(KernelSpec const& spec) {
  GeneratedKernel kernel;
  kernel.target = HardwareTarget::NEURON;
  kernel.kernel_name = "kernel_neuron_" + std::to_string(static_cast<int>(spec.op));
  
  std::ostringstream ss;
  ss << "# NKI (Neuron Kernel Interface) for AWS Trainium/Inferentia\n";
  ss << "import neuronxcc.nki as nki\n";
  ss << "import neuronxcc.nki.language as nl\n\n";
  
  switch (spec.op) {
    case KernelOp::MATMUL:
      ss << "@nki.jit\n";
      ss << "def " << kernel.kernel_name << "(A, B):\n";
      ss << "    # NeuronCore tensor engine\n";
      ss << "    return nl.matmul(A, B)\n";
      break;
    
    case KernelOp::FLASH_ATTENTION:
      ss << "@nki.jit\n";
      ss << "def " << kernel.kernel_name << "(Q, K, V):\n";
      ss << "    # NKI FlashAttention implementation\n";
      ss << "    return nki.kernels.flash_attention(Q, K, V)\n";
      break;
    
    default:
      ss << "@nki.jit\n";
      ss << "def " << kernel.kernel_name << "():\n";
      ss << "    pass\n";
  }
  
  kernel.source_code = ss.str();
  return kernel;
}

GeneratedKernel UniversalKernelGenerator::generate_fpga(KernelSpec const& spec) {
  GeneratedKernel kernel;
  kernel.target = HardwareTarget::FPGA;
  kernel.kernel_name = "kernel_fpga_" + std::to_string(static_cast<int>(spec.op));
  
  std::ostringstream ss;
  ss << "// Vitis HLS kernel for FPGA\n";
  ss << "#include \"ap_int.h\"\n";
  ss << "#include \"hls_stream.h\"\n";
  ss << "#include \"hls_half.h\"\n\n";
  
  switch (spec.op) {
    case KernelOp::MATMUL:
      ss << "void " << kernel.kernel_name << "(\n";
      ss << "    half* A, half* B, half* C,\n";
      ss << "    int M, int N, int K) {\n";
      ss << "    #pragma HLS INTERFACE m_axi port=A bundle=gmem0\n";
      ss << "    #pragma HLS INTERFACE m_axi port=B bundle=gmem1\n";
      ss << "    #pragma HLS INTERFACE m_axi port=C bundle=gmem2\n";
      ss << "    #pragma HLS DATAFLOW\n\n";
      ss << "    // Systolic array implementation\n";
      ss << "    for (int i = 0; i < M; i++) {\n";
      ss << "        #pragma HLS PIPELINE II=1\n";
      ss << "        for (int j = 0; j < N; j++) {\n";
      ss << "            half sum = 0;\n";
      ss << "            for (int k = 0; k < K; k++) {\n";
      ss << "                sum += A[i*K+k] * B[k*N+j];\n";
      ss << "            }\n";
      ss << "            C[i*N+j] = sum;\n";
      ss << "        }\n";
      ss << "    }\n";
      ss << "}\n";
      break;
    
    default:
      ss << "void " << kernel.kernel_name << "() {\n";
      ss << "    #pragma HLS INTERFACE ap_ctrl_none port=return\n";
      ss << "}\n";
  }
  
  kernel.source_code = ss.str();
  return kernel;
}

GeneratedKernel UniversalKernelGenerator::generate_cpu(KernelSpec const& spec) {
  GeneratedKernel kernel;
  kernel.target = HardwareTarget::CPU;
  kernel.kernel_name = "kernel_cpu_" + std::to_string(static_cast<int>(spec.op));
  kernel.compile_flags = {"-O3", "-mavx512f", "-mavx512bw", "-fopenmp"};
  
  std::ostringstream ss;
  ss << "// CPU kernel with AVX-512 and OpenMP\n";
  ss << "#include <immintrin.h>\n";
  ss << "#include <omp.h>\n";
  ss << "#include <cstdint>\n\n";
  
  switch (spec.op) {
    case KernelOp::MATMUL:
      ss << "void " << kernel.kernel_name << "(\n";
      ss << "    const float* A, const float* B, float* C,\n";
      ss << "    int M, int N, int K) {\n";
      ss << "    #pragma omp parallel for collapse(2)\n";
      ss << "    for (int i = 0; i < M; i++) {\n";
      ss << "        for (int j = 0; j < N; j += 16) {\n";
      ss << "            __m512 sum = _mm512_setzero_ps();\n";
      ss << "            for (int k = 0; k < K; k++) {\n";
      ss << "                __m512 a = _mm512_set1_ps(A[i*K+k]);\n";
      ss << "                __m512 b = _mm512_loadu_ps(&B[k*N+j]);\n";
      ss << "                sum = _mm512_fmadd_ps(a, b, sum);\n";
      ss << "            }\n";
      ss << "            _mm512_storeu_ps(&C[i*N+j], sum);\n";
      ss << "        }\n";
      ss << "    }\n";
      ss << "}\n";
      break;
    
    default:
      ss << "void " << kernel.kernel_name << "() {}\n";
  }
  
  kernel.source_code = ss.str();
  return kernel;
}

GeneratedKernel UniversalKernelGenerator::generate_triton(KernelSpec const& spec) {
  GeneratedKernel kernel;
  kernel.target = HardwareTarget::TRITON;
  kernel.kernel_name = "kernel_triton_" + std::to_string(static_cast<int>(spec.op));
  
  std::ostringstream ss;
  ss << "# Triton kernel (cross-platform)\n";
  ss << "import triton\n";
  ss << "import triton.language as tl\n\n";
  
  switch (spec.op) {
    case KernelOp::MATMUL:
      ss << "@triton.jit\n";
      ss << "def " << kernel.kernel_name << "(\n";
      ss << "    A_ptr, B_ptr, C_ptr,\n";
      ss << "    M, N, K,\n";
      ss << "    stride_am, stride_ak, stride_bk, stride_bn,\n";
      ss << "    stride_cm, stride_cn,\n";
      ss << "    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr\n";
      ss << "):\n";
      ss << "    pid_m = tl.program_id(0)\n";
      ss << "    pid_n = tl.program_id(1)\n";
      ss << "    \n";
      ss << "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n";
      ss << "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n";
      ss << "    offs_k = tl.arange(0, BLOCK_K)\n";
      ss << "    \n";
      ss << "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n";
      ss << "    for k in range(0, K, BLOCK_K):\n";
      ss << "        a = tl.load(A_ptr + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak)\n";
      ss << "        b = tl.load(B_ptr + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)\n";
      ss << "        acc += tl.dot(a, b)\n";
      ss << "    \n";
      ss << "    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)\n";
      break;
    
    case KernelOp::RMS_NORM:
      ss << "@triton.jit\n";
      ss << "def " << kernel.kernel_name << "(\n";
      ss << "    X_ptr, W_ptr, Y_ptr,\n";
      ss << "    stride, N,\n";
      ss << "    eps: tl.constexpr,\n";
      ss << "    BLOCK_SIZE: tl.constexpr\n";
      ss << "):\n";
      ss << "    row_idx = tl.program_id(0)\n";
      ss << "    offs = tl.arange(0, BLOCK_SIZE)\n";
      ss << "    mask = offs < N\n";
      ss << "    \n";
      ss << "    x = tl.load(X_ptr + row_idx * stride + offs, mask=mask, other=0.0)\n";
      ss << "    w = tl.load(W_ptr + offs, mask=mask, other=1.0)\n";
      ss << "    \n";
      ss << "    var = tl.sum(x * x, axis=0) / N\n";
      ss << "    rrms = 1.0 / tl.sqrt(var + eps)\n";
      ss << "    y = x * rrms * w\n";
      ss << "    \n";
      ss << "    tl.store(Y_ptr + row_idx * stride + offs, y, mask=mask)\n";
      break;
    
    default:
      ss << "@triton.jit\n";
      ss << "def " << kernel.kernel_name << "():\n";
      ss << "    pass\n";
  }
  
  kernel.source_code = ss.str();
  return kernel;
}

// ============================================================================
// Performance Estimator
// ============================================================================

void KernelPerformanceEstimator::estimate(KernelSpec const& spec,
                                          HardwareTarget target,
                                          float& latency_us,
                                          float& throughput_tflops,
                                          float& memory_bw_gbps) {
  // Calculate FLOPs for the operation
  int64_t flops = 0;
  int64_t bytes = 0;
  
  switch (spec.op) {
    case KernelOp::MATMUL:
    case KernelOp::GEMM: {
      if (spec.inputs.size() >= 2) {
        int64_t M = spec.inputs[0].dims[0];
        int64_t K = spec.inputs[0].dims.size() > 1 ? spec.inputs[0].dims[1] : 1;
        int64_t N = spec.inputs[1].dims.size() > 1 ? spec.inputs[1].dims[1] : 1;
        flops = 2 * M * N * K;  // mul + add
        bytes = (M * K + K * N + M * N) * 2;  // FP16
      }
      break;
    }
    
    case KernelOp::RMS_NORM:
    case KernelOp::LAYER_NORM: {
      if (!spec.inputs.empty()) {
        int64_t numel = spec.inputs[0].numel();
        flops = numel * 5;  // square, sum, sqrt, div, mul
        bytes = numel * 4;  // read + write
      }
      break;
    }
    
    default:
      if (!spec.inputs.empty()) {
        flops = spec.inputs[0].numel() * 2;
        bytes = spec.inputs[0].size_bytes() * 2;
      }
  }
  
  // Estimate based on target hardware
  float peak_tflops = 100.0;  // Default
  float peak_bw_gbps = 1000.0;
  
  switch (target) {
    case HardwareTarget::CUDA:
      peak_tflops = 312.0;   // A100
      peak_bw_gbps = 2039.0;
      break;
    case HardwareTarget::ROCM:
      peak_tflops = 383.0;   // MI250X
      peak_bw_gbps = 3276.0;
      break;
    case HardwareTarget::TPU:
      peak_tflops = 275.0;   // v4
      peak_bw_gbps = 1200.0;
      break;
    case HardwareTarget::ASCEND:
      peak_tflops = 320.0;   // 910B
      peak_bw_gbps = 1200.0;
      break;
    case HardwareTarget::XPU:
      peak_tflops = 839.0;   // Max 1550
      peak_bw_gbps = 3276.0;
      break;
    case HardwareTarget::MPS:
      peak_tflops = 27.0;    // M2 Ultra
      peak_bw_gbps = 800.0;
      break;
    case HardwareTarget::CPU:
      peak_tflops = 2.0;
      peak_bw_gbps = 300.0;
      break;
    default:
      break;
  }
  
  // Roofline model
  float ai = static_cast<float>(flops) / bytes;  // Arithmetic intensity
  float compute_bound = peak_tflops * 1e12;
  float memory_bound = peak_bw_gbps * 1e9 * ai;
  
  float effective_throughput = std::min(compute_bound, memory_bound);
  float efficiency = 0.7;  // Assume 70% efficiency
  
  throughput_tflops = (effective_throughput * efficiency) / 1e12;
  latency_us = flops / (throughput_tflops * 1e12) * 1e6;
  memory_bw_gbps = bytes / (latency_us * 1e-6) / 1e9;
}

void KernelPerformanceEstimator::get_roofline_bounds(
    KernelSpec const& spec,
    HardwareTarget target,
    float& compute_bound_tflops,
    float& memory_bound_tflops) {
  
  float ai = estimate_arithmetic_intensity(spec);
  
  // Get hardware specs
  float peak_tflops = 100.0;
  float peak_bw_gbps = 1000.0;
  
  switch (target) {
    case HardwareTarget::CUDA:
      peak_tflops = 312.0;
      peak_bw_gbps = 2039.0;
      break;
    case HardwareTarget::ROCM:
      peak_tflops = 383.0;
      peak_bw_gbps = 3276.0;
      break;
    default:
      break;
  }
  
  compute_bound_tflops = peak_tflops;
  memory_bound_tflops = peak_bw_gbps * ai / 1000.0;  // Convert to TFLOPS
}

float KernelPerformanceEstimator::estimate_arithmetic_intensity(
    KernelSpec const& spec) {
  
  int64_t flops = 0;
  int64_t bytes = 0;
  
  switch (spec.op) {
    case KernelOp::MATMUL: {
      if (spec.inputs.size() >= 2) {
        int64_t M = spec.inputs[0].dims[0];
        int64_t K = spec.inputs[0].dims.size() > 1 ? spec.inputs[0].dims[1] : 1;
        int64_t N = spec.inputs[1].dims.size() > 1 ? spec.inputs[1].dims[1] : 1;
        flops = 2 * M * N * K;
        bytes = (M * K + K * N + M * N) * 2;
      }
      break;
    }
    default:
      if (!spec.inputs.empty()) {
        flops = spec.inputs[0].numel() * 2;
        bytes = spec.inputs[0].size_bytes() * 2;
      }
  }
  
  return bytes > 0 ? static_cast<float>(flops) / bytes : 0.0f;
}

} // namespace universal
} // namespace kernel
} // namespace yirage
