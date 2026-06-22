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
 * Universal Kernel Template System
 * 
 * Provides unified kernel generation for all hardware backends:
 * - CUDA (NVIDIA)
 * - ROCm (AMD)
 * - MPS (Apple)
 * - Ascend (Huawei)
 * - MACA (MetaX)
 * - XPU (Intel)
 * - TPU (Google)
 * - NKI (AWS Neuron)
 * - FPGA
 * - CPU
 */

#pragma once

#include "kernel/common/kernel_interface.h"
#include "type.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace yirage {
namespace kernel {
namespace universal {

/**
 * @brief Supported hardware targets
 */
enum class HardwareTarget {
  CUDA,
  ROCM,
  MPS,
  ASCEND,
  MACA,
  XPU,
  TPU,
  NEURON,
  FPGA,
  CPU,
  TRITON,   // Cross-platform via Triton
  AUTO      // Auto-detect best target
};

/**
 * @brief Kernel operation types
 */
enum class KernelOp {
  // Matrix operations
  MATMUL,
  BATCHED_MATMUL,
  GEMM,
  GEMV,
  
  // Elementwise unary
  EXP,
  LOG,
  SQRT,
  SQUARE,
  SILU,
  SIGMOID,
  GELU,
  RELU,
  TANH,
  SOFTMAX,
  
  // Elementwise binary
  ADD,
  SUB,
  MUL,
  DIV,
  POW,
  
  // Reductions
  SUM,
  MEAN,
  MAX,
  MIN,
  
  // Normalization
  RMS_NORM,
  LAYER_NORM,
  BATCH_NORM,
  
  // Attention
  ATTENTION,
  FLASH_ATTENTION,
  MULTI_HEAD_ATTENTION,
  GROUPED_QUERY_ATTENTION,
  PAGED_ATTENTION,
  
  // Memory
  CONCAT,
  SPLIT,
  TRANSPOSE,
  RESHAPE,
  GATHER,
  SCATTER,
  
  // Quantization
  QUANTIZE,
  DEQUANTIZE,
  
  // Communication
  ALL_REDUCE,
  ALL_GATHER,
  REDUCE_SCATTER,
  ALL_TO_ALL,
  
  // Custom
  CUSTOM
};

/**
 * @brief Data type for kernel operations
 */
enum class KernelDataType {
  FP32,
  FP16,
  BF16,
  FP8,
  INT8,
  INT4,
  UINT8,
  UINT4
};

/**
 * @brief Tensor shape descriptor
 */
struct TensorShape {
  std::vector<int64_t> dims;
  KernelDataType dtype = KernelDataType::FP16;
  bool is_contiguous = true;
  int64_t offset = 0;
  
  int64_t numel() const {
    int64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
  }
  
  size_t size_bytes() const {
    size_t dtype_size = 2;  // Default FP16
    switch (dtype) {
      case KernelDataType::FP32: dtype_size = 4; break;
      case KernelDataType::FP16:
      case KernelDataType::BF16: dtype_size = 2; break;
      case KernelDataType::FP8:
      case KernelDataType::INT8:
      case KernelDataType::UINT8: dtype_size = 1; break;
      case KernelDataType::INT4:
      case KernelDataType::UINT4: dtype_size = 1; break;  // Packed
    }
    return numel() * dtype_size;
  }
};

/**
 * @brief Kernel specification
 */
struct KernelSpec {
  KernelOp op;
  std::vector<TensorShape> inputs;
  std::vector<TensorShape> outputs;
  std::unordered_map<std::string, float> params;  // Op-specific parameters
  
  // Optional constraints
  bool require_tensor_core = false;
  bool require_flash_attention = false;
  int max_shared_memory = 0;  // 0 = no limit
};

/**
 * @brief Generated kernel code
 */
struct GeneratedKernel {
  HardwareTarget target;
  std::string source_code;
  std::string kernel_name;
  std::vector<std::string> compile_flags;
  std::unique_ptr<KernelConfig> config;
  
  // Performance estimates
  float estimated_latency_us = 0;
  float estimated_throughput_tflops = 0;
  float memory_bandwidth_gbps = 0;
};

/**
 * @brief Universal kernel generator
 */
class UniversalKernelGenerator {
public:
  /**
   * @brief Generate kernel for specified target
   */
  static GeneratedKernel generate(KernelSpec const &spec,
                                  HardwareTarget target);
  
  /**
   * @brief Generate kernels for all available targets
   */
  static std::vector<GeneratedKernel> generate_all(KernelSpec const &spec);
  
  /**
   * @brief Get best target for given operation
   */
  static HardwareTarget get_best_target(KernelSpec const &spec,
                                        std::vector<HardwareTarget> available);
  
  /**
   * @brief Check if operation is supported on target
   */
  static bool is_supported(KernelOp op, HardwareTarget target);
  
  /**
   * @brief Get support level (native, triton, fallback, unsupported)
   */
  static std::string get_support_level(KernelOp op, HardwareTarget target);
  
private:
  // Backend-specific generators
  static GeneratedKernel generate_cuda(KernelSpec const &spec);
  static GeneratedKernel generate_rocm(KernelSpec const &spec);
  static GeneratedKernel generate_mps(KernelSpec const &spec);
  static GeneratedKernel generate_ascend(KernelSpec const &spec);
  static GeneratedKernel generate_maca(KernelSpec const &spec);
  static GeneratedKernel generate_xpu(KernelSpec const &spec);
  static GeneratedKernel generate_tpu(KernelSpec const &spec);
  static GeneratedKernel generate_neuron(KernelSpec const &spec);
  static GeneratedKernel generate_fpga(KernelSpec const &spec);
  static GeneratedKernel generate_cpu(KernelSpec const &spec);
  static GeneratedKernel generate_triton(KernelSpec const &spec);
};

/**
 * @brief Kernel template registry
 */
class KernelTemplateRegistry {
public:
  using TemplateFunc = std::function<std::string(KernelSpec const &)>;
  
  /**
   * @brief Register a kernel template
   */
  static void register_template(KernelOp op, HardwareTarget target,
                                TemplateFunc func);
  
  /**
   * @brief Get kernel template
   */
  static TemplateFunc get_template(KernelOp op, HardwareTarget target);
  
  /**
   * @brief Check if template exists
   */
  static bool has_template(KernelOp op, HardwareTarget target);
  
  /**
   * @brief List all registered templates
   */
  static std::vector<std::pair<KernelOp, HardwareTarget>> list_templates();
  
private:
  static std::unordered_map<int, TemplateFunc> &get_registry();
  static int make_key(KernelOp op, HardwareTarget target);
};

/**
 * @brief Kernel performance estimator
 */
class KernelPerformanceEstimator {
public:
  /**
   * @brief Estimate kernel performance
   */
  static void estimate(KernelSpec const &spec, HardwareTarget target,
                       float &latency_us, float &throughput_tflops,
                       float &memory_bw_gbps);
  
  /**
   * @brief Get roofline model bounds
   */
  static void get_roofline_bounds(KernelSpec const &spec,
                                  HardwareTarget target,
                                  float &compute_bound_tflops,
                                  float &memory_bound_tflops);
  
  /**
   * @brief Estimate arithmetic intensity
   */
  static float estimate_arithmetic_intensity(KernelSpec const &spec);
};

/**
 * @brief Macro for registering kernel templates
 */
#define REGISTER_KERNEL_TEMPLATE(op, target, func) \
  namespace { \
    struct KernelTemplateRegistrar_##op##_##target { \
      KernelTemplateRegistrar_##op##_##target() { \
        KernelTemplateRegistry::register_template( \
          KernelOp::op, HardwareTarget::target, func); \
      } \
    }; \
    static KernelTemplateRegistrar_##op##_##target \
      g_registrar_##op##_##target; \
  }

} // namespace universal
} // namespace kernel
} // namespace yirage
