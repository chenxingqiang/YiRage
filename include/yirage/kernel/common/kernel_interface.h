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
 * This file is part of YiRage (Yi Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */


#pragma once

#include "yirage/type.h"
#include <memory>
#include <string>
#include <vector>

namespace yirage {
namespace kernel {

/**
 * @brief Base kernel configuration applicable to all backends
 */
struct KernelConfig {
  // Grid configuration (varies by backend)
  int grid_dim_x = 1;
  int grid_dim_y = 1;
  int grid_dim_z = 1;

  // Block/workgroup configuration
  int block_dim_x = 1;
  int block_dim_y = 1;
  int block_dim_z = 1;

  // Shared/local memory size (bytes)
  size_t shared_memory_size = 0;

  // Backend type
  type::BackendType backend_type = type::BT_UNKNOWN;

  virtual ~KernelConfig() = default;

  // Get total number of threads
  virtual int get_total_threads() const {
    return block_dim_x * block_dim_y * block_dim_z;
  }

  // Get total number of blocks
  virtual int get_total_blocks() const {
    return grid_dim_x * grid_dim_y * grid_dim_z;
  }
};

/**
 * @brief Performance metrics for kernel execution
 */
struct KernelMetrics {
  float execution_time_ms = 0.0f;    // Execution time in milliseconds
  float memory_bandwidth_gbps = 0.0f; // Memory bandwidth in GB/s
  float compute_throughput_tflops = 0.0f; // Compute throughput in TFLOPS
  float occupancy = 0.0f;            // Resource occupancy (0.0 - 1.0)
  size_t memory_bytes_accessed = 0;  // Total memory accessed
  
  // Efficiency metrics
  float arithmetic_intensity = 0.0f; // FLOPs per byte
  float cache_hit_rate = 0.0f;      // Cache hit rate (0.0 - 1.0)
};

/**
 * @brief Abstract interface for kernel executors
 * 
 * Each backend implements this interface to provide hardware-specific
 * kernel compilation and execution.
 */
class KernelExecutor {
public:
  virtual ~KernelExecutor() = default;

  /**
   * @brief Compile kernel from source code
   * @param source Kernel source code (CUDA, OpenCL, Metal, etc.)
   * @param config Kernel configuration
   * @return true if compilation succeeded, false otherwise
   */
  virtual bool compile(std::string const &source,
                      KernelConfig const &config) = 0;

  /**
   * @brief Execute compiled kernel
   * @param inputs Array of input pointers
   * @param num_inputs Number of inputs
   * @param outputs Array of output pointers
   * @param num_outputs Number of outputs
   * @param config Kernel configuration
   * @return true if execution succeeded, false otherwise
   */
  virtual bool execute(void **inputs, size_t num_inputs, void **outputs,
                      size_t num_outputs, KernelConfig const &config) = 0;

  /**
   * @brief Get execution time of last kernel launch
   * @return Execution time in milliseconds
   */
  virtual float get_execution_time() const = 0;

  /**
   * @brief Get performance metrics of last kernel launch
   * @return KernelMetrics structure
   */
  virtual KernelMetrics get_metrics() const = 0;

  /**
   * @brief Get backend type
   * @return Backend type enum
   */
  virtual type::BackendType get_backend_type() const = 0;

  /**
   * @brief Validate kernel configuration
   * @param config Configuration to validate
   * @return true if valid, false otherwise
   */
  virtual bool validate_config(KernelConfig const &config) const = 0;
};

/**
 * @brief Factory for creating backend-specific kernel executors
 */
class KernelExecutorFactory {
public:
  /**
   * @brief Create a MatMul kernel executor
   * @param backend Backend type
   * @return Unique pointer to kernel executor, or nullptr if backend not supported
   */
  static std::unique_ptr<KernelExecutor>
  create_matmul_executor(type::BackendType backend);

  /**
   * @brief Create a RMSNorm kernel executor
   * @param backend Backend type
   * @return Unique pointer to kernel executor
   */
  static std::unique_ptr<KernelExecutor>
  create_rmsnorm_executor(type::BackendType backend);

  /**
   * @brief Create a reduction kernel executor
   * @param backend Backend type
   * @return Unique pointer to kernel executor
   */
  static std::unique_ptr<KernelExecutor>
  create_reduction_executor(type::BackendType backend);

  /**
   * @brief Create an element-wise unary kernel executor
   * @param backend Backend type
   * @param op_type Operation type (exp, sqrt, etc.)
   * @return Unique pointer to kernel executor
   */
  static std::unique_ptr<KernelExecutor>
  create_element_unary_executor(type::BackendType backend,
                               type::KNOperatorType op_type);

  /**
   * @brief Create an element-wise binary kernel executor
   * @param backend Backend type
   * @param op_type Operation type (add, mul, etc.)
   * @return Unique pointer to kernel executor
   */
  static std::unique_ptr<KernelExecutor>
  create_element_binary_executor(type::BackendType backend,
                                type::KNOperatorType op_type);
};

/**
 * @brief Base class for operator kernels
 */
class OperatorKernel {
public:
  virtual ~OperatorKernel() = default;

  /**
   * @brief Initialize kernel with configuration
   * @param config Kernel configuration
   * @return true if initialization succeeded
   */
  virtual bool initialize(KernelConfig const &config) = 0;

  /**
   * @brief Execute kernel
   * @param inputs Input tensors
   * @param outputs Output tensors
   * @return true if execution succeeded
   */
  virtual bool execute(std::vector<void *> const &inputs,
                      std::vector<void *> &outputs) = 0;

  /**
   * @brief Get kernel configuration
   * @return Kernel configuration
   */
  virtual KernelConfig const &get_config() const = 0;

  /**
   * @brief Get backend type
   * @return Backend type
   */
  virtual type::BackendType get_backend_type() const = 0;

  /**
   * @brief Get operator type
   * @return Operator type
   */
  virtual type::KNOperatorType get_operator_type() const = 0;
};

} // namespace kernel
} // namespace yirage





