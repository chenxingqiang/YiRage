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
 * TPU Kernel Interface
 * 
 * Main header for Google TPU kernel implementations.
 * Corresponds to source files in src/kernel/tpu/
 * 
 * TPU kernels are implemented via:
 * - XLA HLO (High Level Operations)
 * - Pallas (JAX-based kernel DSL)
 */

#pragma once

#include "kernel/tpu/tpu_kernel_config.h"
#include "kernel/graph.h"
#include <string>
#include <vector>

#ifdef YIRAGE_BACKEND_TPU_ENABLED

namespace yirage {
namespace kernel {
namespace tpu {

// =============================================================================
// TPU Kernel Generator
// =============================================================================

/**
 * @brief TPU kernel code generator
 * 
 * Generates XLA HLO or Pallas kernels for TPU
 */
class TPUKernelGenerator {
public:
  /**
   * @brief Generate kernel code for TPU
   * @param graph Kernel graph
   * @param config TPU configuration
   * @return Generated kernel code (Python/Pallas)
   */
  static std::string generate_kernel_code(Graph const &graph,
                                         TPUKernelConfig const &config);
  
  /**
   * @brief Generate Python imports for TPU kernels
   */
  static std::string generate_imports(TPUKernelConfig const &config);
  
  /**
   * @brief Generate matmul kernel (corresponds to matmul_kernel.py)
   * Uses MXU (Matrix Multiply Unit) for optimal performance
   */
  static std::string generate_matmul_kernel(int m, int n, int k,
                                           TPUKernelConfig const &config);
  
  /**
   * @brief Generate attention kernel (corresponds to attention_kernel.py)
   * Flash attention optimized for TPU MXU
   */
  static std::string generate_attention_kernel(int num_heads, int head_dim,
                                               int seq_len, bool causal,
                                               TPUKernelConfig const &config);
  
  /**
   * @brief Generate embedding kernel (corresponds to embedding_kernel.py)
   */
  static std::string generate_embedding_kernel(int vocab_size, int embed_dim,
                                              TPUKernelConfig const &config);
  
  /**
   * @brief Generate softmax kernel (corresponds to softmax_kernel.py)
   */
  static std::string generate_softmax_kernel(int dim,
                                            TPUKernelConfig const &config);
  
  /**
   * @brief Generate RMS normalization kernel
   */
  static std::string generate_rms_norm_kernel(TPUKernelConfig const &config);
  
  /**
   * @brief Generate element-wise kernel
   */
  static std::string generate_elementwise_kernel(std::string const &op_type,
                                                TPUKernelConfig const &config);
  
  /**
   * @brief Generate reduction kernel
   */
  static std::string generate_reduction_kernel(int dim, std::string const &op,
                                              TPUKernelConfig const &config);

private:
  /**
   * @brief Generate Pallas kernel decorator
   */
  static std::string generate_pallas_decorator(TPUKernelConfig const &config);
  
  /**
   * @brief Generate XLA custom call
   */
  static std::string generate_xla_custom_call(std::string const &kernel_name,
                                             TPUKernelConfig const &config);
};

// =============================================================================
// TPU Kernel Compiler
// =============================================================================

/**
 * @brief TPU kernel compiler
 * 
 * Compiles Pallas/XLA kernels for TPU execution
 */
class TPUKernelCompiler {
public:
  /**
   * @brief Compile Pallas kernel
   * @param code Pallas kernel code (Python)
   * @param config Configuration
   * @return Compiled XLA module path
   */
  static std::string compile_pallas_kernel(std::string const &code,
                                          TPUKernelConfig const &config);
  
  /**
   * @brief Compile XLA HLO
   * @param hlo_text XLA HLO text representation
   * @param config Configuration
   * @return Compiled HLO module path
   */
  static std::string compile_xla_hlo(std::string const &hlo_text,
                                    TPUKernelConfig const &config);
  
  /**
   * @brief Generate XLA HLO from graph
   */
  static std::string generate_xla_hlo(Graph const &graph,
                                     TPUKernelConfig const &config);
  
  /**
   * @brief Check if TPU compiler is available
   */
  static bool is_compiler_available();
  
  /**
   * @brief Get JAX version
   */
  static std::string get_jax_version();
};

// =============================================================================
// TPU Kernel Executor
// =============================================================================

/**
 * @brief TPU kernel executor
 * 
 * Executes compiled kernels on TPU via JAX/XLA runtime
 */
class TPUKernelExecutor {
public:
  TPUKernelExecutor();
  ~TPUKernelExecutor();
  
  /**
   * @brief Initialize TPU runtime
   * @param num_devices Number of TPU devices to use
   * @return true if initialization succeeded
   */
  bool initialize(int num_devices = 1);
  
  /**
   * @brief Load compiled kernel
   * @param module_path Path to compiled XLA module
   * @return true if load succeeded
   */
  bool load_kernel(std::string const &module_path);
  
  /**
   * @brief Execute kernel with inputs
   * @param inputs Input tensor pointers
   * @param outputs Output tensor pointers
   * @return true if execution succeeded
   */
  bool execute(std::vector<void*> const &inputs,
              std::vector<void*> &outputs);
  
  /**
   * @brief Synchronize TPU
   */
  void synchronize();
  
  /**
   * @brief Get execution time
   */
  float get_execution_time() const;
  
  /**
   * @brief Get device info
   */
  std::string get_device_info() const;

private:
  int num_devices_;
  void *runtime_;         // JAX runtime handle
  float last_exec_time_;
  bool initialized_;
};

// =============================================================================
// TPU Memory Manager
// =============================================================================

/**
 * @brief TPU memory manager (via JAX)
 */
class TPUMemoryManager {
public:
  /**
   * @brief Allocate TPU memory
   * @param size Size in bytes
   * @return Device buffer handle
   */
  static void* allocate(size_t size);
  
  /**
   * @brief Free TPU memory
   */
  static void free(void *ptr);
  
  /**
   * @brief Copy data from host to TPU
   */
  static bool copy_host_to_device(void *dst, void const *src, size_t size);
  
  /**
   * @brief Copy data from TPU to host
   */
  static bool copy_device_to_host(void *dst, void const *src, size_t size);
  
  /**
   * @brief Get available HBM memory
   */
  static size_t get_available_memory();
  
  /**
   * @brief Get total HBM memory per core
   */
  static size_t get_hbm_per_core();
  
  /**
   * @brief Get VMEM size per core
   */
  static size_t get_vmem_per_core();
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if TPU is available
 */
bool is_tpu_available();

/**
 * @brief Get number of TPU devices
 */
int get_tpu_device_count();

/**
 * @brief Get TPU version
 */
TPUVersion get_tpu_version();

/**
 * @brief Get TPU topology (e.g., "2x2x1" for a 4-chip pod)
 */
std::string get_tpu_topology();

/**
 * @brief Get TPU cores per device
 */
int get_tpu_cores_per_device();

/**
 * @brief Get TPU HBM capacity per device (bytes)
 */
size_t get_tpu_hbm_capacity();

/**
 * @brief Get TPU peak TFLOPS (BF16)
 */
float get_tpu_peak_tflops();

} // namespace tpu
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_TPU_ENABLED
