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
 */

#pragma once

#include "kernel/ascend/ascend_kernel_config.h"
#include "kernel/graph.h"
#include <string>
#include <vector>

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED

namespace yirage {
namespace kernel {
namespace ascend {

/**
 * @brief Ascend kernel code generator
 * 
 * Generates TBE (Tensor Boost Engine) or AscendC kernel code
 */
class AscendKernelGenerator {
public:
  /**
   * @brief Generate kernel code for Ascend NPU
   * @param graph Kernel graph
   * @param config Ascend configuration
   * @return Generated kernel code
   */
  static std::string generate_kernel_code(Graph const &graph,
                                         AscendKernelConfig const &config);
  
  /**
   * @brief Generate header includes
   */
  static std::string generate_includes(AscendKernelConfig const &config);
  
  /**
   * @brief Generate Cube matmul kernel
   */
  static std::string generate_cube_matmul(int m, int n, int k,
                                         AscendKernelConfig const &config);
  
  /**
   * @brief Generate Vector element-wise kernel
   */
  static std::string generate_vector_ops(std::string const &op_type,
                                        AscendKernelConfig const &config);
  
  /**
   * @brief Generate L1 buffer allocation
   */
  static std::string generate_l1_buffer_alloc(AscendKernelConfig const &config);
  
  /**
   * @brief Generate RMS Normalization kernel
   */
  static std::string generate_rms_norm_kernel(AscendKernelConfig const &config);
  
  /**
   * @brief Generate attention kernel (GQA/MHA)
   */
  static std::string generate_attention_kernel(int num_heads, int head_dim,
                                               int num_kv_heads,
                                               AscendKernelConfig const &config);
  
  /**
   * @brief Generate AllReduce kernel for multi-chip communication
   */
  static std::string generate_all_reduce_kernel(int num_elements,
                                                std::string const &reduce_op,
                                                AscendKernelConfig const &config);
  
  /**
   * @brief Generate customized/fused kernel
   */
  static std::string generate_customized_kernel(Graph const &subgraph,
                                               AscendKernelConfig const &config);
  
  /**
   * @brief Generate input data loading kernel
   */
  static std::string generate_input_kernel(std::vector<int> const &dims,
                                          AscendKernelConfig const &config);
  
  /**
   * @brief Generate output data storing kernel
   */
  static std::string generate_output_kernel(std::vector<int> const &dims,
                                           AscendKernelConfig const &config);
  
  /**
   * @brief Generate embedding lookup kernel
   */
  static std::string generate_embedding_kernel(int vocab_size, int embed_dim,
                                              AscendKernelConfig const &config);
  
  /**
   * @brief Generate softmax kernel
   */
  static std::string generate_softmax_kernel(int dim,
                                            AscendKernelConfig const &config);
  
  /**
   * @brief Generate reduction kernel
   */
  static std::string generate_reduction_kernel(int dim, std::string const &op,
                                              AscendKernelConfig const &config);
  
  /**
   * @brief Generate device tensor operations kernel
   */
  static std::string generate_device_tensor_kernel(AscendKernelConfig const &config);
};

/**
 * @brief Ascend kernel compiler
 * 
 * Compiles TBE/AscendC code using CANN toolchain
 */
class AscendKernelCompiler {
public:
  /**
   * @brief Compile kernel code to binary
   * @param code Kernel source code
   * @param config Configuration
   * @return true if compilation succeeded
   */
  static bool compile_kernel(std::string const &code,
                            AscendKernelConfig const &config,
                            std::string const &output_path);
  
  /**
   * @brief Get compiler command
   */
  static std::string get_compiler_command(AscendKernelConfig const &config);
  
  /**
   * @brief Get compiler flags
   */
  static std::vector<std::string> get_compiler_flags(AscendKernelConfig const &config);
};

/**
 * @brief Ascend kernel executor
 * 
 * Loads and executes compiled kernels on Ascend NPU
 */
class AscendKernelExecutor {
public:
  AscendKernelExecutor();
  ~AscendKernelExecutor();
  
  /**
   * @brief Load compiled kernel
   * @param kernel_path Path to compiled kernel binary
   * @return true if load succeeded
   */
  bool load_kernel(std::string const &kernel_path);
  
  /**
   * @brief Execute kernel with inputs
   * @param inputs Input tensors
   * @param outputs Output tensors
   * @return true if execution succeeded
   */
  bool execute(std::vector<void*> const &inputs,
              std::vector<void*> &outputs);
  
  /**
   * @brief Get execution time
   * @return Execution time in milliseconds
   */
  float get_execution_time() const;

private:
  void *stream_;          // ACL stream
  void *kernel_handle_;   // Loaded kernel handle
  float last_exec_time_;
};

/**
 * @brief Ascend device memory manager
 * 
 * Manages memory allocation on Ascend NPU devices
 */
class AscendDeviceMemoryManager {
public:
  /**
   * @brief Initialize Ascend device
   * @param device_id Device ID
   * @return true if initialization succeeded
   */
  static bool initialize_device(int device_id);
  
  /**
   * @brief Finalize Ascend device
   */
  static void finalize_device();
  
  /**
   * @brief Allocate device memory
   * @param size Size in bytes
   * @return Pointer to allocated memory, or nullptr on failure
   */
  static void* allocate(size_t size);
  
  /**
   * @brief Free device memory
   * @param ptr Pointer to memory
   */
  static void free(void *ptr);
  
  /**
   * @brief Copy data from host to device
   * @param dst Device pointer
   * @param src Host pointer
   * @param size Size in bytes
   * @return true if copy succeeded
   */
  static bool copy_host_to_device(void *dst, void const *src, size_t size);
  
  /**
   * @brief Copy data from device to host
   * @param dst Host pointer
   * @param src Device pointer
   * @param size Size in bytes
   * @return true if copy succeeded
   */
  static bool copy_device_to_host(void *dst, void const *src, size_t size);
  
  /**
   * @brief Copy data between devices
   * @param dst Destination device pointer
   * @param src Source device pointer
   * @param size Size in bytes
   * @return true if copy succeeded
   */
  static bool copy_device_to_device(void *dst, void const *src, size_t size);
  
  /**
   * @brief Get available memory on device
   * @return Available memory in bytes
   */
  static size_t get_available_memory();
  
  /**
   * @brief Get total memory on device
   * @return Total memory in bytes
   */
  static size_t get_total_memory();
};

} // namespace ascend
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED

