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
 * CUDA Kernel Interface
 * 
 * Main header for CUDA kernel implementations.
 * Corresponds to source files in src/kernel/cuda/
 */

#pragma once

#include "kernel/cuda/cuda_kernel_config.h"
#include "kernel/graph.h"
#include <string>
#include <vector>

#ifdef YIRAGE_BACKEND_CUDA_ENABLED

namespace yirage {
namespace kernel {
namespace cuda {

// =============================================================================
// CUDA Kernel Generator
// =============================================================================

/**
 * @brief CUDA kernel code generator
 * 
 * Generates optimized CUDA kernels for various operations
 */
class CUDAKernelGenerator {
public:
  /**
   * @brief Generate kernel code for CUDA GPU
   * @param graph Kernel graph
   * @param config CUDA configuration
   * @return Generated kernel code
   */
  static std::string generate_kernel_code(Graph const &graph,
                                         CUDAKernelConfig const &config);
  
  /**
   * @brief Generate header includes
   */
  static std::string generate_includes(CUDAKernelConfig const &config);
  
  /**
   * @brief Generate matmul kernel (corresponds to matmul_kernel.cu)
   */
  static std::string generate_matmul_kernel(int m, int n, int k,
                                           CUDAKernelConfig const &config);
  
  /**
   * @brief Generate element-wise unary kernel (corresponds to element_unary_kernel.cu)
   */
  static std::string generate_element_unary_kernel(std::string const &op_type,
                                                   CUDAKernelConfig const &config);
  
  /**
   * @brief Generate element-wise binary kernel (corresponds to element_binary_kernel.cu)
   */
  static std::string generate_element_binary_kernel(std::string const &op_type,
                                                    CUDAKernelConfig const &config);
  
  /**
   * @brief Generate reduction kernel (corresponds to reduction_kernel.cu)
   */
  static std::string generate_reduction_kernel(int dim, std::string const &op,
                                              CUDAKernelConfig const &config);
  
  /**
   * @brief Generate RMS normalization kernel (corresponds to rms_norm_kernel.cu)
   */
  static std::string generate_rms_norm_kernel(CUDAKernelConfig const &config);
  
  /**
   * @brief Generate input loading kernel (corresponds to input_kernel.cu)
   */
  static std::string generate_input_kernel(std::vector<int> const &dims,
                                          CUDAKernelConfig const &config);
  
  /**
   * @brief Generate output storing kernel (corresponds to output_kernel.cu)
   */
  static std::string generate_output_kernel(std::vector<int> const &dims,
                                           CUDAKernelConfig const &config);
  
  /**
   * @brief Generate customized/fused kernel (corresponds to customized_kernel.cu)
   */
  static std::string generate_customized_kernel(Graph const &subgraph,
                                               CUDAKernelConfig const &config);
  
  /**
   * @brief Generate device tensor operations kernel (corresponds to device_tensor_kernel.cu)
   */
  static std::string generate_device_tensor_kernel(CUDAKernelConfig const &config);
  
  /**
   * @brief Generate chunk kernel (corresponds to chunk_kernel.cu)
   */
  static std::string generate_chunk_kernel(int num_chunks, int chunk_size,
                                          CUDAKernelConfig const &config);
  
  /**
   * @brief Generate all-reduce kernel (corresponds to all_reduce_kernel.cu)
   */
  static std::string generate_all_reduce_kernel(int num_elements,
                                                std::string const &reduce_op,
                                                CUDAKernelConfig const &config);
};

// =============================================================================
// CUDA Kernel Compiler
// =============================================================================

/**
 * @brief CUDA kernel compiler using NVCC
 */
class CUDAKernelCompiler {
public:
  /**
   * @brief Compile kernel code to PTX/CUBIN
   * @param code Kernel source code
   * @param config Configuration
   * @param output_path Output file path
   * @return true if compilation succeeded
   */
  static bool compile_kernel(std::string const &code,
                            CUDAKernelConfig const &config,
                            std::string const &output_path);
  
  /**
   * @brief Get nvcc compiler command
   */
  static std::string get_compiler_command();
  
  /**
   * @brief Get compiler flags for CUDA
   */
  static std::vector<std::string> get_compiler_flags(CUDAKernelConfig const &config);
  
  /**
   * @brief Check if nvcc compiler is available
   */
  static bool is_compiler_available();
  
  /**
   * @brief Get CUDA toolkit path
   */
  static std::string get_cuda_toolkit_path();
};

// =============================================================================
// CUDA Kernel Executor
// =============================================================================

/**
 * @brief CUDA kernel executor
 */
class CUDAKernelExecutor {
public:
  CUDAKernelExecutor();
  ~CUDAKernelExecutor();
  
  /**
   * @brief Initialize CUDA runtime
   * @param device_id GPU device ID
   * @return true if initialization succeeded
   */
  bool initialize(int device_id = 0);
  
  /**
   * @brief Load compiled kernel
   * @param kernel_path Path to compiled kernel (PTX/CUBIN)
   * @return true if load succeeded
   */
  bool load_kernel(std::string const &kernel_path);
  
  /**
   * @brief Execute kernel with inputs
   * @param inputs Input tensor pointers
   * @param outputs Output tensor pointers
   * @param grid Grid dimensions
   * @param block Block dimensions
   * @return true if execution succeeded
   */
  bool execute(std::vector<void*> const &inputs,
              std::vector<void*> &outputs,
              dim3 grid, dim3 block);
  
  /**
   * @brief Synchronize device
   */
  void synchronize();
  
  /**
   * @brief Get execution time
   * @return Execution time in milliseconds
   */
  float get_execution_time() const;
  
  /**
   * @brief Get device info
   */
  std::string get_device_info() const;

private:
  int device_id_;
  void *stream_;          // cudaStream_t
  void *kernel_handle_;   // Loaded kernel handle
  float last_exec_time_;
  bool initialized_;
};

// =============================================================================
// CUDA Device Memory Manager
// =============================================================================

/**
 * @brief CUDA device memory manager
 */
class CUDADeviceMemoryManager {
public:
  /**
   * @brief Initialize CUDA device
   * @param device_id Device ID
   * @return true if initialization succeeded
   */
  static bool initialize_device(int device_id);
  
  /**
   * @brief Finalize CUDA device
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
   */
  static bool copy_host_to_device(void *dst, void const *src, size_t size);
  
  /**
   * @brief Copy data from device to host
   */
  static bool copy_device_to_host(void *dst, void const *src, size_t size);
  
  /**
   * @brief Copy data between devices
   */
  static bool copy_device_to_device(void *dst, void const *src, size_t size);
  
  /**
   * @brief Get available memory on device
   */
  static size_t get_available_memory();
  
  /**
   * @brief Get total memory on device
   */
  static size_t get_total_memory();
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if CUDA is available
 */
bool is_cuda_available();

/**
 * @brief Get number of CUDA devices
 */
int get_cuda_device_count();

/**
 * @brief Get current device
 */
int get_current_cuda_device();

/**
 * @brief Set current device
 */
void set_cuda_device(int device_id);

/**
 * @brief Get device name
 */
std::string get_cuda_device_name(int device_id = -1);

/**
 * @brief Get compute capability
 */
int get_cuda_compute_capability(int device_id = -1);

/**
 * @brief Get device memory (bytes)
 */
size_t get_cuda_device_memory(int device_id = -1);

/**
 * @brief Get shared memory per block
 */
size_t get_cuda_shared_memory_per_block(int device_id = -1);

} // namespace cuda
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_CUDA_ENABLED
