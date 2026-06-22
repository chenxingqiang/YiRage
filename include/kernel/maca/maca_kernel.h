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
 * This file is part of YiRage (Yi Revolutionary AGile Engine)
 * 
 * MACA Kernel Infrastructure
 * 
 * Kernel generator, compiler, and executor for MetaX MACA backend.
 * Uses mxcc compiler and mc* runtime API.
 */

#pragma once

#include "kernel/maca/maca_kernel_config.h"
#include "kernel/graph.h"
#include <string>
#include <vector>

#ifdef YIRAGE_BACKEND_MACA_ENABLED

namespace yirage {
namespace kernel {
namespace maca {

/**
 * @brief MACA kernel code generator
 * 
 * Generates MACA kernel code (CUDA-compatible with mc* API)
 * Optimized for MetaX GPU architecture with 64-thread warps.
 */
class MACAKernelGenerator {
public:
  /**
   * @brief Generate kernel code for MACA GPU
   * @param graph Kernel graph
   * @param config MACA configuration
   * @return Generated kernel code
   */
  static std::string generate_kernel_code(Graph const &graph,
                                         MACAArchConfig const &config);
  
  /**
   * @brief Generate header includes for MACA
   * Includes mc_runtime.h, mc_common.h etc.
   */
  static std::string generate_includes(MACAArchConfig const &config);
  
  /**
   * @brief Generate matmul kernel optimized for MACA
   * Uses 64-thread warp tiles and mctlass patterns
   */
  static std::string generate_matmul_kernel(int m, int n, int k,
                                           MACAMatmulConfig const &config);
  
  /**
   * @brief Generate element-wise kernel
   */
  static std::string generate_elementwise_kernel(std::string const &op_type,
                                                MACAArchConfig const &config);
  
  /**
   * @brief Generate reduction kernel with 64-thread warp reduce
   */
  static std::string generate_reduction_kernel(MACAReductionConfig const &config);
  
  /**
   * @brief Generate RMS normalization kernel
   */
  static std::string generate_rms_norm_kernel(MACANormConfig const &config);
  
  /**
   * @brief Generate attention kernel (flash attention compatible)
   */
  static std::string generate_attention_kernel(MACAAttentionConfig const &config);
  
  /**
   * @brief Generate AllReduce kernel for multi-chip communication
   * Uses mccl (MetaX NCCL equivalent) for collective operations
   * (corresponds to all_reduce_kernel.maca)
   */
  static std::string generate_all_reduce_kernel(int num_elements,
                                                std::string const &reduce_op,
                                                MACAArchConfig const &config);
  
  /**
   * @brief Generate customized/fused kernel
   * (corresponds to customized_kernel.maca)
   */
  static std::string generate_customized_kernel(Graph const &subgraph,
                                               MACAArchConfig const &config);
  
  /**
   * @brief Generate input data loading kernel
   * (corresponds to input_kernel.maca)
   */
  static std::string generate_input_kernel(std::vector<int> const &dims,
                                          MACAArchConfig const &config);
  
  /**
   * @brief Generate output data storing kernel
   * (corresponds to output_kernel.maca)
   */
  static std::string generate_output_kernel(std::vector<int> const &dims,
                                           MACAArchConfig const &config);
  
  /**
   * @brief Generate embedding lookup kernel
   * (corresponds to embedding_kernel.maca)
   */
  static std::string generate_embedding_kernel(int vocab_size, int embed_dim,
                                              MACAArchConfig const &config);
  
  /**
   * @brief Generate softmax kernel
   * (corresponds to softmax_kernel.maca)
   */
  static std::string generate_softmax_kernel(int dim,
                                            MACAArchConfig const &config);
  
  /**
   * @brief Generate device tensor operations kernel
   * (corresponds to device_tensor_kernel.maca)
   */
  static std::string generate_device_tensor_kernel(MACAArchConfig const &config);

private:
  /**
   * @brief Generate warp shuffle code for 64-thread warps
   * Uses 6 iterations instead of 5 for NVIDIA
   */
  static std::string generate_warp_reduce_code(std::string const &type);
};

/**
 * @brief MACA kernel compiler
 * 
 * Compiles kernels using mxcc (MetaX compiler)
 * Command: mxcc -x maca source.cpp -o output --maca-path=/opt/maca
 */
class MACAKernelCompiler {
public:
  /**
   * @brief Compile kernel code to binary
   * @param code Kernel source code
   * @param config Configuration
   * @param output_path Output file path
   * @return true if compilation succeeded
   */
  static bool compile_kernel(std::string const &code,
                            MACAArchConfig const &config,
                            std::string const &output_path);
  
  /**
   * @brief Get mxcc compiler command
   * @return Path to mxcc compiler
   */
  static std::string get_compiler_command();
  
  /**
   * @brief Get compiler flags for MACA
   * Includes -x maca, --maca-path, optimization flags
   */
  static std::vector<std::string> get_compiler_flags(MACAArchConfig const &config);
  
  /**
   * @brief Get MACA SDK path
   * Checks MACA_HOME and MACA_PATH environment variables
   */
  static std::string get_maca_sdk_path();
  
  /**
   * @brief Check if mxcc compiler is available
   */
  static bool is_compiler_available();
};

/**
 * @brief MACA kernel executor
 * 
 * Loads and executes compiled kernels on MetaX GPU
 */
class MACAKernelExecutor {
public:
  MACAKernelExecutor();
  ~MACAKernelExecutor();
  
  /**
   * @brief Initialize MACA runtime
   * @param device_id GPU device ID
   * @return true if initialization succeeded
   */
  bool initialize(int device_id = 0);
  
  /**
   * @brief Load compiled kernel
   * @param kernel_path Path to compiled kernel binary
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
  void *stream_;          // MACA stream (mcStream_t)
  void *kernel_handle_;   // Loaded kernel handle
  float last_exec_time_;
  bool initialized_;
};

/**
 * @brief MACA kernel profiler
 * 
 * Profiles kernel execution using MACA profiling APIs
 */
class MACAKernelProfiler {
public:
  /**
   * @brief Start profiling
   */
  static void start_profiling();
  
  /**
   * @brief Stop profiling
   */
  static void stop_profiling();
  
  /**
   * @brief Get kernel execution time
   * @param kernel_name Kernel name
   * @return Execution time in milliseconds
   */
  static float get_kernel_time(std::string const &kernel_name);
  
  /**
   * @brief Get memory bandwidth utilization
   * @return Bandwidth in GB/s
   */
  static float get_memory_bandwidth();
  
  /**
   * @brief Get compute utilization
   * @return Utilization percentage (0-100)
   */
  static float get_compute_utilization();
  
  /**
   * @brief Get occupancy
   * @return Achieved occupancy (0-1)
   */
  static float get_occupancy();
};

/**
 * @brief MACA device memory manager
 * 
 * Manages memory allocation on MetaX GPU devices
 * Uses mc* memory APIs (mcMalloc, mcFree, mcMemcpy)
 * (corresponds to device_memory_manager.maca)
 */
class MACADeviceMemoryManager {
public:
  /**
   * @brief Initialize MACA device
   * @param device_id Device ID
   * @return true if initialization succeeded
   */
  static bool initialize_device(int device_id);
  
  /**
   * @brief Finalize MACA device
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

} // namespace maca
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED

