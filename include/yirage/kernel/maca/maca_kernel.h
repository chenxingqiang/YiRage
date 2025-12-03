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

#include "yirage/kernel/maca/maca_kernel_config.h"
#include "yirage/kernel/graph.h"
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

} // namespace maca
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED

