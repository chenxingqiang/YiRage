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
 * FPGA Kernel Interface
 * 
 * Main header for FPGA kernel implementations.
 * Corresponds to source files in src/kernel/fpga/
 */

#pragma once

#include "kernel/fpga/fpga_kernel_config.h"
#include "kernel/graph.h"
#include <string>
#include <vector>

#ifdef YIRAGE_BACKEND_FPGA_ENABLED

namespace yirage {
namespace kernel {
namespace fpga {

// =============================================================================
// FPGA Kernel Generator
// =============================================================================

/**
 * @brief FPGA kernel code generator
 * 
 * Generates HLS C++ or OpenCL code for FPGA kernels
 */
class FPGAKernelGenerator {
public:
  /**
   * @brief Generate kernel code for FPGA
   * @param graph Kernel graph
   * @param config FPGA configuration
   * @return Generated kernel code
   */
  static std::string generate_kernel_code(Graph const &graph,
                                         FPGAKernelConfig const &config);
  
  /**
   * @brief Generate header includes
   */
  static std::string generate_includes(FPGAKernelConfig const &config);
  
  /**
   * @brief Generate matmul kernel (corresponds to matmul_kernel.cpp)
   * Uses systolic array pattern for FPGA
   */
  static std::string generate_matmul_kernel(int m, int n, int k,
                                           FPGAKernelConfig const &config);
  
  /**
   * @brief Generate attention kernel (corresponds to attention_kernel.cpp)
   * Implements flash attention pattern for FPGA
   */
  static std::string generate_attention_kernel(int num_heads, int head_dim,
                                               int seq_len,
                                               FPGAKernelConfig const &config);
  
  /**
   * @brief Generate embedding kernel (corresponds to embedding_kernel.cpp)
   */
  static std::string generate_embedding_kernel(int vocab_size, int embed_dim,
                                              FPGAKernelConfig const &config);
  
  /**
   * @brief Generate softmax kernel (corresponds to softmax_kernel.cpp)
   */
  static std::string generate_softmax_kernel(int dim,
                                            FPGAKernelConfig const &config);
  
  /**
   * @brief Generate element-wise kernel
   */
  static std::string generate_elementwise_kernel(std::string const &op_type,
                                                FPGAKernelConfig const &config);
  
  /**
   * @brief Generate reduction kernel
   */
  static std::string generate_reduction_kernel(int dim, std::string const &op,
                                              FPGAKernelConfig const &config);
  
  /**
   * @brief Generate RMS normalization kernel
   */
  static std::string generate_rms_norm_kernel(FPGAKernelConfig const &config);
};

// =============================================================================
// FPGA Kernel Compiler
// =============================================================================

/**
 * @brief FPGA kernel compiler
 * 
 * Compiles HLS C++/OpenCL to FPGA bitstream using vendor tools
 */
class FPGAKernelCompiler {
public:
  /**
   * @brief Compile kernel code to FPGA bitstream
   * @param code Kernel source code
   * @param config Configuration
   * @param output_path Output file path
   * @return true if compilation succeeded
   */
  static bool compile_kernel(std::string const &code,
                            FPGAKernelConfig const &config,
                            std::string const &output_path);
  
  /**
   * @brief Run HLS synthesis (C++ to RTL)
   */
  static bool run_hls_synthesis(std::string const &code,
                               FPGAKernelConfig const &config,
                               std::string const &output_path);
  
  /**
   * @brief Run place and route
   */
  static bool run_place_and_route(std::string const &rtl_path,
                                 FPGAKernelConfig const &config,
                                 std::string const &output_path);
  
  /**
   * @brief Get compiler command (Vitis/Quartus)
   */
  static std::string get_compiler_command(FPGAVendor vendor);
  
  /**
   * @brief Get compiler flags
   */
  static std::vector<std::string> get_compiler_flags(FPGAKernelConfig const &config);
  
  /**
   * @brief Check if compiler is available
   */
  static bool is_compiler_available(FPGAVendor vendor);
  
  /**
   * @brief Estimate compilation time
   */
  static float estimate_compile_time_hours(FPGAKernelConfig const &config);
};

// =============================================================================
// FPGA Kernel Executor
// =============================================================================

/**
 * @brief FPGA kernel executor
 * 
 * Loads and executes compiled kernels on FPGA
 */
class FPGAKernelExecutor {
public:
  FPGAKernelExecutor();
  ~FPGAKernelExecutor();
  
  /**
   * @brief Initialize FPGA runtime (XRT/OpenCL)
   * @param device_id Device ID
   * @return true if initialization succeeded
   */
  bool initialize(int device_id = 0);
  
  /**
   * @brief Load FPGA bitstream
   * @param xclbin_path Path to compiled bitstream (.xclbin/.aocx)
   * @return true if load succeeded
   */
  bool load_bitstream(std::string const &xclbin_path);
  
  /**
   * @brief Execute kernel with inputs
   * @param kernel_name Kernel function name
   * @param inputs Input tensor pointers
   * @param outputs Output tensor pointers
   * @return true if execution succeeded
   */
  bool execute(std::string const &kernel_name,
              std::vector<void*> const &inputs,
              std::vector<void*> &outputs);
  
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
  void *context_;         // OpenCL context
  void *queue_;           // Command queue
  void *program_;         // Loaded program
  float last_exec_time_;
  bool initialized_;
};

// =============================================================================
// FPGA Device Memory Manager
// =============================================================================

/**
 * @brief FPGA device memory manager
 */
class FPGADeviceMemoryManager {
public:
  /**
   * @brief Initialize FPGA device
   */
  static bool initialize_device(int device_id);
  
  /**
   * @brief Finalize FPGA device
   */
  static void finalize_device();
  
  /**
   * @brief Allocate device memory (DDR/HBM)
   * @param size Size in bytes
   * @param memory_bank Memory bank index (for multi-channel memory)
   * @return Pointer to allocated memory
   */
  static void* allocate(size_t size, int memory_bank = 0);
  
  /**
   * @brief Free device memory
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
   * @brief Get available memory on device
   */
  static size_t get_available_memory(int memory_bank = -1);
  
  /**
   * @brief Get total memory on device
   */
  static size_t get_total_memory();
  
  /**
   * @brief Get number of memory banks (DDR channels / HBM stacks)
   */
  static int get_memory_bank_count();
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if FPGA is available
 */
bool is_fpga_available();

/**
 * @brief Get number of FPGA devices
 */
int get_fpga_device_count();

/**
 * @brief Get device name
 */
std::string get_fpga_device_name(int device_id = -1);

/**
 * @brief Get FPGA vendor
 */
FPGAVendor get_fpga_vendor(int device_id = -1);

/**
 * @brief Get FPGA device type
 */
FPGADevice get_fpga_device_type(int device_id = -1);

/**
 * @brief Get shell/platform name
 */
std::string get_fpga_platform_name(int device_id = -1);

} // namespace fpga
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_FPGA_ENABLED
