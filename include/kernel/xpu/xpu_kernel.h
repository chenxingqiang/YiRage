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
 * Intel XPU Kernel Interface
 * 
 * Main header for Intel XPU (Data Center GPU Max / Arc) kernel implementations.
 * Corresponds to source files in src/kernel/xpu/
 * 
 * Uses oneAPI/SYCL for kernel programming.
 */

#pragma once

#include "kernel/xpu/xpu_kernel_config.h"
#include "kernel/graph.h"
#include <string>
#include <vector>

#ifdef YIRAGE_BACKEND_XPU_ENABLED

namespace yirage {
namespace kernel {
namespace xpu {

// =============================================================================
// XPU Kernel Generator
// =============================================================================

/**
 * @brief XPU kernel code generator
 * 
 * Generates SYCL kernels for Intel XPU
 */
class XPUKernelGenerator {
public:
  /**
   * @brief Generate kernel code for Intel XPU
   * @param graph Kernel graph
   * @param config XPU configuration
   * @return Generated SYCL kernel code
   */
  static std::string generate_kernel_code(Graph const &graph,
                                         XPUKernelConfig const &config);
  
  /**
   * @brief Generate SYCL header includes
   */
  static std::string generate_includes(XPUKernelConfig const &config);
  
  /**
   * @brief Generate matmul kernel (corresponds to matmul_kernel.sycl)
   * Uses XMX (Xe Matrix eXtensions) for optimal performance
   */
  static std::string generate_matmul_kernel(int m, int n, int k,
                                           XPUKernelConfig const &config);
  
  /**
   * @brief Generate attention kernel (corresponds to attention_kernel.sycl)
   */
  static std::string generate_attention_kernel(int num_heads, int head_dim,
                                               int seq_len, bool causal,
                                               XPUKernelConfig const &config);
  
  /**
   * @brief Generate embedding kernel (corresponds to embedding_kernel.sycl)
   */
  static std::string generate_embedding_kernel(int vocab_size, int embed_dim,
                                              XPUKernelConfig const &config);
  
  /**
   * @brief Generate softmax kernel (corresponds to softmax_kernel.sycl)
   */
  static std::string generate_softmax_kernel(int dim,
                                            XPUKernelConfig const &config);
  
  /**
   * @brief Generate RMS normalization kernel
   */
  static std::string generate_rms_norm_kernel(XPUKernelConfig const &config);
  
  /**
   * @brief Generate element-wise kernel
   */
  static std::string generate_elementwise_kernel(std::string const &op_type,
                                                XPUKernelConfig const &config);
  
  /**
   * @brief Generate reduction kernel
   */
  static std::string generate_reduction_kernel(int dim, std::string const &op,
                                              XPUKernelConfig const &config);

private:
  /**
   * @brief Generate XMX (Xe Matrix eXtensions) intrinsics code
   */
  static std::string generate_xmx_intrinsics(int m, int n, int k,
                                            XPUKernelConfig const &config);
  
  /**
   * @brief Generate DPAS (Dot Product Accumulate Systolic) code
   */
  static std::string generate_dpas_code(XPUKernelConfig const &config);
};

// =============================================================================
// XPU Kernel Compiler
// =============================================================================

/**
 * @brief XPU kernel compiler using Intel DPC++
 */
class XPUKernelCompiler {
public:
  /**
   * @brief Compile SYCL kernel for XPU
   * @param code SYCL kernel code
   * @param config Configuration
   * @param output_path Output file path
   * @return true if compilation succeeded
   */
  static bool compile_kernel(std::string const &code,
                            XPUKernelConfig const &config,
                            std::string const &output_path);
  
  /**
   * @brief Get DPC++ compiler command (icpx)
   */
  static std::string get_compiler_command();
  
  /**
   * @brief Get compiler flags for XPU
   */
  static std::vector<std::string> get_compiler_flags(XPUKernelConfig const &config);
  
  /**
   * @brief Check if DPC++ compiler is available
   */
  static bool is_compiler_available();
  
  /**
   * @brief Get oneAPI base toolkit path
   */
  static std::string get_oneapi_path();
  
  /**
   * @brief Generate oneDNN integration code
   */
  static std::string generate_onednn_wrapper(std::string const &op_name,
                                            XPUKernelConfig const &config);
};

// =============================================================================
// XPU Kernel Executor
// =============================================================================

/**
 * @brief XPU kernel executor
 * 
 * Executes compiled SYCL kernels on Intel XPU
 */
class XPUKernelExecutor {
public:
  XPUKernelExecutor();
  ~XPUKernelExecutor();
  
  /**
   * @brief Initialize XPU runtime
   * @param device_id Device ID
   * @return true if initialization succeeded
   */
  bool initialize(int device_id = 0);
  
  /**
   * @brief Initialize with specific architecture
   */
  bool initialize(XPUArch arch);
  
  /**
   * @brief Load compiled kernel
   * @param kernel_path Path to compiled SYCL kernel
   * @return true if load succeeded
   */
  bool load_kernel(std::string const &kernel_path);
  
  /**
   * @brief Execute kernel with inputs
   * @param inputs Input tensor pointers
   * @param outputs Output tensor pointers
   * @param global_size Global work size
   * @param local_size Local work size
   * @return true if execution succeeded
   */
  bool execute(std::vector<void*> const &inputs,
              std::vector<void*> &outputs,
              std::vector<size_t> global_size,
              std::vector<size_t> local_size);
  
  /**
   * @brief Synchronize device
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
  int device_id_;
  XPUArch arch_;
  void *queue_;           // sycl::queue
  void *context_;         // sycl::context
  float last_exec_time_;
  bool initialized_;
};

// =============================================================================
// XPU Device Memory Manager
// =============================================================================

/**
 * @brief XPU device memory manager (via SYCL USM)
 */
class XPUDeviceMemoryManager {
public:
  /**
   * @brief Initialize XPU device
   */
  static bool initialize_device(int device_id);
  
  /**
   * @brief Finalize XPU device
   */
  static void finalize_device();
  
  /**
   * @brief Allocate device memory (USM device allocation)
   * @param size Size in bytes
   * @return Pointer to allocated memory
   */
  static void* allocate(size_t size);
  
  /**
   * @brief Allocate shared memory (USM shared allocation)
   */
  static void* allocate_shared(size_t size);
  
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
  
  /**
   * @brief Get L3 cache size
   */
  static size_t get_l3_cache_size();
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Check if XPU is available
 */
bool is_xpu_available();

/**
 * @brief Get number of XPU devices
 */
int get_xpu_device_count();

/**
 * @brief Get current device
 */
int get_current_xpu_device();

/**
 * @brief Set current device
 */
void set_xpu_device(int device_id);

/**
 * @brief Get device name
 */
std::string get_xpu_device_name(int device_id = -1);

/**
 * @brief Get XPU architecture
 */
XPUArch get_xpu_arch(int device_id = -1);

/**
 * @brief Get number of execution units
 */
int get_xpu_eu_count(int device_id = -1);

/**
 * @brief Get number of subslices
 */
int get_xpu_subslice_count(int device_id = -1);

/**
 * @brief Get device memory (bytes)
 */
size_t get_xpu_device_memory(int device_id = -1);

/**
 * @brief Check if XMX (Xe Matrix eXtensions) is available
 */
bool has_xpu_xmx(int device_id = -1);

/**
 * @brief Get XPU peak TFLOPS (BF16)
 */
float get_xpu_peak_tflops(int device_id = -1);

} // namespace xpu
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_XPU_ENABLED
