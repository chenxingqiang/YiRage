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

#include "yirage/kernel/ascend/ascend_kernel_config.h"
#include "yirage/kernel/graph.h"
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

} // namespace ascend
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED

