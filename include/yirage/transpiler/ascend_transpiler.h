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
 * Ascend Transpiler for YiRage
 * 
 * This transpiler generates code for Huawei Ascend NPU devices.
 * 
 * Architecture Strategy:
 * The primary strategy is to REUSE the existing Triton transpiler because:
 * 1. BiSheng Compiler (Huawei's fork) supports Triton natively
 * 2. This eliminates the need for a separate Ascend-specific code generator
 * 3. Triton code can run on Ascend 910B/910B2 without modification
 * 
 * Fallback paths:
 * - Ascend C: For 910B+ devices when Triton is unavailable
 * - TBE (Tensor Boost Engine): For legacy Ascend 910 compatibility
 */

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "yirage/kernel/graph.h"

namespace yirage {
namespace ascend_transpiler {

namespace kn = yirage::kernel;

/**
 * @brief Code generation path selection
 */
enum class CodeGenPath {
  TRITON,    // Triton via BiSheng (RECOMMENDED)
  ASCEND_C,  // Native Ascend C (910B+)
  TBE        // Tensor Boost Engine (910 legacy)
};

/**
 * @brief Ascend device type enumeration
 */
enum class AscendDeviceType {
  ASCEND_910 = 0,    // Original Ascend 910 (Da Vinci)
  ASCEND_910B = 1,   // Ascend 910B (improved)
  ASCEND_910B2 = 2,  // Ascend 910B2 (latest)
  ASCEND_310P = 3,   // Ascend 310P (inference)
  ASCEND_310 = 4     // Ascend 310 (edge inference)
};

/**
 * @brief Configuration for Ascend transpilation
 */
struct AscendTranspilerConfig {
  // Device target
  AscendDeviceType device_type = AscendDeviceType::ASCEND_910B;
  
  // Code generation path
  CodeGenPath codegen_path = CodeGenPath::TRITON;
  
  // Cube operations (matmul acceleration)
  bool use_cube_ops = true;
  
  // Operator fusion
  bool enable_fusion = true;
  
  // AI Cores per block (1-32)
  int ai_cores_per_block = 16;
  
  // L1 buffer size hint (0 = auto)
  size_t l1_buffer_size = 0;
  
  // Optimization level (0-3)
  int opt_level = 3;
  
  // Enable FP16 computation
  bool enable_fp16 = true;
  
  // Enable BF16 computation (910B+)
  bool enable_bf16 = false;
  
  // Debug mode (generate verbose comments)
  bool debug_mode = false;
};

/**
 * @brief Error information from transpilation
 */
struct AscendTranspileError {
  std::vector<std::string> messages;
  
  bool has_error() const { return !messages.empty(); }
  
  std::string to_string() const {
    std::string result;
    for (const auto& msg : messages) {
      result += msg + "\n";
    }
    return result;
  }
};

/**
 * @brief Result of Ascend transpilation
 */
struct AscendTranspileResult {
  // Generated code (Triton Python, Ascend C, or TBE)
  std::string code;
  
  // Compilation command for the generated code
  std::string compile_command;
  
  // Code generation path that was used
  CodeGenPath path_used;
  
  // Output tensor shapes
  std::vector<std::vector<int>> output_shapes;
  
  // Error information
  AscendTranspileError error;
  
  // Whether transpilation succeeded
  bool success() const { return !error.has_error(); }
};

/**
 * @brief Main Ascend Transpiler class
 */
class AscendTranspiler {
public:
  explicit AscendTranspiler(kn::Graph const* graph,
                           AscendTranspilerConfig const& config);
  
  /**
   * @brief Generate code for Ascend
   * @return Transpilation result
   */
  AscendTranspileResult generate_code();
  
  /**
   * @brief Get recommended configuration for detected environment
   */
  static AscendTranspilerConfig get_recommended_config();
  
  /**
   * @brief Detect device type from environment
   */
  static AscendDeviceType detect_device_type();
  
private:
  // The kernel graph to transpile
  kn::Graph const* graph_;
  
  // Configuration
  AscendTranspilerConfig config_;
  
  // Internal methods for different code paths
  AscendTranspileResult transpile_via_triton();
  AscendTranspileResult transpile_to_ascend_c();
  AscendTranspileResult transpile_to_tbe();
  
  // Helper to get SOC version string
  std::string get_soc_version() const;
};

/**
 * @brief Transpile kernel graph to Ascend code (convenience function)
 * 
 * @param graph The kernel graph to transpile
 * @param config Transpiler configuration
 * @return Transpilation result
 */
AscendTranspileResult transpile(kn::Graph const* graph,
                                AscendTranspilerConfig const& config);

/**
 * @brief Transpile with automatic path selection
 * 
 * Automatically chooses the best code generation path based on:
 * 1. Device type (910 vs 910B+)
 * 2. Available compilers (BiSheng, ascendc, tbe)
 * 3. Graph characteristics
 * 
 * @param graph The kernel graph to transpile
 * @param device_type Target device type
 * @return Transpilation result
 */
AscendTranspileResult transpile_auto(kn::Graph const* graph,
                                     AscendDeviceType device_type);

} // namespace ascend_transpiler
} // namespace yirage
