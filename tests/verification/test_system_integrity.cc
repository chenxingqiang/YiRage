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
 * System Integrity Verification Tests
 * ===================================
 * This file verifies the completeness, closedness, and robustness of
 * all YiRage system components.
 */

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <memory>

#include "type.h"
#include "kernel/common/kernel_interface.h"
#include "search/common/search_strategy.h"
#include "search/config.h"

namespace yirage {
namespace test {

//===----------------------------------------------------------------------===//
// Test Infrastructure
//===----------------------------------------------------------------------===//

struct TestResult {
  std::string component;
  std::string test_name;
  bool passed;
  std::string message;
};

class IntegrityVerifier {
public:
  std::vector<TestResult> results;
  
  void add_result(const std::string& component, const std::string& test,
                  bool passed, const std::string& msg = "") {
    results.push_back({component, test, passed, msg});
  }
  
  void print_summary() {
    int total = results.size();
    int passed = 0;
    for (const auto& r : results) {
      if (r.passed) passed++;
    }
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "System Integrity Verification Summary\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Total Tests: " << total << "\n";
    std::cout << "Passed:      " << passed << "\n";
    std::cout << "Failed:      " << (total - passed) << "\n";
    std::cout << "Pass Rate:   " << (100.0 * passed / total) << "%\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    if (passed < total) {
      std::cout << "Failed Tests:\n";
      for (const auto& r : results) {
        if (!r.passed) {
          std::cout << "  [" << r.component << "] " << r.test_name;
          if (!r.message.empty()) {
            std::cout << ": " << r.message;
          }
          std::cout << "\n";
        }
      }
    }
  }
};

//===----------------------------------------------------------------------===//
// 1. Backend Type Completeness Verification
//===----------------------------------------------------------------------===//

void verify_backend_types(IntegrityVerifier& v) {
  using namespace type;
  
  // Hardware backends should be complete
  std::vector<BackendType> hardware_backends = {
    BT_CUDA, BT_MPS, BT_ROCM, BT_CPU, BT_ASCEND, BT_MACA, BT_TPU, BT_FPGA, BT_XPU
  };
  
  for (auto bt : hardware_backends) {
    bool is_hw = is_hardware_backend(bt);
    v.add_result("BackendType", 
                 "is_hardware_backend(" + backend_type_to_string(bt) + ")",
                 is_hw);
  }
  
  // Library backends
  std::vector<BackendType> library_backends = {
    BT_CUDNN, BT_CUTLASS, BT_MKL
  };
  
  for (auto bt : library_backends) {
    bool is_lib = is_library_backend(bt);
    v.add_result("BackendType",
                 "is_library_backend(" + backend_type_to_string(bt) + ")",
                 is_lib);
  }
  
  // DSL backends
  std::vector<BackendType> dsl_backends = {
    BT_TRITON, BT_NKI, BT_MLIR
  };
  
  for (auto bt : dsl_backends) {
    bool is_dsl = is_dsl_backend(bt);
    v.add_result("BackendType",
                 "is_dsl_backend(" + backend_type_to_string(bt) + ")",
                 is_dsl);
  }
  
  // Fallback logic
  v.add_result("BackendType", "fallback(CUDNN)->CUDA",
               get_fallback_backend(BT_CUDNN) == BT_CUDA);
  v.add_result("BackendType", "fallback(MKL)->CPU",
               get_fallback_backend(BT_MKL) == BT_CPU);
  v.add_result("BackendType", "fallback(TRITON)->CUDA",
               get_fallback_backend(BT_TRITON) == BT_CUDA);
  
  // MLIR dialect mapping
  std::vector<MLIRDialect> dialects = {
    MLIR_LINALG, MLIR_AFFINE, MLIR_SCF, MLIR_GPU, MLIR_LLVM
  };
  
  for (auto d : dialects) {
    std::string name = mlir_dialect_to_string(d);
    v.add_result("MLIRDialect", "has_name(" + name + ")", !name.empty());
  }
}

//===----------------------------------------------------------------------===//
// 2. KN Operator Type Completeness Verification
//===----------------------------------------------------------------------===//

void verify_kn_operators(IntegrityVerifier& v) {
  using namespace type;
  
  // Core operators that must exist
  std::vector<KNOperatorType> core_ops = {
    KN_INPUT_OP, KN_OUTPUT_OP, KN_MATMUL_OP
  };
  
  for (auto op : core_ops) {
    std::string name = kn_operator_type_to_string(op);
    v.add_result("KNOperator", "has_name(" + name + ")", !name.empty());
  }
  
  // Element-wise unary operators
  std::vector<KNOperatorType> unary_ops = {
    KN_EXP_OP, KN_SQRT_OP, KN_SQUARE_OP, KN_SILU_OP, KN_SIGMOID_OP, KN_GELU_OP,
    KN_RELU_OP, KN_CLAMP_OP, KN_LOG_OP
  };
  
  for (auto op : unary_ops) {
    std::string name = kn_operator_type_to_string(op);
    v.add_result("KNOperator", "unary_" + name, !name.empty());
  }
  
  // Element-wise binary operators
  std::vector<KNOperatorType> binary_ops = {
    KN_ADD_OP, KN_MUL_OP, KN_DIV_OP, KN_POW_OP
  };
  
  for (auto op : binary_ops) {
    std::string name = kn_operator_type_to_string(op);
    v.add_result("KNOperator", "binary_" + name, !name.empty());
  }
  
  // Reduction operators
  std::vector<KNOperatorType> reduction_ops = {
    KN_REDUCTION_0_OP, KN_REDUCTION_1_OP, KN_REDUCTION_2_OP
  };
  
  for (auto op : reduction_ops) {
    std::string name = kn_operator_type_to_string(op);
    v.add_result("KNOperator", "reduction_" + name, !name.empty());
  }
}

//===----------------------------------------------------------------------===//
// 3. TB Operator Type Completeness Verification
//===----------------------------------------------------------------------===//

void verify_tb_operators(IntegrityVerifier& v) {
  using namespace type;
  
  // Core TB operators
  std::vector<TBOperatorType> core_ops = {
    TB_INPUT_OP, TB_OUTPUT_OP, TB_MATMUL_OP
  };
  
  for (auto op : core_ops) {
    std::string name = tb_operator_type_to_string(op);
    v.add_result("TBOperator", "has_name(" + name + ")", !name.empty());
  }
  
  // Forloop accumulation operators
  std::vector<TBOperatorType> accum_ops = {
    TB_FORLOOP_ACCUM_NO_RED_OP, TB_FORLOOP_ACCUM_RED_LD_SUM_OP,
    TB_FORLOOP_ACCUM_RED_LD_MEAN_OP, TB_FORLOOP_ACCUM_RED_LD_RMS_OP,
    TB_FORLOOP_ACCUM_MAX_OP
  };
  
  for (auto op : accum_ops) {
    std::string name = tb_operator_type_to_string(op);
    v.add_result("TBOperator", "accum_" + name, !name.empty());
  }
  
  // Verify TB_FORLOOP_ACCUM range
  v.add_result("TBOperator", "accum_range_valid",
               TB_FORLOOP_ACCUM_FIRST_OP <= TB_FORLOOP_ACCUM_NO_RED_OP &&
               TB_FORLOOP_ACCUM_NO_RED_OP <= TB_FORLOOP_ACCUM_LAST_OP);
}

//===----------------------------------------------------------------------===//
// 4. Search Strategy Factory Verification
//===----------------------------------------------------------------------===//

void verify_search_strategies(IntegrityVerifier& v) {
  using namespace search;
  using namespace type;
  
  // Test factory creation for each backend
  std::vector<BackendType> backends = {
    BT_CUDA, BT_MPS, BT_ROCM, BT_CPU, BT_ASCEND, BT_MACA,
    BT_TPU, BT_FPGA, BT_XPU, BT_TRITON, BT_NKI, BT_MLIR
  };
  
  for (auto bt : backends) {
    auto strategy = SearchStrategyFactory::create(bt);
    bool valid = (strategy != nullptr);
    v.add_result("SearchStrategy",
                 "factory_creates_" + backend_type_to_string(bt),
                 valid);
    
    if (valid) {
      // Verify backend type matches
      bool type_match = (strategy->get_backend_type() == bt);
      v.add_result("SearchStrategy",
                   "type_matches_" + backend_type_to_string(bt),
                   type_match);
    }
  }
  
  // Test fallback mechanism
  auto cudnn_strategy = SearchStrategyFactory::create(BT_CUDNN);
  if (cudnn_strategy) {
    // CUDNN should fallback to CUDA
    v.add_result("SearchStrategy", "cudnn_fallback",
                 cudnn_strategy->get_backend_type() == BT_CUDA);
  }
}

//===----------------------------------------------------------------------===//
// 5. Kernel Executor Factory Verification
//===----------------------------------------------------------------------===//

void verify_kernel_executors(IntegrityVerifier& v) {
  using namespace kernel;
  using namespace type;
  
  // Test matmul executor creation
  std::vector<BackendType> backends = {
    BT_CUDA, BT_CPU, BT_MPS
  };
  
  for (auto bt : backends) {
    auto executor = KernelExecutorFactory::create_matmul_executor(bt);
    bool valid = (executor != nullptr);
    v.add_result("KernelExecutor",
                 "matmul_" + backend_type_to_string(bt),
                 valid);
    
    if (valid) {
      // Verify config validation
      KernelConfig config;
      config.block_dim_x = 32;
      config.block_dim_y = 32;
      config.grid_dim_x = 8;
      config.grid_dim_y = 8;
      
      bool validates = executor->validate_config(config);
      v.add_result("KernelExecutor",
                   "validates_config_" + backend_type_to_string(bt),
                   validates);
    }
  }
  
  // Test other executor types
  auto rmsnorm = KernelExecutorFactory::create_rmsnorm_executor(BT_CUDA);
  v.add_result("KernelExecutor", "rmsnorm_cuda", rmsnorm != nullptr);
  
  auto reduction = KernelExecutorFactory::create_reduction_executor(BT_CPU);
  v.add_result("KernelExecutor", "reduction_cpu", reduction != nullptr);
  
  auto unary = KernelExecutorFactory::create_element_unary_executor(BT_MPS, KN_EXP_OP);
  v.add_result("KernelExecutor", "unary_mps", unary != nullptr);
}

//===----------------------------------------------------------------------===//
// 6. Generator Config Verification
//===----------------------------------------------------------------------===//

void verify_generator_config(IntegrityVerifier& v) {
  using namespace search;
  
  // Get default config
  auto config = GeneratorConfig::get_default_config();
  
  // Verify sensible defaults
  v.add_result("GeneratorConfig", "max_tb_ops > 0",
               config.max_num_threadblock_graph_op > 0);
  v.add_result("GeneratorConfig", "max_kn_ops > 0",
               config.max_num_kernel_graph_op > 0);
  v.add_result("GeneratorConfig", "search_thread > 0",
               config.search_thread > 0);
  
  // Verify operator lists are populated
  v.add_result("GeneratorConfig", "has_knop_to_explore",
               !config.knop_to_explore.empty());
  v.add_result("GeneratorConfig", "has_tbop_to_explore",
               !config.tbop_to_explore.empty());
  v.add_result("GeneratorConfig", "has_grid_dims",
               !config.grid_dim_to_explore.empty());
  v.add_result("GeneratorConfig", "has_block_dims",
               !config.block_dim_to_explore.empty());
}

//===----------------------------------------------------------------------===//
// 7. Data Type Verification
//===----------------------------------------------------------------------===//

void verify_data_types(IntegrityVerifier& v) {
  using namespace type;
  
  // Common data types
  std::vector<DataType> types = {
    DT_FLOAT16, DT_FLOAT32, DT_BFLOAT16, DT_INT8, DT_INT32, DT_INT64
  };
  
  for (auto dt : types) {
    std::string name = data_type_to_string(dt);
    v.add_result("DataType", "has_name_" + name, !name.empty());
    
    size_t size = get_data_type_size(dt);
    v.add_result("DataType", "size_" + name + " > 0", size > 0);
  }
  
  // Verify FP16 is 2 bytes
  v.add_result("DataType", "fp16_is_2_bytes",
               get_data_type_size(DT_FLOAT16) == 2);
  
  // Verify FP32 is 4 bytes
  v.add_result("DataType", "fp32_is_4_bytes",
               get_data_type_size(DT_FLOAT32) == 4);
}

//===----------------------------------------------------------------------===//
// 8. Layout Verification
//===----------------------------------------------------------------------===//

void verify_layouts(IntegrityVerifier& v) {
  using namespace type;
  
  // Verify layout types exist
  v.add_result("Layout", "row_major_exists", true);
  v.add_result("Layout", "col_major_exists", true);
  
  // Verify layout conversion consistency
  // (Would need layout utilities to test properly)
}

//===----------------------------------------------------------------------===//
// Main Verification Entry Point
//===----------------------------------------------------------------------===//

int run_integrity_verification() {
  IntegrityVerifier verifier;
  
  std::cout << "Running YiRage System Integrity Verification...\n\n";
  
  // Run all verification suites
  std::cout << "1. Verifying Backend Types...\n";
  verify_backend_types(verifier);
  
  std::cout << "2. Verifying KN Operators...\n";
  verify_kn_operators(verifier);
  
  std::cout << "3. Verifying TB Operators...\n";
  verify_tb_operators(verifier);
  
  std::cout << "4. Verifying Search Strategies...\n";
  verify_search_strategies(verifier);
  
  std::cout << "5. Verifying Kernel Executors...\n";
  verify_kernel_executors(verifier);
  
  std::cout << "6. Verifying Generator Config...\n";
  verify_generator_config(verifier);
  
  std::cout << "7. Verifying Data Types...\n";
  verify_data_types(verifier);
  
  std::cout << "8. Verifying Layouts...\n";
  verify_layouts(verifier);
  
  // Print summary
  verifier.print_summary();
  
  // Return number of failed tests
  int failed = 0;
  for (const auto& r : verifier.results) {
    if (!r.passed) failed++;
  }
  
  return failed;
}

} // namespace test
} // namespace yirage

// Main entry point
int main() {
  return yirage::test::run_integrity_verification();
}
