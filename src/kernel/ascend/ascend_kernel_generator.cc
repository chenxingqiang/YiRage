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

#include "yirage/kernel/ascend/ascend_kernel.h"

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef __ASCEND__
#include "acl/acl.h"
#include "acl/acl_rt.h"
#endif

namespace yirage {
namespace kernel {
namespace ascend {

std::string AscendKernelGenerator::generate_includes(
    AscendKernelConfig const &config) {
  std::ostringstream code;
  
  // TBE (Tensor Boost Engine) headers for Ascend 910
  // AscendC headers for newer chips
  code << "#include \"tbe/tbe_api.h\"\n";
  code << "#include \"register/tilingdata_base.h\"\n";
  code << "#include \"register/register.h\"\n";
  code << "\n";
  code << "using namespace tbe;\n";
  code << "\n";
  
  return code.str();
}

std::string AscendKernelGenerator::generate_cube_matmul(
    int m, int n, int k, AscendKernelConfig const &config) {
  std::ostringstream code;
  
  code << "// Cube matmul kernel: C[" << m << "x" << n << "] = A[" 
       << m << "x" << k << "] * B[" << k << "x" << n << "]\n";
  code << "extern \"C\" __global__ __aicore__ void ascend_matmul_kernel(\n";
  code << "    GM_ADDR float16* A,\n";
  code << "    GM_ADDR float16* B,\n";
  code << "    GM_ADDR float* C,\n";
  code << "    GM_ADDR uint8_t* workspace) {\n";
  code << "\n";
  
  // L1 buffer allocation
  code << "  // L1 buffer allocation\n";
  code << "  LocalTensor<float16> A_local;\n";
  code << "  LocalTensor<float16> B_local;\n";
  code << "  LocalTensor<float> C_local;\n";
  code << "\n";
  
  // Tile configuration
  code << "  const int TILE_M = " << config.tile_m << ";\n";
  code << "  const int TILE_N = " << config.tile_n << ";\n";
  code << "  const int TILE_K = " << config.tile_k << ";\n";
  code << "\n";
  
  // Cube matmul operation
  code << "  // Cube matmul operation\n";
  code << "  CubeMatmul(C_local, A_local, B_local, TILE_M, TILE_N, TILE_K);\n";
  code << "\n";
  
  code << "  // Copy result back to global memory\n";
  code << "  DataCopy(C, C_local, TILE_M * TILE_N);\n";
  code << "}\n";
  
  return code.str();
}

std::string AscendKernelGenerator::generate_vector_ops(
    std::string const &op_type, AscendKernelConfig const &config) {
  std::ostringstream code;
  
  code << "// Vector " << op_type << " kernel\n";
  code << "extern \"C\" __global__ __aicore__ void ascend_" << op_type << "_kernel(\n";
  code << "    GM_ADDR float16* input,\n";
  code << "    GM_ADDR float16* output,\n";
  code << "    int size) {\n";
  code << "\n";
  code << "  LocalTensor<float16> local_in;\n";
  code << "  LocalTensor<float16> local_out;\n";
  code << "\n";
  code << "  // Vector operation\n";
  
  if (op_type == "silu") {
    code << "  // SiLU: x * sigmoid(x)\n";
    code << "  LocalTensor<float16> sigmoid_out;\n";
    code << "  Sigmoid(sigmoid_out, local_in, size);\n";
    code << "  Mul(local_out, local_in, sigmoid_out, size);\n";
  } else if (op_type == "relu") {
    code << "  Relu(local_out, local_in, size);\n";
  } else if (op_type == "exp") {
    code << "  Exp(local_out, local_in, size);\n";
  }
  
  code << "\n";
  code << "  DataCopy(output, local_out, size);\n";
  code << "}\n";
  
  return code.str();
}

std::string AscendKernelGenerator::generate_l1_buffer_alloc(
    AscendKernelConfig const &config) {
  std::ostringstream code;
  
  size_t tile_a = config.tile_m * config.tile_k * 2;
  size_t tile_b = config.tile_k * config.tile_n * 2;
  size_t tile_c = config.tile_m * config.tile_n * 4;
  
  code << "  // L1 Buffer allocation (" << (config.l1_buffer_size / 1024) << " KB total)\n";
  code << "  pipe_barrier(PIPE_ALL);\n";
  code << "  AllocTensor(A_local, " << tile_a << ");\n";
  code << "  AllocTensor(B_local, " << tile_b << ");\n";
  code << "  AllocTensor(C_local, " << tile_c << ");\n";
  code << "  // Total: " << ((tile_a + tile_b + tile_c) / 1024) << " KB\n";
  
  return code.str();
}

std::string AscendKernelGenerator::generate_kernel_code(
    Graph const &graph, AscendKernelConfig const &config) {
  std::ostringstream code;
  
  // Generate includes
  code << generate_includes(config);
  code << "\n";
  
  // Analyze graph operators
  bool has_matmul = false;
  bool has_elementwise = false;
  bool has_rms_norm = false;
  int matmul_m = 0, matmul_n = 0, matmul_k = 0;
  std::vector<std::string> elementwise_ops;
  
  for (const auto &op : graph.operators) {
    switch (op->op_type) {
      case type::KN_MATMUL_OP:
        has_matmul = true;
        // Extract dimensions from input tensors
        if (op->input_tensors.size() >= 2) {
          auto const &A = op->input_tensors[0];
          auto const &B = op->input_tensors[1];
          if (A.num_dims >= 2 && B.num_dims >= 2) {
            matmul_m = A.dim[A.num_dims - 2];
            matmul_k = A.dim[A.num_dims - 1];
            matmul_n = B.dim[B.num_dims - 1];
          }
        }
        break;
        
      case type::KN_SILU_OP:
        has_elementwise = true;
        elementwise_ops.push_back("silu");
        break;
        
      case type::KN_RELU_OP:
        has_elementwise = true;
        elementwise_ops.push_back("relu");
        break;
        
      case type::KN_EXP_OP:
        has_elementwise = true;
        elementwise_ops.push_back("exp");
        break;
        
      case type::KN_RMS_NORM_OP:
        has_rms_norm = true;
        break;
        
      case type::KN_ADD_OP:
      case type::KN_MUL_OP:
      case type::KN_DIV_OP:
        has_elementwise = true;
        break;
        
      default:
        break;
    }
  }
  
  // Generate kernels based on graph analysis
  if (has_matmul) {
    if (matmul_m == 0) matmul_m = 1024;
    if (matmul_n == 0) matmul_n = 1024;
    if (matmul_k == 0) matmul_k = 1024;
    code << generate_cube_matmul(matmul_m, matmul_n, matmul_k, config);
    code << "\n";
  }
  
  if (has_elementwise) {
    for (const auto &op_type : elementwise_ops) {
      code << generate_vector_ops(op_type, config);
      code << "\n";
    }
  }
  
  if (has_rms_norm) {
    code << generate_rms_norm_kernel(config);
    code << "\n";
  }
  
  // Generate combined kernel entry point
  code << "// Main kernel dispatcher\n";
  code << "extern \"C\" __global__ __aicore__ void yirage_ascend_main(\n";
  code << "    GM_ADDR void* inputs[],\n";
  code << "    GM_ADDR void* outputs[],\n";
  code << "    GM_ADDR uint8_t* workspace,\n";
  code << "    int* op_sequence) {\n";
  code << "  // Dispatch to appropriate kernel based on op_sequence\n";
  code << "  // AI Cores: " << config.ai_cores_per_block << " per block\n";
  code << "  // L1 Buffer: " << (config.l1_buffer_size / 1024) << " KB\n";
  code << "}\n";
  
  return code.str();
}

std::string AscendKernelGenerator::generate_rms_norm_kernel(
    AscendKernelConfig const &config) {
  std::ostringstream code;
  
  code << "// RMS Normalization kernel (Vector Unit)\n";
  code << "extern \"C\" __global__ __aicore__ void ascend_rms_norm_kernel(\n";
  code << "    GM_ADDR float16* input,\n";
  code << "    GM_ADDR float16* weight,\n";
  code << "    GM_ADDR float16* output,\n";
  code << "    int num_rows,\n";
  code << "    int row_size,\n";
  code << "    float eps) {\n";
  code << "\n";
  code << "  // Each AI Core processes one or more rows\n";
  code << "  int row_idx = GetBlockIdx();\n";
  code << "  if (row_idx >= num_rows) return;\n";
  code << "\n";
  code << "  LocalTensor<float16> local_in;\n";
  code << "  LocalTensor<float> local_sq_sum;\n";
  code << "  LocalTensor<float16> local_out;\n";
  code << "\n";
  code << "  // Load row to L1 buffer\n";
  code << "  DataCopy(local_in, input + row_idx * row_size, row_size);\n";
  code << "\n";
  code << "  // Compute sum of squares\n";
  code << "  float sq_sum = 0.0f;\n";
  code << "  for (int i = 0; i < row_size; i++) {\n";
  code << "    float val = static_cast<float>(local_in[i]);\n";
  code << "    sq_sum += val * val;\n";
  code << "  }\n";
  code << "\n";
  code << "  // Compute RMS\n";
  code << "  float rms = sqrtf(sq_sum / row_size + eps);\n";
  code << "\n";
  code << "  // Normalize and apply weight\n";
  code << "  for (int i = 0; i < row_size; i++) {\n";
  code << "    float val = static_cast<float>(local_in[i]);\n";
  code << "    float w = static_cast<float>(weight[i]);\n";
  code << "    local_out[i] = static_cast<float16>((val / rms) * w);\n";
  code << "  }\n";
  code << "\n";
  code << "  // Store result\n";
  code << "  DataCopy(output + row_idx * row_size, local_out, row_size);\n";
  code << "}\n";
  
  return code.str();
}

// =============================================================================
// AscendKernelCompiler Implementation
// =============================================================================

bool AscendKernelCompiler::compile_kernel(std::string const &code,
                                          AscendKernelConfig const &config,
                                          std::string const &output_path) {
  // Get compiler command
  std::string compiler_cmd = get_compiler_command(config);
  if (compiler_cmd.empty()) {
    std::cerr << "Ascend compiler not found" << std::endl;
    return false;
  }
  
  // Write source to temp file
  std::string source_path = "/tmp/yirage_ascend_kernel_" + 
      std::to_string(std::hash<std::string>{}(code)) + ".cpp";
  
  std::ofstream source_file(source_path);
  if (!source_file.good()) {
    std::cerr << "Failed to create temp source file" << std::endl;
    return false;
  }
  source_file << code;
  source_file.close();
  
  // Build compilation command
  std::ostringstream cmd;
  cmd << compiler_cmd << " -c " << source_path << " -o " << output_path;
  
  // Add flags
  for (const auto &flag : get_compiler_flags(config)) {
    cmd << " " << flag;
  }
  
  // Execute
  int result = std::system(cmd.str().c_str());
  
  // Cleanup temp file
  std::remove(source_path.c_str());
  
  if (result != 0) {
    std::cerr << "Ascend kernel compilation failed" << std::endl;
    return false;
  }
  
  return true;
}

std::string AscendKernelCompiler::get_compiler_command(
    AscendKernelConfig const &config) {
  
  // Check CANN installation paths
  const char *cann_home = std::getenv("CANN_HOME");
  if (!cann_home) {
    cann_home = std::getenv("ASCEND_HOME");
  }
  if (!cann_home) {
    cann_home = "/usr/local/Ascend/ascend-toolkit/latest";
  }
  
  // AscendC compiler for 910B+
  std::string ascendc_path = std::string(cann_home) + "/compiler/bin/ascendc";
  std::ifstream check_ascendc(ascendc_path);
  if (check_ascendc.good()) {
    return ascendc_path;
  }
  
  // Fall back to TBE compiler for 910
  std::string tbe_path = std::string(cann_home) + "/tools/te/bin/te_compile";
  std::ifstream check_tbe(tbe_path);
  if (check_tbe.good()) {
    return tbe_path;
  }
  
  return "";
}

std::vector<std::string> AscendKernelCompiler::get_compiler_flags(
    AscendKernelConfig const &config) {
  
  std::vector<std::string> flags;
  
  // SOC version
  std::string soc;
  switch (config.device_type) {
    case 0: soc = "Ascend910"; break;
    case 1: soc = "Ascend910B"; break;
    case 2: soc = "Ascend310P"; break;
    default: soc = "Ascend910B";
  }
  flags.push_back("--soc_version=" + soc);
  
  // Optimization level
  flags.push_back("-O3");
  
  // Enable Cube operations if configured
  if (config.use_cube_ops) {
    flags.push_back("--enable-cube");
  }
  
  // Enable operator fusion
  if (config.enable_fusion) {
    flags.push_back("--enable-fusion");
  }
  
  return flags;
}

// =============================================================================
// AscendKernelExecutor Implementation
// =============================================================================

AscendKernelExecutor::AscendKernelExecutor() 
    : stream_(nullptr), kernel_handle_(nullptr), last_exec_time_(0.0f) {
#ifdef __ASCEND__
  // Initialize ACL stream
  aclError err = aclrtCreateStream(reinterpret_cast<aclrtStream*>(&stream_));
  if (err != ACL_SUCCESS) {
    std::cerr << "Failed to create ACL stream: " << err << std::endl;
  }
#endif
}

AscendKernelExecutor::~AscendKernelExecutor() {
#ifdef __ASCEND__
  if (stream_) {
    aclrtDestroyStream(reinterpret_cast<aclrtStream>(stream_));
    stream_ = nullptr;
  }
  if (kernel_handle_) {
    // Unload kernel
    kernel_handle_ = nullptr;
  }
#endif
}

bool AscendKernelExecutor::load_kernel(std::string const &kernel_path) {
#ifdef __ASCEND__
  // Load compiled kernel binary
  std::ifstream kernel_file(kernel_path, std::ios::binary);
  if (!kernel_file.good()) {
    std::cerr << "Failed to open kernel file: " << kernel_path << std::endl;
    return false;
  }
  
  // Read kernel binary
  kernel_file.seekg(0, std::ios::end);
  size_t kernel_size = kernel_file.tellg();
  kernel_file.seekg(0, std::ios::beg);
  
  std::vector<char> kernel_data(kernel_size);
  kernel_file.read(kernel_data.data(), kernel_size);
  kernel_file.close();
  
  // Register kernel with ACL runtime
  // Note: Actual implementation depends on CANN version
  // This is a simplified placeholder
  kernel_handle_ = malloc(kernel_size);
  if (kernel_handle_) {
    memcpy(kernel_handle_, kernel_data.data(), kernel_size);
    return true;
  }
  
  return false;
#else
  std::cerr << "Ascend runtime not available" << std::endl;
  return false;
#endif
}

bool AscendKernelExecutor::execute(std::vector<void*> const &inputs,
                                    std::vector<void*> &outputs) {
#ifdef __ASCEND__
  if (!kernel_handle_ || !stream_) {
    std::cerr << "Kernel not loaded or stream not initialized" << std::endl;
    return false;
  }
  
  // Create timing events
  aclrtEvent start_event, end_event;
  aclrtCreateEvent(&start_event);
  aclrtCreateEvent(&end_event);
  
  // Record start time
  aclrtRecordEvent(start_event, reinterpret_cast<aclrtStream>(stream_));
  
  // Launch kernel
  // Note: Actual launch API depends on kernel registration method
  // This is a conceptual implementation
  
  // aclrtLaunchKernel(kernel_handle_, grid_dim, block_dim,
  //                   args, arg_sizes, stream_);
  
  // Record end time
  aclrtRecordEvent(end_event, reinterpret_cast<aclrtStream>(stream_));
  
  // Synchronize
  aclrtSynchronizeStream(reinterpret_cast<aclrtStream>(stream_));
  
  // Calculate execution time
  float elapsed_ms = 0.0f;
  aclrtEventElapsedTime(&elapsed_ms, start_event, end_event);
  last_exec_time_ = elapsed_ms;
  
  // Cleanup events
  aclrtDestroyEvent(start_event);
  aclrtDestroyEvent(end_event);
  
  return true;
#else
  std::cerr << "Ascend runtime not available" << std::endl;
  return false;
#endif
}

float AscendKernelExecutor::get_execution_time() const {
  return last_exec_time_;
}

} // namespace ascend
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED

