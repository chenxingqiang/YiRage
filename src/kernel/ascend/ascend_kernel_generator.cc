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

#include <sstream>

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
  
  // TODO: Analyze graph and generate appropriate kernel
  // For now, generate a simple matmul kernel as placeholder
  code << generate_cube_matmul(1024, 1024, 1024, config);
  
  return code.str();
}

} // namespace ascend
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED

