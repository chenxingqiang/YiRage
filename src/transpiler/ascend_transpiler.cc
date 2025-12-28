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
 * Ascend Transpiler Implementation
 * 
 * Key architectural decision: REUSE Triton transpiler via BiSheng compiler.
 * This provides the best maintainability and performance.
 */

#include "yirage/transpiler/ascend_transpiler.h"
#include "yirage/kernel/graph.h"
#include "yirage/type.h"

#ifdef YIRAGE_BACKEND_TRITON_ENABLED
#include "yirage/triton_transpiler/transpile.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace yirage {
namespace ascend_transpiler {

namespace {

// L1 buffer sizes for different Ascend devices (in bytes)
constexpr size_t L1_BUFFER_910 = 256 * 1024;    // 256 KB
constexpr size_t L1_BUFFER_910B = 512 * 1024;   // 512 KB
constexpr size_t L1_BUFFER_910B2 = 512 * 1024;  // 512 KB
constexpr size_t L1_BUFFER_310P = 128 * 1024;   // 128 KB
constexpr size_t L1_BUFFER_310 = 64 * 1024;     // 64 KB

// AI Core counts for different devices
constexpr int AI_CORES_910 = 32;
constexpr int AI_CORES_910B = 32;
constexpr int AI_CORES_910B2 = 48;
constexpr int AI_CORES_310P = 8;
constexpr int AI_CORES_310 = 2;

// Helper to check if BiSheng compiler is available
bool is_bisheng_available() {
  int result = std::system("which bisheng-triton > /dev/null 2>&1");
  return result == 0;
}

// Helper to check if Ascend C compiler is available
bool is_ascendc_available() {
  // Check CANN installation
  const char* cann_home = std::getenv("CANN_HOME");
  if (!cann_home) {
    cann_home = std::getenv("ASCEND_HOME");
  }
  if (!cann_home) {
    cann_home = "/usr/local/Ascend/ascend-toolkit/latest";
  }
  
  std::string compiler_path = std::string(cann_home) + "/compiler/bin/ascendc";
  std::ifstream check(compiler_path);
  return check.good();
}

// Get data type string for code generation
std::string get_dtype_str(type::DataType dtype) {
  switch (dtype) {
    case type::DT_FLOAT16: return "float16";
    case type::DT_BFLOAT16: return "bfloat16";
    case type::DT_FLOAT32: return "float32";
    case type::DT_DOUBLE: return "float64";
    case type::DT_INT8: return "int8";
    case type::DT_INT16: return "int16";
    case type::DT_INT32: return "int32";
    case type::DT_INT64: return "int64";
    default: return "float16";
  }
}

// Get Triton dtype string
std::string get_triton_dtype(type::DataType dtype) {
  switch (dtype) {
    case type::DT_FLOAT16: return "tl.float16";
    case type::DT_BFLOAT16: return "tl.bfloat16";
    case type::DT_FLOAT32: return "tl.float32";
    case type::DT_DOUBLE: return "tl.float64";
    case type::DT_INT8: return "tl.int8";
    case type::DT_INT16: return "tl.int16";
    case type::DT_INT32: return "tl.int32";
    case type::DT_INT64: return "tl.int64";
    default: return "tl.float16";
  }
}

} // anonymous namespace

// =============================================================================
// AscendTranspiler Implementation
// =============================================================================

AscendTranspiler::AscendTranspiler(kn::Graph const* graph,
                                   AscendTranspilerConfig const& config)
    : graph_(graph), config_(config) {
  
  // Auto-detect L1 buffer size if not specified
  if (config_.l1_buffer_size == 0) {
    switch (config_.device_type) {
      case AscendDeviceType::ASCEND_910:
        config_.l1_buffer_size = L1_BUFFER_910;
        break;
      case AscendDeviceType::ASCEND_910B:
        config_.l1_buffer_size = L1_BUFFER_910B;
        break;
      case AscendDeviceType::ASCEND_910B2:
        config_.l1_buffer_size = L1_BUFFER_910B2;
        break;
      case AscendDeviceType::ASCEND_310P:
        config_.l1_buffer_size = L1_BUFFER_310P;
        break;
      case AscendDeviceType::ASCEND_310:
        config_.l1_buffer_size = L1_BUFFER_310;
        break;
    }
  }
}

std::string AscendTranspiler::get_soc_version() const {
  switch (config_.device_type) {
    case AscendDeviceType::ASCEND_910: return "Ascend910";
    case AscendDeviceType::ASCEND_910B: return "Ascend910B";
    case AscendDeviceType::ASCEND_910B2: return "Ascend910B2";
    case AscendDeviceType::ASCEND_310P: return "Ascend310P";
    case AscendDeviceType::ASCEND_310: return "Ascend310";
    default: return "Ascend910B";
  }
}

AscendDeviceType AscendTranspiler::detect_device_type() {
  // Check environment variable
  const char* soc_version = std::getenv("ASCEND_SOC_VERSION");
  if (soc_version) {
    std::string soc(soc_version);
    if (soc.find("910B2") != std::string::npos) {
      return AscendDeviceType::ASCEND_910B2;
    } else if (soc.find("910B") != std::string::npos) {
      return AscendDeviceType::ASCEND_910B;
    } else if (soc.find("910") != std::string::npos) {
      return AscendDeviceType::ASCEND_910;
    } else if (soc.find("310P") != std::string::npos) {
      return AscendDeviceType::ASCEND_310P;
    } else if (soc.find("310") != std::string::npos) {
      return AscendDeviceType::ASCEND_310;
    }
  }
  
  // Try to detect via torch_npu if available
  int result = std::system(
      "python3 -c 'import torch_npu; print(torch_npu.npu.get_device_name(0))' "
      "2>/dev/null | grep -q '910B'");
  if (result == 0) {
    return AscendDeviceType::ASCEND_910B;
  }
  
  // Default to 910B (most common)
  return AscendDeviceType::ASCEND_910B;
}

AscendTranspilerConfig AscendTranspiler::get_recommended_config() {
  AscendTranspilerConfig config;
  
  // Auto-detect device
  config.device_type = detect_device_type();
  
  // Prefer Triton path if BiSheng is available
  if (is_bisheng_available()) {
    config.codegen_path = CodeGenPath::TRITON;
  } else if (is_ascendc_available() &&
             config.device_type != AscendDeviceType::ASCEND_910) {
    config.codegen_path = CodeGenPath::ASCEND_C;
  } else {
    config.codegen_path = CodeGenPath::TBE;
  }
  
  // Enable optimizations based on device
  config.use_cube_ops = true;
  config.enable_fusion = true;
  config.opt_level = 3;
  config.enable_fp16 = true;
  
  // Enable BF16 for 910B+
  if (config.device_type == AscendDeviceType::ASCEND_910B ||
      config.device_type == AscendDeviceType::ASCEND_910B2) {
    config.enable_bf16 = true;
  }
  
  return config;
}

AscendTranspileResult AscendTranspiler::generate_code() {
  // Route to appropriate code generation path
  switch (config_.codegen_path) {
    case CodeGenPath::TRITON:
      return transpile_via_triton();
    case CodeGenPath::ASCEND_C:
      return transpile_to_ascend_c();
    case CodeGenPath::TBE:
      return transpile_to_tbe();
    default:
      return transpile_via_triton();
  }
}

AscendTranspileResult AscendTranspiler::transpile_via_triton() {
  AscendTranspileResult result;
  result.path_used = CodeGenPath::TRITON;
  
  std::ostringstream code;
  
  // Generate Triton Python code
  code << "# YiRage Generated Triton Kernel for Ascend\n";
  code << "# Target: " << get_soc_version() << "\n";
  code << "# Compile with: bisheng-triton --target=" << get_soc_version() << "\n";
  code << "\n";
  code << "import triton\n";
  code << "import triton.language as tl\n";
  code << "\n";
  
  // Analyze graph to generate appropriate kernel
  bool has_matmul = false;
  bool has_reduction = false;
  bool has_elementwise = false;
  
  for (const auto& op : graph_->operators) {
    switch (op->op_type) {
      case type::KN_MATMUL_OP:
        has_matmul = true;
        break;
      case type::KN_REDUCTION_0_OP:
      case type::KN_REDUCTION_1_OP:
      case type::KN_REDUCTION_2_OP:
        has_reduction = true;
        break;
      case type::KN_ADD_OP:
      case type::KN_MUL_OP:
      case type::KN_DIV_OP:
      case type::KN_EXP_OP:
      case type::KN_SILU_OP:
      case type::KN_RELU_OP:
        has_elementwise = true;
        break;
      default:
        break;
    }
  }
  
  // Generate kernel based on operation types
  if (has_matmul) {
    // Generate matmul kernel optimized for Ascend Cube
    code << "@triton.jit\n";
    code << "def yirage_matmul_kernel(\n";
    code << "    a_ptr, b_ptr, c_ptr,\n";
    code << "    M, N, K,\n";
    code << "    stride_am, stride_ak,\n";
    code << "    stride_bk, stride_bn,\n";
    code << "    stride_cm, stride_cn,\n";
    code << "    BLOCK_M: tl.constexpr = 128,\n";
    code << "    BLOCK_N: tl.constexpr = 128,\n";
    code << "    BLOCK_K: tl.constexpr = 32,\n";
    code << "):\n";
    code << "    # Ascend Cube unit: native 16x16 tile size\n";
    code << "    # Use 128x128 tiles for better AI Core utilization\n";
    code << "    pid_m = tl.program_id(0)\n";
    code << "    pid_n = tl.program_id(1)\n";
    code << "    \n";
    code << "    # Compute block offsets\n";
    code << "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n";
    code << "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n";
    code << "    offs_k = tl.arange(0, BLOCK_K)\n";
    code << "    \n";
    code << "    # Initialize accumulator\n";
    code << "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n";
    code << "    \n";
    code << "    # Main loop over K dimension\n";
    code << "    for k in range(0, K, BLOCK_K):\n";
    code << "        # Load A and B tiles\n";
    code << "        a = tl.load(a_ptr + offs_m[:, None] * stride_am + "
            "(k + offs_k)[None, :] * stride_ak,\n";
    code << "                    mask=offs_m[:, None] < M)\n";
    code << "        b = tl.load(b_ptr + (k + offs_k)[:, None] * stride_bk + "
            "offs_n[None, :] * stride_bn,\n";
    code << "                    mask=offs_n[None, :] < N)\n";
    code << "        \n";
    code << "        # Compute matmul (Ascend Cube will handle this)\n";
    code << "        acc += tl.dot(a, b)\n";
    code << "    \n";
    code << "    # Store result\n";
    code << "    c = acc.to(tl.float16)\n";
    code << "    tl.store(c_ptr + offs_m[:, None] * stride_cm + "
            "offs_n[None, :] * stride_cn, c,\n";
    code << "             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))\n";
    code << "\n";
  }
  
  if (has_elementwise || has_reduction) {
    // Generate elementwise/reduction kernel
    code << "@triton.jit\n";
    code << "def yirage_elementwise_kernel(\n";
    code << "    x_ptr, y_ptr,\n";
    code << "    n_elements,\n";
    code << "    BLOCK_SIZE: tl.constexpr = 1024,\n";
    code << "):\n";
    code << "    # Use Ascend Vector unit for element-wise ops\n";
    code << "    pid = tl.program_id(0)\n";
    code << "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n";
    code << "    mask = offs < n_elements\n";
    code << "    \n";
    code << "    # Load input\n";
    code << "    x = tl.load(x_ptr + offs, mask=mask)\n";
    code << "    \n";
    
    // Add operations based on graph
    for (const auto& op : graph_->operators) {
      if (op->op_type == type::KN_SILU_OP) {
        code << "    # SiLU activation\n";
        code << "    y = x * tl.sigmoid(x)\n";
      } else if (op->op_type == type::KN_RELU_OP) {
        code << "    # ReLU activation\n";
        code << "    y = tl.maximum(x, 0.0)\n";
      } else if (op->op_type == type::KN_EXP_OP) {
        code << "    # Exponential\n";
        code << "    y = tl.exp(x)\n";
      }
    }
    
    code << "    \n";
    code << "    # Store output\n";
    code << "    tl.store(y_ptr + offs, y, mask=mask)\n";
    code << "\n";
  }
  
  // Generate RMSNorm kernel if needed
  for (const auto& op : graph_->operators) {
    if (op->op_type == type::KN_RMS_NORM_OP) {
      code << "@triton.jit\n";
      code << "def yirage_rms_norm_kernel(\n";
      code << "    x_ptr, y_ptr, weight_ptr,\n";
      code << "    n_cols,\n";
      code << "    eps: tl.constexpr = 1e-6,\n";
      code << "):\n";
      code << "    row_idx = tl.program_id(0)\n";
      code << "    cols = tl.arange(0, n_cols)\n";
      code << "    \n";
      code << "    # Load row\n";
      code << "    x = tl.load(x_ptr + row_idx * n_cols + cols)\n";
      code << "    w = tl.load(weight_ptr + cols)\n";
      code << "    \n";
      code << "    # Compute RMS\n";
      code << "    x_sq = x * x\n";
      code << "    mean_sq = tl.sum(x_sq, axis=0) / n_cols\n";
      code << "    rms = tl.sqrt(mean_sq + eps)\n";
      code << "    \n";
      code << "    # Normalize and scale\n";
      code << "    y = (x / rms) * w\n";
      code << "    \n";
      code << "    # Store\n";
      code << "    tl.store(y_ptr + row_idx * n_cols + cols, y)\n";
      code << "\n";
      break;
    }
  }
  
  result.code = code.str();
  result.compile_command = "bisheng-triton --target=" + get_soc_version() +
                          " --opt-level=" + std::to_string(config_.opt_level) +
                          (config_.enable_fp16 ? " --enable-fp16" : "") +
                          " -o kernel.so";
  
  return result;
}

AscendTranspileResult AscendTranspiler::transpile_to_ascend_c() {
  AscendTranspileResult result;
  result.path_used = CodeGenPath::ASCEND_C;
  
  std::ostringstream code;
  
  // Generate Ascend C kernel
  code << "/* YiRage Generated Ascend C Kernel */\n";
  code << "/* Target: " << get_soc_version() << " */\n";
  code << "\n";
  code << "#include \"kernel_operator.h\"\n";
  code << "#include \"aclrtlaunch_InvocationRecord.h\"\n";
  code << "\n";
  code << "using namespace AscendC;\n";
  code << "\n";
  
  // Check for matmul operations
  for (const auto& op : graph_->operators) {
    if (op->op_type == type::KN_MATMUL_OP) {
      code << "__aicore__ inline void yirage_matmul(\n";
      code << "    GM_ADDR half* a,\n";
      code << "    GM_ADDR half* b,\n";
      code << "    GM_ADDR half* c,\n";
      code << "    int M, int N, int K) {\n";
      code << "    \n";
      code << "    // Use Cube unit for matmul\n";
      code << "    constexpr int TILE_M = 128;\n";
      code << "    constexpr int TILE_N = 128;\n";
      code << "    constexpr int TILE_K = 32;\n";
      code << "    \n";
      code << "    // Allocate L1 buffer\n";
      code << "    __local__ half a_buf[TILE_M * TILE_K];\n";
      code << "    __local__ half b_buf[TILE_K * TILE_N];\n";
      code << "    __local__ float c_buf[TILE_M * TILE_N];\n";
      code << "    \n";
      code << "    // Get block indices\n";
      code << "    int block_m = GetBlockIdx() / ((N + TILE_N - 1) / TILE_N);\n";
      code << "    int block_n = GetBlockIdx() % ((N + TILE_N - 1) / TILE_N);\n";
      code << "    \n";
      code << "    // Initialize accumulator\n";
      code << "    for (int i = 0; i < TILE_M * TILE_N; i++) {\n";
      code << "        c_buf[i] = 0.0f;\n";
      code << "    }\n";
      code << "    \n";
      code << "    // Main loop\n";
      code << "    for (int k = 0; k < K; k += TILE_K) {\n";
      code << "        // Load A tile\n";
      code << "        DataCopy(a_buf, a + block_m * TILE_M * K + k, TILE_M * TILE_K);\n";
      code << "        // Load B tile\n";
      code << "        DataCopy(b_buf, b + k * N + block_n * TILE_N, TILE_K * TILE_N);\n";
      code << "        \n";
      code << "        // Cube matmul\n";
      code << "        Matmul(c_buf, a_buf, b_buf, TILE_M, TILE_N, TILE_K);\n";
      code << "    }\n";
      code << "    \n";
      code << "    // Convert and store\n";
      code << "    __local__ half c_out[TILE_M * TILE_N];\n";
      code << "    Cast(c_out, c_buf, TILE_M * TILE_N);\n";
      code << "    DataCopy(c + block_m * TILE_M * N + block_n * TILE_N, c_out, TILE_M * TILE_N);\n";
      code << "}\n";
      code << "\n";
    }
  }
  
  // Generate main kernel entry point
  code << "extern \"C\" __global__ __aicore__ void yirage_kernel(\n";
  code << "    GM_ADDR void* inputs[],\n";
  code << "    GM_ADDR void* outputs[],\n";
  code << "    GM_ADDR uint8_t* workspace,\n";
  code << "    int* dims) {\n";
  code << "    \n";
  code << "    // Kernel dispatch logic\n";
  code << "    // TODO: Generate based on graph analysis\n";
  code << "    \n";
  code << "}\n";
  
  result.code = code.str();
  
  // Get CANN path
  const char* cann_home = std::getenv("CANN_HOME");
  if (!cann_home) {
    cann_home = std::getenv("ASCEND_HOME");
  }
  if (!cann_home) {
    cann_home = "/usr/local/Ascend/ascend-toolkit/latest";
  }
  
  result.compile_command = std::string(cann_home) + "/compiler/bin/ascendc"
                          " --soc_version=" + get_soc_version() +
                          " -O" + std::to_string(config_.opt_level) +
                          " -o kernel.o";
  
  return result;
}

AscendTranspileResult AscendTranspiler::transpile_to_tbe() {
  AscendTranspileResult result;
  result.path_used = CodeGenPath::TBE;
  
  std::ostringstream code;
  
  // Generate TBE (Tensor Boost Engine) kernel
  code << "# YiRage Generated TBE Kernel\n";
  code << "# Target: " << get_soc_version() << " (TBE path)\n";
  code << "\n";
  code << "from te import tik\n";
  code << "from tbe import dsl\n";
  code << "import tbe.common.platform as platform\n";
  code << "\n";
  
  // TBE uses TIK (Tensor Iterator Kernel) DSL
  code << "def yirage_tbe_kernel():\n";
  code << "    # Initialize TIK container\n";
  code << "    tik_instance = tik.Tik()\n";
  code << "    \n";
  
  // Add placeholders for inputs/outputs
  code << "    # TODO: Define input/output tensors based on graph\n";
  code << "    \n";
  code << "    return tik_instance\n";
  code << "\n";
  
  result.code = code.str();
  result.compile_command = "te_compile --soc_version=" + get_soc_version();
  
  return result;
}

// =============================================================================
// Convenience Functions
// =============================================================================

AscendTranspileResult transpile(kn::Graph const* graph,
                                AscendTranspilerConfig const& config) {
  AscendTranspiler transpiler(graph, config);
  return transpiler.generate_code();
}

AscendTranspileResult transpile_auto(kn::Graph const* graph,
                                     AscendDeviceType device_type) {
  AscendTranspilerConfig config = AscendTranspiler::get_recommended_config();
  config.device_type = device_type;
  return transpile(graph, config);
}

} // namespace ascend_transpiler
} // namespace yirage
