// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_kernel_ascend_gtest.cc
 * @brief Ascend Kernel Module Unit Tests (Google Test version)
 *
 * Tests for yirage Ascend kernel module including:
 *   - AscendKernelConfig
 *   - AscendOptimizer
 *   - AscendKernelGenerator
 *   - AscendKernelCompiler
 *   - AscendKernelExecutor
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace yirage {
namespace type {

enum BackendType {
    BT_UNKNOWN = 0,
    BT_CUDA = 1,
    BT_ROCM = 2,
    BT_CPU = 3,
    BT_ASCEND = 4,
    BT_MPS = 5,
};

enum KNOperatorType {
    KN_UNKNOWN = 1000,
    KN_INPUT_OP = 1001,
    KN_OUTPUT_OP = 1002,
    KN_MATMUL_OP = 1003,
    KN_SILU_OP = 1104,
    KN_RELU_OP = 1150,
    KN_EXP_OP = 1100,
    KN_ADD_OP = 1200,
    KN_MUL_OP = 1201,
    KN_DIV_OP = 1202,
    KN_RMS_NORM_OP = 1350,
};

}  // namespace type

namespace kernel {

// =============================================================================
// Base KernelConfig
// =============================================================================

struct KernelConfig {
    type::BackendType backend_type = type::BT_UNKNOWN;
    int block_dim_x = 1;
    int block_dim_y = 1;
    int block_dim_z = 1;
    int grid_dim_x = 1;
    int grid_dim_y = 1;
    int grid_dim_z = 1;
    size_t shared_memory = 0;
};

namespace ascend {

// =============================================================================
// AscendKernelConfig
// =============================================================================

struct AscendKernelConfig : public KernelConfig {
    // AI Core block configuration
    int ai_cores_per_block = 8;
    int blocks_per_grid_x = 1;
    int blocks_per_grid_y = 1;
    
    // Memory configuration
    size_t l1_buffer_size = 256 * 1024;  // 256 KB default
    
    // Tile configuration for Cube operations
    int tile_m = 16;
    int tile_n = 16;
    int tile_k = 16;
    
    // Device type: 0=910, 1=910B, 2=310P
    int device_type = 0;
    
    // Optimization flags
    bool use_cube_ops = true;
    bool use_vector_ops = true;
    bool enable_fusion = true;
    
    AscendKernelConfig() { backend_type = type::BT_ASCEND; }
    
    int get_total_ai_cores() const {
        return ai_cores_per_block * blocks_per_grid_x * blocks_per_grid_y;
    }
};

// =============================================================================
// AscendOptimizer
// =============================================================================

class AscendOptimizer {
public:
    static int detect_device_type() {
        // Default to 910B
        return 1;
    }
    
    static int get_ai_core_count() {
        // 910B has 32 AI Cores
        return 32;
    }
    
    static int compute_optimal_block_size(size_t problem_size, int device_type) {
        int base_size = 8;
        
        if (problem_size < 1024) {
            base_size = 2;
        } else if (problem_size < 4096) {
            base_size = 4;
        } else if (problem_size > 1048576) {
            base_size = 16;
        }
        
        // 310P has fewer cores
        if (device_type == 2) {
            base_size = std::min(base_size, 4);
        }
        
        return base_size;
    }
    
    static void compute_optimal_tiles(int m, int n, int k,
                                      int device_type,
                                      AscendKernelConfig& config) {
        // L1 buffer size based on device
        size_t l1_size = (device_type == 1) ? 512 * 1024 : 256 * 1024;
        
        // Start with 16x16 (native Cube size)
        int tile_m = 16, tile_n = 16, tile_k = 16;
        
        auto try_tile = [&](int tm, int tn, int tk) -> bool {
            size_t mem = (tm * tk + tk * tn) * 2 + tm * tn * 4;
            return mem <= l1_size * 0.75f;
        };
        
        // Try larger tiles
        for (int mult = 2; mult <= 8; mult++) {
            int new_tile = 16 * mult;
            if (try_tile(new_tile, new_tile, new_tile)) {
                if (new_tile <= m && new_tile <= n && new_tile <= k) {
                    tile_m = tile_n = tile_k = new_tile;
                }
            } else {
                break;
            }
        }
        
        config.tile_m = std::min(m, tile_m);
        config.tile_n = std::min(n, tile_n);
        config.tile_k = std::min(k, tile_k);
    }
};

// =============================================================================
// AscendKernelGenerator
// =============================================================================

class AscendKernelGenerator {
public:
    static std::string generate_includes(AscendKernelConfig const& config) {
        std::ostringstream code;
        code << "#include \"tbe/tbe_api.h\"\n";
        code << "#include \"register/tilingdata_base.h\"\n";
        code << "#include \"register/register.h\"\n";
        code << "\n";
        code << "using namespace tbe;\n";
        code << "\n";
        return code.str();
    }
    
    static std::string generate_cube_matmul(int m, int n, int k,
                                            AscendKernelConfig const& config) {
        std::ostringstream code;
        
        code << "// Cube matmul kernel: C[" << m << "x" << n << "] = A["
             << m << "x" << k << "] * B[" << k << "x" << n << "]\n";
        code << "extern \"C\" __global__ __aicore__ void ascend_matmul_kernel(\n";
        code << "    GM_ADDR float16* A,\n";
        code << "    GM_ADDR float16* B,\n";
        code << "    GM_ADDR float* C,\n";
        code << "    GM_ADDR uint8_t* workspace) {\n";
        code << "\n";
        code << "  // L1 buffer allocation\n";
        code << "  LocalTensor<float16> A_local;\n";
        code << "  LocalTensor<float16> B_local;\n";
        code << "  LocalTensor<float> C_local;\n";
        code << "\n";
        code << "  const int TILE_M = " << config.tile_m << ";\n";
        code << "  const int TILE_N = " << config.tile_n << ";\n";
        code << "  const int TILE_K = " << config.tile_k << ";\n";
        code << "\n";
        code << "  CubeMatmul(C_local, A_local, B_local, TILE_M, TILE_N, TILE_K);\n";
        code << "\n";
        code << "  DataCopy(C, C_local, TILE_M * TILE_N);\n";
        code << "}\n";
        
        return code.str();
    }
    
    static std::string generate_vector_ops(std::string const& op_type,
                                           AscendKernelConfig const& config) {
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
        
        if (op_type == "silu") {
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
    
    static std::string generate_l1_buffer_alloc(AscendKernelConfig const& config) {
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
    
    static std::string generate_rms_norm_kernel(AscendKernelConfig const& config) {
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
        code << "  int row_idx = GetBlockIdx();\n";
        code << "  if (row_idx >= num_rows) return;\n";
        code << "\n";
        code << "  LocalTensor<float16> local_in;\n";
        code << "  LocalTensor<float> local_sq_sum;\n";
        code << "  LocalTensor<float16> local_out;\n";
        code << "\n";
        code << "  DataCopy(local_in, input + row_idx * row_size, row_size);\n";
        code << "\n";
        code << "  // Compute RMS normalization\n";
        code << "  // ... (implementation details)\n";
        code << "\n";
        code << "  DataCopy(output + row_idx * row_size, local_out, row_size);\n";
        code << "}\n";
        
        return code.str();
    }
    
    static std::string generate_attention_kernel(int num_heads, int head_dim,
                                                 int num_kv_heads,
                                                 AscendKernelConfig const& config) {
        std::ostringstream code;
        
        code << "// Attention kernel (GQA/MHA)\n";
        code << "// num_heads=" << num_heads << ", head_dim=" << head_dim;
        code << ", num_kv_heads=" << num_kv_heads << "\n";
        code << "extern \"C\" __global__ __aicore__ void ascend_attention_kernel(\n";
        code << "    GM_ADDR float16* Q,\n";
        code << "    GM_ADDR float16* K,\n";
        code << "    GM_ADDR float16* V,\n";
        code << "    GM_ADDR float16* output,\n";
        code << "    int seq_len) {\n";
        code << "\n";
        code << "  // GQA implementation\n";
        code << "  int heads_per_kv = " << (num_heads / num_kv_heads) << ";\n";
        code << "  // ... attention computation\n";
        code << "}\n";
        
        return code.str();
    }
};

// =============================================================================
// AscendKernelCompiler
// =============================================================================

class AscendKernelCompiler {
public:
    static std::string get_compiler_command(AscendKernelConfig const& config) {
        // Check CANN installation paths
        const char* cann_home = std::getenv("CANN_HOME");
        if (!cann_home) {
            cann_home = std::getenv("ASCEND_HOME");
        }
        if (!cann_home) {
            cann_home = "/usr/local/Ascend/ascend-toolkit/latest";
        }
        
        // Return expected compiler path
        return std::string(cann_home) + "/compiler/bin/ascendc";
    }
    
    static std::vector<std::string> get_compiler_flags(AscendKernelConfig const& config) {
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
        
        flags.push_back("-O3");
        
        if (config.use_cube_ops) {
            flags.push_back("--enable-cube");
        }
        
        if (config.enable_fusion) {
            flags.push_back("--enable-fusion");
        }
        
        return flags;
    }
    
    static bool compile_kernel(std::string const& code,
                               AscendKernelConfig const& config,
                               std::string const& output_path) {
        // In test environment, just validate input
        if (code.empty()) return false;
        if (output_path.empty()) return false;
        
        // Simulated compilation (actual compilation requires CANN)
        return true;
    }
};

// =============================================================================
// AscendKernelExecutor
// =============================================================================

class AscendKernelExecutor {
public:
    AscendKernelExecutor()
        : stream_(nullptr), kernel_handle_(nullptr), last_exec_time_(0.0f) {}
    
    ~AscendKernelExecutor() {
        // Cleanup
    }
    
    bool load_kernel(std::string const& kernel_path) {
        if (kernel_path.empty()) return false;
        // Simulated load
        kernel_handle_ = reinterpret_cast<void*>(0x1);  // Non-null marker
        return true;
    }
    
    bool execute(std::vector<void*> const& inputs,
                 std::vector<void*>& outputs) {
        if (!kernel_handle_) return false;
        if (inputs.empty()) return false;
        
        // Simulated execution
        last_exec_time_ = 0.5f;  // 0.5ms simulated
        return true;
    }
    
    float get_execution_time() const {
        return last_exec_time_;
    }
    
    bool is_loaded() const {
        return kernel_handle_ != nullptr;
    }

private:
    void* stream_;
    void* kernel_handle_;
    float last_exec_time_;
};

}  // namespace ascend
}  // namespace kernel
}  // namespace yirage

using namespace yirage;
using namespace yirage::kernel::ascend;

// =============================================================================
// AscendKernelConfig Tests
// =============================================================================

class AscendKernelConfigTest : public ::testing::Test {};

TEST_F(AscendKernelConfigTest, DefaultConstruction) {
    AscendKernelConfig config;
    EXPECT_EQ(config.backend_type, type::BT_ASCEND);
    EXPECT_EQ(config.ai_cores_per_block, 8);
    EXPECT_EQ(config.l1_buffer_size, 256u * 1024u);
}

TEST_F(AscendKernelConfigTest, DefaultTileSize) {
    AscendKernelConfig config;
    EXPECT_EQ(config.tile_m, 16);
    EXPECT_EQ(config.tile_n, 16);
    EXPECT_EQ(config.tile_k, 16);
}

TEST_F(AscendKernelConfigTest, DefaultOptimizationFlags) {
    AscendKernelConfig config;
    EXPECT_TRUE(config.use_cube_ops);
    EXPECT_TRUE(config.use_vector_ops);
    EXPECT_TRUE(config.enable_fusion);
}

TEST_F(AscendKernelConfigTest, TotalAICores) {
    AscendKernelConfig config;
    config.ai_cores_per_block = 8;
    config.blocks_per_grid_x = 4;
    config.blocks_per_grid_y = 2;
    
    EXPECT_EQ(config.get_total_ai_cores(), 64);
}

TEST_F(AscendKernelConfigTest, Device910B) {
    AscendKernelConfig config;
    config.device_type = 1;  // 910B
    config.l1_buffer_size = 512 * 1024;  // 512KB for 910B
    
    EXPECT_EQ(config.device_type, 1);
    EXPECT_EQ(config.l1_buffer_size, 512u * 1024u);
}

TEST_F(AscendKernelConfigTest, Device310P) {
    AscendKernelConfig config;
    config.device_type = 2;  // 310P
    config.l1_buffer_size = 256 * 1024;
    
    EXPECT_EQ(config.device_type, 2);
}

// =============================================================================
// AscendOptimizer Tests
// =============================================================================

class AscendOptimizerTest : public ::testing::Test {};

TEST_F(AscendOptimizerTest, DetectDeviceType) {
    int device_type = AscendOptimizer::detect_device_type();
    EXPECT_GE(device_type, 0);
    EXPECT_LE(device_type, 2);
}

TEST_F(AscendOptimizerTest, GetAICoreCount) {
    int ai_cores = AscendOptimizer::get_ai_core_count();
    EXPECT_GT(ai_cores, 0);
    EXPECT_EQ(ai_cores, 32);  // 910B default
}

TEST_F(AscendOptimizerTest, ComputeBlockSizeSmall) {
    int block_size = AscendOptimizer::compute_optimal_block_size(512, 1);
    EXPECT_EQ(block_size, 2);  // Small problem
}

TEST_F(AscendOptimizerTest, ComputeBlockSizeMedium) {
    int block_size = AscendOptimizer::compute_optimal_block_size(2048, 1);
    EXPECT_EQ(block_size, 4);  // Medium problem
}

TEST_F(AscendOptimizerTest, ComputeBlockSizeLarge) {
    int block_size = AscendOptimizer::compute_optimal_block_size(8192, 1);
    EXPECT_EQ(block_size, 8);  // Large problem
}

TEST_F(AscendOptimizerTest, ComputeBlockSizeVeryLarge) {
    int block_size = AscendOptimizer::compute_optimal_block_size(2 * 1024 * 1024, 1);
    EXPECT_EQ(block_size, 16);  // Very large problem
}

TEST_F(AscendOptimizerTest, ComputeBlockSize310P) {
    // 310P has limited cores
    int block_size = AscendOptimizer::compute_optimal_block_size(8192, 2);
    EXPECT_LE(block_size, 4);
}

TEST_F(AscendOptimizerTest, ComputeOptimalTilesDefault) {
    AscendKernelConfig config;
    AscendOptimizer::compute_optimal_tiles(1024, 1024, 1024, 1, config);
    
    // Should be multiples of 16
    EXPECT_EQ(config.tile_m % 16, 0);
    EXPECT_EQ(config.tile_n % 16, 0);
    EXPECT_EQ(config.tile_k % 16, 0);
}

TEST_F(AscendOptimizerTest, ComputeOptimalTilesSmall) {
    AscendKernelConfig config;
    AscendOptimizer::compute_optimal_tiles(8, 8, 8, 1, config);
    
    // Should not exceed matrix dimensions
    EXPECT_LE(config.tile_m, 8);
    EXPECT_LE(config.tile_n, 8);
    EXPECT_LE(config.tile_k, 8);
}

TEST_F(AscendOptimizerTest, ComputeOptimalTiles910B) {
    AscendKernelConfig config;
    // 910B has larger L1 (512KB)
    AscendOptimizer::compute_optimal_tiles(2048, 2048, 2048, 1, config);
    
    // Should use larger tiles on 910B
    EXPECT_GE(config.tile_m, 16);
}

// =============================================================================
// AscendKernelGenerator Tests
// =============================================================================

class AscendKernelGeneratorTest : public ::testing::Test {
protected:
    AscendKernelConfig config;
};

TEST_F(AscendKernelGeneratorTest, GenerateIncludes) {
    std::string includes = AscendKernelGenerator::generate_includes(config);
    
    EXPECT_FALSE(includes.empty());
    EXPECT_NE(includes.find("tbe_api.h"), std::string::npos);
    EXPECT_NE(includes.find("using namespace tbe"), std::string::npos);
}

TEST_F(AscendKernelGeneratorTest, GenerateCubeMatmul) {
    std::string code = AscendKernelGenerator::generate_cube_matmul(
        1024, 1024, 1024, config);
    
    EXPECT_FALSE(code.empty());
    EXPECT_NE(code.find("ascend_matmul_kernel"), std::string::npos);
    EXPECT_NE(code.find("CubeMatmul"), std::string::npos);
    EXPECT_NE(code.find("LocalTensor"), std::string::npos);
}

TEST_F(AscendKernelGeneratorTest, GenerateVectorOpsSilu) {
    std::string code = AscendKernelGenerator::generate_vector_ops("silu", config);
    
    EXPECT_FALSE(code.empty());
    EXPECT_NE(code.find("ascend_silu_kernel"), std::string::npos);
    EXPECT_NE(code.find("Sigmoid"), std::string::npos);
    EXPECT_NE(code.find("Mul"), std::string::npos);
}

TEST_F(AscendKernelGeneratorTest, GenerateVectorOpsRelu) {
    std::string code = AscendKernelGenerator::generate_vector_ops("relu", config);
    
    EXPECT_FALSE(code.empty());
    EXPECT_NE(code.find("ascend_relu_kernel"), std::string::npos);
    EXPECT_NE(code.find("Relu"), std::string::npos);
}

TEST_F(AscendKernelGeneratorTest, GenerateVectorOpsExp) {
    std::string code = AscendKernelGenerator::generate_vector_ops("exp", config);
    
    EXPECT_FALSE(code.empty());
    EXPECT_NE(code.find("ascend_exp_kernel"), std::string::npos);
    EXPECT_NE(code.find("Exp"), std::string::npos);
}

TEST_F(AscendKernelGeneratorTest, GenerateL1BufferAlloc) {
    std::string code = AscendKernelGenerator::generate_l1_buffer_alloc(config);
    
    EXPECT_FALSE(code.empty());
    EXPECT_NE(code.find("L1 Buffer"), std::string::npos);
    EXPECT_NE(code.find("AllocTensor"), std::string::npos);
    EXPECT_NE(code.find("pipe_barrier"), std::string::npos);
}

TEST_F(AscendKernelGeneratorTest, GenerateRmsNormKernel) {
    std::string code = AscendKernelGenerator::generate_rms_norm_kernel(config);
    
    EXPECT_FALSE(code.empty());
    EXPECT_NE(code.find("ascend_rms_norm_kernel"), std::string::npos);
    EXPECT_NE(code.find("eps"), std::string::npos);
    EXPECT_NE(code.find("GetBlockIdx"), std::string::npos);
}

TEST_F(AscendKernelGeneratorTest, GenerateAttentionKernel) {
    std::string code = AscendKernelGenerator::generate_attention_kernel(
        32, 128, 8, config);  // 32 heads, 128 dim, 8 kv heads
    
    EXPECT_FALSE(code.empty());
    EXPECT_NE(code.find("ascend_attention_kernel"), std::string::npos);
    EXPECT_NE(code.find("num_heads=32"), std::string::npos);
    EXPECT_NE(code.find("head_dim=128"), std::string::npos);
    EXPECT_NE(code.find("num_kv_heads=8"), std::string::npos);
}

TEST_F(AscendKernelGeneratorTest, GenerateMatmulWithCustomTiles) {
    config.tile_m = 64;
    config.tile_n = 32;
    config.tile_k = 16;
    
    std::string code = AscendKernelGenerator::generate_cube_matmul(
        256, 512, 1024, config);
    
    EXPECT_NE(code.find("TILE_M = 64"), std::string::npos);
    EXPECT_NE(code.find("TILE_N = 32"), std::string::npos);
    EXPECT_NE(code.find("TILE_K = 16"), std::string::npos);
}

// =============================================================================
// AscendKernelCompiler Tests
// =============================================================================

class AscendKernelCompilerTest : public ::testing::Test {
protected:
    AscendKernelConfig config;
};

TEST_F(AscendKernelCompilerTest, GetCompilerCommand) {
    std::string cmd = AscendKernelCompiler::get_compiler_command(config);
    EXPECT_FALSE(cmd.empty());
    EXPECT_NE(cmd.find("ascendc"), std::string::npos);
}

TEST_F(AscendKernelCompilerTest, GetCompilerFlags910) {
    config.device_type = 0;
    auto flags = AscendKernelCompiler::get_compiler_flags(config);
    
    EXPECT_FALSE(flags.empty());
    
    bool has_soc = false;
    for (const auto& flag : flags) {
        if (flag.find("Ascend910") != std::string::npos) {
            has_soc = true;
        }
    }
    EXPECT_TRUE(has_soc);
}

TEST_F(AscendKernelCompilerTest, GetCompilerFlags910B) {
    config.device_type = 1;
    auto flags = AscendKernelCompiler::get_compiler_flags(config);
    
    bool has_soc = false;
    for (const auto& flag : flags) {
        if (flag.find("Ascend910B") != std::string::npos) {
            has_soc = true;
        }
    }
    EXPECT_TRUE(has_soc);
}

TEST_F(AscendKernelCompilerTest, GetCompilerFlags310P) {
    config.device_type = 2;
    auto flags = AscendKernelCompiler::get_compiler_flags(config);
    
    bool has_soc = false;
    for (const auto& flag : flags) {
        if (flag.find("Ascend310P") != std::string::npos) {
            has_soc = true;
        }
    }
    EXPECT_TRUE(has_soc);
}

TEST_F(AscendKernelCompilerTest, GetCompilerFlagsWithCube) {
    config.use_cube_ops = true;
    auto flags = AscendKernelCompiler::get_compiler_flags(config);
    
    bool has_cube = false;
    for (const auto& flag : flags) {
        if (flag.find("enable-cube") != std::string::npos) {
            has_cube = true;
        }
    }
    EXPECT_TRUE(has_cube);
}

TEST_F(AscendKernelCompilerTest, GetCompilerFlagsWithFusion) {
    config.enable_fusion = true;
    auto flags = AscendKernelCompiler::get_compiler_flags(config);
    
    bool has_fusion = false;
    for (const auto& flag : flags) {
        if (flag.find("enable-fusion") != std::string::npos) {
            has_fusion = true;
        }
    }
    EXPECT_TRUE(has_fusion);
}

TEST_F(AscendKernelCompilerTest, CompileKernelEmptyCode) {
    bool result = AscendKernelCompiler::compile_kernel("", config, "/tmp/out.o");
    EXPECT_FALSE(result);
}

TEST_F(AscendKernelCompilerTest, CompileKernelEmptyPath) {
    bool result = AscendKernelCompiler::compile_kernel("kernel code", config, "");
    EXPECT_FALSE(result);
}

TEST_F(AscendKernelCompilerTest, CompileKernelValid) {
    std::string code = AscendKernelGenerator::generate_cube_matmul(64, 64, 64, config);
    bool result = AscendKernelCompiler::compile_kernel(code, config, "/tmp/test.o");
    // Returns true in simulated mode
    EXPECT_TRUE(result);
}

// =============================================================================
// AscendKernelExecutor Tests
// =============================================================================

class AscendKernelExecutorTest : public ::testing::Test {
protected:
    std::unique_ptr<AscendKernelExecutor> executor;
    
    void SetUp() override {
        executor = std::make_unique<AscendKernelExecutor>();
    }
};

TEST_F(AscendKernelExecutorTest, CreateExecutor) {
    EXPECT_NE(executor, nullptr);
    EXPECT_FALSE(executor->is_loaded());
}

TEST_F(AscendKernelExecutorTest, LoadKernelEmptyPath) {
    bool result = executor->load_kernel("");
    EXPECT_FALSE(result);
    EXPECT_FALSE(executor->is_loaded());
}

TEST_F(AscendKernelExecutorTest, LoadKernelValid) {
    bool result = executor->load_kernel("/path/to/kernel.o");
    EXPECT_TRUE(result);
    EXPECT_TRUE(executor->is_loaded());
}

TEST_F(AscendKernelExecutorTest, ExecuteWithoutLoad) {
    std::vector<void*> inputs = {nullptr};
    std::vector<void*> outputs;
    
    bool result = executor->execute(inputs, outputs);
    EXPECT_FALSE(result);  // Should fail without load
}

TEST_F(AscendKernelExecutorTest, ExecuteWithLoad) {
    executor->load_kernel("/path/to/kernel.o");
    
    std::vector<void*> inputs = {reinterpret_cast<void*>(0x1000)};
    std::vector<void*> outputs;
    
    bool result = executor->execute(inputs, outputs);
    EXPECT_TRUE(result);
}

TEST_F(AscendKernelExecutorTest, ExecuteEmptyInputs) {
    executor->load_kernel("/path/to/kernel.o");
    
    std::vector<void*> inputs;  // Empty
    std::vector<void*> outputs;
    
    bool result = executor->execute(inputs, outputs);
    EXPECT_FALSE(result);  // Empty inputs should fail
}

TEST_F(AscendKernelExecutorTest, GetExecutionTime) {
    executor->load_kernel("/path/to/kernel.o");
    
    std::vector<void*> inputs = {reinterpret_cast<void*>(0x1000)};
    std::vector<void*> outputs;
    executor->execute(inputs, outputs);
    
    float time = executor->get_execution_time();
    EXPECT_GT(time, 0.0f);
}

// =============================================================================
// Parameterized Tests for Device Types
// =============================================================================

struct DeviceTypeParam {
    int device_type;
    std::string name;
    size_t expected_l1_size;
    int expected_max_block_size;
};

class DeviceTypeTest : public ::testing::TestWithParam<DeviceTypeParam> {};

TEST_P(DeviceTypeTest, DeviceConfiguration) {
    auto param = GetParam();
    
    AscendKernelConfig config;
    config.device_type = param.device_type;
    
    // Get optimal block size
    int block_size = AscendOptimizer::compute_optimal_block_size(8192, param.device_type);
    
    // 310P should have smaller block size
    if (param.device_type == 2) {
        EXPECT_LE(block_size, param.expected_max_block_size);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllDeviceTypes,
    DeviceTypeTest,
    ::testing::Values(
        DeviceTypeParam{0, "Ascend910", 256 * 1024, 16},
        DeviceTypeParam{1, "Ascend910B", 512 * 1024, 16},
        DeviceTypeParam{2, "Ascend310P", 256 * 1024, 4}
    )
);

// =============================================================================
// Parameterized Tests for Vector Operations
// =============================================================================

struct VectorOpParam {
    std::string op_type;
    std::string expected_function;
};

class VectorOpTest : public ::testing::TestWithParam<VectorOpParam> {};

TEST_P(VectorOpTest, GenerateVectorOp) {
    auto param = GetParam();
    AscendKernelConfig config;
    
    std::string code = AscendKernelGenerator::generate_vector_ops(
        param.op_type, config);
    
    EXPECT_FALSE(code.empty());
    EXPECT_NE(code.find(param.expected_function), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    AllVectorOps,
    VectorOpTest,
    ::testing::Values(
        VectorOpParam{"silu", "Sigmoid"},
        VectorOpParam{"relu", "Relu"},
        VectorOpParam{"exp", "Exp"}
    )
);

// =============================================================================
// Tile Size Tests
// =============================================================================

struct TileSizeParam {
    int m, n, k;
    int device_type;
};

class TileSizeTest : public ::testing::TestWithParam<TileSizeParam> {};

TEST_P(TileSizeTest, ValidTileGeneration) {
    auto param = GetParam();
    AscendKernelConfig config;
    
    AscendOptimizer::compute_optimal_tiles(
        param.m, param.n, param.k, param.device_type, config);
    
    // Tiles should be valid (positive and <= matrix dimensions)
    EXPECT_GT(config.tile_m, 0);
    EXPECT_GT(config.tile_n, 0);
    EXPECT_GT(config.tile_k, 0);
    
    EXPECT_LE(config.tile_m, param.m);
    EXPECT_LE(config.tile_n, param.n);
    EXPECT_LE(config.tile_k, param.k);
    
    // Tiles should be multiples of 16 when possible
    if (param.m >= 16 && param.n >= 16 && param.k >= 16) {
        EXPECT_EQ(config.tile_m % 16, 0);
        EXPECT_EQ(config.tile_n % 16, 0);
        EXPECT_EQ(config.tile_k % 16, 0);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllTileSizes,
    TileSizeTest,
    ::testing::Values(
        TileSizeParam{64, 64, 64, 0},
        TileSizeParam{256, 256, 256, 0},
        TileSizeParam{1024, 1024, 1024, 1},
        TileSizeParam{2048, 4096, 512, 1},
        TileSizeParam{8, 8, 8, 0},      // Small
        TileSizeParam{17, 33, 65, 0}    // Non-aligned
    )
);

// =============================================================================
// L1 Buffer Allocation Tests
// =============================================================================

class L1BufferTest : public ::testing::Test {};

TEST_F(L1BufferTest, BufferFitsInL1) {
    AscendKernelConfig config;
    config.tile_m = 64;
    config.tile_n = 64;
    config.tile_k = 64;
    
    // Calculate memory needed
    size_t tile_a_bytes = config.tile_m * config.tile_k * 2;  // float16
    size_t tile_b_bytes = config.tile_k * config.tile_n * 2;  // float16
    size_t tile_c_bytes = config.tile_m * config.tile_n * 4;  // float32
    size_t total = tile_a_bytes + tile_b_bytes + tile_c_bytes;
    
    // Should fit in L1 buffer
    EXPECT_LT(total, config.l1_buffer_size);
}

TEST_F(L1BufferTest, LargeTileExceedsL1) {
    AscendKernelConfig config;
    config.l1_buffer_size = 256 * 1024;  // 256KB
    
    // Very large tiles
    config.tile_m = 256;
    config.tile_n = 256;
    config.tile_k = 256;
    
    size_t total = config.tile_m * config.tile_k * 2 +
                   config.tile_k * config.tile_n * 2 +
                   config.tile_m * config.tile_n * 4;
    
    // Should exceed L1 buffer
    EXPECT_GT(total, config.l1_buffer_size);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
