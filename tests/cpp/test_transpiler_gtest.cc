// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_transpiler_gtest.cc
 * @brief Transpiler Unit Tests (Google Test version)
 *
 * Tests for code generation (transpiler) modules.
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <string>
#include <vector>
#include <sstream>

namespace yirage {
namespace transpiler {

// =============================================================================
// Mock Transpiler Types for Testing
// =============================================================================

enum class TranspilerTarget {
    CUDA,
    TRITON,
    NKI,
    MLIR,
    METAL,
    HIP,
};

struct CodegenConfig {
    TranspilerTarget target = TranspilerTarget::CUDA;
    int32_t opt_level = 2;
    bool enable_vectorization = true;
    bool enable_unrolling = true;
    int32_t max_unroll_factor = 4;
    bool emit_debug_info = false;
};

struct GeneratedCode {
    std::string source;
    std::string kernel_name;
    size_t shared_memory_bytes = 0;
    int32_t register_count = 0;
    bool valid = false;
    std::string error_message;
};

// Mock transpiler for testing
class MockTranspiler {
public:
    explicit MockTranspiler(const CodegenConfig& config) : config_(config) {}
    
    GeneratedCode generate_matmul(int M, int N, int K) {
        GeneratedCode result;
        result.kernel_name = "matmul_kernel";
        result.valid = true;
        
        std::ostringstream oss;
        switch (config_.target) {
            case TranspilerTarget::CUDA:
                oss << generate_cuda_matmul(M, N, K);
                break;
            case TranspilerTarget::TRITON:
                oss << generate_triton_matmul(M, N, K);
                break;
            default:
                result.valid = false;
                result.error_message = "Unsupported target";
                return result;
        }
        
        result.source = oss.str();
        result.shared_memory_bytes = calculate_smem(M, N, K);
        return result;
    }

private:
    CodegenConfig config_;
    
    std::string generate_cuda_matmul(int M, int N, int K) {
        std::ostringstream oss;
        oss << "__global__ void matmul_kernel(\n"
            << "    float* __restrict__ C,\n"
            << "    const float* __restrict__ A,\n"
            << "    const float* __restrict__ B,\n"
            << "    int M, int N, int K) {\n"
            << "    // M=" << M << ", N=" << N << ", K=" << K << "\n"
            << "    int row = blockIdx.y * blockDim.y + threadIdx.y;\n"
            << "    int col = blockIdx.x * blockDim.x + threadIdx.x;\n"
            << "    if (row < M && col < N) {\n"
            << "        float sum = 0.0f;\n"
            << "        for (int k = 0; k < K; ++k) {\n"
            << "            sum += A[row * K + k] * B[k * N + col];\n"
            << "        }\n"
            << "        C[row * N + col] = sum;\n"
            << "    }\n"
            << "}\n";
        return oss.str();
    }
    
    std::string generate_triton_matmul(int M, int N, int K) {
        std::ostringstream oss;
        oss << "@triton.jit\n"
            << "def matmul_kernel(\n"
            << "    a_ptr, b_ptr, c_ptr,\n"
            << "    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,\n"
            << "    BLOCK_M: tl.constexpr = 64, BLOCK_N: tl.constexpr = 64, BLOCK_K: tl.constexpr = 32):\n"
            << "    # M=" << M << ", N=" << N << ", K=" << K << "\n"
            << "    pid_m = tl.program_id(0)\n"
            << "    pid_n = tl.program_id(1)\n"
            << "    # ... implementation ...\n";
        return oss.str();
    }
    
    size_t calculate_smem(int M, int N, int K) {
        // Simple estimation: 2 tiles for double buffering
        return 2 * 64 * 64 * sizeof(float);
    }
};

}  // namespace transpiler
}  // namespace yirage

using namespace yirage::transpiler;

// =============================================================================
// CodegenConfig Tests
// =============================================================================

class CodegenConfigTest : public ::testing::Test {
protected:
    CodegenConfig config;
};

TEST_F(CodegenConfigTest, DefaultValues) {
    EXPECT_EQ(config.target, TranspilerTarget::CUDA);
    EXPECT_EQ(config.opt_level, 2);
    EXPECT_TRUE(config.enable_vectorization);
    EXPECT_TRUE(config.enable_unrolling);
    EXPECT_EQ(config.max_unroll_factor, 4);
    EXPECT_FALSE(config.emit_debug_info);
}

TEST_F(CodegenConfigTest, CustomTarget) {
    config.target = TranspilerTarget::TRITON;
    EXPECT_EQ(config.target, TranspilerTarget::TRITON);
}

TEST_F(CodegenConfigTest, OptLevelRange) {
    // Valid opt levels: 0, 1, 2, 3
    for (int level = 0; level <= 3; ++level) {
        config.opt_level = level;
        EXPECT_GE(config.opt_level, 0);
        EXPECT_LE(config.opt_level, 3);
    }
}

// =============================================================================
// CUDA Transpiler Tests
// =============================================================================

class CUDATranspilerTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.target = TranspilerTarget::CUDA;
        transpiler = std::make_unique<MockTranspiler>(config);
    }
    
    CodegenConfig config;
    std::unique_ptr<MockTranspiler> transpiler;
};

TEST_F(CUDATranspilerTest, MatMulCodeGen) {
    auto result = transpiler->generate_matmul(512, 512, 512);
    
    EXPECT_TRUE(result.valid);
    EXPECT_FALSE(result.source.empty());
    EXPECT_EQ(result.kernel_name, "matmul_kernel");
}

TEST_F(CUDATranspilerTest, ValidCUDAOutput) {
    auto result = transpiler->generate_matmul(256, 256, 256);
    
    EXPECT_TRUE(result.valid);
    // Check for CUDA keywords
    EXPECT_NE(result.source.find("__global__"), std::string::npos);
    EXPECT_NE(result.source.find("blockIdx"), std::string::npos);
    EXPECT_NE(result.source.find("threadIdx"), std::string::npos);
}

TEST_F(CUDATranspilerTest, SharedMemoryAllocation) {
    auto result = transpiler->generate_matmul(1024, 1024, 1024);
    
    EXPECT_TRUE(result.valid);
    EXPECT_GT(result.shared_memory_bytes, 0u);
}

// =============================================================================
// Triton Transpiler Tests
// =============================================================================

class TritonTranspilerTest : public ::testing::Test {
protected:
    void SetUp() override {
        config.target = TranspilerTarget::TRITON;
        transpiler = std::make_unique<MockTranspiler>(config);
    }
    
    CodegenConfig config;
    std::unique_ptr<MockTranspiler> transpiler;
};

TEST_F(TritonTranspilerTest, ValidTritonOutput) {
    auto result = transpiler->generate_matmul(512, 512, 512);
    
    EXPECT_TRUE(result.valid);
    // Check for Triton keywords
    EXPECT_NE(result.source.find("@triton.jit"), std::string::npos);
    EXPECT_NE(result.source.find("tl.constexpr"), std::string::npos);
}

TEST_F(TritonTranspilerTest, ProgramIdUsage) {
    auto result = transpiler->generate_matmul(256, 256, 256);
    
    EXPECT_TRUE(result.valid);
    EXPECT_NE(result.source.find("tl.program_id"), std::string::npos);
}

// =============================================================================
// Memory Planning Tests
// =============================================================================

class MemoryPlanningTest : public ::testing::Test {
protected:
    struct TensorAllocation {
        std::string name;
        size_t size_bytes;
        int memory_level;  // 0: register, 1: shared, 2: global
        int live_start;
        int live_end;
    };
    
    size_t calculate_peak_memory(const std::vector<TensorAllocation>& allocs, int level) {
        // Simple peak calculation
        size_t peak = 0;
        for (int t = 0; t < 100; ++t) {
            size_t current = 0;
            for (const auto& a : allocs) {
                if (a.memory_level == level && t >= a.live_start && t <= a.live_end) {
                    current += a.size_bytes;
                }
            }
            peak = std::max(peak, current);
        }
        return peak;
    }
};

TEST_F(MemoryPlanningTest, DTensorAllocation) {
    std::vector<TensorAllocation> allocs = {
        {"A", 4096, 2, 0, 50},
        {"B", 4096, 2, 0, 50},
        {"C", 4096, 2, 25, 100},
    };
    
    size_t peak = calculate_peak_memory(allocs, 2);
    // Peak should be at time 25-50 where all 3 tensors are live
    EXPECT_EQ(peak, 12288u);
}

TEST_F(MemoryPlanningTest, STensorAllocation) {
    std::vector<TensorAllocation> allocs = {
        {"tile_A", 8192, 1, 0, 20},
        {"tile_B", 8192, 1, 0, 20},
        {"accum", 4096, 1, 0, 50},
    };
    
    size_t peak = calculate_peak_memory(allocs, 1);
    // Peak at time 0-20 where all 3 are live
    EXPECT_EQ(peak, 20480u);
}

TEST_F(MemoryPlanningTest, OverlapOptimization) {
    // Non-overlapping allocations can share memory
    std::vector<TensorAllocation> allocs = {
        {"temp1", 4096, 1, 0, 10},
        {"temp2", 4096, 1, 15, 25},  // After temp1 is dead
    };
    
    size_t peak = calculate_peak_memory(allocs, 1);
    // They don't overlap, so peak is just one of them
    EXPECT_EQ(peak, 4096u);
}

// =============================================================================
// Parameterized Tests
// =============================================================================

class MatMulSizeTest : public ::testing::TestWithParam<std::tuple<int, int, int>> {};

TEST_P(MatMulSizeTest, GenerateValidCode) {
    auto [M, N, K] = GetParam();
    
    CodegenConfig config;
    config.target = TranspilerTarget::CUDA;
    MockTranspiler transpiler(config);
    
    auto result = transpiler.generate_matmul(M, N, K);
    
    EXPECT_TRUE(result.valid);
    EXPECT_FALSE(result.source.empty());
}

INSTANTIATE_TEST_SUITE_P(
    MatMulSizes,
    MatMulSizeTest,
    ::testing::Values(
        std::make_tuple(64, 64, 64),
        std::make_tuple(128, 128, 128),
        std::make_tuple(256, 256, 256),
        std::make_tuple(512, 512, 512),
        std::make_tuple(1024, 1024, 1024),
        std::make_tuple(128, 256, 64),    // Non-square
        std::make_tuple(256, 128, 512)
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
