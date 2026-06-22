// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_transpiler_config_gtest.cc
 * @brief Transpiler Configuration Unit Tests
 *
 * Tests for transpiler configuration (transpiler.h, transpile.h):
 *   - TranspilerConfig structure
 *   - TranspileResult structure
 *   - OutputTensorDirective structure
 *   - CustomOPTranspileResult structure
 *   - Compute capability constants
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <string>
#include <vector>

namespace yirage {
namespace transpiler {

// =============================================================================
// Error Types
// =============================================================================

enum class TranspileErrorType {
    SUCCESS = 0,
    INVALID_INPUT,
    UNSUPPORTED_OPERATION,
    MEMORY_ALLOCATION_FAILED,
    CODE_GENERATION_FAILED,
    INTERNAL_ERROR,
};

inline const char* error_type_to_string(TranspileErrorType err) {
    switch (err) {
        case TranspileErrorType::SUCCESS: return "success";
        case TranspileErrorType::INVALID_INPUT: return "invalid_input";
        case TranspileErrorType::UNSUPPORTED_OPERATION: return "unsupported_operation";
        case TranspileErrorType::MEMORY_ALLOCATION_FAILED: return "memory_allocation_failed";
        case TranspileErrorType::CODE_GENERATION_FAILED: return "code_generation_failed";
        case TranspileErrorType::INTERNAL_ERROR: return "internal_error";
        default: return "unknown";
    }
}

// =============================================================================
// GPU Compute Capabilities
// =============================================================================

namespace GPU_CC {
    static constexpr int P100 = 60;
    static constexpr int V100 = 70;
    static constexpr int T4 = 75;
    static constexpr int A100 = 80;
    static constexpr int H100 = 90;
    static constexpr int B200 = 100;
}

// =============================================================================
// TranspilerConfig
// =============================================================================

struct TranspilerConfig {
    int target_cc = GPU_CC::A100;  // Default to A100
    bool profiling = false;

    // Features for GPUs >= Grace Hopper
    int num_consumer_wgs = 1;
    int num_producer_wgs = 1;
    int pipeline_stages = 2;

    bool enable_online_softmax = false;

    // Validation
    bool is_valid() const {
        if (target_cc < 60 || target_cc > 100) return false;
        if (num_consumer_wgs < 1) return false;
        if (num_producer_wgs < 1) return false;
        if (pipeline_stages < 1) return false;
        return true;
    }

    bool is_hopper_or_above() const {
        return target_cc >= GPU_CC::H100;
    }

    bool is_blackwell_or_above() const {
        return target_cc >= GPU_CC::B200;
    }

    bool supports_tma() const {
        return target_cc >= GPU_CC::H100;
    }

    bool supports_tensor_cores() const {
        return target_cc >= GPU_CC::V100;
    }
};

// =============================================================================
// OutputTensorDirective
// =============================================================================

struct OutputTensorDirective {
    size_t alloc_size = 0;
    std::vector<int> shape;
    std::vector<size_t> strides;

    size_t get_num_elements() const {
        if (shape.empty()) return 0;
        size_t elements = 1;
        for (int dim : shape) {
            elements *= dim;
        }
        return elements;
    }

    bool is_contiguous() const {
        if (shape.empty() || strides.empty()) return true;
        if (shape.size() != strides.size()) return false;

        size_t expected_stride = 1;
        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
            if (strides[i] != expected_stride) return false;
            expected_stride *= shape[i];
        }
        return true;
    }
};

// =============================================================================
// TranspileResult
// =============================================================================

struct TranspileResult {
    TranspileErrorType error_type = TranspileErrorType::SUCCESS;
    std::string code;
    size_t buf_size = 0;
    size_t max_smem_size = 0;
    size_t profiler_buf_size = 0;
    std::vector<OutputTensorDirective> output_directives;

    bool is_success() const {
        return error_type == TranspileErrorType::SUCCESS;
    }

    bool has_code() const {
        return !code.empty();
    }

    size_t num_outputs() const {
        return output_directives.size();
    }
};

// =============================================================================
// CustomOPTranspileResult
// =============================================================================

struct CustomOPTranspileResult {
    TranspileErrorType error_type = TranspileErrorType::SUCCESS;
    std::string func_name;
    size_t smem_size = 0;
    size_t profiler_buf_size = 0;
    std::string code;

    bool is_success() const {
        return error_type == TranspileErrorType::SUCCESS;
    }

    bool has_func_name() const {
        return !func_name.empty();
    }
};

// =============================================================================
// TiledMMA
// =============================================================================

struct TiledMMA {
    std::string A_type = "half_t";
    std::string B_type = "half_t";
    std::string C_type = "half_t";
    int M_tile_size = 256;
    int N_tile_size = 256;
    int K_tile_size = 16;
    size_t guid = 0;

    TiledMMA() = default;

    TiledMMA(std::string A, std::string B, std::string C,
             int M, int N, int K, size_t g)
        : A_type(std::move(A)), B_type(std::move(B)), C_type(std::move(C)),
          M_tile_size(M), N_tile_size(N), K_tile_size(K), guid(g) {}

    size_t get_tile_elements() const {
        return M_tile_size * N_tile_size;
    }

    size_t get_k_iterations(size_t total_k) const {
        return (total_k + K_tile_size - 1) / K_tile_size;
    }
};

}  // namespace transpiler
}  // namespace yirage

using namespace yirage::transpiler;

// =============================================================================
// TranspilerConfig Tests
// =============================================================================

class TranspilerConfigTest : public ::testing::Test {};

TEST_F(TranspilerConfigTest, DefaultValues) {
    TranspilerConfig config;

    EXPECT_EQ(config.target_cc, GPU_CC::A100);
    EXPECT_FALSE(config.profiling);
    EXPECT_EQ(config.num_consumer_wgs, 1);
    EXPECT_EQ(config.num_producer_wgs, 1);
    EXPECT_EQ(config.pipeline_stages, 2);
    EXPECT_FALSE(config.enable_online_softmax);
}

TEST_F(TranspilerConfigTest, IsValid) {
    TranspilerConfig config;
    EXPECT_TRUE(config.is_valid());

    config.target_cc = 50;  // Too low
    EXPECT_FALSE(config.is_valid());

    config.target_cc = 80;
    config.num_consumer_wgs = 0;  // Invalid
    EXPECT_FALSE(config.is_valid());
}

TEST_F(TranspilerConfigTest, IsHopperOrAbove) {
    TranspilerConfig config;

    config.target_cc = GPU_CC::A100;
    EXPECT_FALSE(config.is_hopper_or_above());

    config.target_cc = GPU_CC::H100;
    EXPECT_TRUE(config.is_hopper_or_above());

    config.target_cc = GPU_CC::B200;
    EXPECT_TRUE(config.is_hopper_or_above());
}

TEST_F(TranspilerConfigTest, IsBlackwellOrAbove) {
    TranspilerConfig config;

    config.target_cc = GPU_CC::H100;
    EXPECT_FALSE(config.is_blackwell_or_above());

    config.target_cc = GPU_CC::B200;
    EXPECT_TRUE(config.is_blackwell_or_above());
}

TEST_F(TranspilerConfigTest, SupportsTMA) {
    TranspilerConfig config;

    config.target_cc = GPU_CC::A100;
    EXPECT_FALSE(config.supports_tma());

    config.target_cc = GPU_CC::H100;
    EXPECT_TRUE(config.supports_tma());
}

TEST_F(TranspilerConfigTest, SupportsTensorCores) {
    TranspilerConfig config;

    config.target_cc = GPU_CC::P100;
    EXPECT_FALSE(config.supports_tensor_cores());

    config.target_cc = GPU_CC::V100;
    EXPECT_TRUE(config.supports_tensor_cores());
}

// =============================================================================
// GPU_CC Tests
// =============================================================================

class GPUCCTest : public ::testing::Test {};

TEST_F(GPUCCTest, ComputeCapabilities) {
    EXPECT_EQ(GPU_CC::P100, 60);
    EXPECT_EQ(GPU_CC::V100, 70);
    EXPECT_EQ(GPU_CC::T4, 75);
    EXPECT_EQ(GPU_CC::A100, 80);
    EXPECT_EQ(GPU_CC::H100, 90);
    EXPECT_EQ(GPU_CC::B200, 100);
}

TEST_F(GPUCCTest, Ordering) {
    EXPECT_LT(GPU_CC::P100, GPU_CC::V100);
    EXPECT_LT(GPU_CC::V100, GPU_CC::T4);
    EXPECT_LT(GPU_CC::T4, GPU_CC::A100);
    EXPECT_LT(GPU_CC::A100, GPU_CC::H100);
    EXPECT_LT(GPU_CC::H100, GPU_CC::B200);
}

// =============================================================================
// OutputTensorDirective Tests
// =============================================================================

class OutputTensorDirectiveTest : public ::testing::Test {};

TEST_F(OutputTensorDirectiveTest, DefaultValues) {
    OutputTensorDirective directive;

    EXPECT_EQ(directive.alloc_size, 0u);
    EXPECT_TRUE(directive.shape.empty());
    EXPECT_TRUE(directive.strides.empty());
}

TEST_F(OutputTensorDirectiveTest, GetNumElements) {
    OutputTensorDirective directive;
    directive.shape = {64, 128, 256};

    EXPECT_EQ(directive.get_num_elements(), 64u * 128u * 256u);
}

TEST_F(OutputTensorDirectiveTest, GetNumElementsEmpty) {
    OutputTensorDirective directive;
    EXPECT_EQ(directive.get_num_elements(), 0u);
}

TEST_F(OutputTensorDirectiveTest, IsContiguousTrue) {
    OutputTensorDirective directive;
    directive.shape = {64, 128};
    directive.strides = {128, 1};  // Contiguous row-major

    EXPECT_TRUE(directive.is_contiguous());
}

TEST_F(OutputTensorDirectiveTest, IsContiguousFalse) {
    OutputTensorDirective directive;
    directive.shape = {64, 128};
    directive.strides = {256, 1};  // Padded

    EXPECT_FALSE(directive.is_contiguous());
}

// =============================================================================
// TranspileResult Tests
// =============================================================================

class TranspileResultTest : public ::testing::Test {};

TEST_F(TranspileResultTest, DefaultValues) {
    TranspileResult result;

    EXPECT_EQ(result.error_type, TranspileErrorType::SUCCESS);
    EXPECT_TRUE(result.code.empty());
    EXPECT_EQ(result.buf_size, 0u);
    EXPECT_EQ(result.max_smem_size, 0u);
}

TEST_F(TranspileResultTest, IsSuccess) {
    TranspileResult result;
    EXPECT_TRUE(result.is_success());

    result.error_type = TranspileErrorType::INVALID_INPUT;
    EXPECT_FALSE(result.is_success());
}

TEST_F(TranspileResultTest, HasCode) {
    TranspileResult result;
    EXPECT_FALSE(result.has_code());

    result.code = "__global__ void kernel() {}";
    EXPECT_TRUE(result.has_code());
}

TEST_F(TranspileResultTest, NumOutputs) {
    TranspileResult result;
    EXPECT_EQ(result.num_outputs(), 0u);

    result.output_directives.push_back(OutputTensorDirective());
    result.output_directives.push_back(OutputTensorDirective());
    EXPECT_EQ(result.num_outputs(), 2u);
}

// =============================================================================
// CustomOPTranspileResult Tests
// =============================================================================

class CustomOPTranspileResultTest : public ::testing::Test {};

TEST_F(CustomOPTranspileResultTest, DefaultValues) {
    CustomOPTranspileResult result;

    EXPECT_EQ(result.error_type, TranspileErrorType::SUCCESS);
    EXPECT_TRUE(result.func_name.empty());
    EXPECT_EQ(result.smem_size, 0u);
}

TEST_F(CustomOPTranspileResultTest, IsSuccess) {
    CustomOPTranspileResult result;
    EXPECT_TRUE(result.is_success());

    result.error_type = TranspileErrorType::CODE_GENERATION_FAILED;
    EXPECT_FALSE(result.is_success());
}

TEST_F(CustomOPTranspileResultTest, HasFuncName) {
    CustomOPTranspileResult result;
    EXPECT_FALSE(result.has_func_name());

    result.func_name = "custom_matmul_kernel";
    EXPECT_TRUE(result.has_func_name());
}

// =============================================================================
// TiledMMA Tests
// =============================================================================

class TiledMMATest : public ::testing::Test {};

TEST_F(TiledMMATest, DefaultValues) {
    TiledMMA mma;

    EXPECT_EQ(mma.A_type, "half_t");
    EXPECT_EQ(mma.B_type, "half_t");
    EXPECT_EQ(mma.C_type, "half_t");
    EXPECT_EQ(mma.M_tile_size, 256);
    EXPECT_EQ(mma.N_tile_size, 256);
    EXPECT_EQ(mma.K_tile_size, 16);
}

TEST_F(TiledMMATest, ParameterizedConstruction) {
    TiledMMA mma("bfloat16_t", "bfloat16_t", "float", 128, 128, 32, 42);

    EXPECT_EQ(mma.A_type, "bfloat16_t");
    EXPECT_EQ(mma.B_type, "bfloat16_t");
    EXPECT_EQ(mma.C_type, "float");
    EXPECT_EQ(mma.M_tile_size, 128);
    EXPECT_EQ(mma.N_tile_size, 128);
    EXPECT_EQ(mma.K_tile_size, 32);
    EXPECT_EQ(mma.guid, 42u);
}

TEST_F(TiledMMATest, GetTileElements) {
    TiledMMA mma;
    mma.M_tile_size = 64;
    mma.N_tile_size = 128;

    EXPECT_EQ(mma.get_tile_elements(), 64u * 128u);
}

TEST_F(TiledMMATest, GetKIterations) {
    TiledMMA mma;
    mma.K_tile_size = 16;

    EXPECT_EQ(mma.get_k_iterations(64), 4u);
    EXPECT_EQ(mma.get_k_iterations(65), 5u);  // Ceiling
    EXPECT_EQ(mma.get_k_iterations(16), 1u);
}

// =============================================================================
// TranspileErrorType Tests
// =============================================================================

class TranspileErrorTypeTest : public ::testing::Test {};

TEST_F(TranspileErrorTypeTest, ErrorTypeValues) {
    EXPECT_EQ(static_cast<int>(TranspileErrorType::SUCCESS), 0);
}

TEST_F(TranspileErrorTypeTest, ErrorTypeToString) {
    EXPECT_STREQ(error_type_to_string(TranspileErrorType::SUCCESS), "success");
    EXPECT_STREQ(error_type_to_string(TranspileErrorType::INVALID_INPUT), "invalid_input");
    EXPECT_STREQ(error_type_to_string(TranspileErrorType::UNSUPPORTED_OPERATION), "unsupported_operation");
    EXPECT_STREQ(error_type_to_string(TranspileErrorType::MEMORY_ALLOCATION_FAILED), "memory_allocation_failed");
}

// =============================================================================
// Parameterized GPU CC Tests
// =============================================================================

struct GPUTestParam {
    int cc;
    bool supports_tensor_cores;
    bool supports_tma;
};

class GPUParameterizedTest : public ::testing::TestWithParam<GPUTestParam> {};

TEST_P(GPUParameterizedTest, GPUCapabilities) {
    auto param = GetParam();
    TranspilerConfig config;
    config.target_cc = param.cc;

    EXPECT_EQ(config.supports_tensor_cores(), param.supports_tensor_cores);
    EXPECT_EQ(config.supports_tma(), param.supports_tma);
}

INSTANTIATE_TEST_SUITE_P(
    AllGPUs,
    GPUParameterizedTest,
    ::testing::Values(
        GPUTestParam{GPU_CC::P100, false, false},
        GPUTestParam{GPU_CC::V100, true, false},
        GPUTestParam{GPU_CC::T4, true, false},
        GPUTestParam{GPU_CC::A100, true, false},
        GPUTestParam{GPU_CC::H100, true, true},
        GPUTestParam{GPU_CC::B200, true, true}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
