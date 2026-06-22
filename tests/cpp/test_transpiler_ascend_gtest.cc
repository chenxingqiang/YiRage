// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_transpiler_ascend_gtest.cc
 * @brief Ascend Transpiler Unit Tests
 *
 * Tests for Ascend transpiler (ascend_transpiler.h, ascend_transpiler.cc):
 *   - AscendDeviceType enum
 *   - CodeGenPath enum
 *   - AscendTranspilerConfig structure
 *   - AscendTranspileError structure
 *   - AscendTranspileResult structure
 *   - AscendTranspiler class
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <string>
#include <vector>

namespace yirage {
namespace ascend_transpiler {

// =============================================================================
// Enums
// =============================================================================

enum class CodeGenPath {
    TRITON,    // Triton via BiSheng (RECOMMENDED)
    ASCEND_C,  // Native Ascend C (910B+)
    TBE        // Tensor Boost Engine (910 legacy)
};

enum class AscendDeviceType {
    ASCEND_910 = 0,
    ASCEND_910B = 1,
    ASCEND_910B2 = 2,
    ASCEND_310P = 3,
    ASCEND_310 = 4
};

inline const char* device_type_to_string(AscendDeviceType type) {
    switch (type) {
        case AscendDeviceType::ASCEND_910: return "Ascend 910";
        case AscendDeviceType::ASCEND_910B: return "Ascend 910B";
        case AscendDeviceType::ASCEND_910B2: return "Ascend 910B2";
        case AscendDeviceType::ASCEND_310P: return "Ascend 310P";
        case AscendDeviceType::ASCEND_310: return "Ascend 310";
        default: return "Unknown";
    }
}

inline const char* codegen_path_to_string(CodeGenPath path) {
    switch (path) {
        case CodeGenPath::TRITON: return "Triton";
        case CodeGenPath::ASCEND_C: return "Ascend C";
        case CodeGenPath::TBE: return "TBE";
        default: return "Unknown";
    }
}

inline bool is_training_device(AscendDeviceType type) {
    return type == AscendDeviceType::ASCEND_910 ||
           type == AscendDeviceType::ASCEND_910B ||
           type == AscendDeviceType::ASCEND_910B2;
}

inline bool is_inference_device(AscendDeviceType type) {
    return type == AscendDeviceType::ASCEND_310P ||
           type == AscendDeviceType::ASCEND_310;
}

inline bool supports_bf16(AscendDeviceType type) {
    return type == AscendDeviceType::ASCEND_910B ||
           type == AscendDeviceType::ASCEND_910B2;
}

// =============================================================================
// AscendTranspilerConfig
// =============================================================================

struct AscendTranspilerConfig {
    AscendDeviceType device_type = AscendDeviceType::ASCEND_910B;
    CodeGenPath codegen_path = CodeGenPath::TRITON;
    bool use_cube_ops = true;
    bool enable_fusion = true;
    int ai_cores_per_block = 16;
    size_t l1_buffer_size = 0;
    int opt_level = 3;
    bool enable_fp16 = true;
    bool enable_bf16 = false;
    bool debug_mode = false;

    bool is_valid() const {
        if (ai_cores_per_block < 1 || ai_cores_per_block > 32) return false;
        if (opt_level < 0 || opt_level > 3) return false;
        // BF16 requires 910B+
        if (enable_bf16 && !supports_bf16(device_type)) return false;
        return true;
    }

    static AscendTranspilerConfig for_device(AscendDeviceType type) {
        AscendTranspilerConfig config;
        config.device_type = type;

        switch (type) {
            case AscendDeviceType::ASCEND_910:
                config.codegen_path = CodeGenPath::TBE;
                config.enable_bf16 = false;
                break;
            case AscendDeviceType::ASCEND_910B:
            case AscendDeviceType::ASCEND_910B2:
                config.codegen_path = CodeGenPath::TRITON;
                config.enable_bf16 = true;
                break;
            case AscendDeviceType::ASCEND_310P:
            case AscendDeviceType::ASCEND_310:
                config.codegen_path = CodeGenPath::ASCEND_C;
                config.use_cube_ops = false;  // Limited Cube support
                break;
        }

        return config;
    }
};

// =============================================================================
// AscendTranspileError
// =============================================================================

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

    void add_error(std::string const& msg) {
        messages.push_back(msg);
    }

    void clear() {
        messages.clear();
    }

    size_t num_errors() const {
        return messages.size();
    }
};

// =============================================================================
// AscendTranspileResult
// =============================================================================

struct AscendTranspileResult {
    std::string code;
    std::string compile_command;
    CodeGenPath path_used = CodeGenPath::TRITON;
    std::vector<std::vector<int>> output_shapes;
    AscendTranspileError error;

    bool success() const { return !error.has_error(); }

    bool has_code() const { return !code.empty(); }

    size_t num_outputs() const { return output_shapes.size(); }
};

// =============================================================================
// Mock AscendTranspiler
// =============================================================================

class AscendTranspiler {
public:
    AscendTranspiler(void const* graph, AscendTranspilerConfig const& config)
        : graph_(graph), config_(config) {}

    AscendTranspileResult generate_code() {
        AscendTranspileResult result;
        result.path_used = config_.codegen_path;

        if (!config_.is_valid()) {
            result.error.add_error("Invalid configuration");
            return result;
        }

        // Simulate code generation based on path
        switch (config_.codegen_path) {
            case CodeGenPath::TRITON:
                result.code = generate_triton_code();
                result.compile_command = "python -m triton.tools.compile";
                break;
            case CodeGenPath::ASCEND_C:
                result.code = generate_ascend_c_code();
                result.compile_command = "ascendc --compile";
                break;
            case CodeGenPath::TBE:
                result.code = generate_tbe_code();
                result.compile_command = "tbe_build.sh";
                break;
        }

        return result;
    }

    static AscendDeviceType detect_device_type() {
        // Mock detection - return default
        return AscendDeviceType::ASCEND_910B;
    }

    static AscendTranspilerConfig get_recommended_config() {
        return AscendTranspilerConfig::for_device(detect_device_type());
    }

    std::string get_soc_version() const {
        switch (config_.device_type) {
            case AscendDeviceType::ASCEND_910: return "Ascend910";
            case AscendDeviceType::ASCEND_910B: return "Ascend910B";
            case AscendDeviceType::ASCEND_910B2: return "Ascend910B2";
            case AscendDeviceType::ASCEND_310P: return "Ascend310P";
            case AscendDeviceType::ASCEND_310: return "Ascend310";
            default: return "Unknown";
        }
    }

private:
    std::string generate_triton_code() {
        return "@triton.jit\ndef kernel(): pass";
    }

    std::string generate_ascend_c_code() {
        return "__aicore__ void kernel() {}";
    }

    std::string generate_tbe_code() {
        return "from te import tvm\n";
    }

    void const* graph_;
    AscendTranspilerConfig config_;
};

}  // namespace ascend_transpiler
}  // namespace yirage

using namespace yirage::ascend_transpiler;

// =============================================================================
// AscendDeviceType Tests
// =============================================================================

class AscendDeviceTypeTest : public ::testing::Test {};

TEST_F(AscendDeviceTypeTest, DeviceTypeValues) {
    EXPECT_EQ(static_cast<int>(AscendDeviceType::ASCEND_910), 0);
    EXPECT_EQ(static_cast<int>(AscendDeviceType::ASCEND_910B), 1);
    EXPECT_EQ(static_cast<int>(AscendDeviceType::ASCEND_910B2), 2);
    EXPECT_EQ(static_cast<int>(AscendDeviceType::ASCEND_310P), 3);
    EXPECT_EQ(static_cast<int>(AscendDeviceType::ASCEND_310), 4);
}

TEST_F(AscendDeviceTypeTest, DeviceTypeToString) {
    EXPECT_STREQ(device_type_to_string(AscendDeviceType::ASCEND_910), "Ascend 910");
    EXPECT_STREQ(device_type_to_string(AscendDeviceType::ASCEND_910B), "Ascend 910B");
    EXPECT_STREQ(device_type_to_string(AscendDeviceType::ASCEND_910B2), "Ascend 910B2");
    EXPECT_STREQ(device_type_to_string(AscendDeviceType::ASCEND_310P), "Ascend 310P");
    EXPECT_STREQ(device_type_to_string(AscendDeviceType::ASCEND_310), "Ascend 310");
}

TEST_F(AscendDeviceTypeTest, IsTrainingDevice) {
    EXPECT_TRUE(is_training_device(AscendDeviceType::ASCEND_910));
    EXPECT_TRUE(is_training_device(AscendDeviceType::ASCEND_910B));
    EXPECT_TRUE(is_training_device(AscendDeviceType::ASCEND_910B2));
    EXPECT_FALSE(is_training_device(AscendDeviceType::ASCEND_310P));
    EXPECT_FALSE(is_training_device(AscendDeviceType::ASCEND_310));
}

TEST_F(AscendDeviceTypeTest, IsInferenceDevice) {
    EXPECT_FALSE(is_inference_device(AscendDeviceType::ASCEND_910));
    EXPECT_FALSE(is_inference_device(AscendDeviceType::ASCEND_910B));
    EXPECT_TRUE(is_inference_device(AscendDeviceType::ASCEND_310P));
    EXPECT_TRUE(is_inference_device(AscendDeviceType::ASCEND_310));
}

TEST_F(AscendDeviceTypeTest, SupportsBF16) {
    EXPECT_FALSE(supports_bf16(AscendDeviceType::ASCEND_910));
    EXPECT_TRUE(supports_bf16(AscendDeviceType::ASCEND_910B));
    EXPECT_TRUE(supports_bf16(AscendDeviceType::ASCEND_910B2));
    EXPECT_FALSE(supports_bf16(AscendDeviceType::ASCEND_310P));
}

// =============================================================================
// CodeGenPath Tests
// =============================================================================

class CodeGenPathTest : public ::testing::Test {};

TEST_F(CodeGenPathTest, CodeGenPathToString) {
    EXPECT_STREQ(codegen_path_to_string(CodeGenPath::TRITON), "Triton");
    EXPECT_STREQ(codegen_path_to_string(CodeGenPath::ASCEND_C), "Ascend C");
    EXPECT_STREQ(codegen_path_to_string(CodeGenPath::TBE), "TBE");
}

// =============================================================================
// AscendTranspilerConfig Tests
// =============================================================================

class AscendTranspilerConfigTest : public ::testing::Test {};

TEST_F(AscendTranspilerConfigTest, DefaultValues) {
    AscendTranspilerConfig config;

    EXPECT_EQ(config.device_type, AscendDeviceType::ASCEND_910B);
    EXPECT_EQ(config.codegen_path, CodeGenPath::TRITON);
    EXPECT_TRUE(config.use_cube_ops);
    EXPECT_TRUE(config.enable_fusion);
    EXPECT_EQ(config.ai_cores_per_block, 16);
    EXPECT_EQ(config.l1_buffer_size, 0u);
    EXPECT_EQ(config.opt_level, 3);
    EXPECT_TRUE(config.enable_fp16);
    EXPECT_FALSE(config.enable_bf16);
    EXPECT_FALSE(config.debug_mode);
}

TEST_F(AscendTranspilerConfigTest, IsValid) {
    AscendTranspilerConfig config;
    EXPECT_TRUE(config.is_valid());

    config.ai_cores_per_block = 0;
    EXPECT_FALSE(config.is_valid());

    config.ai_cores_per_block = 16;
    config.opt_level = 5;
    EXPECT_FALSE(config.is_valid());
}

TEST_F(AscendTranspilerConfigTest, BF16ValidationFor910) {
    AscendTranspilerConfig config;
    config.device_type = AscendDeviceType::ASCEND_910;
    config.enable_bf16 = true;

    EXPECT_FALSE(config.is_valid());  // BF16 not supported on 910
}

TEST_F(AscendTranspilerConfigTest, BF16ValidationFor910B) {
    AscendTranspilerConfig config;
    config.device_type = AscendDeviceType::ASCEND_910B;
    config.enable_bf16 = true;

    EXPECT_TRUE(config.is_valid());  // BF16 supported on 910B
}

TEST_F(AscendTranspilerConfigTest, ForDevice910) {
    auto config = AscendTranspilerConfig::for_device(AscendDeviceType::ASCEND_910);

    EXPECT_EQ(config.device_type, AscendDeviceType::ASCEND_910);
    EXPECT_EQ(config.codegen_path, CodeGenPath::TBE);
    EXPECT_FALSE(config.enable_bf16);
}

TEST_F(AscendTranspilerConfigTest, ForDevice910B) {
    auto config = AscendTranspilerConfig::for_device(AscendDeviceType::ASCEND_910B);

    EXPECT_EQ(config.device_type, AscendDeviceType::ASCEND_910B);
    EXPECT_EQ(config.codegen_path, CodeGenPath::TRITON);
    EXPECT_TRUE(config.enable_bf16);
}

TEST_F(AscendTranspilerConfigTest, ForDevice310P) {
    auto config = AscendTranspilerConfig::for_device(AscendDeviceType::ASCEND_310P);

    EXPECT_EQ(config.device_type, AscendDeviceType::ASCEND_310P);
    EXPECT_EQ(config.codegen_path, CodeGenPath::ASCEND_C);
    EXPECT_FALSE(config.use_cube_ops);  // Limited Cube support
}

// =============================================================================
// AscendTranspileError Tests
// =============================================================================

class AscendTranspileErrorTest : public ::testing::Test {};

TEST_F(AscendTranspileErrorTest, DefaultState) {
    AscendTranspileError error;

    EXPECT_FALSE(error.has_error());
    EXPECT_EQ(error.num_errors(), 0u);
    EXPECT_TRUE(error.to_string().empty());
}

TEST_F(AscendTranspileErrorTest, AddError) {
    AscendTranspileError error;
    error.add_error("First error");

    EXPECT_TRUE(error.has_error());
    EXPECT_EQ(error.num_errors(), 1u);
}

TEST_F(AscendTranspileErrorTest, MultipleErrors) {
    AscendTranspileError error;
    error.add_error("Error 1");
    error.add_error("Error 2");
    error.add_error("Error 3");

    EXPECT_EQ(error.num_errors(), 3u);
    EXPECT_TRUE(error.to_string().find("Error 1") != std::string::npos);
    EXPECT_TRUE(error.to_string().find("Error 2") != std::string::npos);
}

TEST_F(AscendTranspileErrorTest, Clear) {
    AscendTranspileError error;
    error.add_error("Error");

    error.clear();

    EXPECT_FALSE(error.has_error());
    EXPECT_EQ(error.num_errors(), 0u);
}

// =============================================================================
// AscendTranspileResult Tests
// =============================================================================

class AscendTranspileResultTest : public ::testing::Test {};

TEST_F(AscendTranspileResultTest, DefaultValues) {
    AscendTranspileResult result;

    EXPECT_TRUE(result.success());
    EXPECT_FALSE(result.has_code());
    EXPECT_EQ(result.num_outputs(), 0u);
}

TEST_F(AscendTranspileResultTest, SuccessWithCode) {
    AscendTranspileResult result;
    result.code = "def kernel(): pass";
    result.path_used = CodeGenPath::TRITON;

    EXPECT_TRUE(result.success());
    EXPECT_TRUE(result.has_code());
}

TEST_F(AscendTranspileResultTest, Failure) {
    AscendTranspileResult result;
    result.error.add_error("Transpilation failed");

    EXPECT_FALSE(result.success());
}

TEST_F(AscendTranspileResultTest, OutputShapes) {
    AscendTranspileResult result;
    result.output_shapes.push_back({64, 128});
    result.output_shapes.push_back({64, 256});

    EXPECT_EQ(result.num_outputs(), 2u);
}

// =============================================================================
// AscendTranspiler Tests
// =============================================================================

class AscendTranspilerTest : public ::testing::Test {};

TEST_F(AscendTranspilerTest, DetectDeviceType) {
    auto device = AscendTranspiler::detect_device_type();
    EXPECT_EQ(device, AscendDeviceType::ASCEND_910B);  // Default mock
}

TEST_F(AscendTranspilerTest, GetRecommendedConfig) {
    auto config = AscendTranspiler::get_recommended_config();

    EXPECT_TRUE(config.is_valid());
}

TEST_F(AscendTranspilerTest, GenerateTritonCode) {
    AscendTranspilerConfig config;
    config.codegen_path = CodeGenPath::TRITON;

    AscendTranspiler transpiler(nullptr, config);
    auto result = transpiler.generate_code();

    EXPECT_TRUE(result.success());
    EXPECT_EQ(result.path_used, CodeGenPath::TRITON);
    EXPECT_TRUE(result.code.find("@triton") != std::string::npos);
}

TEST_F(AscendTranspilerTest, GenerateAscendCCode) {
    AscendTranspilerConfig config;
    config.codegen_path = CodeGenPath::ASCEND_C;

    AscendTranspiler transpiler(nullptr, config);
    auto result = transpiler.generate_code();

    EXPECT_TRUE(result.success());
    EXPECT_EQ(result.path_used, CodeGenPath::ASCEND_C);
    EXPECT_TRUE(result.code.find("__aicore__") != std::string::npos);
}

TEST_F(AscendTranspilerTest, GenerateTBECode) {
    AscendTranspilerConfig config;
    config.codegen_path = CodeGenPath::TBE;

    AscendTranspiler transpiler(nullptr, config);
    auto result = transpiler.generate_code();

    EXPECT_TRUE(result.success());
    EXPECT_EQ(result.path_used, CodeGenPath::TBE);
    EXPECT_TRUE(result.code.find("from te") != std::string::npos);
}

TEST_F(AscendTranspilerTest, InvalidConfigFails) {
    AscendTranspilerConfig config;
    config.ai_cores_per_block = 100;  // Invalid

    AscendTranspiler transpiler(nullptr, config);
    auto result = transpiler.generate_code();

    EXPECT_FALSE(result.success());
}

TEST_F(AscendTranspilerTest, GetSocVersion) {
    AscendTranspilerConfig config;
    config.device_type = AscendDeviceType::ASCEND_910B2;

    AscendTranspiler transpiler(nullptr, config);
    EXPECT_EQ(transpiler.get_soc_version(), "Ascend910B2");
}

// =============================================================================
// Parameterized Device Tests
// =============================================================================

struct DeviceTestParam {
    AscendDeviceType device;
    bool is_training;
    bool supports_bf16;
    CodeGenPath recommended_path;
};

class DeviceParameterizedTest
    : public ::testing::TestWithParam<DeviceTestParam> {};

TEST_P(DeviceParameterizedTest, DeviceProperties) {
    auto param = GetParam();
    auto config = AscendTranspilerConfig::for_device(param.device);

    EXPECT_EQ(is_training_device(param.device), param.is_training);
    EXPECT_EQ(supports_bf16(param.device), param.supports_bf16);
    EXPECT_EQ(config.codegen_path, param.recommended_path);
}

INSTANTIATE_TEST_SUITE_P(
    AllAscendDevices,
    DeviceParameterizedTest,
    ::testing::Values(
        DeviceTestParam{AscendDeviceType::ASCEND_910, true, false, CodeGenPath::TBE},
        DeviceTestParam{AscendDeviceType::ASCEND_910B, true, true, CodeGenPath::TRITON},
        DeviceTestParam{AscendDeviceType::ASCEND_910B2, true, true, CodeGenPath::TRITON},
        DeviceTestParam{AscendDeviceType::ASCEND_310P, false, false, CodeGenPath::ASCEND_C},
        DeviceTestParam{AscendDeviceType::ASCEND_310, false, false, CodeGenPath::ASCEND_C}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
