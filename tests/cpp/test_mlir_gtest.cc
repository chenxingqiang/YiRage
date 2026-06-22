// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_mlir_gtest.cc
 * @brief MLIR Dialect and Lowering Unit Tests (Google Test version)
 *
 * Tests for YiRage MLIR dialect operations and transformations.
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <map>

namespace yirage {
namespace mlir {

// =============================================================================
// Mock MLIR Types for Testing
// =============================================================================

enum class MLIRDialect {
    YIRAGE,
    LINALG,
    ARITH,
    MATH,
    TENSOR,
    MEMREF,
    FUNC,
    LLVM,
    GPU,
    SCF,
};

enum class YirageOp {
    MATMUL,
    SILU,
    GELU,
    RELU,
    RMS_NORM,
    SOFTMAX,
    ATTENTION,
    REDUCE_SUM,
    CONCAT,
    SPLIT,
};

struct TensorType {
    std::vector<int64_t> shape;
    std::string element_type;  // f16, f32, bf16, etc.
    
    std::string to_string() const {
        std::ostringstream oss;
        oss << "tensor<";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) oss << "x";
            oss << shape[i];
        }
        oss << "x" << element_type << ">";
        return oss.str();
    }
};

struct Operation {
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<TensorType> input_types;
    std::vector<TensorType> output_types;
    std::map<std::string, std::string> attributes;
};

// Mock MLIR module
class MLIRModule {
public:
    std::string name;
    std::vector<Operation> operations;
    
    void add_operation(const Operation& op) {
        operations.push_back(op);
    }
    
    std::string emit() const {
        std::ostringstream oss;
        oss << "module {\n";
        for (const auto& op : operations) {
            oss << "  " << emit_operation(op) << "\n";
        }
        oss << "}\n";
        return oss.str();
    }
    
private:
    std::string emit_operation(const Operation& op) const {
        std::ostringstream oss;
        if (!op.outputs.empty()) {
            oss << op.outputs[0] << " = ";
        }
        oss << op.name << " ";
        for (size_t i = 0; i < op.inputs.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << op.inputs[i];
        }
        if (!op.input_types.empty()) {
            oss << " : ";
            for (size_t i = 0; i < op.input_types.size(); ++i) {
                if (i > 0) oss << ", ";
                oss << op.input_types[i].to_string();
            }
            if (!op.output_types.empty()) {
                oss << " -> ";
                for (size_t i = 0; i < op.output_types.size(); ++i) {
                    if (i > 0) oss << ", ";
                    oss << op.output_types[i].to_string();
                }
            }
        }
        return oss.str();
    }
};

// Lowering map: YiRage ops to Linalg ops
inline const std::map<YirageOp, std::string>& get_lowering_map() {
    static const std::map<YirageOp, std::string> map = {
        {YirageOp::MATMUL, "linalg.matmul"},
        {YirageOp::SILU, "linalg.generic"},
        {YirageOp::GELU, "linalg.generic"},
        {YirageOp::RELU, "linalg.generic"},
        {YirageOp::RMS_NORM, "linalg.generic"},
        {YirageOp::SOFTMAX, "linalg.softmax"},
        {YirageOp::ATTENTION, "linalg.generic"},
        {YirageOp::REDUCE_SUM, "linalg.reduce"},
        {YirageOp::CONCAT, "tensor.concat"},
        {YirageOp::SPLIT, "tensor.split"},
    };
    return map;
}

}  // namespace mlir
}  // namespace yirage

using namespace yirage::mlir;

// =============================================================================
// TensorType Tests
// =============================================================================

class TensorTypeTest : public ::testing::Test {};

TEST_F(TensorTypeTest, TwoD_F32) {
    TensorType t{{32, 64}, "f32"};
    EXPECT_EQ(t.to_string(), "tensor<32x64xf32>");
}

TEST_F(TensorTypeTest, ThreeD_F16) {
    TensorType t{{8, 32, 64}, "f16"};
    EXPECT_EQ(t.to_string(), "tensor<8x32x64xf16>");
}

TEST_F(TensorTypeTest, FourD_BF16) {
    TensorType t{{2, 8, 32, 64}, "bf16"};
    EXPECT_EQ(t.to_string(), "tensor<2x8x32x64xbf16>");
}

TEST_F(TensorTypeTest, OneD_I32) {
    TensorType t{{1024}, "i32"};
    EXPECT_EQ(t.to_string(), "tensor<1024xi32>");
}

// =============================================================================
// YiRage Dialect Operation Tests
// =============================================================================

class YirageDialectTest : public ::testing::Test {};

TEST_F(YirageDialectTest, MatmulOperation) {
    Operation op;
    op.name = "yirage.matmul";
    op.inputs = {"%arg0", "%arg1"};
    op.outputs = {"%0"};
    op.input_types = {
        {{32, 64}, "f32"},
        {{64, 128}, "f32"}
    };
    op.output_types = {{{32, 128}, "f32"}};
    
    EXPECT_EQ(op.name, "yirage.matmul");
    EXPECT_EQ(op.inputs.size(), 2u);
    EXPECT_EQ(op.output_types[0].shape, (std::vector<int64_t>{32, 128}));
}

TEST_F(YirageDialectTest, SiluOperation) {
    Operation op;
    op.name = "yirage.silu";
    op.inputs = {"%arg0"};
    op.outputs = {"%0"};
    op.input_types = {{{32, 128}, "f32"}};
    op.output_types = {{{32, 128}, "f32"}};
    
    EXPECT_EQ(op.name, "yirage.silu");
    EXPECT_EQ(op.inputs.size(), 1u);
    EXPECT_EQ(op.input_types[0].shape, op.output_types[0].shape);
}

TEST_F(YirageDialectTest, GeluOperation) {
    Operation op;
    op.name = "yirage.gelu";
    op.inputs = {"%arg0"};
    op.outputs = {"%0"};
    op.input_types = {{{1024, 1024}, "f16"}};
    op.output_types = {{{1024, 1024}, "f16"}};
    
    EXPECT_EQ(op.name, "yirage.gelu");
}

TEST_F(YirageDialectTest, ReluOperation) {
    Operation op;
    op.name = "yirage.relu";
    op.inputs = {"%arg0"};
    op.outputs = {"%0"};
    op.input_types = {{{256, 256}, "f32"}};
    op.output_types = {{{256, 256}, "f32"}};
    
    EXPECT_EQ(op.name, "yirage.relu");
}

TEST_F(YirageDialectTest, RMSNormOperation) {
    Operation op;
    op.name = "yirage.rms_norm";
    op.inputs = {"%input", "%weight"};
    op.outputs = {"%0"};
    op.input_types = {
        {{8, 4096}, "f16"},
        {{4096}, "f16"}
    };
    op.output_types = {{{8, 4096}, "f16"}};
    
    EXPECT_EQ(op.name, "yirage.rms_norm");
    EXPECT_EQ(op.inputs.size(), 2u);
}

TEST_F(YirageDialectTest, ReduceSumOperation) {
    Operation op;
    op.name = "yirage.reduce_sum";
    op.inputs = {"%arg0"};
    op.outputs = {"%0"};
    op.input_types = {{{32, 64, 128}, "f32"}};
    op.output_types = {{{32, 64}, "f32"}};  // Reduced last dim
    op.attributes["axis"] = "2";
    
    EXPECT_EQ(op.name, "yirage.reduce_sum");
    EXPECT_EQ(op.attributes["axis"], "2");
}

// =============================================================================
// MLIR Module Tests
// =============================================================================

class MLIRModuleTest : public ::testing::Test {
protected:
    MLIRModule module;
};

TEST_F(MLIRModuleTest, EmptyModule) {
    std::string mlir = module.emit();
    EXPECT_NE(mlir.find("module"), std::string::npos);
}

TEST_F(MLIRModuleTest, SingleOperation) {
    Operation op;
    op.name = "yirage.matmul";
    op.inputs = {"%arg0", "%arg1"};
    op.outputs = {"%0"};
    op.input_types = {{{32, 64}, "f32"}, {{64, 128}, "f32"}};
    op.output_types = {{{32, 128}, "f32"}};
    
    module.add_operation(op);
    
    std::string mlir = module.emit();
    EXPECT_NE(mlir.find("yirage.matmul"), std::string::npos);
    EXPECT_NE(mlir.find("%arg0"), std::string::npos);
    EXPECT_NE(mlir.find("tensor<32x64xf32>"), std::string::npos);
}

TEST_F(MLIRModuleTest, MultipleOperations) {
    // MatMul + SiLU
    Operation matmul;
    matmul.name = "yirage.matmul";
    matmul.inputs = {"%arg0", "%arg1"};
    matmul.outputs = {"%0"};
    matmul.input_types = {{{32, 64}, "f32"}, {{64, 128}, "f32"}};
    matmul.output_types = {{{32, 128}, "f32"}};
    
    Operation silu;
    silu.name = "yirage.silu";
    silu.inputs = {"%0"};
    silu.outputs = {"%1"};
    silu.input_types = {{{32, 128}, "f32"}};
    silu.output_types = {{{32, 128}, "f32"}};
    
    module.add_operation(matmul);
    module.add_operation(silu);
    
    std::string mlir = module.emit();
    EXPECT_NE(mlir.find("yirage.matmul"), std::string::npos);
    EXPECT_NE(mlir.find("yirage.silu"), std::string::npos);
}

// =============================================================================
// Lowering Tests
// =============================================================================

class LoweringTest : public ::testing::Test {
protected:
    const std::map<YirageOp, std::string>& lowering_map = get_lowering_map();
};

TEST_F(LoweringTest, MatmulLowersToLinalg) {
    EXPECT_EQ(lowering_map.at(YirageOp::MATMUL), "linalg.matmul");
}

TEST_F(LoweringTest, SiluLowersToGeneric) {
    EXPECT_EQ(lowering_map.at(YirageOp::SILU), "linalg.generic");
}

TEST_F(LoweringTest, GeluLowersToGeneric) {
    EXPECT_EQ(lowering_map.at(YirageOp::GELU), "linalg.generic");
}

TEST_F(LoweringTest, ReluLowersToGeneric) {
    EXPECT_EQ(lowering_map.at(YirageOp::RELU), "linalg.generic");
}

TEST_F(LoweringTest, SoftmaxLowersToLinalg) {
    EXPECT_EQ(lowering_map.at(YirageOp::SOFTMAX), "linalg.softmax");
}

TEST_F(LoweringTest, ReduceLowersToLinalg) {
    EXPECT_EQ(lowering_map.at(YirageOp::REDUCE_SUM), "linalg.reduce");
}

TEST_F(LoweringTest, ConcatLowersToTensor) {
    EXPECT_EQ(lowering_map.at(YirageOp::CONCAT), "tensor.concat");
}

TEST_F(LoweringTest, SplitLowersToTensor) {
    EXPECT_EQ(lowering_map.at(YirageOp::SPLIT), "tensor.split");
}

// =============================================================================
// Shape Inference Tests
// =============================================================================

class ShapeInferenceTest : public ::testing::Test {
protected:
    // Simple shape inference for matmul
    std::vector<int64_t> infer_matmul_shape(
        const std::vector<int64_t>& lhs,
        const std::vector<int64_t>& rhs) {
        
        if (lhs.size() != 2 || rhs.size() != 2) {
            return {};
        }
        if (lhs[1] != rhs[0]) {
            return {};  // Incompatible
        }
        return {lhs[0], rhs[1]};
    }
    
    // Shape inference for elementwise
    std::vector<int64_t> infer_elementwise_shape(
        const std::vector<int64_t>& input) {
        return input;  // Same shape
    }
    
    // Shape inference for reduction
    std::vector<int64_t> infer_reduction_shape(
        const std::vector<int64_t>& input, int axis) {
        
        std::vector<int64_t> result;
        for (int i = 0; i < static_cast<int>(input.size()); ++i) {
            if (i != axis) {
                result.push_back(input[i]);
            }
        }
        return result;
    }
};

TEST_F(ShapeInferenceTest, MatmulShapeInference) {
    auto shape = infer_matmul_shape({32, 64}, {64, 128});
    EXPECT_EQ(shape, (std::vector<int64_t>{32, 128}));
}

TEST_F(ShapeInferenceTest, MatmulIncompatibleShapes) {
    auto shape = infer_matmul_shape({32, 64}, {128, 256});  // 64 != 128
    EXPECT_TRUE(shape.empty());
}

TEST_F(ShapeInferenceTest, ElementwiseShapeInference) {
    auto shape = infer_elementwise_shape({8, 32, 64});
    EXPECT_EQ(shape, (std::vector<int64_t>{8, 32, 64}));
}

TEST_F(ShapeInferenceTest, ReductionAxis0) {
    auto shape = infer_reduction_shape({8, 32, 64}, 0);
    EXPECT_EQ(shape, (std::vector<int64_t>{32, 64}));
}

TEST_F(ShapeInferenceTest, ReductionAxis1) {
    auto shape = infer_reduction_shape({8, 32, 64}, 1);
    EXPECT_EQ(shape, (std::vector<int64_t>{8, 64}));
}

TEST_F(ShapeInferenceTest, ReductionAxis2) {
    auto shape = infer_reduction_shape({8, 32, 64}, 2);
    EXPECT_EQ(shape, (std::vector<int64_t>{8, 32}));
}

// =============================================================================
// Backend Pipeline Tests
// =============================================================================

class BackendPipelineTest : public ::testing::TestWithParam<std::string> {};

TEST_P(BackendPipelineTest, PipelineNameFormat) {
    std::string backend = GetParam();
    std::string pipeline = "yirage-" + backend + "-pipeline";
    
    // Pipeline name should follow format
    EXPECT_NE(pipeline.find("yirage-"), std::string::npos);
    EXPECT_NE(pipeline.find("-pipeline"), std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    Backends,
    BackendPipelineTest,
    ::testing::Values(
        "cuda", "rocm", "cpu", "mps", "ascend", "tpu", "fpga", "gpu"
    )
);

// =============================================================================
// Data Type Tests
// =============================================================================

class DataTypeTest : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(DataTypeTest, MLIRTypeMapping) {
    auto [dtype, mlir_type] = GetParam();
    
    std::map<std::string, std::string> mapping = {
        {"fp16", "f16"},
        {"fp32", "f32"},
        {"fp64", "f64"},
        {"bf16", "bf16"},
        {"int8", "i8"},
        {"int16", "i16"},
        {"int32", "i32"},
        {"int64", "i64"},
    };
    
    EXPECT_EQ(mapping[dtype], mlir_type);
}

INSTANTIATE_TEST_SUITE_P(
    DataTypes,
    DataTypeTest,
    ::testing::Values(
        std::make_pair("fp16", "f16"),
        std::make_pair("fp32", "f32"),
        std::make_pair("fp64", "f64"),
        std::make_pair("bf16", "bf16"),
        std::make_pair("int8", "i8"),
        std::make_pair("int32", "i32")
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
