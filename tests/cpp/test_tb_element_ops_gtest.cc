// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_tb_element_ops_gtest.cc
 * @brief Threadblock Element Operations Unit Tests
 *
 * Tests for element-wise operations:
 *   - TBElementUnaryOp (exp, silu, gelu, relu, sqrt, square, clamp)
 *   - TBElementBinaryOp (add, mul, div, sub, pow)
 *   - Shape preservation
 *   - Scalar operations
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cmath>
#include <vector>

namespace yirage {
namespace type {

enum DataType {
    DT_UNKNOWN = 0,
    DT_FLOAT16 = 3,
    DT_FLOAT32 = 4,
};

enum TBOperatorType {
    // Unary ops
    TB_EXP_OP = 300,
    TB_SQUARE_OP = 301,
    TB_SQRT_OP = 302,
    TB_SILU_OP = 303,
    TB_GELU_OP = 304,
    TB_RELU_OP = 305,
    TB_CLAMP_OP = 306,
    TB_MUL_SCALAR_OP = 307,

    // Binary ops
    TB_ADD_OP = 400,
    TB_MUL_OP = 401,
    TB_DIV_OP = 402,
    TB_SUB_OP = 403,
    TB_POW_OP = 404,
};

}  // namespace type

namespace layout {
enum SmemLayout {
    SMEM_LAYOUT_ROW_MAJOR = 1,
};
}  // namespace layout

namespace threadblock {

constexpr int MAX_TENSOR_DIMS = 4;

struct STensor {
    type::DataType data_type = type::DT_FLOAT32;
    layout::SmemLayout layout = layout::SMEM_LAYOUT_ROW_MAJOR;
    int num_dims = 0;
    int dim[MAX_TENSOR_DIMS] = {0};
    int smem_offset = 0;

    size_t num_elements() const {
        if (num_dims == 0) return 0;
        size_t result = 1;
        for (int i = 0; i < num_dims; i++) {
            result *= dim[i];
        }
        return result;
    }

    bool same_shape(STensor const& other) const {
        if (num_dims != other.num_dims) return false;
        for (int i = 0; i < num_dims; i++) {
            if (dim[i] != other.dim[i]) return false;
        }
        return true;
    }
};

// =============================================================================
// TBElementUnaryOp
// =============================================================================

class TBElementUnaryOp {
public:
    TBElementUnaryOp(STensor const& input, type::TBOperatorType op_type,
                     float scalar = 0.0f)
        : input_(input), op_type_(op_type), scalar_(scalar) {
        // Output has same shape as input
        output_ = input;
        output_.smem_offset = 0;  // Will be assigned later
    }

    STensor const& input() const { return input_; }
    STensor const& output() const { return output_; }
    type::TBOperatorType op_type() const { return op_type_; }
    float scalar() const { return scalar_; }

    static bool is_unary_op(type::TBOperatorType op) {
        return op >= type::TB_EXP_OP && op <= type::TB_MUL_SCALAR_OP;
    }

    // Reference implementations for testing
    static float apply_exp(float x) { return std::exp(x); }
    static float apply_square(float x) { return x * x; }
    static float apply_sqrt(float x) { return std::sqrt(x); }
    static float apply_silu(float x) { return x / (1.0f + std::exp(-x)); }
    static float apply_gelu(float x) {
        return 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / M_PI) *
                                             (x + 0.044715f * x * x * x)));
    }
    static float apply_relu(float x) { return x > 0.0f ? x : 0.0f; }
    static float apply_clamp(float x, float min_val, float max_val) {
        return std::min(std::max(x, min_val), max_val);
    }
    static float apply_mul_scalar(float x, float scalar) { return x * scalar; }

private:
    STensor input_;
    STensor output_;
    type::TBOperatorType op_type_;
    float scalar_;
};

// =============================================================================
// TBClampUnaryOp
// =============================================================================

class TBClampUnaryOp : public TBElementUnaryOp {
public:
    TBClampUnaryOp(STensor const& input, float min_val, float max_val)
        : TBElementUnaryOp(input, type::TB_CLAMP_OP, 0.0f),
          min_val_(min_val), max_val_(max_val) {}

    float min_val() const { return min_val_; }
    float max_val() const { return max_val_; }

private:
    float min_val_;
    float max_val_;
};

// =============================================================================
// TBElementBinaryOp
// =============================================================================

class TBElementBinaryOp {
public:
    TBElementBinaryOp(STensor const& A, STensor const& B,
                      type::TBOperatorType op_type)
        : A_(A), B_(B), op_type_(op_type) {
        // Output has same shape as A (with broadcasting)
        output_ = A;
        output_.smem_offset = 0;
    }

    STensor const& input_a() const { return A_; }
    STensor const& input_b() const { return B_; }
    STensor const& output() const { return output_; }
    type::TBOperatorType op_type() const { return op_type_; }

    static bool is_binary_op(type::TBOperatorType op) {
        return op >= type::TB_ADD_OP && op <= type::TB_POW_OP;
    }

    static bool shapes_compatible(STensor const& A, STensor const& B) {
        if (A.num_dims != B.num_dims) return false;
        for (int i = 0; i < A.num_dims; i++) {
            // Allow broadcasting: dim must match or one must be 1
            if (A.dim[i] != B.dim[i] && A.dim[i] != 1 && B.dim[i] != 1) {
                return false;
            }
        }
        return true;
    }

    // Reference implementations
    static float apply_add(float a, float b) { return a + b; }
    static float apply_mul(float a, float b) { return a * b; }
    static float apply_div(float a, float b) { return a / b; }
    static float apply_sub(float a, float b) { return a - b; }
    static float apply_pow(float a, float b) { return std::pow(a, b); }

private:
    STensor A_, B_, output_;
    type::TBOperatorType op_type_;
};

}  // namespace threadblock
}  // namespace yirage

using namespace yirage::threadblock;
using namespace yirage::type;
using namespace yirage::layout;

// =============================================================================
// Element Unary Op Tests
// =============================================================================

class TBElementUnaryOpTest : public ::testing::Test {};

TEST_F(TBElementUnaryOpTest, ExpOp) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;
    input.data_type = DT_FLOAT32;

    TBElementUnaryOp op(input, TB_EXP_OP);

    EXPECT_EQ(op.op_type(), TB_EXP_OP);
    EXPECT_TRUE(op.output().same_shape(input));
}

TEST_F(TBElementUnaryOpTest, SiluOp) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 32;
    input.dim[1] = 64;

    TBElementUnaryOp op(input, TB_SILU_OP);

    EXPECT_EQ(op.op_type(), TB_SILU_OP);
    EXPECT_TRUE(op.output().same_shape(input));
}

TEST_F(TBElementUnaryOpTest, GeluOp) {
    STensor input;
    input.num_dims = 3;
    input.dim[0] = 8;
    input.dim[1] = 32;
    input.dim[2] = 64;

    TBElementUnaryOp op(input, TB_GELU_OP);

    EXPECT_EQ(op.op_type(), TB_GELU_OP);
    EXPECT_EQ(op.output().num_dims, 3);
}

TEST_F(TBElementUnaryOpTest, ReluOp) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 128;
    input.dim[1] = 256;

    TBElementUnaryOp op(input, TB_RELU_OP);

    EXPECT_EQ(op.op_type(), TB_RELU_OP);
}

TEST_F(TBElementUnaryOpTest, SqrtOp) {
    STensor input;
    input.num_dims = 1;
    input.dim[0] = 1024;

    TBElementUnaryOp op(input, TB_SQRT_OP);

    EXPECT_EQ(op.op_type(), TB_SQRT_OP);
    EXPECT_EQ(op.output().dim[0], 1024);
}

TEST_F(TBElementUnaryOpTest, SquareOp) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 64;

    TBElementUnaryOp op(input, TB_SQUARE_OP);

    EXPECT_EQ(op.op_type(), TB_SQUARE_OP);
}

TEST_F(TBElementUnaryOpTest, MulScalarOp) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBElementUnaryOp op(input, TB_MUL_SCALAR_OP, 2.5f);

    EXPECT_EQ(op.op_type(), TB_MUL_SCALAR_OP);
    EXPECT_FLOAT_EQ(op.scalar(), 2.5f);
}

TEST_F(TBElementUnaryOpTest, IsUnaryOp) {
    EXPECT_TRUE(TBElementUnaryOp::is_unary_op(TB_EXP_OP));
    EXPECT_TRUE(TBElementUnaryOp::is_unary_op(TB_SILU_OP));
    EXPECT_TRUE(TBElementUnaryOp::is_unary_op(TB_RELU_OP));
    EXPECT_TRUE(TBElementUnaryOp::is_unary_op(TB_MUL_SCALAR_OP));
    EXPECT_FALSE(TBElementUnaryOp::is_unary_op(TB_ADD_OP));
}

// =============================================================================
// Clamp Op Tests
// =============================================================================

class TBClampOpTest : public ::testing::Test {};

TEST_F(TBClampOpTest, BasicClamp) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBClampUnaryOp op(input, -1.0f, 1.0f);

    EXPECT_FLOAT_EQ(op.min_val(), -1.0f);
    EXPECT_FLOAT_EQ(op.max_val(), 1.0f);
    EXPECT_EQ(op.op_type(), TB_CLAMP_OP);
}

TEST_F(TBClampOpTest, ClampZeroOne) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 32;
    input.dim[1] = 32;

    TBClampUnaryOp op(input, 0.0f, 1.0f);

    EXPECT_FLOAT_EQ(op.min_val(), 0.0f);
    EXPECT_FLOAT_EQ(op.max_val(), 1.0f);
}

// =============================================================================
// Unary Op Reference Implementation Tests
// =============================================================================

class UnaryOpReferenceTest : public ::testing::Test {};

TEST_F(UnaryOpReferenceTest, ExpReference) {
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_exp(0.0f), 1.0f);
    EXPECT_NEAR(TBElementUnaryOp::apply_exp(1.0f), 2.718f, 0.01f);
}

TEST_F(UnaryOpReferenceTest, SquareReference) {
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_square(2.0f), 4.0f);
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_square(-3.0f), 9.0f);
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_square(0.0f), 0.0f);
}

TEST_F(UnaryOpReferenceTest, SqrtReference) {
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_sqrt(4.0f), 2.0f);
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_sqrt(9.0f), 3.0f);
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_sqrt(0.0f), 0.0f);
}

TEST_F(UnaryOpReferenceTest, ReluReference) {
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_relu(1.0f), 1.0f);
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_relu(-1.0f), 0.0f);
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_relu(0.0f), 0.0f);
}

TEST_F(UnaryOpReferenceTest, ClampReference) {
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_clamp(0.5f, 0.0f, 1.0f), 0.5f);
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_clamp(-1.0f, 0.0f, 1.0f), 0.0f);
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_clamp(2.0f, 0.0f, 1.0f), 1.0f);
}

TEST_F(UnaryOpReferenceTest, SiluReference) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    float x = 1.0f;
    float expected = x / (1.0f + std::exp(-x));
    EXPECT_NEAR(TBElementUnaryOp::apply_silu(x), expected, 1e-5f);
}

TEST_F(UnaryOpReferenceTest, MulScalarReference) {
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_mul_scalar(2.0f, 3.0f), 6.0f);
    EXPECT_FLOAT_EQ(TBElementUnaryOp::apply_mul_scalar(-1.0f, 2.0f), -2.0f);
}

// =============================================================================
// Element Binary Op Tests
// =============================================================================

class TBElementBinaryOpTest : public ::testing::Test {};

TEST_F(TBElementBinaryOpTest, AddOp) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 64; B.dim[1] = 128;

    TBElementBinaryOp op(A, B, TB_ADD_OP);

    EXPECT_EQ(op.op_type(), TB_ADD_OP);
    EXPECT_TRUE(op.output().same_shape(A));
}

TEST_F(TBElementBinaryOpTest, MulOp) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 32; A.dim[1] = 64;
    B.num_dims = 2; B.dim[0] = 32; B.dim[1] = 64;

    TBElementBinaryOp op(A, B, TB_MUL_OP);

    EXPECT_EQ(op.op_type(), TB_MUL_OP);
}

TEST_F(TBElementBinaryOpTest, DivOp) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 64; B.dim[1] = 128;

    TBElementBinaryOp op(A, B, TB_DIV_OP);

    EXPECT_EQ(op.op_type(), TB_DIV_OP);
}

TEST_F(TBElementBinaryOpTest, SubOp) {
    STensor A, B;
    A.num_dims = 3; A.dim[0] = 8; A.dim[1] = 32; A.dim[2] = 64;
    B.num_dims = 3; B.dim[0] = 8; B.dim[1] = 32; B.dim[2] = 64;

    TBElementBinaryOp op(A, B, TB_SUB_OP);

    EXPECT_EQ(op.op_type(), TB_SUB_OP);
    EXPECT_EQ(op.output().num_dims, 3);
}

TEST_F(TBElementBinaryOpTest, PowOp) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 64;
    B.num_dims = 2; B.dim[0] = 64; B.dim[1] = 64;

    TBElementBinaryOp op(A, B, TB_POW_OP);

    EXPECT_EQ(op.op_type(), TB_POW_OP);
}

TEST_F(TBElementBinaryOpTest, IsBinaryOp) {
    EXPECT_TRUE(TBElementBinaryOp::is_binary_op(TB_ADD_OP));
    EXPECT_TRUE(TBElementBinaryOp::is_binary_op(TB_MUL_OP));
    EXPECT_TRUE(TBElementBinaryOp::is_binary_op(TB_DIV_OP));
    EXPECT_TRUE(TBElementBinaryOp::is_binary_op(TB_POW_OP));
    EXPECT_FALSE(TBElementBinaryOp::is_binary_op(TB_EXP_OP));
}

// =============================================================================
// Shape Compatibility Tests
// =============================================================================

class BinaryOpShapeTest : public ::testing::Test {};

TEST_F(BinaryOpShapeTest, SameShapes) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 64; B.dim[1] = 128;

    EXPECT_TRUE(TBElementBinaryOp::shapes_compatible(A, B));
}

TEST_F(BinaryOpShapeTest, BroadcastDim1) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 64; B.dim[1] = 1;  // Broadcast dim 1

    EXPECT_TRUE(TBElementBinaryOp::shapes_compatible(A, B));
}

TEST_F(BinaryOpShapeTest, BroadcastDim0) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 1; B.dim[1] = 128;  // Broadcast dim 0

    EXPECT_TRUE(TBElementBinaryOp::shapes_compatible(A, B));
}

TEST_F(BinaryOpShapeTest, IncompatibleShapes) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 32; B.dim[1] = 128;  // Different, not broadcast

    EXPECT_FALSE(TBElementBinaryOp::shapes_compatible(A, B));
}

TEST_F(BinaryOpShapeTest, DifferentNumDims) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 3; B.dim[0] = 1; B.dim[1] = 64; B.dim[2] = 128;

    EXPECT_FALSE(TBElementBinaryOp::shapes_compatible(A, B));
}

// =============================================================================
// Binary Op Reference Implementation Tests
// =============================================================================

class BinaryOpReferenceTest : public ::testing::Test {};

TEST_F(BinaryOpReferenceTest, AddReference) {
    EXPECT_FLOAT_EQ(TBElementBinaryOp::apply_add(2.0f, 3.0f), 5.0f);
    EXPECT_FLOAT_EQ(TBElementBinaryOp::apply_add(-1.0f, 1.0f), 0.0f);
}

TEST_F(BinaryOpReferenceTest, MulReference) {
    EXPECT_FLOAT_EQ(TBElementBinaryOp::apply_mul(2.0f, 3.0f), 6.0f);
    EXPECT_FLOAT_EQ(TBElementBinaryOp::apply_mul(-2.0f, 3.0f), -6.0f);
}

TEST_F(BinaryOpReferenceTest, DivReference) {
    EXPECT_FLOAT_EQ(TBElementBinaryOp::apply_div(6.0f, 2.0f), 3.0f);
    EXPECT_FLOAT_EQ(TBElementBinaryOp::apply_div(1.0f, 4.0f), 0.25f);
}

TEST_F(BinaryOpReferenceTest, SubReference) {
    EXPECT_FLOAT_EQ(TBElementBinaryOp::apply_sub(5.0f, 3.0f), 2.0f);
    EXPECT_FLOAT_EQ(TBElementBinaryOp::apply_sub(3.0f, 5.0f), -2.0f);
}

TEST_F(BinaryOpReferenceTest, PowReference) {
    EXPECT_FLOAT_EQ(TBElementBinaryOp::apply_pow(2.0f, 3.0f), 8.0f);
    EXPECT_FLOAT_EQ(TBElementBinaryOp::apply_pow(4.0f, 0.5f), 2.0f);
}

// =============================================================================
// Parameterized Unary Op Tests
// =============================================================================

struct UnaryOpParam {
    TBOperatorType op_type;
    const char* name;
};

class UnaryOpParameterizedTest : public ::testing::TestWithParam<UnaryOpParam> {};

TEST_P(UnaryOpParameterizedTest, ShapePreservation) {
    auto param = GetParam();

    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBElementUnaryOp op(input, param.op_type);

    EXPECT_TRUE(op.output().same_shape(input));
    EXPECT_TRUE(TBElementUnaryOp::is_unary_op(param.op_type));
}

INSTANTIATE_TEST_SUITE_P(
    AllUnaryOps,
    UnaryOpParameterizedTest,
    ::testing::Values(
        UnaryOpParam{TB_EXP_OP, "exp"},
        UnaryOpParam{TB_SQUARE_OP, "square"},
        UnaryOpParam{TB_SQRT_OP, "sqrt"},
        UnaryOpParam{TB_SILU_OP, "silu"},
        UnaryOpParam{TB_GELU_OP, "gelu"},
        UnaryOpParam{TB_RELU_OP, "relu"},
        UnaryOpParam{TB_CLAMP_OP, "clamp"}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
