// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_tb_reduction_gtest.cc
 * @brief Threadblock Reduction Operations Unit Tests
 *
 * Tests for reduction operations:
 *   - TBReductionOp (sum reduction)
 *   - TBReductionMaxOp (max reduction with indices)
 *   - TBRmsNormOp (RMS normalization)
 *   - Reduction dimension handling
 *   - Output shape computation
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

namespace yirage {
namespace type {

enum DataType {
    DT_UNKNOWN = 0,
    DT_FLOAT16 = 3,
    DT_FLOAT32 = 4,
    DT_INT32 = 5,
};

enum TBOperatorType {
    TB_REDUCTION_OP = 500,
    TB_REDUCTION_TO_DIMX_OP = 501,
    TB_REDUCTION_MAX_OP = 502,
    TB_RMS_NORM_OP = 503,
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
};

// =============================================================================
// TBReductionOp
// =============================================================================

class TBReductionOp {
public:
    TBReductionOp(STensor const& input, int reduce_dim)
        : input_(input), reduce_dim_(reduce_dim) {
        // Validate reduce_dim
        if (reduce_dim < 0 || reduce_dim >= input.num_dims) {
            valid_ = false;
            return;
        }
        valid_ = true;

        // Compute output shape
        output_ = input;
        output_.dim[reduce_dim] = 1;

        reduce_size_ = input.dim[reduce_dim];
    }

    STensor const& input() const { return input_; }
    STensor const& output() const { return output_; }
    int reduce_dim() const { return reduce_dim_; }
    int reduce_size() const { return reduce_size_; }
    bool is_valid() const { return valid_; }

    // Reference implementation
    static std::vector<float> reduce_sum(std::vector<float> const& data,
                                         int dim0, int dim1, int reduce_dim) {
        if (reduce_dim == 0) {
            std::vector<float> result(dim1, 0.0f);
            for (int i = 0; i < dim0; i++) {
                for (int j = 0; j < dim1; j++) {
                    result[j] += data[i * dim1 + j];
                }
            }
            return result;
        } else {
            std::vector<float> result(dim0, 0.0f);
            for (int i = 0; i < dim0; i++) {
                for (int j = 0; j < dim1; j++) {
                    result[i] += data[i * dim1 + j];
                }
            }
            return result;
        }
    }

private:
    STensor input_;
    STensor output_;
    int reduce_dim_;
    int reduce_size_;
    bool valid_;
};

// =============================================================================
// TBReductionToDimxOp
// =============================================================================

class TBReductionToDimxOp {
public:
    TBReductionToDimxOp(STensor const& input, int reduce_dim, int target_dimx)
        : input_(input), reduce_dim_(reduce_dim), target_dimx_(target_dimx) {
        // Output has same shape but reduce_dim becomes target_dimx
        output_ = input;
        output_.dim[reduce_dim] = target_dimx;
    }

    STensor const& output() const { return output_; }
    int target_dimx() const { return target_dimx_; }

private:
    STensor input_;
    STensor output_;
    int reduce_dim_;
    int target_dimx_;
};

// =============================================================================
// TBReductionMaxOp
// =============================================================================

class TBReductionMaxOp {
public:
    TBReductionMaxOp(STensor const& input, int reduce_dim)
        : input_(input), reduce_dim_(reduce_dim) {
        // Output values (max values)
        output_values_ = input;
        output_values_.dim[reduce_dim] = 1;

        // Output indices (argmax)
        output_indices_ = input;
        output_indices_.dim[reduce_dim] = 1;
        output_indices_.data_type = type::DT_INT32;
    }

    STensor const& input() const { return input_; }
    STensor const& output_values() const { return output_values_; }
    STensor const& output_indices() const { return output_indices_; }
    int reduce_dim() const { return reduce_dim_; }

    bool has_two_outputs() const { return true; }

    // Reference implementation
    static std::pair<std::vector<float>, std::vector<int>>
    reduce_max(std::vector<float> const& data, int dim0, int dim1, int reduce_dim) {
        if (reduce_dim == 0) {
            std::vector<float> max_vals(dim1, -std::numeric_limits<float>::infinity());
            std::vector<int> max_idx(dim1, 0);
            for (int i = 0; i < dim0; i++) {
                for (int j = 0; j < dim1; j++) {
                    if (data[i * dim1 + j] > max_vals[j]) {
                        max_vals[j] = data[i * dim1 + j];
                        max_idx[j] = i;
                    }
                }
            }
            return {max_vals, max_idx};
        } else {
            std::vector<float> max_vals(dim0, -std::numeric_limits<float>::infinity());
            std::vector<int> max_idx(dim0, 0);
            for (int i = 0; i < dim0; i++) {
                for (int j = 0; j < dim1; j++) {
                    if (data[i * dim1 + j] > max_vals[i]) {
                        max_vals[i] = data[i * dim1 + j];
                        max_idx[i] = j;
                    }
                }
            }
            return {max_vals, max_idx};
        }
    }

private:
    STensor input_;
    STensor output_values_;
    STensor output_indices_;
    int reduce_dim_;
};

// =============================================================================
// TBRmsNormOp
// =============================================================================

class TBRmsNormOp {
public:
    TBRmsNormOp(STensor const& input, float epsilon = 1e-5f)
        : input_(input), epsilon_(epsilon) {
        // Output has same shape as input
        output_ = input;
    }

    STensor const& input() const { return input_; }
    STensor const& output() const { return output_; }
    float epsilon() const { return epsilon_; }

    // Reference implementation
    static std::vector<float> rms_norm(std::vector<float> const& data,
                                       int rows, int cols, float epsilon) {
        std::vector<float> result(data.size());

        for (int i = 0; i < rows; i++) {
            // Compute RMS for this row
            float sum_sq = 0.0f;
            for (int j = 0; j < cols; j++) {
                float val = data[i * cols + j];
                sum_sq += val * val;
            }
            float rms = std::sqrt(sum_sq / cols + epsilon);

            // Normalize
            for (int j = 0; j < cols; j++) {
                result[i * cols + j] = data[i * cols + j] / rms;
            }
        }

        return result;
    }

private:
    STensor input_;
    STensor output_;
    float epsilon_;
};

}  // namespace threadblock
}  // namespace yirage

using namespace yirage::threadblock;
using namespace yirage::type;

// =============================================================================
// TBReductionOp Tests
// =============================================================================

class TBReductionOpTest : public ::testing::Test {};

TEST_F(TBReductionOpTest, ReduceDim0) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBReductionOp op(input, 0);

    EXPECT_TRUE(op.is_valid());
    EXPECT_EQ(op.output().dim[0], 1);
    EXPECT_EQ(op.output().dim[1], 128);
    EXPECT_EQ(op.reduce_size(), 64);
}

TEST_F(TBReductionOpTest, ReduceDim1) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBReductionOp op(input, 1);

    EXPECT_TRUE(op.is_valid());
    EXPECT_EQ(op.output().dim[0], 64);
    EXPECT_EQ(op.output().dim[1], 1);
    EXPECT_EQ(op.reduce_size(), 128);
}

TEST_F(TBReductionOpTest, ReduceDim2In3D) {
    STensor input;
    input.num_dims = 3;
    input.dim[0] = 8;
    input.dim[1] = 32;
    input.dim[2] = 64;

    TBReductionOp op(input, 2);

    EXPECT_TRUE(op.is_valid());
    EXPECT_EQ(op.output().dim[0], 8);
    EXPECT_EQ(op.output().dim[1], 32);
    EXPECT_EQ(op.output().dim[2], 1);
}

TEST_F(TBReductionOpTest, InvalidReduceDim) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBReductionOp op(input, 3);  // Invalid

    EXPECT_FALSE(op.is_valid());
}

TEST_F(TBReductionOpTest, NegativeReduceDim) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBReductionOp op(input, -1);  // Invalid

    EXPECT_FALSE(op.is_valid());
}

// =============================================================================
// Reduction Reference Implementation Tests
// =============================================================================

class ReductionReferenceTest : public ::testing::Test {};

TEST_F(ReductionReferenceTest, SumReduceDim0) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};  // 2x3 matrix
    auto result = TBReductionOp::reduce_sum(data, 2, 3, 0);

    EXPECT_EQ(result.size(), 3u);
    EXPECT_FLOAT_EQ(result[0], 1 + 4);  // 5
    EXPECT_FLOAT_EQ(result[1], 2 + 5);  // 7
    EXPECT_FLOAT_EQ(result[2], 3 + 6);  // 9
}

TEST_F(ReductionReferenceTest, SumReduceDim1) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6};  // 2x3 matrix
    auto result = TBReductionOp::reduce_sum(data, 2, 3, 1);

    EXPECT_EQ(result.size(), 2u);
    EXPECT_FLOAT_EQ(result[0], 1 + 2 + 3);  // 6
    EXPECT_FLOAT_EQ(result[1], 4 + 5 + 6);  // 15
}

// =============================================================================
// TBReductionToDimxOp Tests
// =============================================================================

class TBReductionToDimxOpTest : public ::testing::Test {};

TEST_F(TBReductionToDimxOpTest, ReduceToSpecificDimx) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBReductionToDimxOp op(input, 1, 32);

    EXPECT_EQ(op.output().dim[0], 64);
    EXPECT_EQ(op.output().dim[1], 32);
    EXPECT_EQ(op.target_dimx(), 32);
}

// =============================================================================
// TBReductionMaxOp Tests
// =============================================================================

class TBReductionMaxOpTest : public ::testing::Test {};

TEST_F(TBReductionMaxOpTest, MaxReduceDim0) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;
    input.data_type = DT_FLOAT32;

    TBReductionMaxOp op(input, 0);

    EXPECT_EQ(op.output_values().dim[0], 1);
    EXPECT_EQ(op.output_values().dim[1], 128);
    EXPECT_EQ(op.output_values().data_type, DT_FLOAT32);
    EXPECT_EQ(op.output_indices().data_type, DT_INT32);
}

TEST_F(TBReductionMaxOpTest, MaxReduceDim1) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBReductionMaxOp op(input, 1);

    EXPECT_EQ(op.output_values().dim[0], 64);
    EXPECT_EQ(op.output_values().dim[1], 1);
    EXPECT_EQ(op.output_indices().dim[0], 64);
    EXPECT_EQ(op.output_indices().dim[1], 1);
}

TEST_F(TBReductionMaxOpTest, HasTwoOutputs) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 32;
    input.dim[1] = 64;

    TBReductionMaxOp op(input, 1);

    EXPECT_TRUE(op.has_two_outputs());
}

// =============================================================================
// Max Reduction Reference Tests
// =============================================================================

class MaxReductionReferenceTest : public ::testing::Test {};

TEST_F(MaxReductionReferenceTest, MaxReduceDim0) {
    std::vector<float> data = {1, 5, 3, 4, 2, 6};  // 2x3 matrix
    auto [max_vals, max_idx] = TBReductionMaxOp::reduce_max(data, 2, 3, 0);

    EXPECT_EQ(max_vals.size(), 3u);
    EXPECT_EQ(max_idx.size(), 3u);
    EXPECT_FLOAT_EQ(max_vals[0], 4);  // max(1, 4)
    EXPECT_FLOAT_EQ(max_vals[1], 5);  // max(5, 2)
    EXPECT_FLOAT_EQ(max_vals[2], 6);  // max(3, 6)
    EXPECT_EQ(max_idx[0], 1);  // row 1
    EXPECT_EQ(max_idx[1], 0);  // row 0
    EXPECT_EQ(max_idx[2], 1);  // row 1
}

TEST_F(MaxReductionReferenceTest, MaxReduceDim1) {
    std::vector<float> data = {1, 5, 3, 4, 2, 6};  // 2x3 matrix
    auto [max_vals, max_idx] = TBReductionMaxOp::reduce_max(data, 2, 3, 1);

    EXPECT_EQ(max_vals.size(), 2u);
    EXPECT_EQ(max_idx.size(), 2u);
    EXPECT_FLOAT_EQ(max_vals[0], 5);  // max(1, 5, 3)
    EXPECT_FLOAT_EQ(max_vals[1], 6);  // max(4, 2, 6)
    EXPECT_EQ(max_idx[0], 1);  // col 1
    EXPECT_EQ(max_idx[1], 2);  // col 2
}

// =============================================================================
// TBRmsNormOp Tests
// =============================================================================

class TBRmsNormOpTest : public ::testing::Test {};

TEST_F(TBRmsNormOpTest, BasicConstruction) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBRmsNormOp op(input);

    EXPECT_EQ(op.output().dim[0], 64);
    EXPECT_EQ(op.output().dim[1], 128);
    EXPECT_NEAR(op.epsilon(), 1e-5f, 1e-10f);
}

TEST_F(TBRmsNormOpTest, CustomEpsilon) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 32;
    input.dim[1] = 64;

    TBRmsNormOp op(input, 1e-6f);

    EXPECT_NEAR(op.epsilon(), 1e-6f, 1e-10f);
}

TEST_F(TBRmsNormOpTest, ShapePreserved) {
    STensor input;
    input.num_dims = 3;
    input.dim[0] = 8;
    input.dim[1] = 32;
    input.dim[2] = 64;

    TBRmsNormOp op(input);

    EXPECT_EQ(op.output().num_dims, 3);
    EXPECT_EQ(op.output().dim[0], 8);
    EXPECT_EQ(op.output().dim[1], 32);
    EXPECT_EQ(op.output().dim[2], 64);
}

// =============================================================================
// RMS Norm Reference Tests
// =============================================================================

class RmsNormReferenceTest : public ::testing::Test {};

TEST_F(RmsNormReferenceTest, SimpleNormalization) {
    // Single row: [3, 4] -> RMS = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
    std::vector<float> data = {3.0f, 4.0f};
    auto result = TBRmsNormOp::rms_norm(data, 1, 2, 1e-5f);

    float rms = std::sqrt((9.0f + 16.0f) / 2.0f + 1e-5f);
    EXPECT_NEAR(result[0], 3.0f / rms, 1e-4f);
    EXPECT_NEAR(result[1], 4.0f / rms, 1e-4f);
}

TEST_F(RmsNormReferenceTest, MultipleRows) {
    std::vector<float> data = {1.0f, 1.0f, 2.0f, 2.0f};  // 2x2
    auto result = TBRmsNormOp::rms_norm(data, 2, 2, 1e-5f);

    // Row 0: RMS = sqrt((1+1)/2) = 1
    float rms0 = std::sqrt(2.0f / 2.0f + 1e-5f);
    EXPECT_NEAR(result[0], 1.0f / rms0, 1e-4f);
    EXPECT_NEAR(result[1], 1.0f / rms0, 1e-4f);

    // Row 1: RMS = sqrt((4+4)/2) = 2
    float rms1 = std::sqrt(8.0f / 2.0f + 1e-5f);
    EXPECT_NEAR(result[2], 2.0f / rms1, 1e-4f);
    EXPECT_NEAR(result[3], 2.0f / rms1, 1e-4f);
}

// =============================================================================
// Parameterized Reduction Tests
// =============================================================================

struct ReductionParam {
    int input_dim0;
    int input_dim1;
    int reduce_dim;
    int expected_out_dim0;
    int expected_out_dim1;
};

class ReductionParameterizedTest : public ::testing::TestWithParam<ReductionParam> {};

TEST_P(ReductionParameterizedTest, OutputShape) {
    auto param = GetParam();

    STensor input;
    input.num_dims = 2;
    input.dim[0] = param.input_dim0;
    input.dim[1] = param.input_dim1;

    TBReductionOp op(input, param.reduce_dim);

    EXPECT_TRUE(op.is_valid());
    EXPECT_EQ(op.output().dim[0], param.expected_out_dim0);
    EXPECT_EQ(op.output().dim[1], param.expected_out_dim1);
}

INSTANTIATE_TEST_SUITE_P(
    CommonReductions,
    ReductionParameterizedTest,
    ::testing::Values(
        ReductionParam{64, 128, 0, 1, 128},
        ReductionParam{64, 128, 1, 64, 1},
        ReductionParam{32, 64, 0, 1, 64},
        ReductionParam{32, 64, 1, 32, 1},
        ReductionParam{128, 256, 0, 1, 256},
        ReductionParam{128, 256, 1, 128, 1}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
