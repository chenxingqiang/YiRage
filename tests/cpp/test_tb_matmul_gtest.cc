// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_tb_matmul_gtest.cc
 * @brief Threadblock Matmul Operation Unit Tests
 *
 * Tests for TBMatmulOp class:
 *   - Matrix multiplication shapes
 *   - Layout compatibility
 *   - Batched matmul
 *   - Data type handling
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <vector>

namespace yirage {
namespace type {

enum DataType {
    DT_UNKNOWN = 0,
    DT_INT8 = 1,
    DT_BFLOAT16 = 2,
    DT_FLOAT16 = 3,
    DT_FLOAT32 = 4,
};

enum TBOperatorType {
    TB_MATMUL_OP = 200,
};

}  // namespace type

namespace layout {

enum SmemLayout {
    SMEM_LAYOUT_UNKNOWN = 0,
    SMEM_LAYOUT_ROW_MAJOR = 1,
    SMEM_LAYOUT_COL_MAJOR = 2,
    SMEM_LAYOUT_SWIZZLE_128B = 3,
    SMEM_LAYOUT_SWIZZLE_64B = 4,
};

}  // namespace layout

namespace threadblock {

constexpr int MAX_TENSOR_DIMS = 4;

struct STensor {
    type::DataType data_type = type::DT_UNKNOWN;
    layout::SmemLayout layout = layout::SMEM_LAYOUT_UNKNOWN;
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

    size_t size() const {
        size_t dtype_size = 4;
        switch (data_type) {
            case type::DT_INT8: dtype_size = 1; break;
            case type::DT_BFLOAT16:
            case type::DT_FLOAT16: dtype_size = 2; break;
            case type::DT_FLOAT32: dtype_size = 4; break;
            default: break;
        }
        return num_elements() * dtype_size;
    }
};

// =============================================================================
// TBMatmulOp
// =============================================================================

class TBMatmulOp {
public:
    TBMatmulOp(STensor const& A, STensor const& B) : A_(A), B_(B) {
        // Compute output shape
        // A: [M, K] or [batch, M, K]
        // B: [K, N] or [batch, K, N]
        // C: [M, N] or [batch, M, N]

        output_.data_type = A.data_type;
        output_.layout = A.layout;

        if (A.num_dims == 2) {
            output_.num_dims = 2;
            output_.dim[0] = A.dim[0];  // M
            output_.dim[1] = B.dim[1];  // N
        } else if (A.num_dims == 3) {
            output_.num_dims = 3;
            output_.dim[0] = A.dim[0];  // batch
            output_.dim[1] = A.dim[1];  // M
            output_.dim[2] = B.dim[2];  // N
        } else if (A.num_dims == 4) {
            output_.num_dims = 4;
            output_.dim[0] = A.dim[0];  // batch0
            output_.dim[1] = A.dim[1];  // batch1
            output_.dim[2] = A.dim[2];  // M
            output_.dim[3] = B.dim[3];  // N
        }
    }

    static bool compatible_layouts(STensor const& A, STensor const& B) {
        // Basic compatibility: same layout type or compatible swizzle patterns
        if (A.layout == B.layout) return true;

        // Row major and swizzle are compatible
        if (A.layout == layout::SMEM_LAYOUT_ROW_MAJOR &&
            (B.layout == layout::SMEM_LAYOUT_SWIZZLE_128B ||
             B.layout == layout::SMEM_LAYOUT_SWIZZLE_64B)) {
            return true;
        }

        return false;
    }

    static bool valid_matmul_shapes(STensor const& A, STensor const& B) {
        if (A.num_dims != B.num_dims) return false;

        // Check K dimension matches
        if (A.num_dims == 2) {
            return A.dim[1] == B.dim[0];  // K
        } else if (A.num_dims == 3) {
            return A.dim[0] == B.dim[0] &&  // batch
                   A.dim[2] == B.dim[1];    // K
        } else if (A.num_dims == 4) {
            return A.dim[0] == B.dim[0] &&  // batch0
                   A.dim[1] == B.dim[1] &&  // batch1
                   A.dim[3] == B.dim[2];    // K
        }
        return false;
    }

    static bool valid_data_types(STensor const& A, STensor const& B) {
        // Same data type
        if (A.data_type == B.data_type) return true;

        // FP16/BF16 mixed is allowed
        if ((A.data_type == type::DT_FLOAT16 || A.data_type == type::DT_BFLOAT16) &&
            (B.data_type == type::DT_FLOAT16 || B.data_type == type::DT_BFLOAT16)) {
            return true;
        }

        return false;
    }

    STensor const& output() const { return output_; }
    STensor const& input_a() const { return A_; }
    STensor const& input_b() const { return B_; }

    int get_M() const {
        if (A_.num_dims == 2) return A_.dim[0];
        if (A_.num_dims == 3) return A_.dim[1];
        if (A_.num_dims == 4) return A_.dim[2];
        return 0;
    }

    int get_N() const {
        if (B_.num_dims == 2) return B_.dim[1];
        if (B_.num_dims == 3) return B_.dim[2];
        if (B_.num_dims == 4) return B_.dim[3];
        return 0;
    }

    int get_K() const {
        if (A_.num_dims == 2) return A_.dim[1];
        if (A_.num_dims == 3) return A_.dim[2];
        if (A_.num_dims == 4) return A_.dim[3];
        return 0;
    }

    size_t compute_flops() const {
        size_t batch = 1;
        if (A_.num_dims == 3) batch = A_.dim[0];
        if (A_.num_dims == 4) batch = A_.dim[0] * A_.dim[1];

        return batch * 2 * get_M() * get_N() * get_K();
    }

private:
    STensor A_, B_, output_;
};

}  // namespace threadblock
}  // namespace yirage

using namespace yirage::threadblock;
using namespace yirage::type;
using namespace yirage::layout;

// =============================================================================
// Basic Matmul Tests
// =============================================================================

class TBMatmulBasicTest : public ::testing::Test {};

TEST_F(TBMatmulBasicTest, Shape2D) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 128; B.dim[1] = 256;
    A.data_type = DT_FLOAT16;
    B.data_type = DT_FLOAT16;

    TBMatmulOp matmul(A, B);
    auto output = matmul.output();

    EXPECT_EQ(output.num_dims, 2);
    EXPECT_EQ(output.dim[0], 64);   // M
    EXPECT_EQ(output.dim[1], 256);  // N
}

TEST_F(TBMatmulBasicTest, Shape3DBatched) {
    STensor A, B;
    A.num_dims = 3; A.dim[0] = 8; A.dim[1] = 64; A.dim[2] = 128;
    B.num_dims = 3; B.dim[0] = 8; B.dim[1] = 128; B.dim[2] = 256;
    A.data_type = DT_FLOAT16;
    B.data_type = DT_FLOAT16;

    TBMatmulOp matmul(A, B);
    auto output = matmul.output();

    EXPECT_EQ(output.num_dims, 3);
    EXPECT_EQ(output.dim[0], 8);    // batch
    EXPECT_EQ(output.dim[1], 64);   // M
    EXPECT_EQ(output.dim[2], 256);  // N
}

TEST_F(TBMatmulBasicTest, Shape4DBatched) {
    STensor A, B;
    A.num_dims = 4; A.dim[0] = 2; A.dim[1] = 4; A.dim[2] = 32; A.dim[3] = 64;
    B.num_dims = 4; B.dim[0] = 2; B.dim[1] = 4; B.dim[2] = 64; B.dim[3] = 128;
    A.data_type = DT_FLOAT16;
    B.data_type = DT_FLOAT16;

    TBMatmulOp matmul(A, B);
    auto output = matmul.output();

    EXPECT_EQ(output.num_dims, 4);
    EXPECT_EQ(output.dim[0], 2);    // batch0
    EXPECT_EQ(output.dim[1], 4);    // batch1
    EXPECT_EQ(output.dim[2], 32);   // M
    EXPECT_EQ(output.dim[3], 128);  // N
}

TEST_F(TBMatmulBasicTest, GetMNK) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 128; B.dim[1] = 256;

    TBMatmulOp matmul(A, B);

    EXPECT_EQ(matmul.get_M(), 64);
    EXPECT_EQ(matmul.get_N(), 256);
    EXPECT_EQ(matmul.get_K(), 128);
}

// =============================================================================
// Shape Validation Tests
// =============================================================================

class TBMatmulShapeValidationTest : public ::testing::Test {};

TEST_F(TBMatmulShapeValidationTest, ValidShapes2D) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 128; B.dim[1] = 256;

    EXPECT_TRUE(TBMatmulOp::valid_matmul_shapes(A, B));
}

TEST_F(TBMatmulShapeValidationTest, InvalidShapes2DKMismatch) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 64; B.dim[1] = 256;  // K mismatch

    EXPECT_FALSE(TBMatmulOp::valid_matmul_shapes(A, B));
}

TEST_F(TBMatmulShapeValidationTest, InvalidShapesDimMismatch) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 3; B.dim[0] = 1; B.dim[1] = 128; B.dim[2] = 256;

    EXPECT_FALSE(TBMatmulOp::valid_matmul_shapes(A, B));
}

TEST_F(TBMatmulShapeValidationTest, ValidShapes3DBatched) {
    STensor A, B;
    A.num_dims = 3; A.dim[0] = 8; A.dim[1] = 64; A.dim[2] = 128;
    B.num_dims = 3; B.dim[0] = 8; B.dim[1] = 128; B.dim[2] = 256;

    EXPECT_TRUE(TBMatmulOp::valid_matmul_shapes(A, B));
}

TEST_F(TBMatmulShapeValidationTest, InvalidShapes3DBatchMismatch) {
    STensor A, B;
    A.num_dims = 3; A.dim[0] = 8; A.dim[1] = 64; A.dim[2] = 128;
    B.num_dims = 3; B.dim[0] = 4; B.dim[1] = 128; B.dim[2] = 256;  // Batch mismatch

    EXPECT_FALSE(TBMatmulOp::valid_matmul_shapes(A, B));
}

// =============================================================================
// Layout Compatibility Tests
// =============================================================================

class TBMatmulLayoutTest : public ::testing::Test {};

TEST_F(TBMatmulLayoutTest, SameLayout) {
    STensor A, B;
    A.layout = SMEM_LAYOUT_ROW_MAJOR;
    B.layout = SMEM_LAYOUT_ROW_MAJOR;

    EXPECT_TRUE(TBMatmulOp::compatible_layouts(A, B));
}

TEST_F(TBMatmulLayoutTest, RowMajorWithSwizzle128B) {
    STensor A, B;
    A.layout = SMEM_LAYOUT_ROW_MAJOR;
    B.layout = SMEM_LAYOUT_SWIZZLE_128B;

    EXPECT_TRUE(TBMatmulOp::compatible_layouts(A, B));
}

TEST_F(TBMatmulLayoutTest, RowMajorWithSwizzle64B) {
    STensor A, B;
    A.layout = SMEM_LAYOUT_ROW_MAJOR;
    B.layout = SMEM_LAYOUT_SWIZZLE_64B;

    EXPECT_TRUE(TBMatmulOp::compatible_layouts(A, B));
}

TEST_F(TBMatmulLayoutTest, IncompatibleLayouts) {
    STensor A, B;
    A.layout = SMEM_LAYOUT_COL_MAJOR;
    B.layout = SMEM_LAYOUT_SWIZZLE_128B;

    EXPECT_FALSE(TBMatmulOp::compatible_layouts(A, B));
}

// =============================================================================
// Data Type Tests
// =============================================================================

class TBMatmulDataTypeTest : public ::testing::Test {};

TEST_F(TBMatmulDataTypeTest, SameDataType) {
    STensor A, B;
    A.data_type = DT_FLOAT16;
    B.data_type = DT_FLOAT16;

    EXPECT_TRUE(TBMatmulOp::valid_data_types(A, B));
}

TEST_F(TBMatmulDataTypeTest, MixedFP16BF16) {
    STensor A, B;
    A.data_type = DT_FLOAT16;
    B.data_type = DT_BFLOAT16;

    EXPECT_TRUE(TBMatmulOp::valid_data_types(A, B));
}

TEST_F(TBMatmulDataTypeTest, Float32) {
    STensor A, B;
    A.data_type = DT_FLOAT32;
    B.data_type = DT_FLOAT32;

    EXPECT_TRUE(TBMatmulOp::valid_data_types(A, B));
}

TEST_F(TBMatmulDataTypeTest, IncompatibleTypes) {
    STensor A, B;
    A.data_type = DT_FLOAT32;
    B.data_type = DT_INT8;

    EXPECT_FALSE(TBMatmulOp::valid_data_types(A, B));
}

TEST_F(TBMatmulDataTypeTest, OutputDataType) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 128; B.dim[1] = 256;
    A.data_type = DT_BFLOAT16;
    B.data_type = DT_BFLOAT16;

    TBMatmulOp matmul(A, B);

    EXPECT_EQ(matmul.output().data_type, DT_BFLOAT16);
}

// =============================================================================
// FLOPS Computation Tests
// =============================================================================

class TBMatmulFlopsTest : public ::testing::Test {};

TEST_F(TBMatmulFlopsTest, BasicFlops2D) {
    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;  // M=64, K=128
    B.num_dims = 2; B.dim[0] = 128; B.dim[1] = 256; // K=128, N=256

    TBMatmulOp matmul(A, B);

    // FLOPs = 2 * M * N * K
    size_t expected = 2 * 64 * 256 * 128;
    EXPECT_EQ(matmul.compute_flops(), expected);
}

TEST_F(TBMatmulFlopsTest, BatchedFlops3D) {
    STensor A, B;
    A.num_dims = 3; A.dim[0] = 8; A.dim[1] = 64; A.dim[2] = 128;
    B.num_dims = 3; B.dim[0] = 8; B.dim[1] = 128; B.dim[2] = 256;

    TBMatmulOp matmul(A, B);

    // FLOPs = batch * 2 * M * N * K
    size_t expected = 8 * 2 * 64 * 256 * 128;
    EXPECT_EQ(matmul.compute_flops(), expected);
}

TEST_F(TBMatmulFlopsTest, BatchedFlops4D) {
    STensor A, B;
    A.num_dims = 4; A.dim[0] = 2; A.dim[1] = 4; A.dim[2] = 32; A.dim[3] = 64;
    B.num_dims = 4; B.dim[0] = 2; B.dim[1] = 4; B.dim[2] = 64; B.dim[3] = 128;

    TBMatmulOp matmul(A, B);

    // FLOPs = batch0 * batch1 * 2 * M * N * K
    size_t expected = 2 * 4 * 2 * 32 * 128 * 64;
    EXPECT_EQ(matmul.compute_flops(), expected);
}

// =============================================================================
// Parameterized Shape Tests
// =============================================================================

struct MatmulShapeParam {
    int M, K, N;
    int expected_output_dim0;
    int expected_output_dim1;
};

class MatmulShapeParameterizedTest
    : public ::testing::TestWithParam<MatmulShapeParam> {};

TEST_P(MatmulShapeParameterizedTest, OutputShape) {
    auto param = GetParam();

    STensor A, B;
    A.num_dims = 2; A.dim[0] = param.M; A.dim[1] = param.K;
    B.num_dims = 2; B.dim[0] = param.K; B.dim[1] = param.N;
    A.data_type = DT_FLOAT16;
    B.data_type = DT_FLOAT16;

    TBMatmulOp matmul(A, B);
    auto output = matmul.output();

    EXPECT_EQ(output.dim[0], param.expected_output_dim0);
    EXPECT_EQ(output.dim[1], param.expected_output_dim1);
}

INSTANTIATE_TEST_SUITE_P(
    CommonMatmulShapes,
    MatmulShapeParameterizedTest,
    ::testing::Values(
        MatmulShapeParam{64, 64, 64, 64, 64},
        MatmulShapeParam{128, 64, 256, 128, 256},
        MatmulShapeParam{32, 128, 512, 32, 512},
        MatmulShapeParam{1024, 1024, 1024, 1024, 1024},
        MatmulShapeParam{16, 4096, 16, 16, 16}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
