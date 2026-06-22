// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_tb_smem_tensor_gtest.cc
 * @brief Threadblock Shared Memory Tensor Unit Tests
 *
 * Tests for STensor (shared memory tensor) components:
 *   - STensor construction and initialization
 *   - Dimension handling (1D, 2D, 3D, 4D)
 *   - Data type support
 *   - Memory layout
 *   - Size calculation
 *   - Equality comparison
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <functional>

namespace yirage {
namespace type {

enum DataType {
    DT_UNKNOWN = 0,
    DT_INT8 = 1,
    DT_BFLOAT16 = 2,
    DT_FLOAT16 = 3,
    DT_FLOAT32 = 4,
    DT_INT32 = 5,
    DT_INT64 = 6,
    DT_DOUBLE = 7,
};

using GuidType = int64_t;

}  // namespace type

namespace layout {

enum SmemLayout {
    SMEM_LAYOUT_UNKNOWN = 0,
    SMEM_LAYOUT_ROW_MAJOR = 1,
    SMEM_LAYOUT_COL_MAJOR = 2,
    SMEM_LAYOUT_SWIZZLE_128B = 3,
    SMEM_LAYOUT_SWIZZLE_64B = 4,
    SMEM_LAYOUT_SWIZZLE_32B = 5,
};

}  // namespace layout

namespace threadblock {

constexpr int MAX_TENSOR_DIMS = 4;

class TBOperator;

// Mock STensor structure
struct STensor {
    STensor() {
        data_type = type::DT_UNKNOWN;
        layout = layout::SMEM_LAYOUT_UNKNOWN;
        num_dims = 0;
        for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
            dim[i] = 0;
        }
        owner_op = nullptr;
        owner_ts_idx = -1000;
        smem_offset = 128;
        after_accum = false;
        store_in_dmem = false;
        guid = next_guid_++;
    }

    bool operator==(STensor const& b) const {
        if (data_type != b.data_type) return false;
        if (layout != b.layout) return false;
        if (num_dims != b.num_dims) return false;
        for (int i = 0; i < num_dims; i++) {
            if (dim[i] != b.dim[i]) return false;
        }
        if (owner_op != b.owner_op) return false;
        if (owner_ts_idx != b.owner_ts_idx) return false;
        if (smem_offset != b.smem_offset) return false;
        return true;
    }

    bool operator!=(STensor const& b) const {
        return !(*this == b);
    }

    size_t size() const {
        if (num_dims == 0) return 0;

        size_t num_elements = 1;
        size_t data_type_size = 1;

        switch (data_type) {
            case type::DT_INT8:
                data_type_size = 1;
                break;
            case type::DT_BFLOAT16:
            case type::DT_FLOAT16:
                data_type_size = 2;
                break;
            case type::DT_FLOAT32:
            case type::DT_INT32:
                data_type_size = 4;
                break;
            case type::DT_INT64:
            case type::DT_DOUBLE:
                data_type_size = 8;
                break;
            default:
                data_type_size = 0;
        }

        for (int i = 0; i < num_dims; i++) {
            num_elements *= dim[i];
        }
        return num_elements * data_type_size;
    }

    size_t num_elements() const {
        if (num_dims == 0) return 0;
        if (num_dims == 4) return dim[0] * dim[1] * dim[2] * dim[3];
        if (num_dims == 3) return dim[0] * dim[1] * dim[2];
        if (num_dims == 2) return dim[0] * dim[1];
        return dim[0];
    }

    type::DataType data_type;
    layout::SmemLayout layout;
    int num_dims;
    int dim[MAX_TENSOR_DIMS];
    type::GuidType guid;
    TBOperator* owner_op;
    int owner_ts_idx;
    int smem_offset;
    bool after_accum;
    bool store_in_dmem;

    static int64_t next_guid_;
};

int64_t STensor::next_guid_ = 0;

// Hash specialization
size_t hash_stensor(STensor const& t) {
    size_t h = std::hash<int>()(t.num_dims);
    for (int i = 0; i < t.num_dims; i++) {
        h ^= std::hash<int>()(t.dim[i]) << (i + 1);
    }
    h ^= std::hash<int>()(static_cast<int>(t.data_type)) << 5;
    h ^= std::hash<int>()(static_cast<int>(t.layout)) << 6;
    return h;
}

}  // namespace threadblock
}  // namespace yirage

using namespace yirage::threadblock;
using namespace yirage::type;
using namespace yirage::layout;

// =============================================================================
// STensor Construction Tests
// =============================================================================

class STensorConstructionTest : public ::testing::Test {
protected:
    void SetUp() override {
        STensor::next_guid_ = 0;
    }
};

TEST_F(STensorConstructionTest, DefaultConstruction) {
    STensor tensor;

    EXPECT_EQ(tensor.data_type, DT_UNKNOWN);
    EXPECT_EQ(tensor.layout, SMEM_LAYOUT_UNKNOWN);
    EXPECT_EQ(tensor.num_dims, 0);
    EXPECT_EQ(tensor.owner_op, nullptr);
    EXPECT_EQ(tensor.owner_ts_idx, -1000);
    EXPECT_EQ(tensor.smem_offset, 128);
    EXPECT_FALSE(tensor.after_accum);
    EXPECT_FALSE(tensor.store_in_dmem);
}

TEST_F(STensorConstructionTest, GuidAssignment) {
    STensor t1;
    STensor t2;
    STensor t3;

    EXPECT_NE(t1.guid, t2.guid);
    EXPECT_NE(t2.guid, t3.guid);
    EXPECT_EQ(t2.guid, t1.guid + 1);
    EXPECT_EQ(t3.guid, t2.guid + 1);
}

TEST_F(STensorConstructionTest, DimensionInitialization) {
    STensor tensor;

    for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
        EXPECT_EQ(tensor.dim[i], 0);
    }
}

// =============================================================================
// STensor Dimension Tests
// =============================================================================

class STensorDimensionTest : public ::testing::Test {};

TEST_F(STensorDimensionTest, OneDimensionalTensor) {
    STensor tensor;
    tensor.num_dims = 1;
    tensor.dim[0] = 128;
    tensor.data_type = DT_FLOAT32;

    EXPECT_EQ(tensor.num_elements(), 128u);
    EXPECT_EQ(tensor.size(), 128u * 4u);
}

TEST_F(STensorDimensionTest, TwoDimensionalTensor) {
    STensor tensor;
    tensor.num_dims = 2;
    tensor.dim[0] = 64;
    tensor.dim[1] = 128;
    tensor.data_type = DT_FLOAT16;

    EXPECT_EQ(tensor.num_elements(), 64u * 128u);
    EXPECT_EQ(tensor.size(), 64u * 128u * 2u);
}

TEST_F(STensorDimensionTest, ThreeDimensionalTensor) {
    STensor tensor;
    tensor.num_dims = 3;
    tensor.dim[0] = 8;
    tensor.dim[1] = 32;
    tensor.dim[2] = 64;
    tensor.data_type = DT_BFLOAT16;

    EXPECT_EQ(tensor.num_elements(), 8u * 32u * 64u);
    EXPECT_EQ(tensor.size(), 8u * 32u * 64u * 2u);
}

TEST_F(STensorDimensionTest, FourDimensionalTensor) {
    STensor tensor;
    tensor.num_dims = 4;
    tensor.dim[0] = 2;
    tensor.dim[1] = 4;
    tensor.dim[2] = 16;
    tensor.dim[3] = 32;
    tensor.data_type = DT_DOUBLE;

    EXPECT_EQ(tensor.num_elements(), 2u * 4u * 16u * 32u);
    EXPECT_EQ(tensor.size(), 2u * 4u * 16u * 32u * 8u);
}

TEST_F(STensorDimensionTest, EmptyTensor) {
    STensor tensor;
    tensor.num_dims = 0;

    EXPECT_EQ(tensor.num_elements(), 0u);
    EXPECT_EQ(tensor.size(), 0u);
}

// =============================================================================
// STensor Data Type Tests
// =============================================================================

class STensorDataTypeTest : public ::testing::Test {};

TEST_F(STensorDataTypeTest, Int8Size) {
    STensor tensor;
    tensor.num_dims = 2;
    tensor.dim[0] = 100;
    tensor.dim[1] = 100;
    tensor.data_type = DT_INT8;

    EXPECT_EQ(tensor.size(), 100u * 100u * 1u);
}

TEST_F(STensorDataTypeTest, Float16Size) {
    STensor tensor;
    tensor.num_dims = 2;
    tensor.dim[0] = 100;
    tensor.dim[1] = 100;
    tensor.data_type = DT_FLOAT16;

    EXPECT_EQ(tensor.size(), 100u * 100u * 2u);
}

TEST_F(STensorDataTypeTest, BFloat16Size) {
    STensor tensor;
    tensor.num_dims = 2;
    tensor.dim[0] = 100;
    tensor.dim[1] = 100;
    tensor.data_type = DT_BFLOAT16;

    EXPECT_EQ(tensor.size(), 100u * 100u * 2u);
}

TEST_F(STensorDataTypeTest, Float32Size) {
    STensor tensor;
    tensor.num_dims = 2;
    tensor.dim[0] = 100;
    tensor.dim[1] = 100;
    tensor.data_type = DT_FLOAT32;

    EXPECT_EQ(tensor.size(), 100u * 100u * 4u);
}

TEST_F(STensorDataTypeTest, Int32Size) {
    STensor tensor;
    tensor.num_dims = 2;
    tensor.dim[0] = 100;
    tensor.dim[1] = 100;
    tensor.data_type = DT_INT32;

    EXPECT_EQ(tensor.size(), 100u * 100u * 4u);
}

TEST_F(STensorDataTypeTest, Int64Size) {
    STensor tensor;
    tensor.num_dims = 2;
    tensor.dim[0] = 100;
    tensor.dim[1] = 100;
    tensor.data_type = DT_INT64;

    EXPECT_EQ(tensor.size(), 100u * 100u * 8u);
}

TEST_F(STensorDataTypeTest, DoubleSize) {
    STensor tensor;
    tensor.num_dims = 2;
    tensor.dim[0] = 100;
    tensor.dim[1] = 100;
    tensor.data_type = DT_DOUBLE;

    EXPECT_EQ(tensor.size(), 100u * 100u * 8u);
}

// =============================================================================
// STensor Layout Tests
// =============================================================================

class STensorLayoutTest : public ::testing::Test {};

TEST_F(STensorLayoutTest, RowMajorLayout) {
    STensor tensor;
    tensor.layout = SMEM_LAYOUT_ROW_MAJOR;

    EXPECT_EQ(tensor.layout, SMEM_LAYOUT_ROW_MAJOR);
}

TEST_F(STensorLayoutTest, ColMajorLayout) {
    STensor tensor;
    tensor.layout = SMEM_LAYOUT_COL_MAJOR;

    EXPECT_EQ(tensor.layout, SMEM_LAYOUT_COL_MAJOR);
}

TEST_F(STensorLayoutTest, Swizzle128BLayout) {
    STensor tensor;
    tensor.layout = SMEM_LAYOUT_SWIZZLE_128B;

    EXPECT_EQ(tensor.layout, SMEM_LAYOUT_SWIZZLE_128B);
}

TEST_F(STensorLayoutTest, Swizzle64BLayout) {
    STensor tensor;
    tensor.layout = SMEM_LAYOUT_SWIZZLE_64B;

    EXPECT_EQ(tensor.layout, SMEM_LAYOUT_SWIZZLE_64B);
}

TEST_F(STensorLayoutTest, Swizzle32BLayout) {
    STensor tensor;
    tensor.layout = SMEM_LAYOUT_SWIZZLE_32B;

    EXPECT_EQ(tensor.layout, SMEM_LAYOUT_SWIZZLE_32B);
}

// =============================================================================
// STensor Equality Tests
// =============================================================================

class STensorEqualityTest : public ::testing::Test {};

TEST_F(STensorEqualityTest, EqualTensors) {
    STensor t1, t2;

    t1.data_type = DT_FLOAT32;
    t1.layout = SMEM_LAYOUT_ROW_MAJOR;
    t1.num_dims = 2;
    t1.dim[0] = 64;
    t1.dim[1] = 128;
    t1.smem_offset = 256;

    t2.data_type = DT_FLOAT32;
    t2.layout = SMEM_LAYOUT_ROW_MAJOR;
    t2.num_dims = 2;
    t2.dim[0] = 64;
    t2.dim[1] = 128;
    t2.smem_offset = 256;

    EXPECT_EQ(t1, t2);
}

TEST_F(STensorEqualityTest, DifferentDataType) {
    STensor t1, t2;

    t1.data_type = DT_FLOAT32;
    t2.data_type = DT_FLOAT16;

    EXPECT_NE(t1, t2);
}

TEST_F(STensorEqualityTest, DifferentLayout) {
    STensor t1, t2;

    t1.layout = SMEM_LAYOUT_ROW_MAJOR;
    t2.layout = SMEM_LAYOUT_COL_MAJOR;

    EXPECT_NE(t1, t2);
}

TEST_F(STensorEqualityTest, DifferentNumDims) {
    STensor t1, t2;

    t1.num_dims = 2;
    t2.num_dims = 3;

    EXPECT_NE(t1, t2);
}

TEST_F(STensorEqualityTest, DifferentDimensions) {
    STensor t1, t2;

    t1.num_dims = 2;
    t1.dim[0] = 64;
    t1.dim[1] = 128;

    t2.num_dims = 2;
    t2.dim[0] = 64;
    t2.dim[1] = 256;  // Different

    EXPECT_NE(t1, t2);
}

TEST_F(STensorEqualityTest, DifferentSmemOffset) {
    STensor t1, t2;

    t1.smem_offset = 128;
    t2.smem_offset = 256;

    EXPECT_NE(t1, t2);
}

TEST_F(STensorEqualityTest, AfterAccumIgnored) {
    STensor t1, t2;

    t1.after_accum = false;
    t2.after_accum = true;

    // after_accum is NOT compared, so these should be equal
    EXPECT_EQ(t1, t2);
}

// =============================================================================
// STensor Hash Tests
// =============================================================================

class STensorHashTest : public ::testing::Test {};

TEST_F(STensorHashTest, SameTensorsSameHash) {
    STensor t1, t2;

    t1.num_dims = 2;
    t1.dim[0] = 64;
    t1.dim[1] = 128;
    t1.data_type = DT_FLOAT32;
    t1.layout = SMEM_LAYOUT_ROW_MAJOR;

    t2.num_dims = 2;
    t2.dim[0] = 64;
    t2.dim[1] = 128;
    t2.data_type = DT_FLOAT32;
    t2.layout = SMEM_LAYOUT_ROW_MAJOR;

    EXPECT_EQ(hash_stensor(t1), hash_stensor(t2));
}

TEST_F(STensorHashTest, DifferentDimensionsDifferentHash) {
    STensor t1, t2;

    t1.num_dims = 2;
    t1.dim[0] = 64;
    t1.dim[1] = 128;

    t2.num_dims = 2;
    t2.dim[0] = 64;
    t2.dim[1] = 256;

    EXPECT_NE(hash_stensor(t1), hash_stensor(t2));
}

// =============================================================================
// STensor Flags Tests
// =============================================================================

class STensorFlagsTest : public ::testing::Test {};

TEST_F(STensorFlagsTest, AfterAccumFlag) {
    STensor tensor;

    EXPECT_FALSE(tensor.after_accum);

    tensor.after_accum = true;
    EXPECT_TRUE(tensor.after_accum);
}

TEST_F(STensorFlagsTest, StoreInDmemFlag) {
    STensor tensor;

    EXPECT_FALSE(tensor.store_in_dmem);

    tensor.store_in_dmem = true;
    EXPECT_TRUE(tensor.store_in_dmem);
}

// =============================================================================
// Parameterized Data Type Size Tests
// =============================================================================

struct DataTypeSizeParam {
    DataType dtype;
    size_t expected_size;
};

class DataTypeSizeParameterizedTest
    : public ::testing::TestWithParam<DataTypeSizeParam> {};

TEST_P(DataTypeSizeParameterizedTest, DataTypeSize) {
    auto param = GetParam();

    STensor tensor;
    tensor.num_dims = 1;
    tensor.dim[0] = 1;
    tensor.data_type = param.dtype;

    EXPECT_EQ(tensor.size(), param.expected_size);
}

INSTANTIATE_TEST_SUITE_P(
    AllDataTypes,
    DataTypeSizeParameterizedTest,
    ::testing::Values(
        DataTypeSizeParam{DT_INT8, 1},
        DataTypeSizeParam{DT_FLOAT16, 2},
        DataTypeSizeParam{DT_BFLOAT16, 2},
        DataTypeSizeParam{DT_FLOAT32, 4},
        DataTypeSizeParam{DT_INT32, 4},
        DataTypeSizeParam{DT_INT64, 8},
        DataTypeSizeParam{DT_DOUBLE, 8}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
