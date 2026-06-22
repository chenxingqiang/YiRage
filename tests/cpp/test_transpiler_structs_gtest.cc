// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_transpiler_structs_gtest.cc
 * @brief Transpiler Metadata Structures Unit Tests
 *
 * Tests for transpiler metadata structures (structs.h):
 *   - DTensorMeta (device tensor metadata)
 *   - STensorMeta (shared memory tensor metadata)
 *   - TMAParams (TMA parameters for Hopper+)
 *   - TBMemoryPlan (threadblock memory planning)
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>

namespace yirage {
namespace transpiler {

constexpr int MAX_TENSOR_DIMS = 4;

using dguid_t = int64_t;
using sguid_t = int64_t;

// =============================================================================
// DTensorMeta
// =============================================================================

struct DTensorMeta {
    bool is_input = false;
    int input_idx = -1;

    bool is_output = false;
    int output_idx = -1;

    size_t strides[MAX_TENSOR_DIMS] = {0};

    int innermost_dim = -1;

    size_t addr = 0;

    size_t num_phy_elems = 0;

    void set_strides(std::vector<size_t> const& s) {
        for (size_t i = 0; i < s.size() && i < MAX_TENSOR_DIMS; ++i) {
            strides[i] = s[i];
        }
    }

    size_t get_stride(int dim) const {
        if (dim < 0 || dim >= MAX_TENSOR_DIMS) return 0;
        return strides[dim];
    }

    bool is_intermediate() const {
        return !is_input && !is_output;
    }

    bool has_valid_layout() const {
        return innermost_dim >= 0;
    }
};

// =============================================================================
// STensorMeta
// =============================================================================

struct STensorMeta {
    int innermost_dim = -1;
    int swizzled_dim = -1;

    size_t strides[MAX_TENSOR_DIMS] = {0};

    size_t num_phy_elems = 0;

    bool is_xor_swizzled = false;

    bool m_input = false;
    bool n_input = false;

    sguid_t m_matrix_guid = 0;
    sguid_t n_matrix_guid = 0;
    sguid_t c_matrix_guid = 0;

    bool is_pipelined_input = false;

    int xor_swizzle_b = 0;
    int xor_swizzle_m = 0;
    int xor_swizzle_s = 0;

    void set_strides(std::vector<size_t> const& s) {
        for (size_t i = 0; i < s.size() && i < MAX_TENSOR_DIMS; ++i) {
            strides[i] = s[i];
        }
    }

    bool is_swizzled() const {
        return swizzled_dim >= 0 || is_xor_swizzled;
    }

    bool is_matrix_input() const {
        return m_input || n_input;
    }

    bool has_matrix_pair() const {
        return m_matrix_guid != 0 || n_matrix_guid != 0 || c_matrix_guid != 0;
    }

    void set_xor_swizzle(int b, int m, int s) {
        is_xor_swizzled = true;
        xor_swizzle_b = b;
        xor_swizzle_m = m;
        xor_swizzle_s = s;
    }
};

// =============================================================================
// TMAParams
// =============================================================================

struct TMAParams {
    size_t input_id = 0;
    size_t guid = 0;
    size_t sguid = 0;
    std::string srcLayout;
    std::string dstLayout;
    std::string tile_size;
    bool m_input = false;
    std::tuple<int, int, int> clusterSize = {1, 1, 1};
    std::vector<int> original_shape;
    std::vector<size_t> original_stride;
    std::vector<int> partition_logic;
    int forloop_range = 1;
    int forloop_dim = -1;
    std::string multicast_direction = "NOT_MULTICAST";

    TMAParams() = default;

    TMAParams(size_t id, size_t g, size_t sg,
              std::string src, std::string dst,
              bool m_in, std::string tile,
              std::tuple<int, int, int> cluster,
              std::vector<int> shape,
              std::vector<size_t> stride,
              std::vector<int> partition,
              int fl_range, int fl_dim,
              std::string multicast = "NOT_MULTICAST")
        : input_id(id), guid(g), sguid(sg),
          srcLayout(std::move(src)), dstLayout(std::move(dst)),
          tile_size(std::move(tile)), m_input(m_in),
          clusterSize(cluster),
          original_shape(std::move(shape)),
          original_stride(std::move(stride)),
          partition_logic(std::move(partition)),
          forloop_range(fl_range), forloop_dim(fl_dim),
          multicast_direction(std::move(multicast)) {}

    bool is_multicast() const {
        return multicast_direction != "NOT_MULTICAST";
    }

    int get_cluster_x() const { return std::get<0>(clusterSize); }
    int get_cluster_y() const { return std::get<1>(clusterSize); }
    int get_cluster_z() const { return std::get<2>(clusterSize); }

    int get_cluster_size() const {
        return get_cluster_x() * get_cluster_y() * get_cluster_z();
    }

    bool has_forloop() const {
        return forloop_dim >= 0 && forloop_range > 1;
    }
};

// =============================================================================
// TBMemoryPlan
// =============================================================================

struct TBMemoryPlan {
    std::unordered_map<sguid_t, size_t> addrs;
    size_t smem_size = 0;
    sguid_t pipelined_input_buf_guid_offset = 0;
    sguid_t tmem_base_ptr_guid = 0;
    sguid_t mbarrier_buf_guid_offset = 0;

    void set_addr(sguid_t guid, size_t addr) {
        addrs[guid] = addr;
    }

    size_t get_addr(sguid_t guid) const {
        auto it = addrs.find(guid);
        return it != addrs.end() ? it->second : 0;
    }

    bool has_tensor(sguid_t guid) const {
        return addrs.find(guid) != addrs.end();
    }

    size_t num_tensors() const {
        return addrs.size();
    }

    bool uses_pipelining() const {
        return pipelined_input_buf_guid_offset != 0;
    }

    bool uses_tmem() const {
        return tmem_base_ptr_guid != 0;
    }
};

}  // namespace transpiler
}  // namespace yirage

using namespace yirage::transpiler;

// =============================================================================
// DTensorMeta Tests
// =============================================================================

class DTensorMetaTest : public ::testing::Test {};

TEST_F(DTensorMetaTest, DefaultValues) {
    DTensorMeta meta;

    EXPECT_FALSE(meta.is_input);
    EXPECT_EQ(meta.input_idx, -1);
    EXPECT_FALSE(meta.is_output);
    EXPECT_EQ(meta.output_idx, -1);
    EXPECT_EQ(meta.innermost_dim, -1);
    EXPECT_EQ(meta.addr, 0u);
    EXPECT_EQ(meta.num_phy_elems, 0u);
}

TEST_F(DTensorMetaTest, SetStrides) {
    DTensorMeta meta;
    meta.set_strides({128, 1});

    EXPECT_EQ(meta.strides[0], 128u);
    EXPECT_EQ(meta.strides[1], 1u);
}

TEST_F(DTensorMetaTest, GetStride) {
    DTensorMeta meta;
    meta.strides[0] = 256;
    meta.strides[1] = 1;

    EXPECT_EQ(meta.get_stride(0), 256u);
    EXPECT_EQ(meta.get_stride(1), 1u);
    EXPECT_EQ(meta.get_stride(5), 0u);  // Invalid dim
}

TEST_F(DTensorMetaTest, IsIntermediate) {
    DTensorMeta meta;
    EXPECT_TRUE(meta.is_intermediate());

    meta.is_input = true;
    EXPECT_FALSE(meta.is_intermediate());

    meta.is_input = false;
    meta.is_output = true;
    EXPECT_FALSE(meta.is_intermediate());
}

TEST_F(DTensorMetaTest, HasValidLayout) {
    DTensorMeta meta;
    EXPECT_FALSE(meta.has_valid_layout());

    meta.innermost_dim = 1;
    EXPECT_TRUE(meta.has_valid_layout());
}

TEST_F(DTensorMetaTest, InputTensor) {
    DTensorMeta meta;
    meta.is_input = true;
    meta.input_idx = 0;
    meta.innermost_dim = 1;
    meta.num_phy_elems = 1024;

    EXPECT_TRUE(meta.is_input);
    EXPECT_EQ(meta.input_idx, 0);
    EXPECT_FALSE(meta.is_intermediate());
}

TEST_F(DTensorMetaTest, OutputTensor) {
    DTensorMeta meta;
    meta.is_output = true;
    meta.output_idx = 2;
    meta.addr = 4096;

    EXPECT_TRUE(meta.is_output);
    EXPECT_EQ(meta.output_idx, 2);
    EXPECT_EQ(meta.addr, 4096u);
}

// =============================================================================
// STensorMeta Tests
// =============================================================================

class STensorMetaTest : public ::testing::Test {};

TEST_F(STensorMetaTest, DefaultValues) {
    STensorMeta meta;

    EXPECT_EQ(meta.innermost_dim, -1);
    EXPECT_EQ(meta.swizzled_dim, -1);
    EXPECT_EQ(meta.num_phy_elems, 0u);
    EXPECT_FALSE(meta.is_xor_swizzled);
    EXPECT_FALSE(meta.m_input);
    EXPECT_FALSE(meta.n_input);
}

TEST_F(STensorMetaTest, SetStrides) {
    STensorMeta meta;
    meta.set_strides({64, 1});

    EXPECT_EQ(meta.strides[0], 64u);
    EXPECT_EQ(meta.strides[1], 1u);
}

TEST_F(STensorMetaTest, IsSwizzled) {
    STensorMeta meta;
    EXPECT_FALSE(meta.is_swizzled());

    meta.swizzled_dim = 0;
    EXPECT_TRUE(meta.is_swizzled());

    meta.swizzled_dim = -1;
    meta.is_xor_swizzled = true;
    EXPECT_TRUE(meta.is_swizzled());
}

TEST_F(STensorMetaTest, IsMatrixInput) {
    STensorMeta meta;
    EXPECT_FALSE(meta.is_matrix_input());

    meta.m_input = true;
    EXPECT_TRUE(meta.is_matrix_input());

    meta.m_input = false;
    meta.n_input = true;
    EXPECT_TRUE(meta.is_matrix_input());
}

TEST_F(STensorMetaTest, HasMatrixPair) {
    STensorMeta meta;
    EXPECT_FALSE(meta.has_matrix_pair());

    meta.m_matrix_guid = 42;
    EXPECT_TRUE(meta.has_matrix_pair());
}

TEST_F(STensorMetaTest, SetXorSwizzle) {
    STensorMeta meta;
    meta.set_xor_swizzle(3, 2, 1);

    EXPECT_TRUE(meta.is_xor_swizzled);
    EXPECT_EQ(meta.xor_swizzle_b, 3);
    EXPECT_EQ(meta.xor_swizzle_m, 2);
    EXPECT_EQ(meta.xor_swizzle_s, 1);
}

TEST_F(STensorMetaTest, PipelinedInput) {
    STensorMeta meta;
    EXPECT_FALSE(meta.is_pipelined_input);

    meta.is_pipelined_input = true;
    EXPECT_TRUE(meta.is_pipelined_input);
}

// =============================================================================
// TMAParams Tests
// =============================================================================

class TMAParamsTest : public ::testing::Test {};

TEST_F(TMAParamsTest, DefaultValues) {
    TMAParams params;

    EXPECT_EQ(params.input_id, 0u);
    EXPECT_EQ(params.guid, 0u);
    EXPECT_FALSE(params.m_input);
    EXPECT_EQ(params.forloop_range, 1);
    EXPECT_EQ(params.forloop_dim, -1);
    EXPECT_EQ(params.multicast_direction, "NOT_MULTICAST");
}

TEST_F(TMAParamsTest, ParameterizedConstruction) {
    TMAParams params(
        1, 100, 200,
        "RowMajor", "Swizzle128B",
        true, "64x64",
        {2, 2, 1},
        {64, 128},
        {128, 1},
        {0, 1},
        8, 0,
        "X_MULTICAST"
    );

    EXPECT_EQ(params.input_id, 1u);
    EXPECT_EQ(params.guid, 100u);
    EXPECT_EQ(params.sguid, 200u);
    EXPECT_EQ(params.srcLayout, "RowMajor");
    EXPECT_EQ(params.dstLayout, "Swizzle128B");
    EXPECT_TRUE(params.m_input);
    EXPECT_EQ(params.tile_size, "64x64");
    EXPECT_EQ(params.forloop_range, 8);
    EXPECT_EQ(params.forloop_dim, 0);
    EXPECT_EQ(params.multicast_direction, "X_MULTICAST");
}

TEST_F(TMAParamsTest, IsMulticast) {
    TMAParams params;
    EXPECT_FALSE(params.is_multicast());

    params.multicast_direction = "X_MULTICAST";
    EXPECT_TRUE(params.is_multicast());
}

TEST_F(TMAParamsTest, GetClusterSize) {
    TMAParams params;
    params.clusterSize = {4, 2, 1};

    EXPECT_EQ(params.get_cluster_x(), 4);
    EXPECT_EQ(params.get_cluster_y(), 2);
    EXPECT_EQ(params.get_cluster_z(), 1);
    EXPECT_EQ(params.get_cluster_size(), 8);
}

TEST_F(TMAParamsTest, HasForloop) {
    TMAParams params;
    EXPECT_FALSE(params.has_forloop());

    params.forloop_dim = 0;
    params.forloop_range = 1;
    EXPECT_FALSE(params.has_forloop());

    params.forloop_range = 8;
    EXPECT_TRUE(params.has_forloop());
}

TEST_F(TMAParamsTest, ShapeAndStride) {
    TMAParams params;
    params.original_shape = {64, 128, 256};
    params.original_stride = {32768, 256, 1};

    EXPECT_EQ(params.original_shape.size(), 3u);
    EXPECT_EQ(params.original_stride.size(), 3u);
    EXPECT_EQ(params.original_shape[0], 64);
    EXPECT_EQ(params.original_stride[0], 32768u);
}

// =============================================================================
// TBMemoryPlan Tests
// =============================================================================

class TBMemoryPlanTest : public ::testing::Test {};

TEST_F(TBMemoryPlanTest, DefaultValues) {
    TBMemoryPlan plan;

    EXPECT_EQ(plan.smem_size, 0u);
    EXPECT_EQ(plan.pipelined_input_buf_guid_offset, 0);
    EXPECT_EQ(plan.tmem_base_ptr_guid, 0);
    EXPECT_TRUE(plan.addrs.empty());
}

TEST_F(TBMemoryPlanTest, SetAndGetAddr) {
    TBMemoryPlan plan;
    plan.set_addr(100, 1024);
    plan.set_addr(200, 2048);

    EXPECT_EQ(plan.get_addr(100), 1024u);
    EXPECT_EQ(plan.get_addr(200), 2048u);
    EXPECT_EQ(plan.get_addr(300), 0u);  // Not found
}

TEST_F(TBMemoryPlanTest, HasTensor) {
    TBMemoryPlan plan;
    plan.set_addr(42, 512);

    EXPECT_TRUE(plan.has_tensor(42));
    EXPECT_FALSE(plan.has_tensor(43));
}

TEST_F(TBMemoryPlanTest, NumTensors) {
    TBMemoryPlan plan;
    EXPECT_EQ(plan.num_tensors(), 0u);

    plan.set_addr(1, 100);
    plan.set_addr(2, 200);
    plan.set_addr(3, 300);

    EXPECT_EQ(plan.num_tensors(), 3u);
}

TEST_F(TBMemoryPlanTest, UsesPipelining) {
    TBMemoryPlan plan;
    EXPECT_FALSE(plan.uses_pipelining());

    plan.pipelined_input_buf_guid_offset = 1000;
    EXPECT_TRUE(plan.uses_pipelining());
}

TEST_F(TBMemoryPlanTest, UsesTmem) {
    TBMemoryPlan plan;
    EXPECT_FALSE(plan.uses_tmem());

    plan.tmem_base_ptr_guid = 500;
    EXPECT_TRUE(plan.uses_tmem());
}

TEST_F(TBMemoryPlanTest, CompleteMemoryPlan) {
    TBMemoryPlan plan;

    // Set up a complete memory plan
    plan.smem_size = 65536;
    plan.set_addr(100, 0);
    plan.set_addr(101, 16384);
    plan.set_addr(102, 32768);
    plan.pipelined_input_buf_guid_offset = 1000;
    plan.mbarrier_buf_guid_offset = 500;

    EXPECT_EQ(plan.smem_size, 65536u);
    EXPECT_EQ(plan.num_tensors(), 3u);
    EXPECT_TRUE(plan.uses_pipelining());
}

// =============================================================================
// Parameterized STensorMeta Tests
// =============================================================================

struct STensorMetaParam {
    bool m_input;
    bool n_input;
    bool expected_is_matrix;
};

class STensorMetaParameterizedTest
    : public ::testing::TestWithParam<STensorMetaParam> {};

TEST_P(STensorMetaParameterizedTest, IsMatrixInput) {
    auto param = GetParam();
    STensorMeta meta;
    meta.m_input = param.m_input;
    meta.n_input = param.n_input;

    EXPECT_EQ(meta.is_matrix_input(), param.expected_is_matrix);
}

INSTANTIATE_TEST_SUITE_P(
    MatrixInputCombinations,
    STensorMetaParameterizedTest,
    ::testing::Values(
        STensorMetaParam{false, false, false},
        STensorMetaParam{true, false, true},
        STensorMetaParam{false, true, true},
        STensorMetaParam{true, true, true}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
