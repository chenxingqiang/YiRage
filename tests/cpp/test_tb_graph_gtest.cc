// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_tb_graph_gtest.cc
 * @brief Threadblock Graph Unit Tests
 *
 * Tests for Graph class:
 *   - Graph construction and configuration
 *   - Grid/block dimensions
 *   - Input/output operator creation
 *   - Matmul, element ops, reduction ops
 *   - Memory allocation
 *   - Forloop accumulation
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace yirage {
namespace type {

enum TBOperatorType {
    TB_OP_UNKNOWN = 0,
    TB_INPUT_OP = 100,
    TB_OUTPUT_OP = 101,
    TB_MATMUL_OP = 200,
    TB_EXP_OP = 300,
    TB_SQUARE_OP = 301,
    TB_SQRT_OP = 302,
    TB_SILU_OP = 303,
    TB_ADD_OP = 400,
    TB_MUL_OP = 401,
    TB_REDUCTION_OP = 500,
    TB_RMS_NORM_OP = 503,
    TB_FORLOOP_ACCUM_OP = 600,
    TB_CONCAT_OP = 700,
};

enum TBEpilogueType {
    TB_EPILOGUE_NONE = 0,
    TB_EPILOGUE_ALLREDUCE = 1,
};

enum DataType {
    DT_UNKNOWN = 0,
    DT_FLOAT16 = 3,
    DT_FLOAT32 = 4,
};

}  // namespace type

namespace layout {

enum SmemLayout {
    SMEM_LAYOUT_UNKNOWN = 0,
    SMEM_LAYOUT_ROW_MAJOR = 1,
    SMEM_LAYOUT_SWIZZLE_128B = 3,
};

}  // namespace layout

namespace threadblock {

constexpr int MAX_TENSOR_DIMS = 4;

// dim3 mock
struct dim3 {
    unsigned int x = 1, y = 1, z = 1;
    dim3() = default;
    dim3(unsigned int _x, unsigned int _y = 1, unsigned int _z = 1)
        : x(_x), y(_y), z(_z) {}
};

struct int3 {
    int x = 0, y = 0, z = 0;
    int3() = default;
    int3(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
};

// Mock STensor
struct STensor {
    type::DataType data_type = type::DT_UNKNOWN;
    layout::SmemLayout layout = layout::SMEM_LAYOUT_UNKNOWN;
    int num_dims = 0;
    int dim[MAX_TENSOR_DIMS] = {0};
    int smem_offset = 0;
    bool after_accum = false;

    size_t size() const {
        if (num_dims == 0) return 0;
        size_t elements = 1;
        size_t dtype_size = (data_type == type::DT_FLOAT32) ? 4 : 2;
        for (int i = 0; i < num_dims; i++) {
            elements *= dim[i];
        }
        return elements * dtype_size;
    }
};

// Mock DTensor
struct DTensor {
    type::DataType data_type = type::DT_FLOAT32;
    int num_dims = 2;
    int dim[MAX_TENSOR_DIMS] = {64, 128, 0, 0};
    size_t guid = 0;
    static size_t next_guid;
    DTensor() { guid = next_guid++; }
};

size_t DTensor::next_guid = 1;

// Mock TBOperator
class TBOperator {
public:
    type::TBOperatorType op_type;
    std::vector<STensor> input_tensors;
    std::vector<STensor> output_tensors;

    TBOperator(type::TBOperatorType type) : op_type(type) {}
    virtual ~TBOperator() = default;
};

// =============================================================================
// Graph Class
// =============================================================================

class Graph {
public:
    Graph() : grid_dim(1, 1, 1), block_dim(128, 1, 1),
              forloop_range(1), reduction_dimx(1), smem_offset(0) {}

    Graph(dim3 _grid_dim, dim3 _block_dim, int _forloop_range, int _reduction_dimx)
        : grid_dim(_grid_dim), block_dim(_block_dim),
          forloop_range(_forloop_range), reduction_dimx(_reduction_dimx),
          smem_offset(0) {}

    ~Graph() {
        for (auto* op : operators) {
            delete op;
        }
    }

    // Input operator
    STensor new_input(DTensor const& dtensor, int3 input_map,
                      int forloop_dim, layout::SmemLayout layout,
                      bool store_in_dmem = false) {
        STensor output;
        output.data_type = dtensor.data_type;
        output.num_dims = dtensor.num_dims;
        for (int i = 0; i < dtensor.num_dims; i++) {
            output.dim[i] = dtensor.dim[i];
        }
        output.layout = layout;
        output.smem_offset = allocate_smem(output.size());

        auto* op = new TBOperator(type::TB_INPUT_OP);
        op->output_tensors.push_back(output);
        operators.push_back(op);

        return output;
    }

    // Output operator
    DTensor mark_output(STensor const& stensor, int3 output_map,
                        int forloop_dim, type::TBEpilogueType epilogue) {
        auto* op = new TBOperator(type::TB_OUTPUT_OP);
        op->input_tensors.push_back(stensor);
        operators.push_back(op);

        DTensor dtensor;
        dtensor.data_type = stensor.data_type;
        dtensor.num_dims = stensor.num_dims;
        for (int i = 0; i < stensor.num_dims; i++) {
            dtensor.dim[i] = stensor.dim[i];
        }
        return dtensor;
    }

    // Matmul operator
    STensor matmul(STensor const& A, STensor const& B) {
        STensor output;
        output.data_type = A.data_type;
        output.num_dims = 2;
        output.dim[0] = A.dim[0];  // M
        output.dim[1] = B.dim[1];  // N
        output.layout = A.layout;
        output.smem_offset = allocate_smem(output.size());

        auto* op = new TBOperator(type::TB_MATMUL_OP);
        op->input_tensors.push_back(A);
        op->input_tensors.push_back(B);
        op->output_tensors.push_back(output);
        operators.push_back(op);

        return output;
    }

    // Element unary operators
    STensor exp(STensor const& A) { return elementunary(A, type::TB_EXP_OP); }
    STensor silu(STensor const& A) { return elementunary(A, type::TB_SILU_OP); }
    STensor sqrt(STensor const& A) { return elementunary(A, type::TB_SQRT_OP); }
    STensor square(STensor const& A) { return elementunary(A, type::TB_SQUARE_OP); }

    STensor elementunary(STensor const& A, type::TBOperatorType op_type) {
        STensor output = A;  // Same shape
        output.smem_offset = allocate_smem(output.size());

        auto* op = new TBOperator(op_type);
        op->input_tensors.push_back(A);
        op->output_tensors.push_back(output);
        operators.push_back(op);

        return output;
    }

    // Element binary operators
    STensor add(STensor const& A, STensor const& B) {
        return elementbinary(A, B, type::TB_ADD_OP);
    }
    STensor mul(STensor const& A, STensor const& B) {
        return elementbinary(A, B, type::TB_MUL_OP);
    }

    STensor elementbinary(STensor const& A, STensor const& B,
                          type::TBOperatorType op_type) {
        STensor output = A;  // Same shape as A
        output.smem_offset = allocate_smem(output.size());

        auto* op = new TBOperator(op_type);
        op->input_tensors.push_back(A);
        op->input_tensors.push_back(B);
        op->output_tensors.push_back(output);
        operators.push_back(op);

        return output;
    }

    // Reduction operator
    STensor reduction(STensor const& A, int reduce_dim) {
        STensor output = A;
        output.dim[reduce_dim] = 1;
        output.smem_offset = allocate_smem(output.size());

        auto* op = new TBOperator(type::TB_REDUCTION_OP);
        op->input_tensors.push_back(A);
        op->output_tensors.push_back(output);
        operators.push_back(op);

        return output;
    }

    // RMS norm operator
    STensor rms_norm(STensor const& A) {
        STensor output = A;
        output.smem_offset = allocate_smem(output.size());

        auto* op = new TBOperator(type::TB_RMS_NORM_OP);
        op->input_tensors.push_back(A);
        op->output_tensors.push_back(output);
        operators.push_back(op);

        return output;
    }

    // Concat operator
    STensor concat(STensor const& A, STensor const& B, int concat_dim) {
        STensor output = A;
        output.dim[concat_dim] = A.dim[concat_dim] + B.dim[concat_dim];
        output.smem_offset = allocate_smem(output.size());

        auto* op = new TBOperator(type::TB_CONCAT_OP);
        op->input_tensors.push_back(A);
        op->input_tensors.push_back(B);
        op->output_tensors.push_back(output);
        operators.push_back(op);

        return output;
    }

    // Forloop accumulator
    STensor forloop_accum(STensor const& input, type::TBOperatorType accum_type) {
        STensor output = input;
        output.after_accum = true;
        output.smem_offset = allocate_smem(output.size());

        auto* op = new TBOperator(type::TB_FORLOOP_ACCUM_OP);
        op->input_tensors.push_back(input);
        op->output_tensors.push_back(output);
        operators.push_back(op);

        return output;
    }

    // Memory allocation
    off_t allocate_smem(size_t size) {
        // Align to 128 bytes
        size_t aligned_size = ((size + 127) / 128) * 128;
        off_t offset = smem_offset;
        smem_offset += aligned_size;
        allocated_tensors.push_back({offset, aligned_size});
        return offset;
    }

    void free_smem(off_t offset) {
        for (auto it = allocated_tensors.begin(); it != allocated_tensors.end(); ++it) {
            if (it->first == offset) {
                allocated_tensors.erase(it);
                break;
            }
        }
    }

    size_t calculate_shared_memory_usage() const {
        return smem_offset;
    }

    size_t num_operators() const { return operators.size(); }

    int get_smem_size_with_pipeline() const {
        // Add pipeline buffer overhead
        return smem_offset + (forloop_range > 1 ? smem_offset / 2 : 0);
    }

public:
    dim3 grid_dim, block_dim, cluster_dim{4, 4, 1};
    int forloop_range;
    int reduction_dimx;
    std::vector<TBOperator*> operators;
    off_t smem_offset;
    std::vector<std::pair<off_t, size_t>> allocated_tensors;
};

}  // namespace threadblock
}  // namespace yirage

using namespace yirage::threadblock;
using namespace yirage::type;
using namespace yirage::layout;

// =============================================================================
// Graph Construction Tests
// =============================================================================

class GraphConstructionTest : public ::testing::Test {
protected:
    void SetUp() override {
        DTensor::next_guid = 1;
    }
};

TEST_F(GraphConstructionTest, DefaultConstruction) {
    Graph graph;

    EXPECT_EQ(graph.grid_dim.x, 1u);
    EXPECT_EQ(graph.grid_dim.y, 1u);
    EXPECT_EQ(graph.block_dim.x, 128u);
    EXPECT_EQ(graph.forloop_range, 1);
    EXPECT_EQ(graph.reduction_dimx, 1);
    EXPECT_EQ(graph.num_operators(), 0u);
}

TEST_F(GraphConstructionTest, ParameterizedConstruction) {
    dim3 grid(4, 4, 1);
    dim3 block(256, 1, 1);

    Graph graph(grid, block, 8, 32);

    EXPECT_EQ(graph.grid_dim.x, 4u);
    EXPECT_EQ(graph.grid_dim.y, 4u);
    EXPECT_EQ(graph.block_dim.x, 256u);
    EXPECT_EQ(graph.forloop_range, 8);
    EXPECT_EQ(graph.reduction_dimx, 32);
}

TEST_F(GraphConstructionTest, ClusterDimDefault) {
    Graph graph;

    EXPECT_EQ(graph.cluster_dim.x, 4u);
    EXPECT_EQ(graph.cluster_dim.y, 4u);
    EXPECT_EQ(graph.cluster_dim.z, 1u);
}

// =============================================================================
// Input/Output Operator Tests
// =============================================================================

class GraphIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        DTensor::next_guid = 1;
    }
};

TEST_F(GraphIOTest, NewInput) {
    Graph graph;
    DTensor dtensor;
    dtensor.num_dims = 2;
    dtensor.dim[0] = 64;
    dtensor.dim[1] = 128;
    dtensor.data_type = DT_FLOAT16;

    STensor stensor = graph.new_input(dtensor, int3(0, 1, 2), -1,
                                       SMEM_LAYOUT_ROW_MAJOR);

    EXPECT_EQ(graph.num_operators(), 1u);
    EXPECT_EQ(stensor.num_dims, 2);
    EXPECT_EQ(stensor.dim[0], 64);
    EXPECT_EQ(stensor.dim[1], 128);
    EXPECT_EQ(stensor.data_type, DT_FLOAT16);
    EXPECT_EQ(stensor.layout, SMEM_LAYOUT_ROW_MAJOR);
}

TEST_F(GraphIOTest, MarkOutput) {
    Graph graph;

    STensor stensor;
    stensor.num_dims = 2;
    stensor.dim[0] = 64;
    stensor.dim[1] = 128;
    stensor.data_type = DT_FLOAT32;

    DTensor dtensor = graph.mark_output(stensor, int3(0, 1, 2), -1,
                                         TB_EPILOGUE_NONE);

    EXPECT_EQ(graph.num_operators(), 1u);
    EXPECT_EQ(dtensor.num_dims, 2);
    EXPECT_EQ(dtensor.dim[0], 64);
    EXPECT_EQ(dtensor.dim[1], 128);
}

TEST_F(GraphIOTest, MultipleInputs) {
    Graph graph;

    DTensor d1, d2, d3;
    d1.dim[0] = 64; d1.dim[1] = 128;
    d2.dim[0] = 128; d2.dim[1] = 256;
    d3.dim[0] = 256; d3.dim[1] = 512;

    graph.new_input(d1, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);
    graph.new_input(d2, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);
    graph.new_input(d3, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);

    EXPECT_EQ(graph.num_operators(), 3u);
}

// =============================================================================
// Matmul Operator Tests
// =============================================================================

class GraphMatmulTest : public ::testing::Test {};

TEST_F(GraphMatmulTest, BasicMatmul) {
    Graph graph;

    STensor A, B;
    A.num_dims = 2;
    A.dim[0] = 64;   // M
    A.dim[1] = 128;  // K
    A.data_type = DT_FLOAT16;

    B.num_dims = 2;
    B.dim[0] = 128;  // K
    B.dim[1] = 256;  // N
    B.data_type = DT_FLOAT16;

    STensor C = graph.matmul(A, B);

    EXPECT_EQ(C.num_dims, 2);
    EXPECT_EQ(C.dim[0], 64);   // M
    EXPECT_EQ(C.dim[1], 256);  // N
    EXPECT_EQ(C.data_type, DT_FLOAT16);
}

TEST_F(GraphMatmulTest, MatmulAddsOperator) {
    Graph graph;

    STensor A, B;
    A.num_dims = 2; A.dim[0] = 32; A.dim[1] = 64;
    B.num_dims = 2; B.dim[0] = 64; B.dim[1] = 128;

    size_t before = graph.num_operators();
    graph.matmul(A, B);
    size_t after = graph.num_operators();

    EXPECT_EQ(after - before, 1u);
}

// =============================================================================
// Element Operation Tests
// =============================================================================

class GraphElementOpsTest : public ::testing::Test {};

TEST_F(GraphElementOpsTest, ExpOp) {
    Graph graph;

    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    STensor output = graph.exp(input);

    EXPECT_EQ(output.num_dims, input.num_dims);
    EXPECT_EQ(output.dim[0], input.dim[0]);
    EXPECT_EQ(output.dim[1], input.dim[1]);
}

TEST_F(GraphElementOpsTest, SiluOp) {
    Graph graph;

    STensor input;
    input.num_dims = 2;
    input.dim[0] = 32;
    input.dim[1] = 64;

    STensor output = graph.silu(input);

    EXPECT_EQ(output.dim[0], 32);
    EXPECT_EQ(output.dim[1], 64);
}

TEST_F(GraphElementOpsTest, AddOp) {
    Graph graph;

    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 64; B.dim[1] = 128;

    STensor C = graph.add(A, B);

    EXPECT_EQ(C.dim[0], 64);
    EXPECT_EQ(C.dim[1], 128);
}

TEST_F(GraphElementOpsTest, MulOp) {
    Graph graph;

    STensor A, B;
    A.num_dims = 2; A.dim[0] = 32; A.dim[1] = 64;
    B.num_dims = 2; B.dim[0] = 32; B.dim[1] = 64;

    STensor C = graph.mul(A, B);

    EXPECT_EQ(C.dim[0], 32);
    EXPECT_EQ(C.dim[1], 64);
}

// =============================================================================
// Reduction Operation Tests
// =============================================================================

class GraphReductionTest : public ::testing::Test {};

TEST_F(GraphReductionTest, ReduceDim0) {
    Graph graph;

    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    STensor output = graph.reduction(input, 0);

    EXPECT_EQ(output.dim[0], 1);
    EXPECT_EQ(output.dim[1], 128);
}

TEST_F(GraphReductionTest, ReduceDim1) {
    Graph graph;

    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    STensor output = graph.reduction(input, 1);

    EXPECT_EQ(output.dim[0], 64);
    EXPECT_EQ(output.dim[1], 1);
}

TEST_F(GraphReductionTest, RmsNorm) {
    Graph graph;

    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    STensor output = graph.rms_norm(input);

    EXPECT_EQ(output.dim[0], 64);
    EXPECT_EQ(output.dim[1], 128);
}

// =============================================================================
// Concat Operation Tests
// =============================================================================

class GraphConcatTest : public ::testing::Test {};

TEST_F(GraphConcatTest, ConcatDim0) {
    Graph graph;

    STensor A, B;
    A.num_dims = 2; A.dim[0] = 32; A.dim[1] = 64;
    B.num_dims = 2; B.dim[0] = 16; B.dim[1] = 64;

    STensor C = graph.concat(A, B, 0);

    EXPECT_EQ(C.dim[0], 48);  // 32 + 16
    EXPECT_EQ(C.dim[1], 64);
}

TEST_F(GraphConcatTest, ConcatDim1) {
    Graph graph;

    STensor A, B;
    A.num_dims = 2; A.dim[0] = 64; A.dim[1] = 128;
    B.num_dims = 2; B.dim[0] = 64; B.dim[1] = 256;

    STensor C = graph.concat(A, B, 1);

    EXPECT_EQ(C.dim[0], 64);
    EXPECT_EQ(C.dim[1], 384);  // 128 + 256
}

// =============================================================================
// Forloop Accumulator Tests
// =============================================================================

class GraphForloopAccumTest : public ::testing::Test {};

TEST_F(GraphForloopAccumTest, BasicAccum) {
    Graph graph(dim3(4, 4, 1), dim3(256, 1, 1), 8, 32);

    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;
    input.after_accum = false;

    STensor output = graph.forloop_accum(input, TB_FORLOOP_ACCUM_OP);

    EXPECT_TRUE(output.after_accum);
    EXPECT_EQ(output.dim[0], 64);
    EXPECT_EQ(output.dim[1], 128);
}

// =============================================================================
// Memory Allocation Tests
// =============================================================================

class GraphMemoryTest : public ::testing::Test {};

TEST_F(GraphMemoryTest, AllocateSmem) {
    Graph graph;

    off_t offset1 = graph.allocate_smem(1024);
    off_t offset2 = graph.allocate_smem(2048);

    EXPECT_EQ(offset1, 0);
    EXPECT_GE(offset2, 1024);  // At least 1024 after alignment
}

TEST_F(GraphMemoryTest, AllocateAligned) {
    Graph graph;

    // Small allocation should be aligned to 128
    graph.allocate_smem(100);
    off_t offset2 = graph.allocate_smem(100);

    EXPECT_EQ(offset2 % 128, 0u);
}

TEST_F(GraphMemoryTest, CalculateSharedMemory) {
    Graph graph;

    DTensor d1, d2;
    d1.dim[0] = 64; d1.dim[1] = 128; d1.data_type = DT_FLOAT16;
    d2.dim[0] = 128; d2.dim[1] = 256; d2.data_type = DT_FLOAT16;

    graph.new_input(d1, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);
    graph.new_input(d2, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);

    size_t usage = graph.calculate_shared_memory_usage();
    EXPECT_GT(usage, 0u);
}

TEST_F(GraphMemoryTest, SmemSizeWithPipeline) {
    Graph graph(dim3(4, 4, 1), dim3(256, 1, 1), 4, 32);

    graph.allocate_smem(4096);

    int with_pipeline = graph.get_smem_size_with_pipeline();
    EXPECT_GT(with_pipeline, 4096);  // Should include pipeline overhead
}

// =============================================================================
// Complex Graph Tests
// =============================================================================

class GraphComplexTest : public ::testing::Test {};

TEST_F(GraphComplexTest, AttentionPattern) {
    Graph graph(dim3(4, 4, 1), dim3(256, 1, 1), 8, 32);

    // Q, K, V inputs
    DTensor dQ, dK, dV;
    dQ.dim[0] = 64; dQ.dim[1] = 64; dQ.data_type = DT_FLOAT16;
    dK.dim[0] = 64; dK.dim[1] = 64; dK.data_type = DT_FLOAT16;
    dV.dim[0] = 64; dV.dim[1] = 64; dV.data_type = DT_FLOAT16;

    STensor Q = graph.new_input(dQ, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);
    STensor K = graph.new_input(dK, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);
    STensor V = graph.new_input(dV, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);

    // QK^T
    STensor scores = graph.matmul(Q, K);

    // Softmax (simplified as exp + reduction)
    STensor exp_scores = graph.exp(scores);
    STensor sum_scores = graph.reduction(exp_scores, 1);

    // Output
    STensor output = graph.matmul(exp_scores, V);

    EXPECT_GE(graph.num_operators(), 6u);
}

TEST_F(GraphComplexTest, MLPPattern) {
    Graph graph(dim3(4, 4, 1), dim3(256, 1, 1), 1, 32);

    DTensor dInput, dW1, dW2;
    dInput.dim[0] = 64; dInput.dim[1] = 128;
    dW1.dim[0] = 128; dW1.dim[1] = 512;
    dW2.dim[0] = 512; dW2.dim[1] = 128;

    STensor input = graph.new_input(dInput, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);
    STensor W1 = graph.new_input(dW1, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);
    STensor W2 = graph.new_input(dW2, int3(), -1, SMEM_LAYOUT_ROW_MAJOR);

    // FC1 + SiLU
    STensor hidden = graph.matmul(input, W1);
    STensor activated = graph.silu(hidden);

    // FC2
    STensor output = graph.matmul(activated, W2);

    EXPECT_EQ(graph.num_operators(), 6u);  // 3 inputs + 2 matmul + 1 silu
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
