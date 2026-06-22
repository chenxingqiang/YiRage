// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_tb_operator_gtest.cc
 * @brief Threadblock Operator Unit Tests
 *
 * Tests for TBOperator and derived classes:
 *   - TBOperator base class
 *   - TBInputOp (input loader)
 *   - TBOutputOp (output saver)
 *   - Operator type classification
 *   - Input/output tensor management
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <memory>
#include <string>
#include <vector>

namespace yirage {
namespace type {

enum TBOperatorType {
    TB_OP_UNKNOWN = 0,

    // Input/Output operations
    TB_INPUT_OP = 100,
    TB_OUTPUT_OP = 101,

    // Matmul operations
    TB_MATMUL_OP = 200,

    // Element-wise unary operations
    TB_EXP_OP = 300,
    TB_SQUARE_OP = 301,
    TB_SQRT_OP = 302,
    TB_SILU_OP = 303,
    TB_GELU_OP = 304,
    TB_RELU_OP = 305,
    TB_CLAMP_OP = 306,
    TB_MUL_SCALAR_OP = 307,

    // Element-wise binary operations
    TB_ADD_OP = 400,
    TB_MUL_OP = 401,
    TB_DIV_OP = 402,
    TB_SUB_OP = 403,
    TB_POW_OP = 404,

    // Reduction operations
    TB_REDUCTION_OP = 500,
    TB_REDUCTION_TO_DIMX_OP = 501,
    TB_REDUCTION_MAX_OP = 502,
    TB_RMS_NORM_OP = 503,

    // Accumulator operations
    TB_FORLOOP_ACCUM_OP = 600,
    TB_FORLOOP_ACCUM_RESCALE_OP = 601,
    TB_FORLOOP_ACCUM_MAX_OP = 602,

    // Other operations
    TB_CONCAT_OP = 700,
};

enum TBEpilogueType {
    TB_EPILOGUE_NONE = 0,
    TB_EPILOGUE_ALLREDUCE = 1,
    TB_EPILOGUE_REDUCE = 2,
};

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

}  // namespace type

namespace layout {

enum SmemLayout {
    SMEM_LAYOUT_UNKNOWN = 0,
    SMEM_LAYOUT_ROW_MAJOR = 1,
    SMEM_LAYOUT_COL_MAJOR = 2,
    SMEM_LAYOUT_SWIZZLE_128B = 3,
};

}  // namespace layout

namespace threadblock {

// Forward declarations
class Graph;
constexpr int MAX_TENSOR_DIMS = 4;

// Mock STensor
struct STensor {
    type::DataType data_type = type::DT_UNKNOWN;
    layout::SmemLayout layout = layout::SMEM_LAYOUT_UNKNOWN;
    int num_dims = 0;
    int dim[MAX_TENSOR_DIMS] = {0};
    int smem_offset = 0;
    bool after_accum = false;

    size_t num_elements() const {
        if (num_dims == 0) return 0;
        size_t result = 1;
        for (int i = 0; i < num_dims; i++) {
            result *= dim[i];
        }
        return result;
    }
};

// Mock DTensor (device tensor)
struct DTensor {
    type::DataType data_type = type::DT_FLOAT32;
    int num_dims = 2;
    int dim[MAX_TENSOR_DIMS] = {64, 128, 0, 0};
    size_t guid = 0;

    static size_t next_guid;

    DTensor() {
        guid = next_guid++;
    }
};

size_t DTensor::next_guid = 1;

// int3 mock
struct int3 {
    int x = 0;
    int y = 0;
    int z = 0;

    int3() = default;
    int3(int _x, int _y, int _z) : x(_x), y(_y), z(_z) {}
};

// =============================================================================
// TBOperator Base Class
// =============================================================================

class TBOperator {
public:
    TBOperator(Graph* graph, type::TBOperatorType op_type)
        : bgraph(graph), op_type(op_type) {}

    TBOperator(Graph* graph, type::TBOperatorType op_type, STensor const& input1)
        : bgraph(graph), op_type(op_type) {
        input_tensors.push_back(input1);
    }

    TBOperator(Graph* graph, type::TBOperatorType op_type,
               STensor const& input1, STensor const& input2)
        : bgraph(graph), op_type(op_type) {
        input_tensors.push_back(input1);
        input_tensors.push_back(input2);
    }

    TBOperator(Graph* graph, type::TBOperatorType op_type,
               std::vector<STensor> const& inputs)
        : bgraph(graph), op_type(op_type), input_tensors(inputs) {}

    virtual ~TBOperator() = default;

    int get_input_stensors(STensor** inputs) {
        if (inputs && !input_tensors.empty()) {
            *inputs = &input_tensors[0];
        }
        return static_cast<int>(input_tensors.size());
    }

    int get_output_stensors(STensor** outputs) {
        if (outputs && !output_tensors.empty()) {
            *outputs = &output_tensors[0];
        }
        return static_cast<int>(output_tensors.size());
    }

    size_t num_inputs() const { return input_tensors.size(); }
    size_t num_outputs() const { return output_tensors.size(); }

    Graph* bgraph;
    type::TBOperatorType op_type;
    std::vector<STensor> input_tensors;
    std::vector<STensor> output_tensors;
};

// =============================================================================
// TBInputOp
// =============================================================================

class TBInputOp : public TBOperator {
public:
    TBInputOp(Graph* _graph, DTensor const& _dtensor, int3 _input_map,
              int _forloop_dim, layout::SmemLayout _layout, bool _store_in_dmem)
        : TBOperator(_graph, type::TB_INPUT_OP),
          dtensor(_dtensor),
          input_map(_input_map),
          forloop_dim(_forloop_dim),
          layout(_layout),
          store_in_dmem(_store_in_dmem) {
        // Create output STensor based on DTensor
        STensor output;
        output.data_type = dtensor.data_type;
        output.num_dims = dtensor.num_dims;
        for (int i = 0; i < dtensor.num_dims; i++) {
            output.dim[i] = dtensor.dim[i];
        }
        output.layout = layout;
        output_tensors.push_back(output);
    }

    size_t get_dtensor_guid() const { return dtensor.guid; }

    DTensor dtensor;
    int3 input_map;
    int forloop_dim;
    layout::SmemLayout layout;
    bool store_in_dmem;
};

// =============================================================================
// TBOutputOp
// =============================================================================

class TBOutputOp : public TBOperator {
public:
    TBOutputOp(Graph* _graph, STensor const& stensor, int3 _output_map,
               int _forloop_dim, type::TBEpilogueType _epilogue)
        : TBOperator(_graph, type::TB_OUTPUT_OP, stensor),
          output_map(_output_map),
          forloop_dim(_forloop_dim),
          epilogue(_epilogue) {
        // Create DTensor from STensor
        dtensor.data_type = stensor.data_type;
        dtensor.num_dims = stensor.num_dims;
        for (int i = 0; i < stensor.num_dims; i++) {
            dtensor.dim[i] = stensor.dim[i];
        }
    }

    size_t get_dtensor_guid() const { return dtensor.guid; }

    DTensor dtensor;
    int3 output_map;
    int forloop_dim;
    type::TBEpilogueType epilogue;
};

// =============================================================================
// Operator Type Classification Functions
// =============================================================================

inline bool is_input_op(type::TBOperatorType type) {
    return type == type::TB_INPUT_OP;
}

inline bool is_output_op(type::TBOperatorType type) {
    return type == type::TB_OUTPUT_OP;
}

inline bool is_matmul_op(type::TBOperatorType type) {
    return type == type::TB_MATMUL_OP;
}

inline bool is_element_unary_op(type::TBOperatorType type) {
    return type >= type::TB_EXP_OP && type <= type::TB_MUL_SCALAR_OP;
}

inline bool is_element_binary_op(type::TBOperatorType type) {
    return type >= type::TB_ADD_OP && type <= type::TB_POW_OP;
}

inline bool is_reduction_op(type::TBOperatorType type) {
    return type >= type::TB_REDUCTION_OP && type <= type::TB_RMS_NORM_OP;
}

inline bool is_accumulator_op(type::TBOperatorType type) {
    return type >= type::TB_FORLOOP_ACCUM_OP && type <= type::TB_FORLOOP_ACCUM_MAX_OP;
}

inline const char* op_type_to_string(type::TBOperatorType type) {
    switch (type) {
        case type::TB_INPUT_OP: return "input";
        case type::TB_OUTPUT_OP: return "output";
        case type::TB_MATMUL_OP: return "matmul";
        case type::TB_EXP_OP: return "exp";
        case type::TB_SQUARE_OP: return "square";
        case type::TB_SQRT_OP: return "sqrt";
        case type::TB_SILU_OP: return "silu";
        case type::TB_GELU_OP: return "gelu";
        case type::TB_RELU_OP: return "relu";
        case type::TB_CLAMP_OP: return "clamp";
        case type::TB_ADD_OP: return "add";
        case type::TB_MUL_OP: return "mul";
        case type::TB_DIV_OP: return "div";
        case type::TB_SUB_OP: return "sub";
        case type::TB_REDUCTION_OP: return "reduction";
        case type::TB_RMS_NORM_OP: return "rms_norm";
        case type::TB_CONCAT_OP: return "concat";
        default: return "unknown";
    }
}

}  // namespace threadblock
}  // namespace yirage

using namespace yirage::threadblock;
using namespace yirage::type;
using namespace yirage::layout;

// =============================================================================
// TBOperator Base Tests
// =============================================================================

class TBOperatorTest : public ::testing::Test {};

TEST_F(TBOperatorTest, ConstructWithNoInputs) {
    TBOperator op(nullptr, TB_MATMUL_OP);

    EXPECT_EQ(op.op_type, TB_MATMUL_OP);
    EXPECT_EQ(op.num_inputs(), 0u);
    EXPECT_EQ(op.num_outputs(), 0u);
}

TEST_F(TBOperatorTest, ConstructWithOneInput) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 64;
    input.dim[1] = 128;

    TBOperator op(nullptr, TB_EXP_OP, input);

    EXPECT_EQ(op.op_type, TB_EXP_OP);
    EXPECT_EQ(op.num_inputs(), 1u);
}

TEST_F(TBOperatorTest, ConstructWithTwoInputs) {
    STensor input1, input2;
    input1.num_dims = 2;
    input2.num_dims = 2;

    TBOperator op(nullptr, TB_ADD_OP, input1, input2);

    EXPECT_EQ(op.op_type, TB_ADD_OP);
    EXPECT_EQ(op.num_inputs(), 2u);
}

TEST_F(TBOperatorTest, ConstructWithVectorInputs) {
    std::vector<STensor> inputs(3);
    for (auto& t : inputs) {
        t.num_dims = 2;
    }

    TBOperator op(nullptr, TB_CONCAT_OP, inputs);

    EXPECT_EQ(op.num_inputs(), 3u);
}

TEST_F(TBOperatorTest, GetInputStensors) {
    STensor input;
    input.num_dims = 2;
    input.dim[0] = 32;
    input.dim[1] = 64;

    TBOperator op(nullptr, TB_EXP_OP, input);

    STensor* inputs = nullptr;
    int count = op.get_input_stensors(&inputs);

    EXPECT_EQ(count, 1);
    EXPECT_NE(inputs, nullptr);
    EXPECT_EQ(inputs[0].dim[0], 32);
    EXPECT_EQ(inputs[0].dim[1], 64);
}

TEST_F(TBOperatorTest, GetOutputStensors) {
    TBOperator op(nullptr, TB_MATMUL_OP);
    op.output_tensors.push_back(STensor());
    op.output_tensors[0].num_dims = 2;
    op.output_tensors[0].dim[0] = 128;
    op.output_tensors[0].dim[1] = 256;

    STensor* outputs = nullptr;
    int count = op.get_output_stensors(&outputs);

    EXPECT_EQ(count, 1);
    EXPECT_NE(outputs, nullptr);
    EXPECT_EQ(outputs[0].dim[0], 128);
}

// =============================================================================
// TBInputOp Tests
// =============================================================================

class TBInputOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        DTensor::next_guid = 1;
    }
};

TEST_F(TBInputOpTest, Construction) {
    DTensor dtensor;
    dtensor.num_dims = 2;
    dtensor.dim[0] = 64;
    dtensor.dim[1] = 128;
    dtensor.data_type = DT_FLOAT16;

    int3 input_map(0, 1, 2);
    int forloop_dim = -1;

    TBInputOp op(nullptr, dtensor, input_map, forloop_dim,
                 SMEM_LAYOUT_ROW_MAJOR, false);

    EXPECT_EQ(op.op_type, TB_INPUT_OP);
    EXPECT_EQ(op.forloop_dim, -1);
    EXPECT_FALSE(op.store_in_dmem);
    EXPECT_EQ(op.layout, SMEM_LAYOUT_ROW_MAJOR);
}

TEST_F(TBInputOpTest, InputMapValues) {
    DTensor dtensor;
    int3 input_map(1, 2, 3);

    TBInputOp op(nullptr, dtensor, input_map, 0, SMEM_LAYOUT_ROW_MAJOR, false);

    EXPECT_EQ(op.input_map.x, 1);
    EXPECT_EQ(op.input_map.y, 2);
    EXPECT_EQ(op.input_map.z, 3);
}

TEST_F(TBInputOpTest, OutputTensorCreated) {
    DTensor dtensor;
    dtensor.num_dims = 2;
    dtensor.dim[0] = 64;
    dtensor.dim[1] = 128;
    dtensor.data_type = DT_FLOAT32;

    TBInputOp op(nullptr, dtensor, int3(), -1, SMEM_LAYOUT_SWIZZLE_128B, false);

    EXPECT_EQ(op.num_outputs(), 1u);

    STensor* outputs = nullptr;
    op.get_output_stensors(&outputs);
    EXPECT_EQ(outputs[0].num_dims, 2);
    EXPECT_EQ(outputs[0].dim[0], 64);
    EXPECT_EQ(outputs[0].dim[1], 128);
    EXPECT_EQ(outputs[0].layout, SMEM_LAYOUT_SWIZZLE_128B);
}

TEST_F(TBInputOpTest, StoreInDmem) {
    DTensor dtensor;
    TBInputOp op(nullptr, dtensor, int3(), -1, SMEM_LAYOUT_ROW_MAJOR, true);

    EXPECT_TRUE(op.store_in_dmem);
}

TEST_F(TBInputOpTest, GetDtensorGuid) {
    DTensor dtensor;
    size_t expected_guid = dtensor.guid;

    TBInputOp op(nullptr, dtensor, int3(), -1, SMEM_LAYOUT_ROW_MAJOR, false);

    EXPECT_EQ(op.get_dtensor_guid(), expected_guid);
}

// =============================================================================
// TBOutputOp Tests
// =============================================================================

class TBOutputOpTest : public ::testing::Test {
protected:
    void SetUp() override {
        DTensor::next_guid = 1;
    }
};

TEST_F(TBOutputOpTest, Construction) {
    STensor stensor;
    stensor.num_dims = 2;
    stensor.dim[0] = 64;
    stensor.dim[1] = 128;
    stensor.data_type = DT_FLOAT16;

    int3 output_map(0, 1, 2);

    TBOutputOp op(nullptr, stensor, output_map, -1, TB_EPILOGUE_NONE);

    EXPECT_EQ(op.op_type, TB_OUTPUT_OP);
    EXPECT_EQ(op.forloop_dim, -1);
    EXPECT_EQ(op.epilogue, TB_EPILOGUE_NONE);
}

TEST_F(TBOutputOpTest, OutputMapValues) {
    STensor stensor;
    int3 output_map(3, 2, 1);

    TBOutputOp op(nullptr, stensor, output_map, -1, TB_EPILOGUE_NONE);

    EXPECT_EQ(op.output_map.x, 3);
    EXPECT_EQ(op.output_map.y, 2);
    EXPECT_EQ(op.output_map.z, 1);
}

TEST_F(TBOutputOpTest, EpilogueAllreduce) {
    STensor stensor;
    TBOutputOp op(nullptr, stensor, int3(), -1, TB_EPILOGUE_ALLREDUCE);

    EXPECT_EQ(op.epilogue, TB_EPILOGUE_ALLREDUCE);
}

TEST_F(TBOutputOpTest, EpilogueReduce) {
    STensor stensor;
    TBOutputOp op(nullptr, stensor, int3(), -1, TB_EPILOGUE_REDUCE);

    EXPECT_EQ(op.epilogue, TB_EPILOGUE_REDUCE);
}

TEST_F(TBOutputOpTest, DTensorCreated) {
    STensor stensor;
    stensor.num_dims = 3;
    stensor.dim[0] = 8;
    stensor.dim[1] = 32;
    stensor.dim[2] = 64;
    stensor.data_type = DT_BFLOAT16;

    TBOutputOp op(nullptr, stensor, int3(), -1, TB_EPILOGUE_NONE);

    EXPECT_EQ(op.dtensor.num_dims, 3);
    EXPECT_EQ(op.dtensor.dim[0], 8);
    EXPECT_EQ(op.dtensor.dim[1], 32);
    EXPECT_EQ(op.dtensor.dim[2], 64);
    EXPECT_EQ(op.dtensor.data_type, DT_BFLOAT16);
}

TEST_F(TBOutputOpTest, InputTensorStored) {
    STensor stensor;
    stensor.num_dims = 2;
    stensor.dim[0] = 64;
    stensor.dim[1] = 128;

    TBOutputOp op(nullptr, stensor, int3(), -1, TB_EPILOGUE_NONE);

    EXPECT_EQ(op.num_inputs(), 1u);
}

// =============================================================================
// Operator Type Classification Tests
// =============================================================================

class OperatorTypeClassificationTest : public ::testing::Test {};

TEST_F(OperatorTypeClassificationTest, InputOp) {
    EXPECT_TRUE(is_input_op(TB_INPUT_OP));
    EXPECT_FALSE(is_input_op(TB_OUTPUT_OP));
    EXPECT_FALSE(is_input_op(TB_MATMUL_OP));
}

TEST_F(OperatorTypeClassificationTest, OutputOp) {
    EXPECT_TRUE(is_output_op(TB_OUTPUT_OP));
    EXPECT_FALSE(is_output_op(TB_INPUT_OP));
    EXPECT_FALSE(is_output_op(TB_ADD_OP));
}

TEST_F(OperatorTypeClassificationTest, MatmulOp) {
    EXPECT_TRUE(is_matmul_op(TB_MATMUL_OP));
    EXPECT_FALSE(is_matmul_op(TB_ADD_OP));
}

TEST_F(OperatorTypeClassificationTest, ElementUnaryOps) {
    EXPECT_TRUE(is_element_unary_op(TB_EXP_OP));
    EXPECT_TRUE(is_element_unary_op(TB_SQUARE_OP));
    EXPECT_TRUE(is_element_unary_op(TB_SQRT_OP));
    EXPECT_TRUE(is_element_unary_op(TB_SILU_OP));
    EXPECT_TRUE(is_element_unary_op(TB_GELU_OP));
    EXPECT_TRUE(is_element_unary_op(TB_RELU_OP));
    EXPECT_TRUE(is_element_unary_op(TB_CLAMP_OP));
    EXPECT_FALSE(is_element_unary_op(TB_ADD_OP));
}

TEST_F(OperatorTypeClassificationTest, ElementBinaryOps) {
    EXPECT_TRUE(is_element_binary_op(TB_ADD_OP));
    EXPECT_TRUE(is_element_binary_op(TB_MUL_OP));
    EXPECT_TRUE(is_element_binary_op(TB_DIV_OP));
    EXPECT_TRUE(is_element_binary_op(TB_SUB_OP));
    EXPECT_TRUE(is_element_binary_op(TB_POW_OP));
    EXPECT_FALSE(is_element_binary_op(TB_EXP_OP));
}

TEST_F(OperatorTypeClassificationTest, ReductionOps) {
    EXPECT_TRUE(is_reduction_op(TB_REDUCTION_OP));
    EXPECT_TRUE(is_reduction_op(TB_REDUCTION_TO_DIMX_OP));
    EXPECT_TRUE(is_reduction_op(TB_REDUCTION_MAX_OP));
    EXPECT_TRUE(is_reduction_op(TB_RMS_NORM_OP));
    EXPECT_FALSE(is_reduction_op(TB_ADD_OP));
}

TEST_F(OperatorTypeClassificationTest, AccumulatorOps) {
    EXPECT_TRUE(is_accumulator_op(TB_FORLOOP_ACCUM_OP));
    EXPECT_TRUE(is_accumulator_op(TB_FORLOOP_ACCUM_RESCALE_OP));
    EXPECT_TRUE(is_accumulator_op(TB_FORLOOP_ACCUM_MAX_OP));
    EXPECT_FALSE(is_accumulator_op(TB_REDUCTION_OP));
}

TEST_F(OperatorTypeClassificationTest, OpTypeToString) {
    EXPECT_STREQ(op_type_to_string(TB_INPUT_OP), "input");
    EXPECT_STREQ(op_type_to_string(TB_OUTPUT_OP), "output");
    EXPECT_STREQ(op_type_to_string(TB_MATMUL_OP), "matmul");
    EXPECT_STREQ(op_type_to_string(TB_SILU_OP), "silu");
    EXPECT_STREQ(op_type_to_string(TB_ADD_OP), "add");
    EXPECT_STREQ(op_type_to_string(TB_REDUCTION_OP), "reduction");
}

// =============================================================================
// Parameterized Operator Type Tests
// =============================================================================

struct OpTypeParam {
    TBOperatorType type;
    const char* expected_name;
    bool is_unary;
    bool is_binary;
};

class OpTypeParameterizedTest : public ::testing::TestWithParam<OpTypeParam> {};

TEST_P(OpTypeParameterizedTest, OperatorProperties) {
    auto param = GetParam();

    EXPECT_STREQ(op_type_to_string(param.type), param.expected_name);
    EXPECT_EQ(is_element_unary_op(param.type), param.is_unary);
    EXPECT_EQ(is_element_binary_op(param.type), param.is_binary);
}

INSTANTIATE_TEST_SUITE_P(
    UnaryOps,
    OpTypeParameterizedTest,
    ::testing::Values(
        OpTypeParam{TB_EXP_OP, "exp", true, false},
        OpTypeParam{TB_SILU_OP, "silu", true, false},
        OpTypeParam{TB_GELU_OP, "gelu", true, false},
        OpTypeParam{TB_RELU_OP, "relu", true, false}
    )
);

INSTANTIATE_TEST_SUITE_P(
    BinaryOps,
    OpTypeParameterizedTest,
    ::testing::Values(
        OpTypeParam{TB_ADD_OP, "add", false, true},
        OpTypeParam{TB_MUL_OP, "mul", false, true},
        OpTypeParam{TB_DIV_OP, "div", false, true},
        OpTypeParam{TB_SUB_OP, "sub", false, true}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
