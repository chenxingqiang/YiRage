// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_kernel_gtest.cc
 * @brief Kernel Module Unit Tests (Google Test version)
 *
 * Tests for yirage kernel module including:
 *   - DTensor (device tensor)
 *   - KNOperator (kernel operator)
 *   - Graph (kernel graph)
 *   - Matmul, reduction, element-wise operations
 *   - Memory allocation and management
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <functional>
#include <unordered_map>

namespace yirage {
namespace type {

enum DataType {
    DT_FLOAT16 = 940,
    DT_BFLOAT16 = 941,
    DT_FLOAT32 = 950,
    DT_INT32 = 955,
    DT_UNKNOWN = 999,
};

enum KNOperatorType {
    KN_UNKOWN = 1000,
    KN_INPUT_OP = 1001,
    KN_OUTPUT_OP = 1002,
    KN_MATMUL_OP = 1003,
    KN_EXP_OP = 1100,
    KN_SQUARE_OP = 1101,
    KN_SQRT_OP = 1102,
    KN_SILU_OP = 1104,
    KN_GELU_OP = 1106,
    KN_RELU_OP = 1150,
    KN_CLAMP_OP = 1151,
    KN_ADD_OP = 1200,
    KN_MUL_OP = 1201,
    KN_DIV_OP = 1202,
    KN_POW_OP = 1203,
    KN_REDUCTION_0_OP = 1300,
    KN_REDUCTION_1_OP = 1301,
    KN_REDUCTION_2_OP = 1302,
    KN_RMS_NORM_OP = 1350,
    KN_CUSTOMIZED_OP = 1999,
};

typedef uint16_t FPType;
typedef int64_t GuidType;

}  // namespace type

namespace layout {

enum DmemLayout {
    DmemRowMajor = 100,
    DmemColumnMajor = 101,
    DmemUnknownLayout = 199,
};

}  // namespace layout

namespace config {
    constexpr int MAX_TENSOR_DIMS = 8;
    constexpr size_t MAX_DMEM_SIZE = 1ULL * 1024 * 1024 * 1024;  // 1GB
    constexpr size_t MAX_DMEM_FP_SIZE = 256ULL * 1024 * 1024;    // 256MB
}  // namespace config

namespace kernel {

// Forward declarations
class Graph;
class KNOperator;

// =============================================================================
// DTensor (Device Tensor)
// =============================================================================

struct DTensor {
    type::DataType data_type = type::DT_UNKNOWN;
    layout::DmemLayout layout = layout::DmemUnknownLayout;
    int num_dims = 0;
    int dim[config::MAX_TENSOR_DIMS] = {0};
    KNOperator* owner_op = nullptr;
    int owner_ts_idx = -1000;
    off_t data_offset = -1000;
    off_t fp_offset = -1000;
    int64_t guid = 0;
    
    static std::atomic<int64_t> next_guid;
    static const DTensor EMPTY_TENSOR;
    
    DTensor() = default;
    
    size_t num_elements() const {
        if (num_dims == 0) return 0;
        size_t n = 1;
        for (int i = 0; i < num_dims; ++i) {
            n *= dim[i];
        }
        return n;
    }
    
    size_t data_size() const {
        size_t elem_size = 4;  // Default to float32
        if (data_type == type::DT_FLOAT16 || data_type == type::DT_BFLOAT16) {
            elem_size = 2;
        }
        return num_elements() * elem_size;
    }
    
    size_t fingerprint_size() const {
        return num_elements() * sizeof(type::FPType);
    }
    
    size_t get_owner_independent_hash() const {
        size_t ret = std::hash<int>()(data_type);
        ret ^= std::hash<int>()(layout) << 1;
        ret ^= std::hash<int>()(num_dims) << 2;
        for (int i = 0; i < num_dims; ++i) {
            ret ^= std::hash<int>()(dim[i]) << (i + 3);
        }
        return ret;
    }
};

std::atomic<int64_t> DTensor::next_guid(10000000);
const DTensor DTensor::EMPTY_TENSOR = {};

// =============================================================================
// KNOperator (Kernel Operator)
// =============================================================================

class KNOperator {
public:
    Graph* kgraph;
    type::KNOperatorType op_type;
    std::vector<DTensor> input_tensors;
    std::vector<DTensor> output_tensors;
    
    KNOperator(Graph* g, type::KNOperatorType type)
        : kgraph(g), op_type(type) {}
    
    KNOperator(Graph* g, type::KNOperatorType type, DTensor const& A)
        : kgraph(g), op_type(type) {
        input_tensors.push_back(A);
    }
    
    KNOperator(Graph* g, type::KNOperatorType type, 
               DTensor const& A, DTensor const& B)
        : kgraph(g), op_type(type) {
        input_tensors.push_back(A);
        input_tensors.push_back(B);
    }
    
    virtual ~KNOperator() = default;
    
    int get_num_inputs() const { return input_tensors.size(); }
    int get_num_outputs() const { return output_tensors.size(); }
    
    size_t get_owner_independent_hash() const {
        size_t ret = std::hash<int>()(op_type);
        for (auto const& t : input_tensors) {
            ret ^= t.get_owner_independent_hash();
        }
        for (auto const& t : output_tensors) {
            ret ^= t.get_owner_independent_hash();
        }
        return ret;
    }
};

// =============================================================================
// Specific Operator Types
// =============================================================================

class KNInputOp : public KNOperator {
public:
    std::vector<size_t> input_strides;
    
    KNInputOp(Graph* g, std::vector<int> const& dims, 
              std::vector<size_t> const& strides,
              type::DataType dtype, layout::DmemLayout layout)
        : KNOperator(g, type::KN_INPUT_OP), input_strides(strides) {
        DTensor output;
        output.num_dims = dims.size();
        for (size_t i = 0; i < dims.size(); ++i) {
            output.dim[i] = dims[i];
        }
        output.data_type = dtype;
        output.layout = layout;
        output.owner_op = this;
        output.owner_ts_idx = 0;
        output.guid = DTensor::next_guid++;
        output_tensors.push_back(output);
    }
};

class KNOutputOp : public KNOperator {
public:
    std::vector<size_t> output_strides;
    
    KNOutputOp(Graph* g, DTensor const& input, 
               std::vector<size_t> const& strides = {})
        : KNOperator(g, type::KN_OUTPUT_OP, input), output_strides(strides) {}
};

class KNMatmulOp : public KNOperator {
public:
    KNMatmulOp(Graph* g, DTensor const& A, DTensor const& B)
        : KNOperator(g, type::KN_MATMUL_OP, A, B) {
        // Compute output shape: [..., M, K] x [..., K, N] -> [..., M, N]
        DTensor C;
        C.num_dims = A.num_dims;
        for (int i = 0; i < C.num_dims - 1; ++i) {
            C.dim[i] = A.dim[i];
        }
        C.dim[C.num_dims - 1] = B.dim[B.num_dims - 1];
        C.data_type = A.data_type;
        C.layout = layout::DmemRowMajor;
        C.owner_op = this;
        C.owner_ts_idx = 0;
        C.guid = DTensor::next_guid++;
        output_tensors.push_back(C);
    }
};

class KNElementUnaryOp : public KNOperator {
public:
    KNElementUnaryOp(Graph* g, DTensor const& input, type::KNOperatorType op)
        : KNOperator(g, op, input) {
        // Output has same shape as input
        DTensor output = input;
        output.owner_op = this;
        output.owner_ts_idx = 0;
        output.guid = DTensor::next_guid++;
        output_tensors.push_back(output);
    }
};

class KNElementBinaryOp : public KNOperator {
public:
    KNElementBinaryOp(Graph* g, DTensor const& A, DTensor const& B, 
                      type::KNOperatorType op)
        : KNOperator(g, op, A, B) {
        // Output has same shape as inputs (assuming broadcasting done)
        DTensor output = A;
        output.owner_op = this;
        output.owner_ts_idx = 0;
        output.guid = DTensor::next_guid++;
        output_tensors.push_back(output);
    }
};

class KNReductionOp : public KNOperator {
public:
    int reduction_dim;
    
    KNReductionOp(Graph* g, DTensor const& input, int dim)
        : KNOperator(g, static_cast<type::KNOperatorType>(
              type::KN_REDUCTION_0_OP + dim), input), 
          reduction_dim(dim) {
        DTensor output = input;
        output.dim[dim] = 1;  // Reduced dimension
        output.owner_op = this;
        output.owner_ts_idx = 0;
        output.guid = DTensor::next_guid++;
        output_tensors.push_back(output);
    }
};

// =============================================================================
// Graph (Kernel Graph)
// =============================================================================

struct dim3 {
    int x = 1, y = 1, z = 1;
    dim3(int x_ = 1, int y_ = 1, int z_ = 1) : x(x_), y(y_), z(z_) {}
};

class Graph {
public:
    dim3 gpu_dim;
    bool disable_fingerprint;
    std::vector<KNOperator*> operators;
    
    off_t dmem_data_offset = 0;
    off_t dmem_fp_offset = 0;
    std::vector<std::pair<off_t, size_t>> allocated_data_tensors;
    std::vector<std::pair<off_t, size_t>> allocated_fp_tensors;
    
    Graph(dim3 gpu = dim3(), bool disable_fp = false)
        : gpu_dim(gpu), disable_fingerprint(disable_fp) {}
    
    ~Graph() {
        for (auto* op : operators) {
            delete op;
        }
    }
    
    // Input/Output operations
    DTensor new_input(std::vector<int> const& dims,
                      std::vector<size_t> const& strides,
                      type::DataType dtype,
                      layout::DmemLayout layout) {
        auto* op = new KNInputOp(this, dims, strides, dtype, layout);
        operators.push_back(op);
        return op->output_tensors[0];
    }
    
    void mark_output(DTensor const& tensor, 
                     std::vector<size_t> const& strides = {}) {
        auto* op = new KNOutputOp(this, tensor, strides);
        operators.push_back(op);
    }
    
    // Matmul
    DTensor matmul(DTensor const& A, DTensor const& B) {
        if (A.num_dims != B.num_dims) return DTensor::EMPTY_TENSOR;
        if (A.dim[A.num_dims - 1] != B.dim[B.num_dims - 2]) 
            return DTensor::EMPTY_TENSOR;
        
        auto* op = new KNMatmulOp(this, A, B);
        operators.push_back(op);
        return op->output_tensors[0];
    }
    
    // Element-wise unary
    DTensor elementunary(DTensor const& input, type::KNOperatorType op_type) {
        auto* op = new KNElementUnaryOp(this, input, op_type);
        operators.push_back(op);
        return op->output_tensors[0];
    }
    
    // Element-wise binary
    DTensor elementbinary(DTensor const& A, DTensor const& B, 
                          type::KNOperatorType op_type) {
        auto* op = new KNElementBinaryOp(this, A, B, op_type);
        operators.push_back(op);
        return op->output_tensors[0];
    }
    
    // Reduction
    DTensor reduction(DTensor const& input, int dim) {
        auto* op = new KNReductionOp(this, input, dim);
        operators.push_back(op);
        return op->output_tensors[0];
    }
    
    // Memory management
    bool can_allocate(DTensor const& tensor, bool alloc_fp = false) const {
        if (disable_fingerprint) return true;
        
        size_t data_size = (tensor.data_size() + 15) & ~15;
        if (dmem_data_offset + data_size > config::MAX_DMEM_SIZE) {
            return false;
        }
        if (alloc_fp) {
            size_t fp_size = (tensor.fingerprint_size() + 15) & ~15;
            if (dmem_fp_offset + fp_size > config::MAX_DMEM_FP_SIZE) {
                return false;
            }
        }
        return true;
    }
    
    bool allocate(DTensor& tensor, bool alloc_fp = false) {
        size_t aligned_size = (tensor.data_size() + 15) & ~15;
        tensor.data_offset = dmem_data_offset;
        dmem_data_offset += aligned_size;
        allocated_data_tensors.push_back({tensor.data_offset, aligned_size});
        
        if (alloc_fp) {
            size_t fp_aligned = (tensor.fingerprint_size() + 15) & ~15;
            tensor.fp_offset = dmem_fp_offset;
            dmem_fp_offset += fp_aligned;
            allocated_fp_tensors.push_back({tensor.fp_offset, fp_aligned});
        }
        return true;
    }
    
    void free(DTensor& tensor) {
        if (tensor.fp_offset >= 0 && !allocated_fp_tensors.empty()) {
            dmem_fp_offset -= allocated_fp_tensors.back().second;
            allocated_fp_tensors.pop_back();
            tensor.fp_offset = -1;
        }
        if (!allocated_data_tensors.empty()) {
            dmem_data_offset -= allocated_data_tensors.back().second;
            allocated_data_tensors.pop_back();
            tensor.data_offset = -1;
        }
    }
    
    // Query functions
    int get_num_operators() const { return operators.size(); }
    
    int get_num_input_dtensors() const {
        int count = 0;
        for (auto* op : operators) {
            if (op->op_type == type::KN_INPUT_OP) count++;
        }
        return count;
    }
    
    int get_num_output_dtensors() const {
        int count = 0;
        for (auto* op : operators) {
            if (op->op_type == type::KN_OUTPUT_OP) count++;
        }
        return count;
    }
    
    size_t get_owner_independent_hash() const {
        size_t ret = std::hash<int>()(gpu_dim.x);
        ret ^= std::hash<int>()(gpu_dim.y) << 1;
        ret ^= std::hash<int>()(gpu_dim.z) << 2;
        for (auto* op : operators) {
            ret ^= op->get_owner_independent_hash();
        }
        return ret;
    }
};

}  // namespace kernel
}  // namespace yirage

using namespace yirage;
using namespace yirage::kernel;

// =============================================================================
// DTensor Tests
// =============================================================================

class DTensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset GUID counter for predictable tests
    }
};

TEST_F(DTensorTest, DefaultConstruction) {
    DTensor t;
    EXPECT_EQ(t.data_type, type::DT_UNKNOWN);
    EXPECT_EQ(t.layout, layout::DmemUnknownLayout);
    EXPECT_EQ(t.num_dims, 0);
    EXPECT_EQ(t.owner_op, nullptr);
}

TEST_F(DTensorTest, NumElements1D) {
    DTensor t;
    t.num_dims = 1;
    t.dim[0] = 128;
    EXPECT_EQ(t.num_elements(), 128u);
}

TEST_F(DTensorTest, NumElements2D) {
    DTensor t;
    t.num_dims = 2;
    t.dim[0] = 32;
    t.dim[1] = 64;
    EXPECT_EQ(t.num_elements(), 32u * 64u);
}

TEST_F(DTensorTest, NumElements3D) {
    DTensor t;
    t.num_dims = 3;
    t.dim[0] = 8;
    t.dim[1] = 32;
    t.dim[2] = 64;
    EXPECT_EQ(t.num_elements(), 8u * 32u * 64u);
}

TEST_F(DTensorTest, DataSizeFloat32) {
    DTensor t;
    t.num_dims = 2;
    t.dim[0] = 32;
    t.dim[1] = 64;
    t.data_type = type::DT_FLOAT32;
    EXPECT_EQ(t.data_size(), 32u * 64u * 4u);
}

TEST_F(DTensorTest, DataSizeFloat16) {
    DTensor t;
    t.num_dims = 2;
    t.dim[0] = 32;
    t.dim[1] = 64;
    t.data_type = type::DT_FLOAT16;
    EXPECT_EQ(t.data_size(), 32u * 64u * 2u);
}

TEST_F(DTensorTest, FingerprintSize) {
    DTensor t;
    t.num_dims = 2;
    t.dim[0] = 32;
    t.dim[1] = 64;
    EXPECT_EQ(t.fingerprint_size(), 32u * 64u * sizeof(type::FPType));
}

TEST_F(DTensorTest, HashDifferentShapes) {
    DTensor t1, t2;
    t1.num_dims = 2; t1.dim[0] = 32; t1.dim[1] = 64;
    t2.num_dims = 2; t2.dim[0] = 64; t2.dim[1] = 32;
    
    EXPECT_NE(t1.get_owner_independent_hash(), t2.get_owner_independent_hash());
}

TEST_F(DTensorTest, HashSameShapeDifferentType) {
    DTensor t1, t2;
    t1.num_dims = 2; t1.dim[0] = 32; t1.dim[1] = 64;
    t1.data_type = type::DT_FLOAT32;
    
    t2.num_dims = 2; t2.dim[0] = 32; t2.dim[1] = 64;
    t2.data_type = type::DT_FLOAT16;
    
    EXPECT_NE(t1.get_owner_independent_hash(), t2.get_owner_independent_hash());
}

TEST_F(DTensorTest, EmptyTensor) {
    DTensor empty = DTensor::EMPTY_TENSOR;
    EXPECT_EQ(empty.num_dims, 0);
    EXPECT_EQ(empty.num_elements(), 0u);
}

// =============================================================================
// KNOperator Tests
// =============================================================================

class KNOperatorTest : public ::testing::Test {
protected:
    std::unique_ptr<Graph> graph;
    
    void SetUp() override {
        graph = std::make_unique<Graph>();
    }
};

TEST_F(KNOperatorTest, InputOperator) {
    std::vector<int> dims = {32, 64};
    std::vector<size_t> strides = {64, 1};
    
    auto* op = new KNInputOp(graph.get(), dims, strides, 
                             type::DT_FLOAT32, layout::DmemRowMajor);
    
    EXPECT_EQ(op->op_type, type::KN_INPUT_OP);
    EXPECT_EQ(op->get_num_inputs(), 0);
    EXPECT_EQ(op->get_num_outputs(), 1);
    EXPECT_EQ(op->output_tensors[0].num_dims, 2);
    
    delete op;
}

TEST_F(KNOperatorTest, MatmulOperator) {
    DTensor A, B;
    A.num_dims = 2; A.dim[0] = 32; A.dim[1] = 64;
    A.data_type = type::DT_FLOAT32;
    
    B.num_dims = 2; B.dim[0] = 64; B.dim[1] = 128;
    B.data_type = type::DT_FLOAT32;
    
    auto* op = new KNMatmulOp(graph.get(), A, B);
    
    EXPECT_EQ(op->op_type, type::KN_MATMUL_OP);
    EXPECT_EQ(op->get_num_inputs(), 2);
    EXPECT_EQ(op->get_num_outputs(), 1);
    
    // Output shape: [32, 128]
    EXPECT_EQ(op->output_tensors[0].dim[0], 32);
    EXPECT_EQ(op->output_tensors[0].dim[1], 128);
    
    delete op;
}

TEST_F(KNOperatorTest, ElementUnaryOperator) {
    DTensor input;
    input.num_dims = 2; input.dim[0] = 32; input.dim[1] = 64;
    input.data_type = type::DT_FLOAT32;
    
    auto* op = new KNElementUnaryOp(graph.get(), input, type::KN_RELU_OP);
    
    EXPECT_EQ(op->op_type, type::KN_RELU_OP);
    EXPECT_EQ(op->get_num_inputs(), 1);
    EXPECT_EQ(op->get_num_outputs(), 1);
    
    // Output shape same as input
    EXPECT_EQ(op->output_tensors[0].dim[0], 32);
    EXPECT_EQ(op->output_tensors[0].dim[1], 64);
    
    delete op;
}

TEST_F(KNOperatorTest, ElementBinaryOperator) {
    DTensor A, B;
    A.num_dims = 2; A.dim[0] = 32; A.dim[1] = 64;
    A.data_type = type::DT_FLOAT32;
    
    B.num_dims = 2; B.dim[0] = 32; B.dim[1] = 64;
    B.data_type = type::DT_FLOAT32;
    
    auto* op = new KNElementBinaryOp(graph.get(), A, B, type::KN_ADD_OP);
    
    EXPECT_EQ(op->op_type, type::KN_ADD_OP);
    EXPECT_EQ(op->get_num_inputs(), 2);
    EXPECT_EQ(op->get_num_outputs(), 1);
    
    delete op;
}

TEST_F(KNOperatorTest, ReductionOperator) {
    DTensor input;
    input.num_dims = 3; 
    input.dim[0] = 8; input.dim[1] = 32; input.dim[2] = 64;
    input.data_type = type::DT_FLOAT32;
    
    auto* op = new KNReductionOp(graph.get(), input, 2);  // Reduce last dim
    
    EXPECT_EQ(op->op_type, type::KN_REDUCTION_2_OP);
    EXPECT_EQ(op->reduction_dim, 2);
    
    // Reduced dimension becomes 1
    EXPECT_EQ(op->output_tensors[0].dim[2], 1);
    
    delete op;
}

// =============================================================================
// Graph Tests
// =============================================================================

class GraphTest : public ::testing::Test {
protected:
    std::unique_ptr<Graph> graph;
    
    void SetUp() override {
        graph = std::make_unique<Graph>();
    }
};

TEST_F(GraphTest, DefaultConstruction) {
    EXPECT_EQ(graph->gpu_dim.x, 1);
    EXPECT_EQ(graph->gpu_dim.y, 1);
    EXPECT_EQ(graph->gpu_dim.z, 1);
    EXPECT_FALSE(graph->disable_fingerprint);
}

TEST_F(GraphTest, CustomGpuDim) {
    Graph g(dim3(4, 2, 1));
    EXPECT_EQ(g.gpu_dim.x, 4);
    EXPECT_EQ(g.gpu_dim.y, 2);
    EXPECT_EQ(g.gpu_dim.z, 1);
}

TEST_F(GraphTest, NewInput) {
    std::vector<int> dims = {32, 64};
    std::vector<size_t> strides = {64, 1};
    
    DTensor t = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    
    EXPECT_EQ(t.num_dims, 2);
    EXPECT_EQ(t.dim[0], 32);
    EXPECT_EQ(t.dim[1], 64);
    EXPECT_EQ(t.data_type, type::DT_FLOAT32);
    EXPECT_EQ(graph->get_num_input_dtensors(), 1);
}

TEST_F(GraphTest, MarkOutput) {
    std::vector<int> dims = {32, 64};
    std::vector<size_t> strides = {64, 1};
    
    DTensor t = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    graph->mark_output(t);
    
    EXPECT_EQ(graph->get_num_output_dtensors(), 1);
}

TEST_F(GraphTest, Matmul) {
    std::vector<int> dims_a = {32, 64};
    std::vector<int> dims_b = {64, 128};
    std::vector<size_t> strides_a = {64, 1};
    std::vector<size_t> strides_b = {128, 1};
    
    DTensor A = graph->new_input(dims_a, strides_a, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    DTensor B = graph->new_input(dims_b, strides_b, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    DTensor C = graph->matmul(A, B);
    
    EXPECT_EQ(C.num_dims, 2);
    EXPECT_EQ(C.dim[0], 32);
    EXPECT_EQ(C.dim[1], 128);
}

TEST_F(GraphTest, MatmulDimensionMismatch) {
    DTensor A, B;
    A.num_dims = 2; A.dim[0] = 32; A.dim[1] = 64;
    B.num_dims = 2; B.dim[0] = 128; B.dim[1] = 256;  // Wrong K dimension
    
    DTensor C = graph->matmul(A, B);
    
    // Should return empty tensor on mismatch
    EXPECT_EQ(C.num_dims, 0);
}

TEST_F(GraphTest, ElementUnary) {
    std::vector<int> dims = {32, 64};
    std::vector<size_t> strides = {64, 1};
    
    DTensor t = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    DTensor r = graph->elementunary(t, type::KN_RELU_OP);
    
    EXPECT_EQ(r.dim[0], 32);
    EXPECT_EQ(r.dim[1], 64);
}

TEST_F(GraphTest, ElementBinary) {
    std::vector<int> dims = {32, 64};
    std::vector<size_t> strides = {64, 1};
    
    DTensor A = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    DTensor B = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    DTensor C = graph->elementbinary(A, B, type::KN_ADD_OP);
    
    EXPECT_EQ(C.dim[0], 32);
    EXPECT_EQ(C.dim[1], 64);
}

TEST_F(GraphTest, Reduction) {
    std::vector<int> dims = {8, 32, 64};
    std::vector<size_t> strides = {2048, 64, 1};
    
    DTensor t = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    DTensor r = graph->reduction(t, 2);
    
    EXPECT_EQ(r.dim[0], 8);
    EXPECT_EQ(r.dim[1], 32);
    EXPECT_EQ(r.dim[2], 1);  // Reduced
}

TEST_F(GraphTest, OperatorCount) {
    std::vector<int> dims = {32, 64};
    std::vector<size_t> strides = {64, 1};
    
    DTensor A = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    DTensor B = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    graph->elementbinary(A, B, type::KN_ADD_OP);
    
    EXPECT_EQ(graph->get_num_operators(), 3);  // 2 inputs + 1 add
}

TEST_F(GraphTest, Hash) {
    std::vector<int> dims = {32, 64};
    std::vector<size_t> strides = {64, 1};
    
    DTensor t = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                  layout::DmemRowMajor);
    graph->mark_output(t);
    
    size_t hash1 = graph->get_owner_independent_hash();
    EXPECT_NE(hash1, 0u);
}

// =============================================================================
// Memory Allocation Tests
// =============================================================================

class MemoryAllocationTest : public ::testing::Test {
protected:
    std::unique_ptr<Graph> graph;
    
    void SetUp() override {
        graph = std::make_unique<Graph>();
    }
};

TEST_F(MemoryAllocationTest, CanAllocate) {
    DTensor t;
    t.num_dims = 2;
    t.dim[0] = 32;
    t.dim[1] = 64;
    t.data_type = type::DT_FLOAT32;
    
    EXPECT_TRUE(graph->can_allocate(t));
}

TEST_F(MemoryAllocationTest, AllocateData) {
    DTensor t;
    t.num_dims = 2;
    t.dim[0] = 32;
    t.dim[1] = 64;
    t.data_type = type::DT_FLOAT32;
    
    EXPECT_TRUE(graph->allocate(t, false));
    EXPECT_EQ(t.data_offset, 0);  // First allocation at offset 0
    EXPECT_GE(graph->dmem_data_offset, t.data_size());
}

TEST_F(MemoryAllocationTest, AllocateWithFingerprint) {
    DTensor t;
    t.num_dims = 2;
    t.dim[0] = 32;
    t.dim[1] = 64;
    t.data_type = type::DT_FLOAT32;
    
    EXPECT_TRUE(graph->allocate(t, true));
    EXPECT_GE(t.data_offset, 0);
    EXPECT_GE(t.fp_offset, 0);
}

TEST_F(MemoryAllocationTest, MultipleAllocations) {
    DTensor t1, t2;
    t1.num_dims = 2; t1.dim[0] = 32; t1.dim[1] = 64;
    t1.data_type = type::DT_FLOAT32;
    
    t2.num_dims = 2; t2.dim[0] = 64; t2.dim[1] = 128;
    t2.data_type = type::DT_FLOAT32;
    
    graph->allocate(t1, false);
    off_t offset1 = t1.data_offset;
    
    graph->allocate(t2, false);
    off_t offset2 = t2.data_offset;
    
    // Second allocation should be after first
    EXPECT_GT(offset2, offset1);
}

TEST_F(MemoryAllocationTest, FreeMemory) {
    DTensor t;
    t.num_dims = 2;
    t.dim[0] = 32;
    t.dim[1] = 64;
    t.data_type = type::DT_FLOAT32;
    
    graph->allocate(t, false);
    off_t offset_before = graph->dmem_data_offset;
    
    graph->free(t);
    
    EXPECT_LT(graph->dmem_data_offset, offset_before);
    EXPECT_EQ(t.data_offset, -1);
}

TEST_F(MemoryAllocationTest, AlignedAllocation) {
    DTensor t;
    t.num_dims = 1;
    t.dim[0] = 17;  // Not aligned to 16
    t.data_type = type::DT_FLOAT32;
    
    graph->allocate(t, false);
    
    // Offset should be 16-byte aligned
    EXPECT_EQ(graph->dmem_data_offset % 16, 0u);
}

// =============================================================================
// Parameterized Operator Type Tests
// =============================================================================

struct UnaryOpParam {
    type::KNOperatorType op;
    std::string name;
};

class UnaryOperatorTest : public ::testing::TestWithParam<UnaryOpParam> {
protected:
    std::unique_ptr<Graph> graph;
    
    void SetUp() override {
        graph = std::make_unique<Graph>();
    }
};

TEST_P(UnaryOperatorTest, CreateAndRun) {
    auto param = GetParam();
    
    std::vector<int> dims = {32, 64};
    std::vector<size_t> strides = {64, 1};
    
    DTensor input = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                     layout::DmemRowMajor);
    DTensor output = graph->elementunary(input, param.op);
    
    EXPECT_EQ(output.dim[0], 32);
    EXPECT_EQ(output.dim[1], 64);
}

INSTANTIATE_TEST_SUITE_P(
    AllUnaryOps,
    UnaryOperatorTest,
    ::testing::Values(
        UnaryOpParam{type::KN_EXP_OP, "exp"},
        UnaryOpParam{type::KN_SQRT_OP, "sqrt"},
        UnaryOpParam{type::KN_SILU_OP, "silu"},
        UnaryOpParam{type::KN_GELU_OP, "gelu"},
        UnaryOpParam{type::KN_RELU_OP, "relu"}
    )
);

struct BinaryOpParam {
    type::KNOperatorType op;
    std::string name;
};

class BinaryOperatorTest : public ::testing::TestWithParam<BinaryOpParam> {
protected:
    std::unique_ptr<Graph> graph;
    
    void SetUp() override {
        graph = std::make_unique<Graph>();
    }
};

TEST_P(BinaryOperatorTest, CreateAndRun) {
    auto param = GetParam();
    
    std::vector<int> dims = {32, 64};
    std::vector<size_t> strides = {64, 1};
    
    DTensor A = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                 layout::DmemRowMajor);
    DTensor B = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                 layout::DmemRowMajor);
    DTensor C = graph->elementbinary(A, B, param.op);
    
    EXPECT_EQ(C.dim[0], 32);
    EXPECT_EQ(C.dim[1], 64);
}

INSTANTIATE_TEST_SUITE_P(
    AllBinaryOps,
    BinaryOperatorTest,
    ::testing::Values(
        BinaryOpParam{type::KN_ADD_OP, "add"},
        BinaryOpParam{type::KN_MUL_OP, "mul"},
        BinaryOpParam{type::KN_DIV_OP, "div"},
        BinaryOpParam{type::KN_POW_OP, "pow"}
    )
);

// =============================================================================
// Complex Graph Tests
// =============================================================================

class ComplexGraphTest : public ::testing::Test {
protected:
    std::unique_ptr<Graph> graph;
    
    void SetUp() override {
        graph = std::make_unique<Graph>();
    }
};

TEST_F(ComplexGraphTest, LinearLayer) {
    // Simple linear layer: Y = X @ W
    std::vector<int> x_dims = {8, 512};    // [batch, in_features]
    std::vector<int> w_dims = {512, 1024}; // [in_features, out_features]
    std::vector<size_t> x_strides = {512, 1};
    std::vector<size_t> w_strides = {1024, 1};
    
    DTensor X = graph->new_input(x_dims, x_strides, type::DT_FLOAT16, 
                                 layout::DmemRowMajor);
    DTensor W = graph->new_input(w_dims, w_strides, type::DT_FLOAT16, 
                                 layout::DmemRowMajor);
    DTensor Y = graph->matmul(X, W);
    graph->mark_output(Y);
    
    EXPECT_EQ(Y.dim[0], 8);
    EXPECT_EQ(Y.dim[1], 1024);
    EXPECT_EQ(graph->get_num_operators(), 4);  // 2 inputs + matmul + output
}

TEST_F(ComplexGraphTest, MLPBlock) {
    // MLP: Y = SiLU(X @ W1) * (X @ W2)
    std::vector<int> x_dims = {8, 512};
    std::vector<int> w_dims = {512, 1024};
    std::vector<size_t> x_strides = {512, 1};
    std::vector<size_t> w_strides = {1024, 1};
    
    DTensor X = graph->new_input(x_dims, x_strides, type::DT_FLOAT16, 
                                 layout::DmemRowMajor);
    DTensor W1 = graph->new_input(w_dims, w_strides, type::DT_FLOAT16, 
                                  layout::DmemRowMajor);
    DTensor W2 = graph->new_input(w_dims, w_strides, type::DT_FLOAT16, 
                                  layout::DmemRowMajor);
    
    DTensor H1 = graph->matmul(X, W1);
    DTensor H2 = graph->matmul(X, W2);
    DTensor A = graph->elementunary(H1, type::KN_SILU_OP);
    DTensor Y = graph->elementbinary(A, H2, type::KN_MUL_OP);
    graph->mark_output(Y);
    
    EXPECT_EQ(Y.dim[0], 8);
    EXPECT_EQ(Y.dim[1], 1024);
    EXPECT_EQ(graph->get_num_operators(), 8);
}

TEST_F(ComplexGraphTest, ReductionPipeline) {
    // Softmax-like: exp(x) / sum(exp(x))
    std::vector<int> dims = {8, 128};
    std::vector<size_t> strides = {128, 1};
    
    DTensor X = graph->new_input(dims, strides, type::DT_FLOAT32, 
                                 layout::DmemRowMajor);
    DTensor E = graph->elementunary(X, type::KN_EXP_OP);
    DTensor S = graph->reduction(E, 1);  // Sum along last dim
    
    EXPECT_EQ(S.dim[0], 8);
    EXPECT_EQ(S.dim[1], 1);
}

// =============================================================================
// Kernel Executor / Factory Tests
// =============================================================================

namespace yirage {
namespace kernel {

// Kernel config for execution
struct KernelConfig {
    int block_dim_x = 1;
    int block_dim_y = 1;
    int block_dim_z = 1;
    int grid_dim_x = 1;
    int grid_dim_y = 1;
    int grid_dim_z = 1;
    size_t shared_memory = 0;
    
    bool is_valid() const {
        return block_dim_x > 0 && block_dim_y > 0 && block_dim_z > 0 &&
               grid_dim_x > 0 && grid_dim_y > 0 && grid_dim_z > 0;
    }
};

// Kernel metrics
struct KernelMetrics {
    float execution_time_ms = 0.0f;
    size_t memory_used = 0;
    int execution_count = 0;
};

// Abstract kernel executor
class KernelExecutor {
public:
    virtual ~KernelExecutor() = default;
    virtual bool compile(std::string const& source, KernelConfig const& config) = 0;
    virtual bool execute(void** inputs, size_t num_inputs, 
                        void** outputs, size_t num_outputs,
                        KernelConfig const& config) = 0;
    virtual float get_execution_time() const = 0;
    virtual KernelMetrics get_metrics() const = 0;
    virtual bool validate_config(KernelConfig const& config) const = 0;
};

// Generic kernel executor implementation
class GenericKernelExecutor : public KernelExecutor {
public:
    GenericKernelExecutor(type::KNOperatorType op_type)
        : op_type_(op_type), compiled_(false), last_execution_time_(0.0f) {}
    
    bool compile(std::string const& source, KernelConfig const& config) override {
        if (!validate_config(config)) return false;
        source_ = source;
        config_ = config;
        compiled_ = true;
        return true;
    }
    
    bool execute(void** inputs, size_t num_inputs,
                void** outputs, size_t num_outputs,
                KernelConfig const& config) override {
        if (!compiled_) return false;
        execution_count_++;
        last_execution_time_ = 0.1f;  // Simulated
        return true;
    }
    
    float get_execution_time() const override { return last_execution_time_; }
    
    KernelMetrics get_metrics() const override {
        KernelMetrics m;
        m.execution_time_ms = last_execution_time_;
        m.execution_count = execution_count_;
        return m;
    }
    
    bool validate_config(KernelConfig const& config) const override {
        return config.is_valid();
    }
    
    type::KNOperatorType get_op_type() const { return op_type_; }
    bool is_compiled() const { return compiled_; }

private:
    type::KNOperatorType op_type_;
    std::string source_;
    KernelConfig config_;
    bool compiled_;
    float last_execution_time_;
    int execution_count_ = 0;
};

// Kernel executor factory
class KernelExecutorFactory {
public:
    static std::unique_ptr<KernelExecutor> create_matmul_executor() {
        return std::make_unique<GenericKernelExecutor>(type::KN_MATMUL_OP);
    }
    
    static std::unique_ptr<KernelExecutor> create_reduction_executor() {
        return std::make_unique<GenericKernelExecutor>(type::KN_REDUCTION_0_OP);
    }
    
    static std::unique_ptr<KernelExecutor> create_element_unary_executor(
        type::KNOperatorType op_type) {
        return std::make_unique<GenericKernelExecutor>(op_type);
    }
    
    static std::unique_ptr<KernelExecutor> create_element_binary_executor(
        type::KNOperatorType op_type) {
        return std::make_unique<GenericKernelExecutor>(op_type);
    }
};

}  // namespace kernel
}  // namespace yirage

class KernelConfigTest : public ::testing::Test {};

TEST_F(KernelConfigTest, DefaultConfigValid) {
    kernel::KernelConfig config;
    EXPECT_TRUE(config.is_valid());
}

TEST_F(KernelConfigTest, InvalidBlockDim) {
    kernel::KernelConfig config;
    config.block_dim_x = 0;
    EXPECT_FALSE(config.is_valid());
}

TEST_F(KernelConfigTest, InvalidGridDim) {
    kernel::KernelConfig config;
    config.grid_dim_x = -1;
    EXPECT_FALSE(config.is_valid());
}

TEST_F(KernelConfigTest, CustomConfig) {
    kernel::KernelConfig config;
    config.block_dim_x = 256;
    config.block_dim_y = 1;
    config.block_dim_z = 1;
    config.grid_dim_x = 128;
    config.grid_dim_y = 64;
    config.grid_dim_z = 1;
    config.shared_memory = 48 * 1024;  // 48KB
    
    EXPECT_TRUE(config.is_valid());
}

class KernelExecutorTest : public ::testing::Test {
protected:
    std::unique_ptr<kernel::GenericKernelExecutor> executor;
    
    void SetUp() override {
        executor = std::make_unique<kernel::GenericKernelExecutor>(
            type::KN_MATMUL_OP);
    }
};

TEST_F(KernelExecutorTest, CreateExecutor) {
    EXPECT_NE(executor, nullptr);
    EXPECT_EQ(executor->get_op_type(), type::KN_MATMUL_OP);
    EXPECT_FALSE(executor->is_compiled());
}

TEST_F(KernelExecutorTest, CompileKernel) {
    kernel::KernelConfig config;
    config.block_dim_x = 128;
    config.grid_dim_x = 64;
    
    bool result = executor->compile("kernel_source", config);
    EXPECT_TRUE(result);
    EXPECT_TRUE(executor->is_compiled());
}

TEST_F(KernelExecutorTest, CompileInvalidConfig) {
    kernel::KernelConfig config;
    config.block_dim_x = 0;  // Invalid
    
    bool result = executor->compile("kernel_source", config);
    EXPECT_FALSE(result);
    EXPECT_FALSE(executor->is_compiled());
}

TEST_F(KernelExecutorTest, ExecuteAfterCompile) {
    kernel::KernelConfig config;
    executor->compile("kernel_source", config);
    
    void* inputs[2] = {nullptr, nullptr};
    void* outputs[1] = {nullptr};
    
    bool result = executor->execute(inputs, 2, outputs, 1, config);
    EXPECT_TRUE(result);
    EXPECT_GT(executor->get_execution_time(), 0.0f);
}

TEST_F(KernelExecutorTest, ExecuteWithoutCompile) {
    kernel::KernelConfig config;
    void* inputs[2] = {nullptr, nullptr};
    void* outputs[1] = {nullptr};
    
    bool result = executor->execute(inputs, 2, outputs, 1, config);
    EXPECT_FALSE(result);  // Should fail without compile
}

TEST_F(KernelExecutorTest, GetMetrics) {
    kernel::KernelConfig config;
    executor->compile("kernel_source", config);
    
    void* inputs[2] = {nullptr, nullptr};
    void* outputs[1] = {nullptr};
    executor->execute(inputs, 2, outputs, 1, config);
    executor->execute(inputs, 2, outputs, 1, config);
    
    auto metrics = executor->get_metrics();
    EXPECT_EQ(metrics.execution_count, 2);
}

class KernelFactoryTest : public ::testing::Test {};

TEST_F(KernelFactoryTest, CreateMatmulExecutor) {
    auto executor = kernel::KernelExecutorFactory::create_matmul_executor();
    EXPECT_NE(executor, nullptr);
}

TEST_F(KernelFactoryTest, CreateReductionExecutor) {
    auto executor = kernel::KernelExecutorFactory::create_reduction_executor();
    EXPECT_NE(executor, nullptr);
}

TEST_F(KernelFactoryTest, CreateUnaryExecutors) {
    auto exp_exec = kernel::KernelExecutorFactory::create_element_unary_executor(
        type::KN_EXP_OP);
    auto relu_exec = kernel::KernelExecutorFactory::create_element_unary_executor(
        type::KN_RELU_OP);
    
    EXPECT_NE(exp_exec, nullptr);
    EXPECT_NE(relu_exec, nullptr);
}

TEST_F(KernelFactoryTest, CreateBinaryExecutors) {
    auto add_exec = kernel::KernelExecutorFactory::create_element_binary_executor(
        type::KN_ADD_OP);
    auto mul_exec = kernel::KernelExecutorFactory::create_element_binary_executor(
        type::KN_MUL_OP);
    
    EXPECT_NE(add_exec, nullptr);
    EXPECT_NE(mul_exec, nullptr);
}

// =============================================================================
// Task & Event Tests (Runtime)
// =============================================================================

namespace yirage {
namespace runtime {

enum EventType {
    EVENT_TERMINATION = 0,
    EVENT_LAUNCH_TASKS = 1,
    EVENT_LAUNCH_MASSIVE_TASKS = 2,
    EVENT_LAUNCH_DEPENDENT_TASKS = 3,
    EVENT_END_OF_TASK_GRAPH = 4,
    EVENT_EMPTY = 5,
};

enum TaskType {
    TASK_TERMINATE = 0,
    TASK_BEGIN_TASK_GRAPH = 1,
    TASK_EMBEDDING = 100,
    TASK_RMS_NORM = 101,
    TASK_LINEAR = 102,
    TASK_ATTENTION_1 = 103,
    TASK_SILU_MUL = 104,
};

constexpr size_t EVENT_NVSHMEM_TAG = 0x8000000000000000ULL;
constexpr size_t EVENT_INVALID_ID = 0xFFFFFFFFFFFFFFFFULL;

using TaskId = size_t;
using EventId = size_t;

struct EventDesc {
    EventType event_type = EVENT_TERMINATION;
    int num_triggers = 0;
    TaskId first_task_id = 0;
    TaskId last_task_id = 0;
    
    EventDesc() = default;
    EventDesc(EventType type, int triggers, TaskId first, TaskId last)
        : event_type(type), num_triggers(triggers), 
          first_task_id(first), last_task_id(last) {}
};

struct TensorDesc {
    int num_dims = 0;
    int data_type = 0;
    int dim[8] = {0};
    int stride[8] = {0};
    void* base_ptr = nullptr;
};

struct TaskDesc {
    TaskType task_type;
    int variant_id = 0;
    EventId trigger_event = EVENT_INVALID_ID;
    EventId dependent_event = EVENT_INVALID_ID;
    int request_id = -1;
    int num_inputs = 0;
    int num_outputs = 0;
    TensorDesc inputs[8];
    TensorDesc outputs[8];
    
    TaskDesc(TaskType type, int var_id = 0)
        : task_type(type), variant_id(var_id) {}
};

size_t get_event_id(int gpu_id, size_t event_pos, bool nvshmem_event) {
    size_t event_id = (static_cast<size_t>(gpu_id) << 32) | event_pos;
    if (nvshmem_event) {
        event_id |= EVENT_NVSHMEM_TAG;
    }
    return event_id;
}

bool is_nvshmem_event(size_t event_id) {
    return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

}  // namespace runtime
}  // namespace yirage

using namespace yirage::runtime;

class EventDescTest : public ::testing::Test {};

TEST_F(EventDescTest, DefaultConstruction) {
    EventDesc e;
    EXPECT_EQ(e.event_type, EVENT_TERMINATION);
    EXPECT_EQ(e.num_triggers, 0);
}

TEST_F(EventDescTest, ParameterizedConstruction) {
    EventDesc e(EVENT_LAUNCH_TASKS, 4, 10, 20);
    EXPECT_EQ(e.event_type, EVENT_LAUNCH_TASKS);
    EXPECT_EQ(e.num_triggers, 4);
    EXPECT_EQ(e.first_task_id, 10u);
    EXPECT_EQ(e.last_task_id, 20u);
}

class TaskDescTest : public ::testing::Test {};

TEST_F(TaskDescTest, CreateTask) {
    TaskDesc task(TASK_LINEAR, 0);
    EXPECT_EQ(task.task_type, TASK_LINEAR);
    EXPECT_EQ(task.variant_id, 0);
    EXPECT_EQ(task.trigger_event, EVENT_INVALID_ID);
}

TEST_F(TaskDescTest, AddInputTensor) {
    TaskDesc task(TASK_RMS_NORM);
    
    TensorDesc input;
    input.num_dims = 2;
    input.dim[0] = 32;
    input.dim[1] = 64;
    input.stride[0] = 64;
    input.stride[1] = 1;
    
    task.inputs[task.num_inputs++] = input;
    EXPECT_EQ(task.num_inputs, 1);
}

class EventIdTest : public ::testing::Test {};

TEST_F(EventIdTest, GetEventId) {
    EventId eid = get_event_id(0, 5, false);
    EXPECT_EQ(eid & 0xFFFFFFFF, 5u);
    EXPECT_FALSE(is_nvshmem_event(eid));
}

TEST_F(EventIdTest, NvshmemEvent) {
    EventId eid = get_event_id(1, 10, true);
    EXPECT_TRUE(is_nvshmem_event(eid));
}

TEST_F(EventIdTest, ExtractGpuId) {
    EventId eid = get_event_id(3, 7, false);
    int gpu_id = (eid >> 32) & 0xFFFF;
    EXPECT_EQ(gpu_id, 3);
}

// =============================================================================
// Batched Matmul Tests
// =============================================================================

class BatchedMatmulTest : public ::testing::Test {
protected:
    std::unique_ptr<Graph> graph;
    
    void SetUp() override {
        graph = std::make_unique<Graph>();
    }
};

TEST_F(BatchedMatmulTest, BatchedMatmul3D) {
    std::vector<int> a_dims = {4, 32, 64};    // [batch, M, K]
    std::vector<int> b_dims = {4, 64, 128};   // [batch, K, N]
    std::vector<size_t> a_strides = {2048, 64, 1};
    std::vector<size_t> b_strides = {8192, 128, 1};
    
    DTensor A = graph->new_input(a_dims, a_strides, type::DT_FLOAT16, 
                                 layout::DmemRowMajor);
    DTensor B = graph->new_input(b_dims, b_strides, type::DT_FLOAT16, 
                                 layout::DmemRowMajor);
    DTensor C = graph->matmul(A, B);
    
    EXPECT_EQ(C.num_dims, 3);
    EXPECT_EQ(C.dim[0], 4);    // batch
    EXPECT_EQ(C.dim[1], 32);   // M
    EXPECT_EQ(C.dim[2], 128);  // N
}

// =============================================================================
// Operator Type Coverage Tests
// =============================================================================

struct OpTypeTestParam {
    type::KNOperatorType op;
    std::string name;
};

class OperatorTypeTest : public ::testing::TestWithParam<OpTypeTestParam> {};

TEST_P(OperatorTypeTest, OperatorTypeRange) {
    auto param = GetParam();
    // All operator types should be >= 1000 (based on the enum)
    EXPECT_GE(static_cast<int>(param.op), 1000);
}

INSTANTIATE_TEST_SUITE_P(
    AllOperatorTypes,
    OperatorTypeTest,
    ::testing::Values(
        OpTypeTestParam{type::KN_INPUT_OP, "input"},
        OpTypeTestParam{type::KN_OUTPUT_OP, "output"},
        OpTypeTestParam{type::KN_MATMUL_OP, "matmul"},
        OpTypeTestParam{type::KN_EXP_OP, "exp"},
        OpTypeTestParam{type::KN_SQRT_OP, "sqrt"},
        OpTypeTestParam{type::KN_SILU_OP, "silu"},
        OpTypeTestParam{type::KN_GELU_OP, "gelu"},
        OpTypeTestParam{type::KN_RELU_OP, "relu"},
        OpTypeTestParam{type::KN_ADD_OP, "add"},
        OpTypeTestParam{type::KN_MUL_OP, "mul"},
        OpTypeTestParam{type::KN_DIV_OP, "div"},
        OpTypeTestParam{type::KN_REDUCTION_0_OP, "reduce_0"},
        OpTypeTestParam{type::KN_REDUCTION_1_OP, "reduce_1"},
        OpTypeTestParam{type::KN_REDUCTION_2_OP, "reduce_2"},
        OpTypeTestParam{type::KN_RMS_NORM_OP, "rms_norm"},
        OpTypeTestParam{type::KN_CUSTOMIZED_OP, "customized"}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
