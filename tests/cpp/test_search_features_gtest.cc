// Copyright 2025 Chen Xingqiang (YiRage Project)
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_search_features_gtest.cc
 * @brief Graph Features Module Unit Tests
 *
 * Tests for graph feature extraction:
 *   - OperatorFeatures
 *   - TensorFeatures
 *   - GraphStructureFeatures
 *   - ConfigFeatures
 *   - PerformanceFeatures
 *   - MuGraphFeatures
 *   - GraphFeatureExtractor
 */

#include <gtest/gtest.h>
#include <gtest/gtest-param-test.h>
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <sstream>

namespace yirage {
namespace features {

// =============================================================================
// OperatorFeatures
// =============================================================================

struct OperatorFeatures {
    int op_id = 0;
    std::string op_type = "";
    int op_type_id = 0;
    int num_inputs = 0;
    int num_outputs = 0;
    
    float flops = 0.0f;
    float memory_read_bytes = 0.0f;
    float memory_write_bytes = 0.0f;
    
    std::vector<int> input_tensor_ids;
    std::vector<int> output_tensor_ids;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << "{";
        ss << "\"op_id\":" << op_id << ",";
        ss << "\"op_type\":\"" << op_type << "\",";
        ss << "\"op_type_id\":" << op_type_id << ",";
        ss << "\"num_inputs\":" << num_inputs << ",";
        ss << "\"num_outputs\":" << num_outputs << ",";
        ss << "\"flops\":" << flops << ",";
        ss << "\"memory_read_bytes\":" << memory_read_bytes << ",";
        ss << "\"memory_write_bytes\":" << memory_write_bytes;
        ss << "}";
        return ss.str();
    }
};

// =============================================================================
// TensorFeatures
// =============================================================================

struct TensorFeatures {
    int tensor_id = 0;
    std::vector<int> dims;
    std::string dtype = "float16";
    int dtype_id = 0;
    size_t size_bytes = 0;
    int memory_level = 2;  // 0=register, 1=shared, 2=global
    bool is_input = false;
    bool is_output = false;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << "{";
        ss << "\"tensor_id\":" << tensor_id << ",";
        ss << "\"dims\":[";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i > 0) ss << ",";
            ss << dims[i];
        }
        ss << "],";
        ss << "\"dtype\":\"" << dtype << "\",";
        ss << "\"dtype_id\":" << dtype_id << ",";
        ss << "\"size_bytes\":" << size_bytes << ",";
        ss << "\"memory_level\":" << memory_level << ",";
        ss << "\"is_input\":" << (is_input ? "true" : "false") << ",";
        ss << "\"is_output\":" << (is_output ? "true" : "false");
        ss << "}";
        return ss.str();
    }
    
    size_t compute_size() const {
        if (dims.empty()) return 0;
        size_t elements = 1;
        for (int d : dims) elements *= d;
        int bytes_per_element = (dtype == "float32") ? 4 : 2;  // float16 default
        return elements * bytes_per_element;
    }
};

// =============================================================================
// GraphStructureFeatures
// =============================================================================

struct GraphStructureFeatures {
    int num_operators = 0;
    int num_tensors = 0;
    int graph_depth = 0;
    int graph_width = 0;
    int critical_path_length = 0;
    float parallelism_degree = 0.0f;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << "{";
        ss << "\"num_operators\":" << num_operators << ",";
        ss << "\"num_tensors\":" << num_tensors << ",";
        ss << "\"graph_depth\":" << graph_depth << ",";
        ss << "\"graph_width\":" << graph_width << ",";
        ss << "\"critical_path_length\":" << critical_path_length << ",";
        ss << "\"parallelism_degree\":" << parallelism_degree;
        ss << "}";
        return ss.str();
    }
};

// =============================================================================
// ConfigFeatures
// =============================================================================

struct ConfigFeatures {
    std::array<int, 3> grid_dim = {1, 1, 1};
    std::array<int, 3> block_dim = {128, 1, 1};
    int forloop_range = 1;
    int reduction_dimx = 1;
    
    float occupancy = 0.0f;
    float shared_mem_usage = 0.0f;
    float register_usage = 0.0f;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << "{";
        ss << "\"grid_dim\":[" << grid_dim[0] << "," << grid_dim[1] << "," << grid_dim[2] << "],";
        ss << "\"block_dim\":[" << block_dim[0] << "," << block_dim[1] << "," << block_dim[2] << "],";
        ss << "\"forloop_range\":" << forloop_range << ",";
        ss << "\"reduction_dimx\":" << reduction_dimx << ",";
        ss << "\"occupancy\":" << occupancy << ",";
        ss << "\"shared_mem_usage\":" << shared_mem_usage << ",";
        ss << "\"register_usage\":" << register_usage;
        ss << "}";
        return ss.str();
    }
    
    int total_threads() const {
        return block_dim[0] * block_dim[1] * block_dim[2];
    }
    
    int total_blocks() const {
        return grid_dim[0] * grid_dim[1] * grid_dim[2];
    }
};

// =============================================================================
// PerformanceFeatures
// =============================================================================

struct PerformanceFeatures {
    float theoretical_flops = 0.0f;
    float memory_bandwidth_utilization = 0.0f;
    float arithmetic_intensity = 0.0f;
    float estimated_latency_ms = 0.0f;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << "{";
        ss << "\"theoretical_flops\":" << theoretical_flops << ",";
        ss << "\"memory_bandwidth_utilization\":" << memory_bandwidth_utilization << ",";
        ss << "\"arithmetic_intensity\":" << arithmetic_intensity << ",";
        ss << "\"estimated_latency_ms\":" << estimated_latency_ms;
        ss << "}";
        return ss.str();
    }
    
    void compute_arithmetic_intensity(float total_flops, float total_bytes) {
        if (total_bytes > 0) {
            arithmetic_intensity = total_flops / total_bytes;
        }
    }
};

// =============================================================================
// MuGraphFeatures
// =============================================================================

struct MuGraphFeatures {
    std::vector<OperatorFeatures> operators;
    std::vector<TensorFeatures> tensors;
    std::vector<std::pair<int, int>> edges;
    
    GraphStructureFeatures structure;
    ConfigFeatures config;
    PerformanceFeatures performance;
    
    int search_level = 0;
    int search_depth = 0;
    
    std::string to_json() const {
        std::ostringstream ss;
        ss << "{";
        ss << "\"operators\":[";
        for (size_t i = 0; i < operators.size(); ++i) {
            if (i > 0) ss << ",";
            ss << operators[i].to_json();
        }
        ss << "],";
        ss << "\"tensors\":[";
        for (size_t i = 0; i < tensors.size(); ++i) {
            if (i > 0) ss << ",";
            ss << tensors[i].to_json();
        }
        ss << "],";
        ss << "\"structure\":" << structure.to_json() << ",";
        ss << "\"config\":" << config.to_json() << ",";
        ss << "\"performance\":" << performance.to_json() << ",";
        ss << "\"search_level\":" << search_level << ",";
        ss << "\"search_depth\":" << search_depth;
        ss << "}";
        return ss.str();
    }
    
    float total_flops() const {
        float total = 0.0f;
        for (auto const& op : operators) {
            total += op.flops;
        }
        return total;
    }
    
    float total_memory_bytes() const {
        float total = 0.0f;
        for (auto const& op : operators) {
            total += op.memory_read_bytes + op.memory_write_bytes;
        }
        return total;
    }
};

// =============================================================================
// GraphFeatureExtractor
// =============================================================================

class GraphFeatureExtractor {
public:
    static int get_operator_type_id(std::string const& op_type) {
        static std::unordered_map<std::string, int> type_map = {
            {"matmul", 0},
            {"conv2d", 1},
            {"relu", 2},
            {"silu", 3},
            {"gelu", 4},
            {"add", 5},
            {"mul", 6},
            {"div", 7},
            {"reduction", 8},
            {"rms_norm", 9},
            {"softmax", 10},
            {"attention", 11},
        };
        
        auto it = type_map.find(op_type);
        return (it != type_map.end()) ? it->second : -1;
    }
    
    static float estimate_operator_flops(
            std::string const& op_type,
            std::vector<std::vector<int>> const& input_shapes,
            std::vector<int> const& output_shape) {
        
        if (op_type == "matmul" && input_shapes.size() >= 2) {
            // C[M,N] = A[M,K] * B[K,N] => 2*M*N*K FLOPs
            int m = input_shapes[0][0];
            int k = input_shapes[0][1];
            int n = input_shapes[1][1];
            return 2.0f * m * n * k;
        }
        
        if (op_type == "relu" || op_type == "silu" || op_type == "gelu") {
            // Element-wise: N FLOPs
            float elements = 1.0f;
            for (int d : output_shape) elements *= d;
            return elements;
        }
        
        if (op_type == "add" || op_type == "mul" || op_type == "div") {
            float elements = 1.0f;
            for (int d : output_shape) elements *= d;
            return elements;
        }
        
        if (op_type == "reduction") {
            // Sum reduction: N FLOPs
            float elements = 1.0f;
            if (!input_shapes.empty()) {
                for (int d : input_shapes[0]) elements *= d;
            }
            return elements;
        }
        
        return 0.0f;
    }
    
    static GraphStructureFeatures compute_structure_features(
            std::vector<OperatorFeatures> const& operators,
            std::vector<std::pair<int, int>> const& edges) {
        
        GraphStructureFeatures features;
        features.num_operators = static_cast<int>(operators.size());
        features.num_tensors = 0;
        
        // Compute depth using BFS
        if (!operators.empty()) {
            // Simplified: assume linear depth
            features.graph_depth = static_cast<int>(operators.size());
            features.graph_width = 1;
            features.critical_path_length = features.graph_depth;
            features.parallelism_degree = 1.0f;
        }
        
        return features;
    }
    
    static PerformanceFeatures predict_performance(
            MuGraphFeatures const& features,
            std::string const& backend) {
        
        PerformanceFeatures perf;
        perf.theoretical_flops = features.total_flops();
        
        float total_bytes = features.total_memory_bytes();
        perf.compute_arithmetic_intensity(perf.theoretical_flops, total_bytes);
        
        // Estimate latency based on backend
        float peak_tflops = 100.0f;  // Default
        if (backend == "cuda") peak_tflops = 312.0f;  // A100
        else if (backend == "rocm") peak_tflops = 95.7f;  // MI250
        else if (backend == "mps") peak_tflops = 10.0f;   // M2
        
        perf.estimated_latency_ms = perf.theoretical_flops / (peak_tflops * 1e9f);
        
        return perf;
    }
};

}  // namespace features
}  // namespace yirage

using namespace yirage::features;

// =============================================================================
// OperatorFeatures Tests
// =============================================================================

class OperatorFeaturesTest : public ::testing::Test {};

TEST_F(OperatorFeaturesTest, DefaultConstruction) {
    OperatorFeatures op;
    EXPECT_EQ(op.op_id, 0);
    EXPECT_TRUE(op.op_type.empty());
    EXPECT_EQ(op.num_inputs, 0);
}

TEST_F(OperatorFeaturesTest, SetProperties) {
    OperatorFeatures op;
    op.op_id = 1;
    op.op_type = "matmul";
    op.op_type_id = 0;
    op.num_inputs = 2;
    op.num_outputs = 1;
    op.flops = 1e9f;
    
    EXPECT_EQ(op.op_type, "matmul");
    EXPECT_EQ(op.num_inputs, 2);
    EXPECT_FLOAT_EQ(op.flops, 1e9f);
}

TEST_F(OperatorFeaturesTest, ToJson) {
    OperatorFeatures op;
    op.op_id = 0;
    op.op_type = "relu";
    
    std::string json = op.to_json();
    EXPECT_NE(json.find("\"op_id\":0"), std::string::npos);
    EXPECT_NE(json.find("\"op_type\":\"relu\""), std::string::npos);
}

TEST_F(OperatorFeaturesTest, InputOutputTensorIds) {
    OperatorFeatures op;
    op.input_tensor_ids = {0, 1};
    op.output_tensor_ids = {2};
    
    EXPECT_EQ(op.input_tensor_ids.size(), 2u);
    EXPECT_EQ(op.output_tensor_ids.size(), 1u);
}

// =============================================================================
// TensorFeatures Tests
// =============================================================================

class TensorFeaturesTest : public ::testing::Test {};

TEST_F(TensorFeaturesTest, DefaultConstruction) {
    TensorFeatures tensor;
    EXPECT_EQ(tensor.tensor_id, 0);
    EXPECT_EQ(tensor.dtype, "float16");
    EXPECT_EQ(tensor.memory_level, 2);
}

TEST_F(TensorFeaturesTest, SetDimensions) {
    TensorFeatures tensor;
    tensor.dims = {128, 256};
    
    EXPECT_EQ(tensor.dims.size(), 2u);
    EXPECT_EQ(tensor.dims[0], 128);
    EXPECT_EQ(tensor.dims[1], 256);
}

TEST_F(TensorFeaturesTest, ComputeSizeFloat16) {
    TensorFeatures tensor;
    tensor.dims = {128, 256};
    tensor.dtype = "float16";
    
    size_t size = tensor.compute_size();
    EXPECT_EQ(size, 128u * 256u * 2u);  // 2 bytes per float16
}

TEST_F(TensorFeaturesTest, ComputeSizeFloat32) {
    TensorFeatures tensor;
    tensor.dims = {128, 256};
    tensor.dtype = "float32";
    
    size_t size = tensor.compute_size();
    EXPECT_EQ(size, 128u * 256u * 4u);  // 4 bytes per float32
}

TEST_F(TensorFeaturesTest, ToJson) {
    TensorFeatures tensor;
    tensor.tensor_id = 0;
    tensor.dims = {64, 128};
    tensor.is_input = true;
    
    std::string json = tensor.to_json();
    EXPECT_NE(json.find("\"tensor_id\":0"), std::string::npos);
    EXPECT_NE(json.find("\"is_input\":true"), std::string::npos);
}

// =============================================================================
// GraphStructureFeatures Tests
// =============================================================================

class GraphStructureFeaturesTest : public ::testing::Test {};

TEST_F(GraphStructureFeaturesTest, DefaultConstruction) {
    GraphStructureFeatures features;
    EXPECT_EQ(features.num_operators, 0);
    EXPECT_EQ(features.graph_depth, 0);
}

TEST_F(GraphStructureFeaturesTest, ToJson) {
    GraphStructureFeatures features;
    features.num_operators = 5;
    features.num_tensors = 10;
    features.graph_depth = 3;
    
    std::string json = features.to_json();
    EXPECT_NE(json.find("\"num_operators\":5"), std::string::npos);
    EXPECT_NE(json.find("\"num_tensors\":10"), std::string::npos);
}

// =============================================================================
// ConfigFeatures Tests
// =============================================================================

class ConfigFeaturesTest : public ::testing::Test {};

TEST_F(ConfigFeaturesTest, DefaultConstruction) {
    ConfigFeatures config;
    EXPECT_EQ(config.grid_dim[0], 1);
    EXPECT_EQ(config.block_dim[0], 128);
    EXPECT_EQ(config.forloop_range, 1);
}

TEST_F(ConfigFeaturesTest, TotalThreads) {
    ConfigFeatures config;
    config.block_dim = {256, 1, 1};
    
    EXPECT_EQ(config.total_threads(), 256);
    
    config.block_dim = {32, 8, 1};
    EXPECT_EQ(config.total_threads(), 256);
}

TEST_F(ConfigFeaturesTest, TotalBlocks) {
    ConfigFeatures config;
    config.grid_dim = {64, 64, 1};
    
    EXPECT_EQ(config.total_blocks(), 4096);
}

TEST_F(ConfigFeaturesTest, ToJson) {
    ConfigFeatures config;
    config.occupancy = 0.75f;
    
    std::string json = config.to_json();
    EXPECT_NE(json.find("\"grid_dim\""), std::string::npos);
    EXPECT_NE(json.find("\"occupancy\""), std::string::npos);
}

// =============================================================================
// PerformanceFeatures Tests
// =============================================================================

class PerformanceFeaturesTest : public ::testing::Test {};

TEST_F(PerformanceFeaturesTest, DefaultConstruction) {
    PerformanceFeatures perf;
    EXPECT_FLOAT_EQ(perf.theoretical_flops, 0.0f);
    EXPECT_FLOAT_EQ(perf.arithmetic_intensity, 0.0f);
}

TEST_F(PerformanceFeaturesTest, ComputeArithmeticIntensity) {
    PerformanceFeatures perf;
    perf.compute_arithmetic_intensity(1e12f, 1e9f);  // 1 TFLOP, 1 GB
    
    EXPECT_FLOAT_EQ(perf.arithmetic_intensity, 1000.0f);  // 1000 FLOP/byte
}

TEST_F(PerformanceFeaturesTest, ToJson) {
    PerformanceFeatures perf;
    perf.estimated_latency_ms = 1.5f;
    
    std::string json = perf.to_json();
    EXPECT_NE(json.find("\"estimated_latency_ms\""), std::string::npos);
}

// =============================================================================
// MuGraphFeatures Tests
// =============================================================================

class MuGraphFeaturesTest : public ::testing::Test {};

TEST_F(MuGraphFeaturesTest, DefaultConstruction) {
    MuGraphFeatures features;
    EXPECT_TRUE(features.operators.empty());
    EXPECT_TRUE(features.tensors.empty());
    EXPECT_EQ(features.search_level, 0);
}

TEST_F(MuGraphFeaturesTest, TotalFlops) {
    MuGraphFeatures features;
    
    OperatorFeatures op1;
    op1.flops = 1e9f;
    
    OperatorFeatures op2;
    op2.flops = 2e9f;
    
    features.operators.push_back(op1);
    features.operators.push_back(op2);
    
    EXPECT_FLOAT_EQ(features.total_flops(), 3e9f);
}

TEST_F(MuGraphFeaturesTest, TotalMemoryBytes) {
    MuGraphFeatures features;
    
    OperatorFeatures op;
    op.memory_read_bytes = 1e6f;
    op.memory_write_bytes = 0.5e6f;
    
    features.operators.push_back(op);
    
    EXPECT_FLOAT_EQ(features.total_memory_bytes(), 1.5e6f);
}

TEST_F(MuGraphFeaturesTest, ToJson) {
    MuGraphFeatures features;
    features.search_level = 1;
    features.search_depth = 5;
    
    std::string json = features.to_json();
    EXPECT_NE(json.find("\"search_level\":1"), std::string::npos);
    EXPECT_NE(json.find("\"search_depth\":5"), std::string::npos);
}

// =============================================================================
// GraphFeatureExtractor Tests
// =============================================================================

class GraphFeatureExtractorTest : public ::testing::Test {};

TEST_F(GraphFeatureExtractorTest, GetOperatorTypeId) {
    EXPECT_EQ(GraphFeatureExtractor::get_operator_type_id("matmul"), 0);
    EXPECT_EQ(GraphFeatureExtractor::get_operator_type_id("relu"), 2);
    EXPECT_EQ(GraphFeatureExtractor::get_operator_type_id("softmax"), 10);
    EXPECT_EQ(GraphFeatureExtractor::get_operator_type_id("unknown"), -1);
}

TEST_F(GraphFeatureExtractorTest, EstimateMatmulFlops) {
    std::vector<std::vector<int>> inputs = {{128, 256}, {256, 512}};
    std::vector<int> output = {128, 512};
    
    float flops = GraphFeatureExtractor::estimate_operator_flops("matmul", inputs, output);
    
    // 2 * M * N * K = 2 * 128 * 512 * 256
    EXPECT_FLOAT_EQ(flops, 2.0f * 128 * 512 * 256);
}

TEST_F(GraphFeatureExtractorTest, EstimateReluFlops) {
    std::vector<std::vector<int>> inputs = {{128, 256}};
    std::vector<int> output = {128, 256};
    
    float flops = GraphFeatureExtractor::estimate_operator_flops("relu", inputs, output);
    
    EXPECT_FLOAT_EQ(flops, 128.0f * 256);
}

TEST_F(GraphFeatureExtractorTest, ComputeStructureFeatures) {
    std::vector<OperatorFeatures> ops(5);
    std::vector<std::pair<int, int>> edges;
    
    auto features = GraphFeatureExtractor::compute_structure_features(ops, edges);
    
    EXPECT_EQ(features.num_operators, 5);
    EXPECT_GT(features.graph_depth, 0);
}

TEST_F(GraphFeatureExtractorTest, PredictPerformance) {
    MuGraphFeatures features;
    OperatorFeatures op;
    op.flops = 1e12f;  // 1 TFLOP
    op.memory_read_bytes = 1e9f;
    op.memory_write_bytes = 1e9f;
    features.operators.push_back(op);
    
    auto perf = GraphFeatureExtractor::predict_performance(features, "cuda");
    
    EXPECT_FLOAT_EQ(perf.theoretical_flops, 1e12f);
    EXPECT_GT(perf.arithmetic_intensity, 0.0f);
}

// =============================================================================
// Parameterized Operator Type Tests
// =============================================================================

struct OpTypeTestParam {
    std::string op_type;
    int expected_id;
};

class OpTypeParameterizedTest : public ::testing::TestWithParam<OpTypeTestParam> {};

TEST_P(OpTypeParameterizedTest, OperatorTypeId) {
    auto param = GetParam();
    int id = GraphFeatureExtractor::get_operator_type_id(param.op_type);
    EXPECT_EQ(id, param.expected_id);
}

INSTANTIATE_TEST_SUITE_P(
    AllOperatorTypes,
    OpTypeParameterizedTest,
    ::testing::Values(
        OpTypeTestParam{"matmul", 0},
        OpTypeTestParam{"conv2d", 1},
        OpTypeTestParam{"relu", 2},
        OpTypeTestParam{"silu", 3},
        OpTypeTestParam{"gelu", 4},
        OpTypeTestParam{"add", 5},
        OpTypeTestParam{"mul", 6},
        OpTypeTestParam{"div", 7},
        OpTypeTestParam{"reduction", 8},
        OpTypeTestParam{"rms_norm", 9},
        OpTypeTestParam{"softmax", 10},
        OpTypeTestParam{"attention", 11},
        OpTypeTestParam{"invalid", -1}
    )
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
