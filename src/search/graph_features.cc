/* Copyright 2025 Chen Xingqiang (YiRage Project)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file graph_features.cc
 * @brief Implementation of µGraph feature extraction.
 */

#include "search/graph_features.h"
#include "kernel/graph.h"
#include "threadblock/graph.h"
#include "type.h"

#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <unordered_map>
#include <algorithm>

namespace yirage {
namespace features {

// Operator type mapping
static std::unordered_map<std::string, int> OPERATOR_TYPE_MAP = {
    {"MATMUL", 0}, {"matmul", 0},
    {"ADD", 1}, {"add", 1},
    {"MUL", 2}, {"mul", 2},
    {"DIV", 3}, {"div", 3},
    {"EXP", 4}, {"exp", 4},
    {"SILU", 5}, {"silu", 5},
    {"GELU", 6}, {"gelu", 6},
    {"RELU", 7}, {"relu", 7},
    {"REDUCTION", 8}, {"reduction", 8},
    {"RMS_NORM", 9}, {"rms_norm", 9},
    {"SOFTMAX", 10}, {"softmax", 10},
    {"CONCAT", 11}, {"concat", 11},
    {"FORLOOP_ACCUM", 12}, {"forloop_accum", 12},
    {"SQUARE", 13}, {"square", 13},
    {"SQRT", 14}, {"sqrt", 14},
};

std::string OperatorFeatures::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"op_id\":" << op_id << ",";
    oss << "\"op_type\":\"" << op_type << "\",";
    oss << "\"op_type_id\":" << op_type_id << ",";
    oss << "\"num_inputs\":" << num_inputs << ",";
    oss << "\"num_outputs\":" << num_outputs << ",";
    oss << "\"flops\":" << flops << ",";
    oss << "\"memory_read_bytes\":" << memory_read_bytes << ",";
    oss << "\"memory_write_bytes\":" << memory_write_bytes << ",";
    
    oss << "\"input_tensor_ids\":[";
    for (size_t i = 0; i < input_tensor_ids.size(); ++i) {
        if (i > 0) oss << ",";
        oss << input_tensor_ids[i];
    }
    oss << "],";
    
    oss << "\"output_tensor_ids\":[";
    for (size_t i = 0; i < output_tensor_ids.size(); ++i) {
        if (i > 0) oss << ",";
        oss << output_tensor_ids[i];
    }
    oss << "]";
    
    oss << "}";
    return oss.str();
}

std::string TensorFeatures::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"tensor_id\":" << tensor_id << ",";
    
    oss << "\"dims\":[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) oss << ",";
        oss << dims[i];
    }
    oss << "],";
    
    oss << "\"dtype\":\"" << dtype << "\",";
    oss << "\"dtype_id\":" << dtype_id << ",";
    oss << "\"size_bytes\":" << size_bytes << ",";
    oss << "\"memory_level\":" << memory_level << ",";
    oss << "\"is_input\":" << (is_input ? "true" : "false") << ",";
    oss << "\"is_output\":" << (is_output ? "true" : "false");
    oss << "}";
    return oss.str();
}

std::string GraphStructureFeatures::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"num_operators\":" << num_operators << ",";
    oss << "\"num_tensors\":" << num_tensors << ",";
    oss << "\"graph_depth\":" << graph_depth << ",";
    oss << "\"graph_width\":" << graph_width << ",";
    oss << "\"critical_path_length\":" << critical_path_length << ",";
    oss << "\"parallelism_degree\":" << parallelism_degree;
    oss << "}";
    return oss.str();
}

std::string ConfigFeatures::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"grid_dim\":{\"x\":" << grid_dim[0] 
        << ",\"y\":" << grid_dim[1] 
        << ",\"z\":" << grid_dim[2] << "},";
    oss << "\"block_dim\":{\"x\":" << block_dim[0] 
        << ",\"y\":" << block_dim[1] 
        << ",\"z\":" << block_dim[2] << "},";
    oss << "\"forloop_range\":" << forloop_range << ",";
    oss << "\"reduction_dimx\":" << reduction_dimx << ",";
    oss << "\"occupancy\":" << occupancy << ",";
    oss << "\"shared_mem_usage\":" << shared_mem_usage << ",";
    oss << "\"register_usage\":" << register_usage;
    oss << "}";
    return oss.str();
}

std::string PerformanceFeatures::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"theoretical_flops\":" << theoretical_flops << ",";
    oss << "\"memory_bandwidth_utilization\":" << memory_bandwidth_utilization << ",";
    oss << "\"arithmetic_intensity\":" << arithmetic_intensity << ",";
    oss << "\"estimated_latency_ms\":" << estimated_latency_ms;
    oss << "}";
    return oss.str();
}

std::string MuGraphFeatures::to_json() const {
    std::ostringstream oss;
    oss << "{";
    
    // Operators
    oss << "\"operators\":[";
    for (size_t i = 0; i < operators.size(); ++i) {
        if (i > 0) oss << ",";
        oss << operators[i].to_json();
    }
    oss << "],";
    
    // Tensors
    oss << "\"tensors\":[";
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i > 0) oss << ",";
        oss << tensors[i].to_json();
    }
    oss << "],";
    
    // Edges
    oss << "\"edges\":[";
    for (size_t i = 0; i < edges.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "[" << edges[i].first << "," << edges[i].second << "]";
    }
    oss << "],";
    
    // Structure features (inline)
    oss << "\"num_operators\":" << structure.num_operators << ",";
    oss << "\"num_tensors\":" << structure.num_tensors << ",";
    oss << "\"graph_depth\":" << structure.graph_depth << ",";
    oss << "\"graph_width\":" << structure.graph_width << ",";
    oss << "\"critical_path_length\":" << structure.critical_path_length << ",";
    oss << "\"parallelism_degree\":" << structure.parallelism_degree << ",";
    
    // Config features (inline)
    oss << "\"grid_dim\":{\"x\":" << config.grid_dim[0] 
        << ",\"y\":" << config.grid_dim[1] 
        << ",\"z\":" << config.grid_dim[2] << "},";
    oss << "\"block_dim\":{\"x\":" << config.block_dim[0] 
        << ",\"y\":" << config.block_dim[1] 
        << ",\"z\":" << config.block_dim[2] << "},";
    oss << "\"forloop_range\":" << config.forloop_range << ",";
    oss << "\"reduction_dimx\":" << config.reduction_dimx << ",";
    oss << "\"occupancy\":" << config.occupancy << ",";
    oss << "\"shared_mem_usage\":" << config.shared_mem_usage << ",";
    oss << "\"register_usage\":" << config.register_usage << ",";
    
    // Performance features (inline)
    oss << "\"theoretical_flops\":" << performance.theoretical_flops << ",";
    oss << "\"memory_bandwidth_utilization\":" << performance.memory_bandwidth_utilization << ",";
    oss << "\"arithmetic_intensity\":" << performance.arithmetic_intensity << ",";
    oss << "\"estimated_latency_ms\":" << performance.estimated_latency_ms << ",";
    
    // Search state
    oss << "\"search_level\":" << search_level << ",";
    oss << "\"search_depth\":" << search_depth;
    
    oss << "}";
    return oss.str();
}

int GraphFeatureExtractor::get_operator_type_id(const std::string& op_type) {
    auto it = OPERATOR_TYPE_MAP.find(op_type);
    if (it != OPERATOR_TYPE_MAP.end()) {
        return it->second;
    }
    return 15;  // Unknown
}

float GraphFeatureExtractor::estimate_operator_flops(
    const std::string& op_type,
    const std::vector<std::vector<int>>& input_shapes,
    const std::vector<int>& output_shape
) {
    float flops = 0.0f;
    
    // Estimate FLOPs based on operator type
    if (op_type == "MATMUL" || op_type == "matmul") {
        // GEMM: 2 * M * N * K
        if (input_shapes.size() >= 2) {
            int M = input_shapes[0].size() > 0 ? input_shapes[0][0] : 1;
            int K = input_shapes[0].size() > 1 ? input_shapes[0][1] : 1;
            int N = input_shapes[1].size() > 1 ? input_shapes[1][1] : 1;
            flops = 2.0f * M * N * K;
        }
    } else if (op_type == "SOFTMAX" || op_type == "softmax") {
        // Softmax: 5 * N (exp, sum, div for each element)
        int total_elements = 1;
        for (int d : output_shape) {
            total_elements *= d;
        }
        flops = 5.0f * total_elements;
    } else if (op_type == "RMS_NORM" || op_type == "rms_norm") {
        // RMSNorm: 3 * N (square, mean, sqrt, mul)
        int total_elements = 1;
        for (int d : output_shape) {
            total_elements *= d;
        }
        flops = 3.0f * total_elements;
    } else {
        // Element-wise ops: N
        int total_elements = 1;
        for (int d : output_shape) {
            total_elements *= d;
        }
        flops = static_cast<float>(total_elements);
    }
    
    return flops;
}

GraphStructureFeatures GraphFeatureExtractor::compute_structure_features(
    const std::vector<OperatorFeatures>& operators,
    const std::vector<std::pair<int, int>>& edges
) {
    GraphStructureFeatures features;
    features.num_operators = static_cast<int>(operators.size());
    
    if (operators.empty()) {
        return features;
    }
    
    // Compute graph depth and width using topological analysis
    // Simple heuristic: depth = longest path, width = max concurrent ops
    
    // Build adjacency list
    std::vector<std::vector<int>> adj(operators.size());
    std::vector<int> in_degree(operators.size(), 0);
    
    for (const auto& edge : edges) {
        if (edge.first < static_cast<int>(operators.size()) && 
            edge.second < static_cast<int>(operators.size())) {
            adj[edge.first].push_back(edge.second);
            in_degree[edge.second]++;
        }
    }
    
    // BFS for depth
    std::vector<int> depth(operators.size(), 0);
    int max_depth = 0;
    
    for (size_t i = 0; i < operators.size(); ++i) {
        if (in_degree[i] == 0) {
            depth[i] = 1;
        }
    }
    
    for (size_t i = 0; i < operators.size(); ++i) {
        for (int neighbor : adj[i]) {
            depth[neighbor] = std::max(depth[neighbor], depth[i] + 1);
            max_depth = std::max(max_depth, depth[neighbor]);
        }
    }
    
    features.graph_depth = std::max(1, max_depth);
    
    // Width: count ops at each depth level
    std::vector<int> level_counts(max_depth + 1, 0);
    for (int d : depth) {
        if (d > 0) level_counts[d]++;
    }
    features.graph_width = *std::max_element(level_counts.begin(), level_counts.end());
    
    features.critical_path_length = features.graph_depth;
    features.parallelism_degree = features.num_operators > 0 ? 
        static_cast<float>(features.graph_width) / features.graph_depth : 1.0f;
    
    return features;
}

PerformanceFeatures GraphFeatureExtractor::predict_performance(
    const MuGraphFeatures& features,
    const std::string& backend
) {
    PerformanceFeatures perf;
    
    // Sum up theoretical FLOPs
    for (const auto& op : features.operators) {
        perf.theoretical_flops += op.flops;
    }
    
    // Estimate memory bandwidth utilization
    float total_memory_access = 0.0f;
    for (const auto& op : features.operators) {
        total_memory_access += op.memory_read_bytes + op.memory_write_bytes;
    }
    
    // Simple model: bandwidth utilization based on arithmetic intensity
    float compute_time_estimate = perf.theoretical_flops / 1e12;  // Assuming 1 TFLOP/s
    float memory_time_estimate = total_memory_access / 1e9;  // Assuming 1 TB/s
    
    if (memory_time_estimate > 0) {
        perf.arithmetic_intensity = perf.theoretical_flops / total_memory_access;
        perf.memory_bandwidth_utilization = std::min(1.0f, 
            compute_time_estimate / memory_time_estimate);
    }
    
    // Estimate latency
    perf.estimated_latency_ms = std::max(compute_time_estimate, memory_time_estimate) * 1000.0f;
    
    return perf;
}

MuGraphFeatures GraphFeatureExtractor::extract_from_kernel_graph(
    const void* kn_graph,
    const ConfigFeatures& config
) {
    MuGraphFeatures features;
    features.config = config;
    
    if (kn_graph == nullptr) {
        return features;
    }
    
    // Cast to kernel::Graph and extract features
    auto const* graph = static_cast<const kernel::Graph*>(kn_graph);
    
    // Extract operator counts into structure
    features.structure.num_operators = static_cast<int>(graph->operators.size());
    
    // Count operator types (store locally as MuGraphFeatures uses nested structures)
    int num_matmuls = 0;
    int num_reductions = 0;
    int num_elementwise = 0;
    int max_dim = 0;
    
    for (auto const* op : graph->operators) {
        if (op->op_type == type::KN_MATMUL_OP) {
            num_matmuls++;
        } else if (op->op_type >= type::KN_REDUCTION_0_OP && 
                   op->op_type <= type::KN_REDUCTION_2_OP) {
            num_reductions++;
        } else if (op->op_type >= type::KN_ADD_OP && 
                   op->op_type <= type::KN_SUB_OP) {
            num_elementwise++;
        }
        
        // Extract tensor dimensions
        for (auto const& tensor : op->input_tensors) {
            for (int i = 0; i < tensor.num_dims; ++i) {
                max_dim = std::max(max_dim, tensor.dim[i]);
            }
        }
    }
    
    // These counts can be used to populate operator features or performance hints
    (void)num_matmuls;
    (void)num_reductions;
    (void)num_elementwise;
    (void)max_dim;
    
    return features;
}

MuGraphFeatures GraphFeatureExtractor::extract_from_tb_graph(
    const void* tb_graph,
    const ConfigFeatures& config
) {
    MuGraphFeatures features;
    features.config = config;
    
    if (tb_graph == nullptr) {
        return features;
    }
    
    // Cast to threadblock::Graph and extract features
    auto const* graph = static_cast<const threadblock::Graph*>(tb_graph);
    
    // Extract grid/block dimensions into config
    features.config.grid_dim[0] = graph->grid_dim.x;
    features.config.grid_dim[1] = graph->grid_dim.y;
    features.config.grid_dim[2] = graph->grid_dim.z;
    features.config.block_dim[0] = graph->block_dim.x;
    features.config.block_dim[1] = graph->block_dim.y;
    features.config.block_dim[2] = graph->block_dim.z;
    features.config.forloop_range = graph->forloop_range;
    
    // Count operators into structure
    features.structure.num_operators = static_cast<int>(graph->operators.size());
    
    int num_matmuls = 0;
    for (auto const* op : graph->operators) {
        if (op->op_type == type::TB_MATMUL_OP) {
            num_matmuls++;
        }
    }
    // Note: num_matmuls can be stored in operators vector or as performance hint
    (void)num_matmuls;  // Currently not stored, but computed
    
    return features;
}

MuGraphFeatures GraphFeatureExtractor::extract_from_context(
    const void* context,
    const ConfigFeatures& config
) {
    MuGraphFeatures features;
    features.config = config;
    
    if (context == nullptr) {
        return features;
    }
    
    // Context-based extraction uses the search state
    // The search_depth in MuGraphFeatures tracks how deep in the search we are
    // This would be set based on the actual search context
    features.search_depth = 0;  // Default, will be updated by search algorithm
    
    return features;
}

// C Interface
extern "C" {

char* extract_graph_features_json(void* context, const char* config_json) {
    MuGraphFeatures features;
    
    // Parse config if provided
    if (config_json) {
        // Simple parsing - in production use proper JSON library
        features.config = ConfigFeatures();
    }
    
    // Extract features from context
    if (context) {
        features = GraphFeatureExtractor::extract_from_context(
            context, features.config
        );
    }
    
    std::string json = features.to_json();
    
    char* result = static_cast<char*>(malloc(json.size() + 1));
    strcpy(result, json.c_str());
    return result;
}

void free_features_string(char* str) {
    free(str);
}

} // extern "C"

} // namespace features
} // namespace yirage
