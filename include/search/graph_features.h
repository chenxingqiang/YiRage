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

#pragma once

/**
 * @file graph_features.h
 * @brief Feature extraction from µGraph for RL model input.
 * 
 * This header defines the features extracted from KernelGraph and 
 * ThreadblockGraph for use in RL-guided search.
 * 
 * Feature flow:
 *   C++ µGraph → GraphFeatureExtractor → JSON → Python FeatureProcessor → RL Model
 */

#include <string>
#include <vector>
#include <array>
#include <memory>

namespace yirage {
namespace features {

/**
 * @brief Features for a single operator.
 */
struct OperatorFeatures {
    int op_id = 0;
    std::string op_type = "";
    int op_type_id = 0;
    int num_inputs = 0;
    int num_outputs = 0;
    
    // Performance characteristics
    float flops = 0.0f;
    float memory_read_bytes = 0.0f;
    float memory_write_bytes = 0.0f;
    
    // Connectivity
    std::vector<int> input_tensor_ids;
    std::vector<int> output_tensor_ids;
    
    std::string to_json() const;
};

/**
 * @brief Features for a single tensor.
 */
struct TensorFeatures {
    int tensor_id = 0;
    std::vector<int> dims;
    std::string dtype = "float16";
    int dtype_id = 0;
    size_t size_bytes = 0;
    
    // Memory level: 0=register, 1=shared, 2=global
    int memory_level = 2;
    
    bool is_input = false;
    bool is_output = false;
    
    std::string to_json() const;
};

/**
 * @brief Graph structure features.
 */
struct GraphStructureFeatures {
    int num_operators = 0;
    int num_tensors = 0;
    int graph_depth = 0;
    int graph_width = 0;
    int critical_path_length = 0;
    float parallelism_degree = 0.0f;
    
    std::string to_json() const;
};

/**
 * @brief Hardware configuration features.
 */
struct ConfigFeatures {
    std::array<int, 3> grid_dim = {1, 1, 1};
    std::array<int, 3> block_dim = {128, 1, 1};
    int forloop_range = 1;
    int reduction_dimx = 1;
    
    // Resource utilization
    float occupancy = 0.0f;
    float shared_mem_usage = 0.0f;
    float register_usage = 0.0f;
    
    std::string to_json() const;
};

/**
 * @brief Performance prediction features.
 */
struct PerformanceFeatures {
    float theoretical_flops = 0.0f;
    float memory_bandwidth_utilization = 0.0f;
    float arithmetic_intensity = 0.0f;
    float estimated_latency_ms = 0.0f;
    
    std::string to_json() const;
};

/**
 * @brief Complete µGraph features.
 * 
 * This is the main structure passed to Python RL layer.
 */
struct MuGraphFeatures {
    // Node features
    std::vector<OperatorFeatures> operators;
    std::vector<TensorFeatures> tensors;
    
    // Edge features
    std::vector<std::pair<int, int>> edges;
    
    // Aggregated features
    GraphStructureFeatures structure;
    ConfigFeatures config;
    PerformanceFeatures performance;
    
    // Search state
    int search_level = 0;
    int search_depth = 0;
    
    /**
     * @brief Serialize to JSON for Python.
     */
    std::string to_json() const;
    
    /**
     * @brief Deserialize from JSON.
     */
    static MuGraphFeatures from_json(const std::string& json);
};

/**
 * @brief Feature extractor for µGraph.
 * 
 * Extracts features from KernelGraph and ThreadblockGraph
 * for use in RL model input.
 */
class GraphFeatureExtractor {
public:
    /**
     * @brief Extract features from kernel graph.
     * 
     * @param kn_graph Kernel graph pointer
     * @param config Hardware configuration
     * @return Extracted features
     */
    static MuGraphFeatures extract_from_kernel_graph(
        const void* kn_graph,  // kernel::KNGraph*
        const ConfigFeatures& config
    );
    
    /**
     * @brief Extract features from threadblock graph.
     * 
     * @param tb_graph Threadblock graph pointer
     * @param config Hardware configuration
     * @return Extracted features
     */
    static MuGraphFeatures extract_from_tb_graph(
        const void* tb_graph,  // threadblock::TBGraph*
        const ConfigFeatures& config
    );
    
    /**
     * @brief Extract features from search context.
     * 
     * @param context Search context
     * @param config Hardware configuration
     * @return Extracted features
     */
    static MuGraphFeatures extract_from_context(
        const void* context,  // SearchContext*
        const ConfigFeatures& config
    );
    
    /**
     * @brief Compute operator type ID from name.
     */
    static int get_operator_type_id(const std::string& op_type);
    
    /**
     * @brief Estimate FLOPs for an operator.
     */
    static float estimate_operator_flops(
        const std::string& op_type,
        const std::vector<std::vector<int>>& input_shapes,
        const std::vector<int>& output_shape
    );
    
    /**
     * @brief Compute graph structure features.
     */
    static GraphStructureFeatures compute_structure_features(
        const std::vector<OperatorFeatures>& operators,
        const std::vector<std::pair<int, int>>& edges
    );
    
    /**
     * @brief Predict performance features.
     */
    static PerformanceFeatures predict_performance(
        const MuGraphFeatures& features,
        const std::string& backend
    );
};

// ============================================
// C Interface for Python bindings
// ============================================

extern "C" {

/**
 * @brief Extract features from context and return JSON.
 * 
 * @param context RLSearchContext pointer
 * @param config_json Hardware config JSON
 * @return JSON string (caller must free with free_features_string)
 */
char* extract_graph_features_json(
    void* context,
    const char* config_json
);

/**
 * @brief Free features JSON string.
 */
void free_features_string(char* str);

} // extern "C"

} // namespace features
} // namespace yirage
