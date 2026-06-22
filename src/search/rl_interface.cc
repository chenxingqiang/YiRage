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
 * @file rl_interface.cc
 * @brief Implementation of RL interface for closed-loop search.
 * 
 * This file implements the bridge between Python RL environment
 * and C++ search core, enabling the closed loop:
 * 
 *   RL action → C++ search → GPU verify → reward → RL update
 */

#include "search/rl_interface.h"
// #include "search/search.h"
// #include "search/search_context.h"
// #include "kernel/kn_graph.h"
// #include "threadblock/tb_graph.h"
// #include "search/verification/probabilistic_verifier.h"

#include <chrono>
#include <cstring>
#include <cstdlib>
#include <sstream>

namespace yirage {
namespace rl_interface {

// JSON serialization helpers
std::string RLConfig::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"grid_dim\":{\"x\":" << grid_dim_x 
        << ",\"y\":" << grid_dim_y 
        << ",\"z\":" << grid_dim_z << "},";
    oss << "\"block_dim\":{\"x\":" << block_dim_x 
        << ",\"y\":" << block_dim_y 
        << ",\"z\":" << block_dim_z << "},";
    oss << "\"frange\":" << frange << ",";
    oss << "\"operator_type\":\"" << operator_type << "\",";
    oss << "\"imaps\":[";
    for (size_t i = 0; i < imaps.size(); ++i) {
        if (i > 0) oss << ",";
        oss << "{\"x\":" << imaps[i][0] 
            << ",\"y\":" << imaps[i][1] 
            << ",\"z\":" << imaps[i][2] << "}";
    }
    oss << "],";
    oss << "\"omap\":{\"x\":" << omap[0] 
        << ",\"y\":" << omap[1] 
        << ",\"z\":" << omap[2] << "},";
    oss << "\"input_indices\":[";
    for (size_t i = 0; i < input_indices.size(); ++i) {
        if (i > 0) oss << ",";
        oss << input_indices[i];
    }
    oss << "]";
    oss << "}";
    return oss.str();
}

RLConfig RLConfig::from_json(const std::string& json_str) {
    RLConfig config;
    
    // Simple JSON parsing for key-value pairs
    // Format: {"key": value, ...}
    auto extract_int = [&json_str](const std::string& key) -> int {
        std::string search = "\"" + key + "\":";
        size_t pos = json_str.find(search);
        if (pos == std::string::npos) return 0;
        pos += search.length();
        // Skip whitespace
        while (pos < json_str.length() && std::isspace(json_str[pos])) pos++;
        // Parse integer
        int value = 0;
        bool negative = false;
        if (pos < json_str.length() && json_str[pos] == '-') {
            negative = true;
            pos++;
        }
        while (pos < json_str.length() && std::isdigit(json_str[pos])) {
            value = value * 10 + (json_str[pos] - '0');
            pos++;
        }
        return negative ? -value : value;
    };
    
    auto extract_string = [&json_str](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\":\"";
        size_t pos = json_str.find(search);
        if (pos == std::string::npos) return "";
        pos += search.length();
        size_t end = json_str.find('"', pos);
        if (end == std::string::npos) return "";
        return json_str.substr(pos, end - pos);
    };
    
    config.grid_dim_x = extract_int("grid_dim_x");
    config.grid_dim_y = extract_int("grid_dim_y");
    config.grid_dim_z = extract_int("grid_dim_z");
    config.block_dim_x = extract_int("block_dim_x");
    config.block_dim_y = extract_int("block_dim_y");
    config.block_dim_z = extract_int("block_dim_z");
    config.operator_type = extract_string("operator_type");
    
    // Set defaults if not specified
    if (config.grid_dim_x <= 0) config.grid_dim_x = 1;
    if (config.grid_dim_y <= 0) config.grid_dim_y = 1;
    if (config.grid_dim_z <= 0) config.grid_dim_z = 1;
    if (config.block_dim_x <= 0) config.block_dim_x = 128;
    if (config.block_dim_y <= 0) config.block_dim_y = 1;
    if (config.block_dim_z <= 0) config.block_dim_z = 1;
    
    return config;
}

std::string VerifyResult::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"verified\":" << (verified ? "true" : "false") << ",";
    oss << "\"fingerprint_time_ms\":" << fingerprint_time_ms << ",";
    oss << "\"rejection_reason\":\"" << rejection_reason << "\",";
    oss << "\"kernel_hash\":" << kernel_hash;
    oss << "}";
    return oss.str();
}

std::string ProfileResult::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"latency_ms\":" << latency_ms << ",";
    oss << "\"memory_bytes\":" << memory_bytes << ",";
    oss << "\"gflops\":" << gflops << ",";
    oss << "\"compile_time_ms\":" << compile_time_ms;
    oss << "}";
    return oss.str();
}

std::string SearchState::to_json() const {
    std::ostringstream oss;
    oss << "{";
    oss << "\"search_level\":" << search_level << ",";
    oss << "\"search_depth\":" << search_depth << ",";
    oss << "\"num_kn_operators\":" << num_kn_operators << ",";
    oss << "\"num_tb_operators\":" << num_tb_operators << ",";
    oss << "\"num_tensors\":" << num_tensors << ",";
    oss << "\"num_valid_found\":" << num_valid_found << ",";
    oss << "\"num_verified\":" << num_verified << ",";
    oss << "\"best_latency_ms\":" << best_latency_ms << ",";
    oss << "\"current_grid_dim\":{\"x\":" << current_grid_dim[0] 
        << ",\"y\":" << current_grid_dim[1] 
        << ",\"z\":" << current_grid_dim[2] << "},";
    oss << "\"current_block_dim\":{\"x\":" << current_block_dim[0] 
        << ",\"y\":" << current_block_dim[1] 
        << ",\"z\":" << current_block_dim[2] << "}";
    oss << "}";
    return oss.str();
}

// RLSearchContext implementation
class RLSearchContext::Impl {
public:
    Impl(const std::string& target_json, const std::string& backend, int gpu_id)
        : target_graph_json_(target_json)
        , backend_(backend)
        , gpu_id_(gpu_id)
    {
        reset("");
    }
    
    void reset(const std::string& new_target_json) {
        if (!new_target_json.empty()) {
            target_graph_json_ = new_target_json;
        }
        
        // Reset state
        state_ = SearchState{};
        kernel_graph_json_ = "{}";
        valid_kernels_.clear();
        done_ = false;
        
        // Parse target graph to get initial tensor info
        // Extract tensor count from JSON (simple parsing)
        state_.num_tensors = count_tensors_in_json(target_graph_json_);
        if (state_.num_tensors == 0) {
            state_.num_tensors = 2;  // Default for simple matmul
        }
    }
    
    int count_tensors_in_json(const std::string& json) {
        // Simple JSON parsing - count "tensor" occurrences
        int count = 0;
        size_t pos = 0;
        while ((pos = json.find("tensor", pos)) != std::string::npos) {
            count++;
            pos++;
        }
        return std::max(1, count);
    }
    
    bool apply_action(int action_type, const RLConfig& config) {
        state_.search_depth++;
        
        switch (action_type) {
            case 0:  // ADD_KN_OP
                return add_kn_operator(config);
            case 1:  // CREATE_TB
                return create_threadblock(config);
            case 2:  // ADD_TB_OP
                return add_tb_operator(config);
            case 3:  // FINISH
                done_ = true;
                return true;
            default:
                return false;
        }
    }
    
    bool add_kn_operator(const RLConfig& config) {
        // Validate operator type
        if (config.operator_type.empty()) {
            return false;
        }
        
        // Track operator addition
        state_.num_kn_operators++;
        state_.num_tensors++;  // New output tensor
        
        // Update kernel graph JSON
        update_kernel_graph_json(config);
        
        return true;
    }
    
    void update_kernel_graph_json(const RLConfig& config) {
        std::ostringstream oss;
        oss << "{\"operators\":[";
        for (int i = 0; i < state_.num_kn_operators; ++i) {
            if (i > 0) oss << ",";
            oss << "{\"type\":\"" << config.operator_type << "\"}";
        }
        oss << "],\"tensors\":" << state_.num_tensors << "}";
        kernel_graph_json_ = oss.str();
    }
    
    bool create_threadblock(const RLConfig& config) {
        // Validate grid/block dimensions
        if (config.grid_dim_x <= 0 || config.grid_dim_y <= 0 || config.grid_dim_z <= 0) {
            return false;
        }
        if (config.block_dim_x <= 0 || config.block_dim_y <= 0 || config.block_dim_z <= 0) {
            return false;
        }
        
        // Check thread count limits (max 1024 threads per block for most GPUs)
        int threads_per_block = config.block_dim_x * config.block_dim_y * config.block_dim_z;
        if (threads_per_block > 1024) {
            return false;
        }
        
        state_.search_level = 1;
        state_.current_grid_dim = {config.grid_dim_x, config.grid_dim_y, config.grid_dim_z};
        state_.current_block_dim = {config.block_dim_x, config.block_dim_y, config.block_dim_z};
        return true;
    }
    
    bool add_tb_operator(const RLConfig& config) {
        // Must have created threadblock first
        if (state_.search_level < 1) {
            return false;
        }
        
        state_.num_tb_operators++;
        return true;
    }
    
    VerifyResult verify() {
        VerifyResult result;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Verification logic:
        // 1. Check that we have both kernel and threadblock operators
        // 2. Verify dimensions are consistent
        // 3. Check fingerprint match (simulated for now)
        
        bool has_valid_structure = (state_.num_kn_operators > 0 && 
                                    state_.num_tb_operators > 0);
        
        bool dimensions_valid = (state_.current_grid_dim[0] > 0 &&
                                 state_.current_block_dim[0] > 0);
        
        // Compute a simple hash for the kernel configuration
        result.kernel_hash = compute_kernel_hash();
        
        // Verification succeeds if structure and dimensions are valid
        result.verified = has_valid_structure && dimensions_valid;
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        result.fingerprint_time_ms = duration.count() / 1000.0f;
        
        if (!result.verified) {
            if (!has_valid_structure) {
                result.rejection_reason = "incomplete_graph";
            } else if (!dimensions_valid) {
                result.rejection_reason = "invalid_dimensions";
            } else {
                result.rejection_reason = "fingerprint_mismatch";
            }
        }
        
        state_.num_verified++;
        
        if (result.verified) {
            state_.num_valid_found++;
            valid_kernels_.push_back(kernel_graph_json_);
        }
        
        return result;
    }
    
    uint64_t compute_kernel_hash() {
        // Simple hash based on configuration
        uint64_t hash = 0;
        hash ^= static_cast<uint64_t>(state_.num_kn_operators) << 32;
        hash ^= static_cast<uint64_t>(state_.num_tb_operators) << 16;
        hash ^= static_cast<uint64_t>(state_.current_grid_dim[0]);
        hash ^= static_cast<uint64_t>(state_.current_block_dim[0]) << 48;
        return hash;
    }
    
    ProfileResult profile(int warmup_iters, int profile_iters) {
        ProfileResult result;
        
        auto compile_start = std::chrono::high_resolution_clock::now();
        
        // Simulate compilation time based on complexity
        int complexity = state_.num_kn_operators + state_.num_tb_operators;
        
        auto compile_end = std::chrono::high_resolution_clock::now();
        auto compile_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            compile_end - compile_start);
        result.compile_time_ms = compile_duration.count() + complexity * 10.0f;
        
        // Estimate latency based on configuration
        // Larger grids generally mean more parallel work
        int grid_size = state_.current_grid_dim[0] * state_.current_grid_dim[1] * 
                        state_.current_grid_dim[2];
        int block_size = state_.current_block_dim[0] * state_.current_block_dim[1] * 
                         state_.current_block_dim[2];
        
        // Simulate latency (smaller grids with larger blocks tend to be faster for memory-bound ops)
        float base_latency = 1.0f;
        float grid_factor = std::log2(grid_size + 1) * 0.1f;
        float block_factor = (1024 - block_size) / 1024.0f * 0.5f;
        
        result.latency_ms = base_latency + grid_factor + block_factor;
        
        // Estimate memory based on tensor count and dimensions
        result.memory_bytes = state_.num_tensors * 1024 * 1024;  // 1 MB per tensor
        
        // Estimate GFLOPS based on operation count
        float ops_per_element = state_.num_kn_operators * 2.0f;  // Rough estimate
        float elements = grid_size * block_size * 1000.0f;  // Rough estimate
        result.gflops = (ops_per_element * elements) / (result.latency_ms * 1e6f);
        
        if (result.latency_ms < state_.best_latency_ms) {
            state_.best_latency_ms = result.latency_ms;
        }
        
        return result;
    }
    
    SearchState get_state() const { return state_; }
    std::string get_kernel_graph_json() const { return kernel_graph_json_; }
    std::vector<std::string> get_valid_kernels() const { return valid_kernels_; }
    bool is_done() const { return done_; }
    
    std::string extract_features() const {
        // Build feature JSON for RL model input
        std::ostringstream oss;
        oss << "{";
        
        // Operators
        oss << "\"operators\":[";
        for (int i = 0; i < state_.num_kn_operators + state_.num_tb_operators; ++i) {
            if (i > 0) oss << ",";
            oss << "{";
            oss << "\"op_id\":" << i << ",";
            oss << "\"op_type\":\"unknown\",";
            oss << "\"op_type_id\":0,";
            oss << "\"num_inputs\":2,";
            oss << "\"num_outputs\":1,";
            oss << "\"flops\":1000000,";
            oss << "\"memory_read_bytes\":4096,";
            oss << "\"memory_write_bytes\":4096,";
            oss << "\"input_tensor_ids\":[" << (i > 0 ? i-1 : 0) << "," << i << "],";
            oss << "\"output_tensor_ids\":[" << (i+1) << "]";
            oss << "}";
        }
        oss << "],";
        
        // Tensors
        oss << "\"tensors\":[";
        for (int i = 0; i < state_.num_tensors; ++i) {
            if (i > 0) oss << ",";
            oss << "{";
            oss << "\"tensor_id\":" << i << ",";
            oss << "\"dims\":[8,4096],";
            oss << "\"dtype\":\"float16\",";
            oss << "\"dtype_id\":1,";
            oss << "\"size_bytes\":65536,";
            oss << "\"memory_level\":" << (i < 2 ? 2 : 1) << ",";
            oss << "\"is_input\":" << (i < 2 ? "true" : "false") << ",";
            oss << "\"is_output\":" << (i == state_.num_tensors - 1 ? "true" : "false");
            oss << "}";
        }
        oss << "],";
        
        // Edges
        oss << "\"edges\":[],";
        
        // Structure features
        oss << "\"num_operators\":" << (state_.num_kn_operators + state_.num_tb_operators) << ",";
        oss << "\"num_tensors\":" << state_.num_tensors << ",";
        oss << "\"graph_depth\":" << std::max(1, state_.num_kn_operators) << ",";
        oss << "\"graph_width\":" << std::max(1, state_.num_tb_operators) << ",";
        oss << "\"critical_path_length\":" << state_.num_kn_operators << ",";
        oss << "\"parallelism_degree\":1.0,";
        
        // Config features
        oss << "\"grid_dim\":{\"x\":" << state_.current_grid_dim[0] 
            << ",\"y\":" << state_.current_grid_dim[1] 
            << ",\"z\":" << state_.current_grid_dim[2] << "},";
        oss << "\"block_dim\":{\"x\":" << state_.current_block_dim[0] 
            << ",\"y\":" << state_.current_block_dim[1] 
            << ",\"z\":" << state_.current_block_dim[2] << "},";
        oss << "\"forloop_range\":1,";
        oss << "\"reduction_dimx\":1,";
        
        // Resource usage
        oss << "\"occupancy\":0.5,";
        oss << "\"shared_mem_usage\":0.3,";
        oss << "\"register_usage\":0.4,";
        
        // Performance prediction
        oss << "\"theoretical_flops\":1e12,";
        oss << "\"memory_bandwidth_utilization\":0.6,";
        oss << "\"arithmetic_intensity\":100.0,";
        oss << "\"estimated_latency_ms\":1.0,";
        
        // Search state
        oss << "\"search_level\":" << state_.search_level << ",";
        oss << "\"search_depth\":" << state_.search_depth;
        
        oss << "}";
        
        return oss.str();
    }
    
private:
    std::string target_graph_json_;
    std::string backend_;
    int gpu_id_;
    
    SearchState state_;
    std::string kernel_graph_json_;
    std::vector<std::string> valid_kernels_;
    bool done_ = false;
};

RLSearchContext::RLSearchContext(
    const std::string& target_graph_json,
    const std::string& backend,
    int gpu_id
) : impl_(std::make_unique<Impl>(target_graph_json, backend, gpu_id))
{}

RLSearchContext::~RLSearchContext() = default;

void RLSearchContext::reset(const std::string& new_target_json) {
    impl_->reset(new_target_json);
}

bool RLSearchContext::apply_action(int action_type, const RLConfig& config) {
    return impl_->apply_action(action_type, config);
}

VerifyResult RLSearchContext::verify() {
    return impl_->verify();
}

ProfileResult RLSearchContext::profile(int warmup_iters, int profile_iters) {
    return impl_->profile(warmup_iters, profile_iters);
}

SearchState RLSearchContext::get_state() const {
    return impl_->get_state();
}

std::string RLSearchContext::get_kernel_graph_json() const {
    return impl_->get_kernel_graph_json();
}

std::vector<std::string> RLSearchContext::get_valid_kernels() const {
    return impl_->get_valid_kernels();
}

bool RLSearchContext::is_done() const {
    return impl_->is_done();
}

std::string RLSearchContext::extract_features() const {
    return impl_->extract_features();
}

// C Interface Implementation

extern "C" {

void* rl_context_create(
    const char* target_graph_json,
    const char* backend,
    int gpu_id
) {
    try {
        return new RLSearchContext(
            target_graph_json ? target_graph_json : "{}",
            backend ? backend : "cuda",
            gpu_id
        );
    } catch (...) {
        return nullptr;
    }
}

void rl_context_destroy(void* ctx) {
    delete static_cast<RLSearchContext*>(ctx);
}

void rl_context_reset(void* ctx, const char* new_target_json) {
    if (ctx) {
        static_cast<RLSearchContext*>(ctx)->reset(
            new_target_json ? new_target_json : ""
        );
    }
}

int rl_context_apply_action(void* ctx, int action_type, const char* config_json) {
    if (!ctx) return 0;
    
    RLConfig config;
    if (config_json) {
        config = RLConfig::from_json(config_json);
    }
    
    return static_cast<RLSearchContext*>(ctx)->apply_action(action_type, config) ? 1 : 0;
}

char* rl_context_verify(void* ctx) {
    if (!ctx) {
        char* result = static_cast<char*>(malloc(3));
        strcpy(result, "{}");
        return result;
    }
    
    auto verify_result = static_cast<RLSearchContext*>(ctx)->verify();
    std::string json = verify_result.to_json();
    
    char* result = static_cast<char*>(malloc(json.size() + 1));
    strcpy(result, json.c_str());
    return result;
}

char* rl_context_profile(void* ctx, int warmup_iters, int profile_iters) {
    if (!ctx) {
        char* result = static_cast<char*>(malloc(3));
        strcpy(result, "{}");
        return result;
    }
    
    auto profile_result = static_cast<RLSearchContext*>(ctx)->profile(
        warmup_iters, profile_iters
    );
    std::string json = profile_result.to_json();
    
    char* result = static_cast<char*>(malloc(json.size() + 1));
    strcpy(result, json.c_str());
    return result;
}

char* rl_context_get_state(void* ctx) {
    if (!ctx) {
        char* result = static_cast<char*>(malloc(3));
        strcpy(result, "{}");
        return result;
    }
    
    auto state = static_cast<RLSearchContext*>(ctx)->get_state();
    std::string json = state.to_json();
    
    char* result = static_cast<char*>(malloc(json.size() + 1));
    strcpy(result, json.c_str());
    return result;
}

char* rl_context_get_kernel_graph(void* ctx) {
    if (!ctx) {
        char* result = static_cast<char*>(malloc(3));
        strcpy(result, "{}");
        return result;
    }
    
    std::string json = static_cast<RLSearchContext*>(ctx)->get_kernel_graph_json();
    
    char* result = static_cast<char*>(malloc(json.size() + 1));
    strcpy(result, json.c_str());
    return result;
}

int rl_context_is_done(void* ctx) {
    if (!ctx) return 1;
    return static_cast<RLSearchContext*>(ctx)->is_done() ? 1 : 0;
}

char* rl_context_extract_features(void* ctx) {
    if (!ctx) {
        char* result = static_cast<char*>(malloc(3));
        strcpy(result, "{}");
        return result;
    }
    
    std::string features = static_cast<RLSearchContext*>(ctx)->extract_features();
    
    char* result = static_cast<char*>(malloc(features.size() + 1));
    strcpy(result, features.c_str());
    return result;
}

void rl_free_string(char* str) {
    free(str);
}

} // extern "C"

} // namespace rl_interface
} // namespace yirage
