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
 * @file rl_interface.h
 * @brief C++ interface for RL-guided kernel search closed loop.
 * 
 * This header defines the interface between the Python RL environment
 * and the C++ search core. It enables:
 * 
 * 1. RL policy → C++ search: Apply configuration choices
 * 2. C++ verification → RL: Return validation results (GPU)
 * 3. C++ profiling → RL: Return performance metrics (GPU)
 * 
 * Closed loop:
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │  RL Policy ──action──> C++ Core ──GPU verify──> reward     │
 *   │      │                     │                       │        │
 *   │      └──────── obs <───────┴───── feedback <───────┘        │
 *   └─────────────────────────────────────────────────────────────┘
 */

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace yirage {
namespace rl_interface {

/**
 * @brief Configuration from RL action.
 * 
 * This structure represents the decoded RL action
 * as a search configuration for kernel generation.
 */
struct RLConfig {
    // Grid dimensions
    int grid_dim_x = 1;
    int grid_dim_y = 1;
    int grid_dim_z = 1;
    
    // Block dimensions
    int block_dim_x = 128;
    int block_dim_y = 1;
    int block_dim_z = 1;
    
    // Input mappings: (x, y, z) for each input
    std::vector<std::array<int, 3>> imaps;
    
    // Output mapping
    std::array<int, 3> omap = {0, 0, 0};
    
    // Forloop range
    int frange = 1;
    
    // Operator to add
    std::string operator_type;
    std::vector<int> input_indices;
    
    // Serialization
    std::string to_json() const;
    static RLConfig from_json(const std::string& json);
};

/**
 * @brief Result from fingerprint verification.
 * 
 * This is the critical feedback from GPU verification
 * that closes the RL loop.
 */
struct VerifyResult {
    bool verified = false;
    float fingerprint_time_ms = 0.0f;
    std::string rejection_reason;
    uint64_t kernel_hash = 0;
    
    std::string to_json() const;
};

/**
 * @brief Result from performance profiling.
 * 
 * Provides performance metrics for reward computation.
 */
struct ProfileResult {
    float latency_ms = std::numeric_limits<float>::infinity();
    size_t memory_bytes = 0;
    float gflops = 0.0f;
    float compile_time_ms = 0.0f;
    
    std::string to_json() const;
};

/**
 * @brief Search state for observation.
 * 
 * Encapsulates current search state for RL observation.
 */
struct SearchState {
    // Search progress
    int search_level = 0;  // 0=kernel, 1=threadblock
    int search_depth = 0;
    int num_kn_operators = 0;
    int num_tb_operators = 0;
    
    // Graph statistics
    int num_tensors = 0;
    std::vector<std::vector<int>> tensor_dims;
    
    // History
    int num_valid_found = 0;
    int num_verified = 0;
    float best_latency_ms = std::numeric_limits<float>::infinity();
    
    // Current configuration
    std::array<int, 3> current_grid_dim = {1, 1, 1};
    std::array<int, 3> current_block_dim = {128, 1, 1};
    
    std::string to_json() const;
};

/**
 * @brief RL Search Context.
 * 
 * Maintains state across RL steps for a single episode.
 */
class RLSearchContext {
public:
    /**
     * @brief Create new context for target graph.
     * 
     * @param target_graph_json Target computation graph (JSON)
     * @param backend Target backend (cuda, maca, etc.)
     * @param gpu_id GPU to use for verification
     */
    RLSearchContext(
        const std::string& target_graph_json,
        const std::string& backend = "cuda",
        int gpu_id = 0
    );
    
    ~RLSearchContext();
    
    /**
     * @brief Reset context for new episode.
     * 
     * @param new_target_json Optional new target graph
     */
    void reset(const std::string& new_target_json = "");
    
    /**
     * @brief Apply RL action to search.
     * 
     * This is the entry point for RL decisions.
     * 
     * @param action_type Action type (ADD_KN_OP, CREATE_TB, etc.)
     * @param config Configuration from decoded action
     * @return true if action was successfully applied
     */
    bool apply_action(int action_type, const RLConfig& config);
    
    /**
     * @brief Verify current kernel on GPU.
     * 
     * This is the critical GPU operation that provides
     * ground truth for the RL reward.
     * 
     * @return Verification result with validity and timing
     */
    VerifyResult verify();
    
    /**
     * @brief Profile kernel performance on GPU.
     * 
     * @param warmup_iters Warmup iterations
     * @param profile_iters Profile iterations
     * @return Performance metrics
     */
    ProfileResult profile(int warmup_iters = 10, int profile_iters = 100);
    
    /**
     * @brief Get current search state.
     * 
     * Used to construct RL observation.
     */
    SearchState get_state() const;
    
    /**
     * @brief Get current kernel graph as JSON.
     */
    std::string get_kernel_graph_json() const;
    
    /**
     * @brief Get list of valid kernels found.
     */
    std::vector<std::string> get_valid_kernels() const;
    
    /**
     * @brief Check if search is complete.
     */
    bool is_done() const;
    
    /**
     * @brief Extract features from current µGraph for RL model input.
     * 
     * This is the key interface for the closed loop:
     * C++ µGraph features → Python RL model → action → C++ apply
     * 
     * @return JSON string with complete graph features
     */
    std::string extract_features() const;
    
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================
// C Interface for Python bindings
// ============================================

extern "C" {

/**
 * @brief Create new RL search context.
 * 
 * @param target_graph_json Target graph JSON string
 * @param backend Backend name
 * @param gpu_id GPU ID for verification
 * @return Opaque handle to context
 */
void* rl_context_create(
    const char* target_graph_json,
    const char* backend,
    int gpu_id
);

/**
 * @brief Destroy RL search context.
 */
void rl_context_destroy(void* ctx);

/**
 * @brief Reset context for new episode.
 */
void rl_context_reset(void* ctx, const char* new_target_json);

/**
 * @brief Apply action to search.
 * 
 * @param ctx Context handle
 * @param action_type Action type
 * @param config_json Configuration JSON
 * @return 1 if successful, 0 otherwise
 */
int rl_context_apply_action(void* ctx, int action_type, const char* config_json);

/**
 * @brief Verify kernel on GPU.
 * 
 * @param ctx Context handle
 * @return JSON string with verification result (caller must free)
 */
char* rl_context_verify(void* ctx);

/**
 * @brief Profile kernel on GPU.
 * 
 * @param ctx Context handle
 * @param warmup_iters Warmup iterations
 * @param profile_iters Profile iterations
 * @return JSON string with profile result (caller must free)
 */
char* rl_context_profile(void* ctx, int warmup_iters, int profile_iters);

/**
 * @brief Get current search state.
 * 
 * @param ctx Context handle
 * @return JSON string with state (caller must free)
 */
char* rl_context_get_state(void* ctx);

/**
 * @brief Get current kernel graph.
 * 
 * @param ctx Context handle
 * @return JSON string (caller must free)
 */
char* rl_context_get_kernel_graph(void* ctx);

/**
 * @brief Check if search is complete.
 */
int rl_context_is_done(void* ctx);

/**
 * @brief Extract features from current µGraph.
 * 
 * Features are used as input to the RL model.
 * 
 * @param ctx Context handle
 * @return JSON string with features (caller must free)
 */
char* rl_context_extract_features(void* ctx);

/**
 * @brief Free string allocated by RL interface.
 */
void rl_free_string(char* str);

} // extern "C"

} // namespace rl_interface
} // namespace yirage
