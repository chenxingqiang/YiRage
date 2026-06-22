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
 *
 * ROCm Search Strategy
 * 
 * Search strategy optimized for AMD ROCm/HIP GPU architecture.
 * Key differences from CUDA:
 * - 64-thread wavefronts (vs 32-thread warps)
 * - LDS (Local Data Share) instead of shared memory
 * - MFMA (Matrix Fused Multiply-Add) instructions
 */

#pragma once

#include "search/common/search_strategy.h"
#include "kernel/rocm/rocm_kernel_config.h"

#ifdef YIRAGE_BACKEND_ROCM_ENABLED

namespace yirage {
namespace search {

/**
 * @brief ROCm kernel configuration
 */
struct ROCmKernelConfig : public kernel::KernelConfig {
    int block_x = 256;
    int block_y = 1;
    int block_z = 1;
    int grid_x = 1;
    int grid_y = 1;
    int grid_z = 1;
    size_t lds_size = 0;            // Local Data Share size
    int wavefronts_per_block = 4;
    bool use_mfma = false;          // Matrix Fused Multiply-Add
    bool use_fp16 = false;
    bool use_bf16 = false;
    int tile_m = 64;
    int tile_n = 64;
    int tile_k = 16;
    int unroll_factor = 4;
    
    kernel::KernelConfig* clone() const override {
        return new ROCmKernelConfig(*this);
    }
};

/**
 * @brief Search strategy for AMD ROCm/HIP GPUs
 * 
 * Optimizes kernel configurations for AMD GPU architecture:
 * - CDNA (MI100, MI200, MI250, MI300)
 * - RDNA (consumer GPUs)
 */
class ROCmSearchStrategy : public SearchStrategy {
public:
    /**
     * @brief Construct with architecture detection
     */
    ROCmSearchStrategy();
    
    /**
     * @brief Construct with specific architecture
     * @param arch Architecture identifier (e.g., "gfx90a" for MI200)
     */
    explicit ROCmSearchStrategy(std::string const& arch);
    
    /**
     * @brief Initialize strategy with config
     */
    bool initialize(SearchConfig const& config) override;
    
    /**
     * @brief Generate candidate configurations
     */
    std::vector<CandidateConfig> generate_candidates(
        kernel::Graph const& graph) override;
    
    /**
     * @brief Evaluate a candidate configuration
     */
    float evaluate_candidate(CandidateConfig& candidate,
                            kernel::Graph const& graph) override;
    
    /**
     * @brief Select best configuration from candidates
     */
    kernel::KernelConfig* select_best_config(
        std::vector<CandidateConfig>& candidates) override;
    
    /**
     * @brief Run full optimization
     */
    std::unique_ptr<kernel::KernelConfig> optimize(
        kernel::Graph const& graph) override;
    
    /**
     * @brief Get search statistics
     */
    std::string get_statistics() const override;

private:
    // Hardware constants
    static constexpr int WAVEFRONT_SIZE = 64;
    static constexpr int MAX_THREADS_PER_BLOCK = 1024;
    static constexpr int MAX_WAVEFRONTS_PER_CU = 32;
    static constexpr int MAX_LDS_PER_BLOCK = 65536;  // 64 KB
    static constexpr int REGISTERS_PER_CU = 65536;
    
    // Architecture-specific parameters
    std::string architecture_;
    int compute_units_ = 120;       // Default for MI250X
    int mfma_size_ = 32;            // MFMA tile size
    bool supports_fp8_ = false;
    bool supports_sparsity_ = false;
    
    // Search state
    int total_candidates_ = 0;
    int evaluated_candidates_ = 0;
    float best_score_ = 0.0f;
    
    /**
     * @brief Generate wavefront configurations
     */
    std::vector<int> generate_wavefront_configs(size_t problem_size);
    
    /**
     * @brief Generate LDS configurations
     */
    std::vector<size_t> generate_lds_configs(size_t data_size);
    
    /**
     * @brief Generate grid/block configurations
     */
    std::vector<std::pair<int, int>> generate_grid_block_configs(int m, int n);
    
    /**
     * @brief Generate MFMA-optimized matmul configs
     */
    std::vector<ROCmKernelConfig> generate_mfma_configs(int m, int n, int k);
    
    /**
     * @brief Evaluate occupancy for ROCm
     */
    float evaluate_occupancy(ROCmKernelConfig const& config);
    
    /**
     * @brief Evaluate LDS efficiency
     */
    float evaluate_lds_efficiency(ROCmKernelConfig const& config);
    
    /**
     * @brief Evaluate compute throughput
     */
    float evaluate_compute_throughput(ROCmKernelConfig const& config);
    
    /**
     * @brief Evaluate LDS bank conflicts
     */
    float evaluate_bank_conflicts(ROCmKernelConfig const& config);
    
    /**
     * @brief Check if configuration is valid
     */
    bool is_valid_config(ROCmKernelConfig const& config);
    
    /**
     * @brief Detect architecture from hardware
     */
    void detect_architecture();
    
    /**
     * @brief Configure architecture-specific parameters
     */
    void configure_for_architecture();
};

}  // namespace search
}  // namespace yirage

#endif  // YIRAGE_BACKEND_ROCM_ENABLED
