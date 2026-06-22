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
 * ROCm Search Strategy Implementation
 * 
 * Search strategy optimized for AMD ROCm/HIP GPU hardware.
 * Key optimizations:
 * - 64-thread wavefront configurations
 * - LDS (Local Data Share) optimization
 * - MFMA instruction utilization
 * - Occupancy calculation for GCN/CDNA architecture
 */

#include "search/backend_strategies/rocm_strategy.h"

#ifdef YIRAGE_BACKEND_ROCM_ENABLED

#include <algorithm>
#include <cmath>
#include <sstream>
#include <cstdlib>

namespace yirage {
namespace search {

// =============================================================================
// Constructor and Initialization
// =============================================================================

ROCmSearchStrategy::ROCmSearchStrategy()
    : architecture_("gfx90a") {  // Default to MI200
    detect_architecture();
    configure_for_architecture();
}

ROCmSearchStrategy::ROCmSearchStrategy(std::string const& arch)
    : architecture_(arch) {
    configure_for_architecture();
}

bool ROCmSearchStrategy::initialize(SearchConfig const& config) {
    // Use config to adjust search parameters
    return true;
}

void ROCmSearchStrategy::detect_architecture() {
    // Try environment variable first
    const char* arch_env = std::getenv("YIRAGE_ROCM_ARCH");
    if (arch_env) {
        architecture_ = arch_env;
        return;
    }
    
    // Try rocm-smi or hipinfo
    // For now, default to MI200 series (gfx90a)
    architecture_ = "gfx90a";
}

void ROCmSearchStrategy::configure_for_architecture() {
    // Configure based on architecture
    if (architecture_ == "gfx908" || architecture_.find("mi100") != std::string::npos) {
        // MI100 (CDNA1)
        compute_units_ = 120;
        mfma_size_ = 32;
        supports_fp8_ = false;
        supports_sparsity_ = false;
    } else if (architecture_ == "gfx90a" || architecture_.find("mi200") != std::string::npos ||
               architecture_.find("mi250") != std::string::npos) {
        // MI200/MI250 (CDNA2)
        compute_units_ = 110;  // Per die
        mfma_size_ = 32;
        supports_fp8_ = false;
        supports_sparsity_ = false;
    } else if (architecture_ == "gfx940" || architecture_ == "gfx941" || architecture_ == "gfx942" ||
               architecture_.find("mi300") != std::string::npos) {
        // MI300 (CDNA3)
        compute_units_ = 304;  // MI300X
        mfma_size_ = 32;
        supports_fp8_ = true;
        supports_sparsity_ = true;
    } else if (architecture_.find("gfx11") != std::string::npos) {
        // RDNA3
        compute_units_ = 96;
        mfma_size_ = 16;
        supports_fp8_ = false;
        supports_sparsity_ = false;
    } else {
        // Default
        compute_units_ = 60;
        mfma_size_ = 16;
    }
}

// =============================================================================
// Candidate Generation
// =============================================================================

std::vector<CandidateConfig>
ROCmSearchStrategy::generate_candidates(kernel::Graph const& graph) {
    std::vector<CandidateConfig> candidates;
    
    // Get problem dimensions from graph
    int m = 1024, n = 1024, k = 1024;  // Default, should extract from graph
    
    // Generate wavefront configurations
    auto wavefront_configs = generate_wavefront_configs(m * n);
    
    // Generate LDS configurations
    size_t data_size = (m * k + k * n + m * n) * sizeof(float);
    auto lds_configs = generate_lds_configs(data_size);
    
    // Generate grid/block configurations
    auto grid_block_configs = generate_grid_block_configs(m, n);
    
    // Generate MFMA configurations if applicable
    auto mfma_configs = generate_mfma_configs(m, n, k);
    
    // Add MFMA-based candidates
    for (auto const& mfma_cfg : mfma_configs) {
        CandidateConfig candidate;
        candidate.config = std::make_unique<ROCmKernelConfig>(mfma_cfg);
        candidate.score = 0.0f;
        candidates.push_back(std::move(candidate));
    }
    
    // Add wavefront-based candidates
    for (int num_wavefronts : wavefront_configs) {
        for (size_t lds_size : lds_configs) {
            auto config = std::make_unique<ROCmKernelConfig>();
            config->block_x = num_wavefronts * WAVEFRONT_SIZE;
            config->lds_size = lds_size;
            config->wavefronts_per_block = num_wavefronts;
            
            // Determine tile sizes based on wavefronts
            config->tile_m = 32 * std::min(num_wavefronts, 4);
            config->tile_n = 32 * std::min(num_wavefronts, 4);
            config->tile_k = 16;
            
            if (is_valid_config(*config)) {
                CandidateConfig candidate;
                candidate.config = std::move(config);
                candidate.score = 0.0f;
                candidates.push_back(std::move(candidate));
            }
        }
    }
    
    total_candidates_ = candidates.size();
    return candidates;
}

std::vector<int> ROCmSearchStrategy::generate_wavefront_configs(size_t problem_size) {
    std::vector<int> configs;
    
    // ROCm uses 64-thread wavefronts
    // Valid block sizes: 64, 128, 192, 256, 320, 384, 448, 512, ..., 1024
    for (int num_wavefronts = 1; num_wavefronts <= 16; num_wavefronts++) {
        int threads = num_wavefronts * WAVEFRONT_SIZE;
        if (threads <= MAX_THREADS_PER_BLOCK) {
            configs.push_back(num_wavefronts);
        }
    }
    
    return configs;
}

std::vector<size_t> ROCmSearchStrategy::generate_lds_configs(size_t data_size) {
    std::vector<size_t> configs;
    
    // LDS sizes from 16KB to 64KB
    for (size_t lds = 16 * 1024; lds <= MAX_LDS_PER_BLOCK; lds += 16 * 1024) {
        configs.push_back(lds);
    }
    
    return configs;
}

std::vector<std::pair<int, int>>
ROCmSearchStrategy::generate_grid_block_configs(int m, int n) {
    std::vector<std::pair<int, int>> configs;
    
    // Block sizes must be multiples of 64 (wavefront size)
    std::vector<int> block_sizes = {64, 128, 256, 512, 1024};
    
    for (int block_x : block_sizes) {
        for (int block_y : {1, 2, 4, 8}) {
            if (block_x * block_y <= MAX_THREADS_PER_BLOCK) {
                configs.emplace_back(block_x, block_y);
            }
        }
    }
    
    return configs;
}

std::vector<ROCmKernelConfig>
ROCmSearchStrategy::generate_mfma_configs(int m, int n, int k) {
    std::vector<ROCmKernelConfig> configs;
    
    // MFMA tile sizes for different architectures
    std::vector<std::tuple<int, int, int>> mfma_tiles;
    
    if (architecture_.find("gfx90") != std::string::npos ||
        architecture_.find("gfx94") != std::string::npos) {
        // CDNA: 32x32x8 and 16x16x16 MFMA
        mfma_tiles = {
            {32, 32, 8},
            {16, 16, 16},
            {64, 64, 8},
            {128, 128, 8}
        };
    } else {
        // RDNA/GCN: Smaller tiles
        mfma_tiles = {
            {16, 16, 16},
            {32, 32, 16}
        };
    }
    
    for (auto const& [tile_m, tile_n, tile_k] : mfma_tiles) {
        ROCmKernelConfig config;
        config.tile_m = tile_m;
        config.tile_n = tile_n;
        config.tile_k = tile_k;
        config.use_mfma = true;
        
        // Calculate block dimensions
        int warps_m = (tile_m + 31) / 32;
        int warps_n = (tile_n + 31) / 32;
        config.wavefronts_per_block = warps_m * warps_n;
        config.block_x = config.wavefronts_per_block * WAVEFRONT_SIZE;
        
        // Calculate LDS requirements
        config.lds_size = (tile_m * tile_k + tile_k * tile_n) * sizeof(float);
        
        // Grid dimensions
        config.grid_x = (m + tile_m - 1) / tile_m;
        config.grid_y = (n + tile_n - 1) / tile_n;
        
        // FP16/BF16 options
        for (bool use_fp16 : {false, true}) {
            config.use_fp16 = use_fp16;
            if (is_valid_config(config)) {
                configs.push_back(config);
            }
        }
    }
    
    return configs;
}

// =============================================================================
// Candidate Evaluation
// =============================================================================

float ROCmSearchStrategy::evaluate_candidate(CandidateConfig& candidate,
                                             kernel::Graph const& graph) {
    auto* config = static_cast<ROCmKernelConfig*>(candidate.config.get());
    
    // Weighted combination of metrics
    float occupancy = evaluate_occupancy(*config);
    float lds_efficiency = evaluate_lds_efficiency(*config);
    float compute = evaluate_compute_throughput(*config);
    float bank_conflicts = evaluate_bank_conflicts(*config);
    
    // Weights
    float score = 0.3f * occupancy +
                  0.25f * lds_efficiency +
                  0.3f * compute +
                  0.15f * (1.0f - bank_conflicts);
    
    // MFMA bonus
    if (config->use_mfma) {
        score *= 1.2f;
    }
    
    // FP16 bonus for memory-bound kernels
    if (config->use_fp16) {
        score *= 1.1f;
    }
    
    candidate.score = score;
    evaluated_candidates_++;
    
    if (score > best_score_) {
        best_score_ = score;
    }
    
    return score;
}

float ROCmSearchStrategy::evaluate_occupancy(ROCmKernelConfig const& config) {
    int threads_per_block = config.block_x * config.block_y * config.block_z;
    int wavefronts_per_block = (threads_per_block + WAVEFRONT_SIZE - 1) / WAVEFRONT_SIZE;
    
    // Limit by wavefronts
    int blocks_by_wavefronts = MAX_WAVEFRONTS_PER_CU / wavefronts_per_block;
    
    // Limit by LDS
    int blocks_by_lds = (config.lds_size > 0) ?
        MAX_LDS_PER_BLOCK / config.lds_size : 16;
    
    // Limit by registers (estimate)
    int regs_per_thread = 64;  // Typical for matmul
    int regs_per_block = regs_per_thread * threads_per_block;
    int blocks_by_regs = REGISTERS_PER_CU / regs_per_block;
    
    int max_blocks = std::min({blocks_by_wavefronts, blocks_by_lds, blocks_by_regs, 16});
    int active_wavefronts = max_blocks * wavefronts_per_block;
    
    return static_cast<float>(active_wavefronts) / MAX_WAVEFRONTS_PER_CU;
}

float ROCmSearchStrategy::evaluate_lds_efficiency(ROCmKernelConfig const& config) {
    if (config.lds_size == 0) return 0.5f;
    
    // Optimal LDS usage is 50-80% of max
    float usage_ratio = static_cast<float>(config.lds_size) / MAX_LDS_PER_BLOCK;
    
    if (usage_ratio < 0.3f) {
        return 0.5f + usage_ratio;  // Under-utilized
    } else if (usage_ratio > 0.8f) {
        return 1.0f - (usage_ratio - 0.8f) * 2.0f;  // Over-utilized limits occupancy
    } else {
        return 0.9f + 0.1f * (usage_ratio - 0.3f) / 0.5f;  // Sweet spot
    }
}

float ROCmSearchStrategy::evaluate_compute_throughput(ROCmKernelConfig const& config) {
    float score = 0.5f;
    
    // Higher wavefront count generally better
    score += 0.05f * std::min(config.wavefronts_per_block, 8);
    
    // MFMA provides higher throughput
    if (config.use_mfma) {
        score += 0.2f;
    }
    
    // Larger tiles amortize overhead
    float tile_efficiency = std::min(1.0f, 
        (config.tile_m * config.tile_n) / (128.0f * 128.0f));
    score += 0.2f * tile_efficiency;
    
    return std::min(1.0f, score);
}

float ROCmSearchStrategy::evaluate_bank_conflicts(ROCmKernelConfig const& config) {
    // LDS has 32 banks, 4 bytes per bank
    // Access patterns divisible by 32 or 64 may cause conflicts
    
    int tile_m = config.tile_m;
    int tile_n = config.tile_n;
    
    // Check for power-of-2 dimensions that cause conflicts
    bool m_conflict = (tile_m % 32 == 0) && (tile_m % 33 != 0);
    bool n_conflict = (tile_n % 32 == 0) && (tile_n % 33 != 0);
    
    float conflict_ratio = 0.0f;
    if (m_conflict) conflict_ratio += 0.3f;
    if (n_conflict) conflict_ratio += 0.3f;
    
    // Padding usually helps
    if (config.lds_size % 128 == 0) {
        conflict_ratio *= 0.5f;
    }
    
    return std::min(1.0f, conflict_ratio);
}

bool ROCmSearchStrategy::is_valid_config(ROCmKernelConfig const& config) {
    int total_threads = config.block_x * config.block_y * config.block_z;
    
    if (total_threads == 0 || total_threads > MAX_THREADS_PER_BLOCK) {
        return false;
    }
    
    if (total_threads % WAVEFRONT_SIZE != 0) {
        return false;
    }
    
    if (config.lds_size > MAX_LDS_PER_BLOCK) {
        return false;
    }
    
    if (config.tile_m <= 0 || config.tile_n <= 0 || config.tile_k <= 0) {
        return false;
    }
    
    return true;
}

// =============================================================================
// Configuration Selection and Optimization
// =============================================================================

kernel::KernelConfig* ROCmSearchStrategy::select_best_config(
    std::vector<CandidateConfig>& candidates) {
    
    if (candidates.empty()) return nullptr;
    
    auto best_it = std::max_element(candidates.begin(), candidates.end(),
        [](CandidateConfig const& a, CandidateConfig const& b) {
            return a.score < b.score;
        });
    
    return best_it->config.get();
}

std::unique_ptr<kernel::KernelConfig>
ROCmSearchStrategy::optimize(kernel::Graph const& graph) {
    auto candidates = generate_candidates(graph);
    
    for (auto& candidate : candidates) {
        evaluate_candidate(candidate, graph);
    }
    
    auto* best = select_best_config(candidates);
    if (!best) {
        // Return default configuration
        return std::make_unique<ROCmKernelConfig>();
    }
    
    return std::make_unique<ROCmKernelConfig>(
        *static_cast<ROCmKernelConfig*>(best));
}

std::string ROCmSearchStrategy::get_statistics() const {
    std::ostringstream oss;
    oss << "ROCm Search Statistics:\n";
    oss << "  Architecture: " << architecture_ << "\n";
    oss << "  Compute Units: " << compute_units_ << "\n";
    oss << "  Wavefront Size: " << WAVEFRONT_SIZE << "\n";
    oss << "  Supports FP8: " << (supports_fp8_ ? "yes" : "no") << "\n";
    oss << "  Supports Sparsity: " << (supports_sparsity_ ? "yes" : "no") << "\n";
    oss << "  Total candidates: " << total_candidates_ << "\n";
    oss << "  Evaluated: " << evaluated_candidates_ << "\n";
    oss << "  Best score: " << best_score_ << "\n";
    return oss.str();
}

}  // namespace search
}  // namespace yirage

#endif  // YIRAGE_BACKEND_ROCM_ENABLED
