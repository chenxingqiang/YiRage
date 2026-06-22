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
 * TPU Search Strategy Implementation
 */

#include "search/backend_strategies/tpu_strategy.h"

#ifdef YIRAGE_BACKEND_TPU_ENABLED

#include <algorithm>
#include <cmath>
#include <sstream>

namespace yirage {
namespace search {

using namespace kernel::tpu;

// =============================================================================
// Constructor and Initialization
// =============================================================================

TPUSearchStrategy::TPUSearchStrategy()
    : version_(TPUVersion::V4), mxu_size_(128), 
      vmem_size_(32 * 1024 * 1024), num_cores_(1) {}

TPUSearchStrategy::TPUSearchStrategy(TPUVersion version)
    : version_(version) {
    switch (version) {
        case TPUVersion::V2:
            mxu_size_ = 128;
            vmem_size_ = 8 * 1024 * 1024;
            break;
        case TPUVersion::V3:
            mxu_size_ = 128;
            vmem_size_ = 16 * 1024 * 1024;
            break;
        case TPUVersion::V4:
        case TPUVersion::V5E:
        case TPUVersion::V5P:
            mxu_size_ = 128;
            vmem_size_ = 32 * 1024 * 1024;
            break;
        default:
            mxu_size_ = 128;
            vmem_size_ = 16 * 1024 * 1024;
    }
}

bool TPUSearchStrategy::initialize(SearchConfig const& config) {
    return true;
}

// =============================================================================
// Candidate Generation
// =============================================================================

std::vector<CandidateConfig>
TPUSearchStrategy::generate_candidates(kernel::Graph const& graph) {
    std::vector<CandidateConfig> candidates;
    
    int m = 1024, n = 1024, k = 1024;  // Default dimensions
    
    auto mxu_configs = generate_mxu_configs(m, n, k);
    
    for (auto const& cfg : mxu_configs) {
        CandidateConfig candidate;
        candidate.config = std::make_unique<TPUKernelConfig>(cfg);
        candidate.score = 0.0f;
        candidates.push_back(std::move(candidate));
    }
    
    total_candidates_ = candidates.size();
    return candidates;
}

std::vector<TPUKernelConfig>
TPUSearchStrategy::generate_mxu_configs(int m, int n, int k) {
    std::vector<TPUKernelConfig> configs;
    
    // MXU is 128x128, so tiles should be multiples of 128
    std::vector<int> tile_sizes = {128, 256, 512, 1024};
    
    for (int tile_m : tile_sizes) {
        for (int tile_n : tile_sizes) {
            for (int tile_k : {128, 256}) {
                TPUKernelConfig config;
                config.version = version_;
                config.mxu_size = mxu_size_;
                config.tile_m = tile_m;
                config.tile_n = tile_n;
                config.tile_k = tile_k;
                config.use_mxu = true;
                config.use_bf16 = true;
                
                // Check VMEM constraint
                size_t vmem_needed = (tile_m * tile_k + tile_k * tile_n + tile_m * tile_n) * 2;
                if (vmem_needed <= vmem_size_) {
                    // Pipeline configurations
                    for (int pipeline : {1, 2, 4}) {
                        config.pipeline_depth = pipeline;
                        config.enable_double_buffering = (pipeline >= 2);
                        configs.push_back(config);
                    }
                }
            }
        }
    }
    
    return configs;
}

// =============================================================================
// Candidate Evaluation
// =============================================================================

float TPUSearchStrategy::evaluate_candidate(CandidateConfig& candidate,
                                           kernel::Graph const& graph) {
    auto* config = static_cast<TPUKernelConfig*>(candidate.config.get());
    
    float mxu_util = evaluate_mxu_utilization(*config);
    float vmem_eff = evaluate_vmem_efficiency(*config);
    float pipeline_eff = evaluate_pipeline_efficiency(*config);
    
    float score = 0.4f * mxu_util + 0.3f * vmem_eff + 0.3f * pipeline_eff;
    
    // BF16 bonus
    if (config->use_bf16) {
        score *= 1.1f;
    }
    
    candidate.score = score;
    evaluated_candidates_++;
    
    if (score > best_score_) {
        best_score_ = score;
    }
    
    return score;
}

float TPUSearchStrategy::evaluate_mxu_utilization(TPUKernelConfig const& config) {
    // MXU is 128x128, tiles should align for full utilization
    int tile_m = config.tile_m;
    int tile_n = config.tile_n;
    int tile_k = config.tile_k;
    
    float m_util = static_cast<float>(tile_m % mxu_size_ == 0 ? 1.0f : 
                   static_cast<float>(tile_m % mxu_size_) / mxu_size_);
    float n_util = static_cast<float>(tile_n % mxu_size_ == 0 ? 1.0f :
                   static_cast<float>(tile_n % mxu_size_) / mxu_size_);
    
    return (m_util + n_util) / 2.0f;
}

float TPUSearchStrategy::evaluate_vmem_efficiency(TPUKernelConfig const& config) {
    size_t tile_bytes = (config.tile_m * config.tile_k + 
                         config.tile_k * config.tile_n +
                         config.tile_m * config.tile_n) * 2;  // BF16
    
    float usage = static_cast<float>(tile_bytes) / vmem_size_;
    
    // Optimal: 60-80% utilization
    if (usage < 0.3f) return 0.5f + usage;
    if (usage > 0.9f) return 1.0f - (usage - 0.9f) * 5.0f;
    return 0.8f + 0.2f * (usage - 0.3f) / 0.6f;
}

float TPUSearchStrategy::evaluate_pipeline_efficiency(TPUKernelConfig const& config) {
    // Deeper pipeline = better for large problems
    float base = 0.5f;
    base += 0.15f * std::min(config.pipeline_depth, 4);
    
    if (config.enable_double_buffering) {
        base += 0.2f;
    }
    
    return std::min(1.0f, base);
}

bool TPUSearchStrategy::is_valid_config(TPUKernelConfig const& config) {
    if (config.tile_m <= 0 || config.tile_n <= 0 || config.tile_k <= 0) {
        return false;
    }
    
    size_t vmem_needed = (config.tile_m * config.tile_k + 
                          config.tile_k * config.tile_n +
                          config.tile_m * config.tile_n) * 2;
    
    return vmem_needed <= vmem_size_;
}

// =============================================================================
// Configuration Selection
// =============================================================================

kernel::KernelConfig* TPUSearchStrategy::select_best_config(
    std::vector<CandidateConfig>& candidates) {
    
    if (candidates.empty()) return nullptr;
    
    auto best_it = std::max_element(candidates.begin(), candidates.end(),
        [](CandidateConfig const& a, CandidateConfig const& b) {
            return a.score < b.score;
        });
    
    return best_it->config.get();
}

std::unique_ptr<kernel::KernelConfig>
TPUSearchStrategy::optimize(kernel::Graph const& graph) {
    auto candidates = generate_candidates(graph);
    
    for (auto& candidate : candidates) {
        evaluate_candidate(candidate, graph);
    }
    
    auto* best = select_best_config(candidates);
    if (!best) {
        return std::make_unique<TPUKernelConfig>();
    }
    
    return std::make_unique<TPUKernelConfig>(
        *static_cast<TPUKernelConfig*>(best));
}

std::string TPUSearchStrategy::get_statistics() const {
    std::ostringstream oss;
    oss << "TPU Search Statistics:\n";
    oss << "  Version: TPU v" << static_cast<int>(version_) + 2 << "\n";
    oss << "  MXU Size: " << mxu_size_ << "x" << mxu_size_ << "\n";
    oss << "  VMEM Size: " << (vmem_size_ / 1024 / 1024) << " MB\n";
    oss << "  Total candidates: " << total_candidates_ << "\n";
    oss << "  Evaluated: " << evaluated_candidates_ << "\n";
    oss << "  Best score: " << best_score_ << "\n";
    return oss.str();
}

}  // namespace search
}  // namespace yirage

#endif  // YIRAGE_BACKEND_TPU_ENABLED
