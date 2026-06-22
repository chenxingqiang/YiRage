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
 * Intel XPU Search Strategy Implementation
 */

#include "search/backend_strategies/xpu_strategy.h"

#ifdef YIRAGE_BACKEND_XPU_ENABLED

#include <algorithm>
#include <cmath>
#include <sstream>

namespace yirage {
namespace search {

using namespace kernel::xpu;

// =============================================================================
// Constructor and Initialization
// =============================================================================

XPUSearchStrategy::XPUSearchStrategy()
    : arch_(XPUArch::PONTE_VECCHIO) {
    configure_for_arch();
}

XPUSearchStrategy::XPUSearchStrategy(XPUArch arch)
    : arch_(arch) {
    configure_for_arch();
}

void XPUSearchStrategy::configure_for_arch() {
    switch (arch_) {
        case XPUArch::PONTE_VECCHIO:
            total_eus_ = 512;           // 128 per tile * 4 (2 tiles * 2 stacks)
            slm_size_ = 128 * 1024;     // 128KB SLM
            num_tiles_ = 2;
            break;
        case XPUArch::ARC_A770:
            total_eus_ = 512;           // 32 Xe-cores * 16 EUs
            slm_size_ = 64 * 1024;
            num_tiles_ = 1;
            break;
        case XPUArch::ARC_A750:
            total_eus_ = 448;
            slm_size_ = 64 * 1024;
            num_tiles_ = 1;
            break;
        case XPUArch::FLEX_170:
            total_eus_ = 256;
            slm_size_ = 64 * 1024;
            num_tiles_ = 1;
            break;
        default:
            total_eus_ = 256;
            slm_size_ = 64 * 1024;
            num_tiles_ = 1;
    }
}

bool XPUSearchStrategy::initialize(SearchConfig const& config) {
    return true;
}

// =============================================================================
// Candidate Generation
// =============================================================================

std::vector<CandidateConfig>
XPUSearchStrategy::generate_candidates(kernel::Graph const& graph) {
    std::vector<CandidateConfig> candidates;
    
    int m = 1024, n = 1024, k = 1024;
    
    auto xmx_configs = generate_xmx_configs(m, n, k);
    
    for (auto const& cfg : xmx_configs) {
        CandidateConfig candidate;
        candidate.config = std::make_unique<XPUKernelConfig>(cfg);
        candidate.score = 0.0f;
        candidates.push_back(std::move(candidate));
    }
    
    total_candidates_ = candidates.size();
    return candidates;
}

std::vector<XPUKernelConfig>
XPUSearchStrategy::generate_xmx_configs(int m, int n, int k) {
    std::vector<XPUKernelConfig> configs;
    
    // SIMD widths
    std::vector<int> simd_widths = {8, 16, 32};
    
    // Sub-group configurations
    std::vector<int> num_subgroups = {2, 4, 8};
    
    // XMX tile sizes
    std::vector<std::tuple<int, int, int>> xmx_tiles = {
        {8, 16, 16},    // Standard XMX
        {16, 16, 16},   // Larger tiles
        {8, 32, 16}     // Wide output
    };
    
    for (int simd : simd_widths) {
        for (int sg : num_subgroups) {
            for (auto const& [xm, xn, xk] : xmx_tiles) {
                XPUKernelConfig config;
                config.arch = arch_;
                config.simd_width = simd;
                config.num_sub_groups = sg;
                config.xmx_m = xm;
                config.xmx_n = xn;
                config.xmx_k = xk;
                config.use_xmx = true;
                config.use_dpas = true;
                config.slm_size = slm_size_;
                config.num_tiles = num_tiles_;
                
                // Check SLM constraint
                size_t slm_needed = (xm * xk + xk * xn) * sg * 2;  // BF16
                if (slm_needed <= slm_size_) {
                    // BF16 option
                    config.use_bf16 = true;
                    configs.push_back(config);
                    
                    // INT8 option
                    config.use_bf16 = false;
                    config.use_int8 = true;
                    configs.push_back(config);
                }
            }
        }
    }
    
    return configs;
}

// =============================================================================
// Candidate Evaluation
// =============================================================================

float XPUSearchStrategy::evaluate_candidate(CandidateConfig& candidate,
                                           kernel::Graph const& graph) {
    auto* config = static_cast<XPUKernelConfig*>(candidate.config.get());
    
    float xmx_util = evaluate_xmx_utilization(*config);
    float slm_eff = evaluate_slm_efficiency(*config);
    float occupancy = evaluate_occupancy(*config);
    
    float score = 0.35f * xmx_util + 0.3f * slm_eff + 0.35f * occupancy;
    
    // XMX bonus
    if (config->use_xmx) {
        score *= 1.2f;
    }
    
    // DPAS bonus
    if (config->use_dpas) {
        score *= 1.1f;
    }
    
    // Multi-tile bonus
    if (config->enable_multi_tile && num_tiles_ > 1) {
        score *= 1.15f;
    }
    
    candidate.score = std::min(1.0f, score);
    evaluated_candidates_++;
    
    if (candidate.score > best_score_) {
        best_score_ = candidate.score;
    }
    
    return candidate.score;
}

float XPUSearchStrategy::evaluate_xmx_utilization(XPUKernelConfig const& config) {
    // XMX is 8x16 systolic array
    int tile_m = config.xmx_m;
    int tile_n = config.xmx_n;
    
    float m_util = (tile_m % 8 == 0) ? 1.0f : static_cast<float>(tile_m % 8) / 8.0f;
    float n_util = (tile_n % 16 == 0) ? 1.0f : static_cast<float>(tile_n % 16) / 16.0f;
    
    return (m_util + n_util) / 2.0f;
}

float XPUSearchStrategy::evaluate_slm_efficiency(XPUKernelConfig const& config) {
    size_t slm_used = (config.xmx_m * config.xmx_k + 
                       config.xmx_k * config.xmx_n) * 
                      config.num_sub_groups * 2;
    
    float usage = static_cast<float>(slm_used) / slm_size_;
    
    // Optimal: 50-80%
    if (usage < 0.3f) return 0.5f + usage;
    if (usage > 0.85f) return 1.0f - (usage - 0.85f) * 3.0f;
    return 0.8f + 0.2f * (usage - 0.3f) / 0.55f;
}

float XPUSearchStrategy::evaluate_occupancy(XPUKernelConfig const& config) {
    // Threads = SIMD width * sub-groups
    int threads_per_wg = config.simd_width * config.num_sub_groups;
    int threads_per_eu = 8;  // PVC has 8 threads per EU
    
    float eu_util = static_cast<float>(threads_per_wg) / (threads_per_eu * 16);
    
    return std::min(1.0f, eu_util);
}

bool XPUSearchStrategy::is_valid_config(XPUKernelConfig const& config) {
    if (config.simd_width <= 0 || config.num_sub_groups <= 0) {
        return false;
    }
    
    size_t slm_needed = config.xmx_m * config.xmx_k * config.num_sub_groups * 2;
    return slm_needed <= slm_size_;
}

// =============================================================================
// Configuration Selection
// =============================================================================

kernel::KernelConfig* XPUSearchStrategy::select_best_config(
    std::vector<CandidateConfig>& candidates) {
    
    if (candidates.empty()) return nullptr;
    
    auto best_it = std::max_element(candidates.begin(), candidates.end(),
        [](CandidateConfig const& a, CandidateConfig const& b) {
            return a.score < b.score;
        });
    
    return best_it->config.get();
}

std::unique_ptr<kernel::KernelConfig>
XPUSearchStrategy::optimize(kernel::Graph const& graph) {
    auto candidates = generate_candidates(graph);
    
    for (auto& candidate : candidates) {
        evaluate_candidate(candidate, graph);
    }
    
    auto* best = select_best_config(candidates);
    if (!best) {
        return std::make_unique<XPUKernelConfig>();
    }
    
    return std::make_unique<XPUKernelConfig>(
        *static_cast<XPUKernelConfig*>(best));
}

std::string XPUSearchStrategy::get_statistics() const {
    std::ostringstream oss;
    oss << "Intel XPU Search Statistics:\n";
    oss << "  Architecture: ";
    switch (arch_) {
        case XPUArch::PONTE_VECCHIO: oss << "Ponte Vecchio"; break;
        case XPUArch::ARC_A770: oss << "Arc A770"; break;
        case XPUArch::ARC_A750: oss << "Arc A750"; break;
        case XPUArch::FLEX_170: oss << "Flex 170"; break;
        default: oss << "Unknown";
    }
    oss << "\n";
    oss << "  Total EUs: " << total_eus_ << "\n";
    oss << "  SLM Size: " << (slm_size_ / 1024) << " KB\n";
    oss << "  Tiles: " << num_tiles_ << "\n";
    oss << "  Total candidates: " << total_candidates_ << "\n";
    oss << "  Evaluated: " << evaluated_candidates_ << "\n";
    oss << "  Best score: " << best_score_ << "\n";
    return oss.str();
}

}  // namespace search
}  // namespace yirage

#endif  // YIRAGE_BACKEND_XPU_ENABLED
