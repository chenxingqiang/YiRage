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
 * FPGA Search Strategy Implementation
 */

#include "search/backend_strategies/fpga_strategy.h"

#ifdef YIRAGE_BACKEND_FPGA_ENABLED

#include <algorithm>
#include <cmath>
#include <sstream>

namespace yirage {
namespace search {

using namespace kernel::fpga;

// =============================================================================
// Constructor and Initialization
// =============================================================================

FPGASearchStrategy::FPGASearchStrategy()
    : device_(FPGADevice::ALVEO_U280), vendor_(FPGAVendor::XILINX) {
    configure_for_device();
}

FPGASearchStrategy::FPGASearchStrategy(FPGADevice device)
    : device_(device) {
    configure_for_device();
}

void FPGASearchStrategy::configure_for_device() {
    switch (device_) {
        case FPGADevice::ALVEO_U200:
            vendor_ = FPGAVendor::XILINX;
            dsp_slices_ = 6840;
            bram_kb_ = 2160;
            target_freq_mhz_ = 300;
            break;
        case FPGADevice::ALVEO_U250:
            vendor_ = FPGAVendor::XILINX;
            dsp_slices_ = 12288;
            bram_kb_ = 5376;
            target_freq_mhz_ = 300;
            break;
        case FPGADevice::ALVEO_U280:
            vendor_ = FPGAVendor::XILINX;
            dsp_slices_ = 9024;
            bram_kb_ = 4032;
            target_freq_mhz_ = 300;
            break;
        case FPGADevice::AGILEX_F:
        case FPGADevice::STRATIX_10:
            vendor_ = FPGAVendor::INTEL;
            dsp_slices_ = 5760;
            bram_kb_ = 2880;
            target_freq_mhz_ = 400;
            break;
        case FPGADevice::AWS_F1:
            vendor_ = FPGAVendor::XILINX;
            dsp_slices_ = 6840;
            bram_kb_ = 2160;
            target_freq_mhz_ = 250;
            break;
        default:
            dsp_slices_ = 4096;
            bram_kb_ = 2048;
            target_freq_mhz_ = 250;
    }
}

bool FPGASearchStrategy::initialize(SearchConfig const& config) {
    return true;
}

// =============================================================================
// Candidate Generation
// =============================================================================

std::vector<CandidateConfig>
FPGASearchStrategy::generate_candidates(kernel::Graph const& graph) {
    std::vector<CandidateConfig> candidates;
    
    int m = 1024, n = 1024, k = 1024;
    
    auto hls_configs = generate_hls_configs(m, n, k);
    
    for (auto const& cfg : hls_configs) {
        CandidateConfig candidate;
        candidate.config = std::make_unique<FPGAKernelConfig>(cfg);
        candidate.score = 0.0f;
        candidates.push_back(std::move(candidate));
    }
    
    total_candidates_ = candidates.size();
    return candidates;
}

std::vector<FPGAKernelConfig>
FPGASearchStrategy::generate_hls_configs(int m, int n, int k) {
    std::vector<FPGAKernelConfig> configs;
    
    // Parallelism factors
    std::vector<int> parallel_ops = {4, 8, 16, 32};
    
    // Compute unit configurations
    std::vector<int> compute_units = {1, 2, 4};
    
    // Pipeline II (initiation interval)
    std::vector<int> pipeline_iis = {1, 2, 4};
    
    for (int num_cu : compute_units) {
        for (int num_parallel : parallel_ops) {
            for (int ii : pipeline_iis) {
                FPGAKernelConfig config;
                config.vendor = vendor_;
                config.device = device_;
                config.target_frequency_mhz = target_freq_mhz_;
                config.num_compute_units = num_cu;
                config.num_parallel_ops = num_parallel;
                config.hls.pipeline_ii = ii;
                config.hls.enable_dataflow = true;
                config.hls.array_partition_factor = num_parallel;
                
                // Estimate DSP usage
                int dsp_per_cu = num_parallel * 2;  // Rough estimate
                if (dsp_per_cu * num_cu <= dsp_slices_ * 0.8) {
                    // Precision options
                    for (bool use_int8 : {false, true}) {
                        config.use_int8 = use_int8;
                        config.use_half = !use_int8;
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

float FPGASearchStrategy::evaluate_candidate(CandidateConfig& candidate,
                                            kernel::Graph const& graph) {
    auto* config = static_cast<FPGAKernelConfig*>(candidate.config.get());
    
    float resource_util = evaluate_resource_utilization(*config);
    float throughput = evaluate_throughput(*config);
    float latency = evaluate_latency(*config);
    
    float score = 0.3f * resource_util + 0.4f * throughput + 0.3f * (1.0f - latency);
    
    // INT8 bonus (FPGA excels at low precision)
    if (config->use_int8) {
        score *= 1.15f;
    }
    
    // Dataflow bonus
    if (config->hls.enable_dataflow) {
        score *= 1.1f;
    }
    
    candidate.score = std::min(1.0f, score);
    evaluated_candidates_++;
    
    if (candidate.score > best_score_) {
        best_score_ = candidate.score;
    }
    
    return candidate.score;
}

float FPGASearchStrategy::evaluate_resource_utilization(FPGAKernelConfig const& config) {
    // Estimate DSP utilization
    int dsp_per_cu = config.num_parallel_ops * (config.use_int8 ? 1 : 2);
    float dsp_util = static_cast<float>(dsp_per_cu * config.num_compute_units) / dsp_slices_;
    
    // Optimal: 60-80% utilization
    if (dsp_util < 0.4f) return 0.5f + dsp_util;
    if (dsp_util > 0.85f) return 1.0f - (dsp_util - 0.85f) * 3.0f;
    return 0.8f + 0.2f * (dsp_util - 0.4f) / 0.45f;
}

float FPGASearchStrategy::evaluate_throughput(FPGAKernelConfig const& config) {
    // Throughput = CUs * parallelism * frequency / II
    float ops_per_cycle = config.num_compute_units * config.num_parallel_ops;
    float effective_freq = config.target_frequency_mhz / config.hls.pipeline_ii;
    float gops = ops_per_cycle * effective_freq / 1000.0f;
    
    // Normalize (assume 1000 GOPS is max)
    return std::min(1.0f, gops / 1000.0f);
}

float FPGASearchStrategy::evaluate_latency(FPGAKernelConfig const& config) {
    // Lower II = lower latency
    float base = config.hls.pipeline_ii / 4.0f;
    
    // More CUs = higher latency due to coordination
    base += 0.1f * (config.num_compute_units - 1);
    
    return std::min(1.0f, base);
}

bool FPGASearchStrategy::is_valid_config(FPGAKernelConfig const& config) {
    if (config.num_compute_units <= 0 || config.num_parallel_ops <= 0) {
        return false;
    }
    
    int dsp_needed = config.num_compute_units * config.num_parallel_ops * 2;
    return dsp_needed <= dsp_slices_;
}

// =============================================================================
// Configuration Selection
// =============================================================================

kernel::KernelConfig* FPGASearchStrategy::select_best_config(
    std::vector<CandidateConfig>& candidates) {
    
    if (candidates.empty()) return nullptr;
    
    auto best_it = std::max_element(candidates.begin(), candidates.end(),
        [](CandidateConfig const& a, CandidateConfig const& b) {
            return a.score < b.score;
        });
    
    return best_it->config.get();
}

std::unique_ptr<kernel::KernelConfig>
FPGASearchStrategy::optimize(kernel::Graph const& graph) {
    auto candidates = generate_candidates(graph);
    
    for (auto& candidate : candidates) {
        evaluate_candidate(candidate, graph);
    }
    
    auto* best = select_best_config(candidates);
    if (!best) {
        return std::make_unique<FPGAKernelConfig>();
    }
    
    return std::make_unique<FPGAKernelConfig>(
        *static_cast<FPGAKernelConfig*>(best));
}

std::string FPGASearchStrategy::get_statistics() const {
    std::ostringstream oss;
    oss << "FPGA Search Statistics:\n";
    oss << "  Device: " << static_cast<int>(device_) << "\n";
    oss << "  Vendor: " << (vendor_ == FPGAVendor::XILINX ? "Xilinx" : "Intel") << "\n";
    oss << "  DSP Slices: " << dsp_slices_ << "\n";
    oss << "  Target Freq: " << target_freq_mhz_ << " MHz\n";
    oss << "  Total candidates: " << total_candidates_ << "\n";
    oss << "  Evaluated: " << evaluated_candidates_ << "\n";
    oss << "  Best score: " << best_score_ << "\n";
    return oss.str();
}

}  // namespace search
}  // namespace yirage

#endif  // YIRAGE_BACKEND_FPGA_ENABLED
