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
 * Intel XPU Search Strategy
 * 
 * Search strategy for Intel Data Center GPU Max (Ponte Vecchio) and Arc.
 * Uses oneAPI/SYCL programming model.
 * Key features:
 * - XMX (Xe Matrix eXtensions)
 * - DPAS (Dot Product Accumulate Systolic)
 * - SLM (Shared Local Memory) optimization
 */

#pragma once

#include "search/common/search_strategy.h"
#include "kernel/xpu/xpu_kernel_config.h"

#ifdef YIRAGE_BACKEND_XPU_ENABLED

namespace yirage {
namespace search {

/**
 * @brief Search strategy for Intel XPU (Data Center GPU Max)
 */
class XPUSearchStrategy : public SearchStrategy {
public:
    XPUSearchStrategy();
    explicit XPUSearchStrategy(kernel::xpu::XPUArch arch);
    
    bool initialize(SearchConfig const& config) override;
    
    std::vector<CandidateConfig> generate_candidates(
        kernel::Graph const& graph) override;
    
    float evaluate_candidate(CandidateConfig& candidate,
                            kernel::Graph const& graph) override;
    
    kernel::KernelConfig* select_best_config(
        std::vector<CandidateConfig>& candidates) override;
    
    std::unique_ptr<kernel::KernelConfig> optimize(
        kernel::Graph const& graph) override;
    
    std::string get_statistics() const override;

private:
    kernel::xpu::XPUArch arch_;
    int total_eus_;
    size_t slm_size_;
    int num_tiles_;
    
    int total_candidates_ = 0;
    int evaluated_candidates_ = 0;
    float best_score_ = 0.0f;
    
    std::vector<kernel::xpu::XPUKernelConfig> generate_xmx_configs(int m, int n, int k);
    float evaluate_xmx_utilization(kernel::xpu::XPUKernelConfig const& config);
    float evaluate_slm_efficiency(kernel::xpu::XPUKernelConfig const& config);
    float evaluate_occupancy(kernel::xpu::XPUKernelConfig const& config);
    bool is_valid_config(kernel::xpu::XPUKernelConfig const& config);
    void configure_for_arch();
};

}  // namespace search
}  // namespace yirage

#endif  // YIRAGE_BACKEND_XPU_ENABLED
