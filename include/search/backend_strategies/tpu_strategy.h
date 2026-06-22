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
 * TPU Search Strategy
 * 
 * Search strategy optimized for Google TPU architecture.
 * Key features:
 * - 128x128 MXU (Matrix Multiply Unit)
 * - Large VMEM (16-32MB per core)
 * - BF16/INT8 optimized
 * - XLA/Pallas kernel generation
 */

#pragma once

#include "search/common/search_strategy.h"
#include "kernel/tpu/tpu_kernel_config.h"

#ifdef YIRAGE_BACKEND_TPU_ENABLED

namespace yirage {
namespace search {

/**
 * @brief Search strategy for Google TPU
 */
class TPUSearchStrategy : public SearchStrategy {
public:
    TPUSearchStrategy();
    explicit TPUSearchStrategy(kernel::tpu::TPUVersion version);
    
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
    kernel::tpu::TPUVersion version_;
    int mxu_size_;
    size_t vmem_size_;
    int num_cores_;
    
    int total_candidates_ = 0;
    int evaluated_candidates_ = 0;
    float best_score_ = 0.0f;
    
    std::vector<kernel::tpu::TPUKernelConfig> generate_mxu_configs(int m, int n, int k);
    float evaluate_mxu_utilization(kernel::tpu::TPUKernelConfig const& config);
    float evaluate_vmem_efficiency(kernel::tpu::TPUKernelConfig const& config);
    float evaluate_pipeline_efficiency(kernel::tpu::TPUKernelConfig const& config);
    bool is_valid_config(kernel::tpu::TPUKernelConfig const& config);
};

}  // namespace search
}  // namespace yirage

#endif  // YIRAGE_BACKEND_TPU_ENABLED
