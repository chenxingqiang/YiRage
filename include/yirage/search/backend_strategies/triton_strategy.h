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
 * This file is part of YiRage (Yi Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */


#pragma once

#include "yirage/kernel/triton/triton_kernel_config.h"
#include "yirage/search/common/search_strategy.h"

#ifdef YIRAGE_BACKEND_TRITON_ENABLED

namespace yirage {
namespace search {

/**
 * @brief Triton compiler-based search strategy
 * 
 * Optimizes by exploring Triton-specific configurations:
 * - Block sizes
 * - Number of warps
 * - Software pipelining stages
 * - Auto-tuning integration
 */
class TritonSearchStrategy : public SearchStrategy {
public:
  explicit TritonSearchStrategy(int compute_capability = 80);

  bool initialize(SearchConfig const &config) override;

  std::vector<CandidateConfig>
  generate_candidates(kernel::Graph const &graph) override;

  float evaluate_candidate(CandidateConfig &candidate,
                          kernel::Graph const &graph) override;

  kernel::KernelConfig *
  select_best_config(std::vector<CandidateConfig> &candidates) override;

  std::unique_ptr<kernel::KernelConfig>
  optimize(kernel::Graph const &graph) override;

  type::BackendType get_backend_type() const override {
    return type::BT_TRITON;
  }

  std::string get_statistics() const override;

private:
  int compute_capability_;

  // Generate block size candidates
  std::vector<std::tuple<int, int, int>>
  generate_block_size_configs(int m, int n, int k);

  // Generate warp count candidates
  std::vector<int> generate_warp_configs();

  // Generate stage count candidates
  std::vector<int> generate_stage_configs();

  // Evaluate Triton-specific metrics
  float evaluate_block_efficiency(
      kernel::triton::TritonKernelConfig const &config);

  bool is_valid_config(kernel::triton::TritonKernelConfig const &config);
};

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_TRITON_ENABLED





