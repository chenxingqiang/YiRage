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

#include "yirage/kernel/nki/nki_kernel_config.h"
#include "yirage/search/common/search_strategy.h"

#ifdef YIRAGE_BACKEND_NKI_ENABLED

namespace yirage {
namespace search {

/**
 * @brief NKI (Neuron Kernel Interface) search strategy
 *
 * Optimizes for AWS Inferentia/Trainium by exploring:
 * - NeuronCore tile configurations
 * - SBUF (State Buffer) usage
 * - DMA scheduling
 * - Collective communication patterns
 */
class NKISearchStrategy : public SearchStrategy {
public:
  explicit NKISearchStrategy(int num_neuron_cores = 1);

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
    return type::BT_NKI;
  }

  std::string get_statistics() const override;

private:
  int num_neuron_cores_;

  // NKI-specific candidate generation
  std::vector<std::tuple<int, int, int>>
      generate_tile_configs(int m, int n, int k);

  std::vector<kernel::nki::NKIKernelConfig::ScheduleStrategy>
      generate_schedule_strategies();

  // NKI-specific evaluation
  float evaluate_sbuf_efficiency(kernel::nki::NKIKernelConfig const &config);

  float evaluate_dma_efficiency(kernel::nki::NKIKernelConfig const &config);

  bool is_valid_config(kernel::nki::NKIKernelConfig const &config);
};

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_NKI_ENABLED
