/* Copyright 2023-2024 CMU
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

#include "yirage/search/backend_strategies/nki_strategy.h"

#ifdef YIRAGE_BACKEND_NKI_ENABLED

#include "yirage/kernel/graph.h"
#include <algorithm>
#include <sstream>

namespace yirage {
namespace search {

NKISearchStrategy::NKISearchStrategy(int num_neuron_cores)
    : num_neuron_cores_(num_neuron_cores) {}

bool NKISearchStrategy::initialize(SearchConfig const &config) {
  config_ = config;
  num_candidates_generated_ = 0;
  num_candidates_evaluated_ = 0;
  best_score_ = 0.0f;
  return true;
}

std::vector<CandidateConfig>
NKISearchStrategy::generate_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;

  int m = 1024, n = 1024, k = 1024;

  auto tile_configs = generate_tile_configs(m, n, k);
  auto schedule_strategies = generate_schedule_strategies();

  for (auto const &[tile_m, tile_n, tile_k] : tile_configs) {
    for (auto strategy : schedule_strategies) {
      auto config = std::make_unique<kernel::nki::NKIKernelConfig>();

      config->tile_m = tile_m;
      config->tile_n = tile_n;
      config->tile_k = tile_k;
      config->schedule_strategy = strategy;
      config->num_neuron_cores = num_neuron_cores_;

      // Optimize for Neuron
      kernel::nki::NKIOptimizer::optimize_for_neuron(m, n, k, *config);

      if (is_valid_config(*config)) {
        candidates.emplace_back(std::move(config));
      }
    }
  }

  num_candidates_generated_ += candidates.size();
  return candidates;
}

float NKISearchStrategy::evaluate_candidate(CandidateConfig &candidate,
                                            kernel::Graph const &graph) {
  auto *nki_config =
      static_cast<kernel::nki::NKIKernelConfig *>(candidate.config.get());

  float sbuf_score = evaluate_sbuf_efficiency(*nki_config);
  float dma_score = evaluate_dma_efficiency(*nki_config);

  float score = 0.5f * sbuf_score + 0.5f * dma_score;

  candidate.score = score;
  num_candidates_evaluated_++;

  if (score > best_score_) {
    best_score_ = score;
  }

  return score;
}

kernel::KernelConfig *NKISearchStrategy::select_best_config(
    std::vector<CandidateConfig> &candidates) {
  if (candidates.empty()) {
    return nullptr;
  }

  auto best_it = std::max_element(
      candidates.begin(), candidates.end(),
      [](CandidateConfig const &a, CandidateConfig const &b) {
        return a.score < b.score;
      });

  return best_it->config.get();
}

std::unique_ptr<kernel::KernelConfig>
NKISearchStrategy::optimize(kernel::Graph const &graph) {
  auto candidates = generate_candidates(graph);

  for (auto &candidate : candidates) {
    evaluate_candidate(candidate, graph);
  }

  auto *best = select_best_config(candidates);
  if (!best) {
    return nullptr;
  }

  return std::make_unique<kernel::nki::NKIKernelConfig>(
      *static_cast<kernel::nki::NKIKernelConfig *>(best));
}

std::string NKISearchStrategy::get_statistics() const {
  std::ostringstream oss;
  oss << "NKI Search Statistics:\n";
  oss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  oss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  oss << "  Best score: " << best_score_ << "\n";
  oss << "  NeuronCores: " << num_neuron_cores_ << "\n";
  return oss.str();
}

std::vector<std::tuple<int, int, int>>
NKISearchStrategy::generate_tile_configs(int m, int n, int k) {
  std::vector<std::tuple<int, int, int>> configs;

  // NeuronCore prefers specific tile sizes
  // K dimension typically 512 for best utilization
  std::vector<int> k_tiles = {256, 512, 1024};
  std::vector<int> m_n_tiles = {128, 256};

  for (int tk : k_tiles) {
    for (int tm : m_n_tiles) {
      for (int tn : m_n_tiles) {
        configs.emplace_back(
            std::min(m, tm),
            std::min(n, tn),
            std::min(k, tk));
      }
    }
  }

  return configs;
}

std::vector<kernel::nki::NKIKernelConfig::ScheduleStrategy>
NKISearchStrategy::generate_schedule_strategies() {
  return {
      kernel::nki::NKIKernelConfig::ScheduleStrategy::PIPELINED,
      kernel::nki::NKIKernelConfig::ScheduleStrategy::ASYNC_DMA
  };
}

float NKISearchStrategy::evaluate_sbuf_efficiency(
    kernel::nki::NKIKernelConfig const &config) {
  // Calculate required SBUF
  size_t required = kernel::nki::NKIOptimizer::optimize_sbuf_usage(
      config.tile_m, config.tile_n, config.tile_k);

  float ratio = static_cast<float>(required) / config.sbuf_size;

  // Penalize over-allocation or under-utilization
  return 1.0f - std::abs(1.0f - ratio);
}

float NKISearchStrategy::evaluate_dma_efficiency(
    kernel::nki::NKIKernelConfig const &config) {
  // Pipelined and async DMA are more efficient
  switch (config.schedule_strategy) {
  case kernel::nki::NKIKernelConfig::ScheduleStrategy::ASYNC_DMA:
    return 1.0f;
  case kernel::nki::NKIKernelConfig::ScheduleStrategy::PIPELINED:
    return 0.9f;
  default:
    return 0.7f;
  }
}

bool NKISearchStrategy::is_valid_config(
    kernel::nki::NKIKernelConfig const &config) {
  // Check tile sizes
  if (config.tile_m <= 0 || config.tile_n <= 0 || config.tile_k <= 0) {
    return false;
  }

  // Check SBUF size
  size_t required = kernel::nki::NKIOptimizer::optimize_sbuf_usage(
      config.tile_m, config.tile_n, config.tile_k);
  if (required > config.sbuf_size) {
    return false;
  }

  return true;
}

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_NKI_ENABLED





