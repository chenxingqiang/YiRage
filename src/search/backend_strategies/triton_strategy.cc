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

#include "yirage/search/backend_strategies/triton_strategy.h"

#ifdef YIRAGE_BACKEND_TRITON_ENABLED

#include "yirage/kernel/graph.h"
#include <algorithm>
#include <sstream>

namespace yirage {
namespace search {

TritonSearchStrategy::TritonSearchStrategy(int compute_capability)
    : compute_capability_(compute_capability) {}

bool TritonSearchStrategy::initialize(SearchConfig const &config) {
  config_ = config;
  num_candidates_generated_ = 0;
  num_candidates_evaluated_ = 0;
  best_score_ = 0.0f;
  return true;
}

std::vector<CandidateConfig>
TritonSearchStrategy::generate_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;

  int m = 1024, n = 1024, k = 1024; // TODO: Extract from graph

  // Generate block size configurations
  auto block_configs = generate_block_size_configs(m, n, k);

  // Generate warp configurations
  auto warp_configs = generate_warp_configs();

  // Generate stage configurations
  auto stage_configs = generate_stage_configs();

  // Combine configurations
  for (auto const &[block_m, block_n, block_k] : block_configs) {
    for (int num_warps : warp_configs) {
      for (int num_stages : stage_configs) {
        auto config = std::make_unique<kernel::triton::TritonKernelConfig>();

        config->block_size_m = block_m;
        config->block_size_n = block_n;
        config->block_size_k = block_k;
        config->num_warps = num_warps;
        config->num_stages = num_stages;

        // Triton-specific optimizations
        kernel::triton::TritonOptimizer::compute_optimal_blocks(
            m, n, k, compute_capability_, *config);

        if (is_valid_config(*config)) {
          candidates.emplace_back(std::move(config));
        }
      }
    }
  }

  num_candidates_generated_ += candidates.size();
  return candidates;
}

float TritonSearchStrategy::evaluate_candidate(CandidateConfig &candidate,
                                               kernel::Graph const &graph) {
  auto *triton_config =
      static_cast<kernel::triton::TritonKernelConfig *>(candidate.config.get());

  // Triton's auto-tuning handles most optimization
  // Score based on configuration sanity
  float block_score = evaluate_block_efficiency(*triton_config);

  candidate.score = block_score;
  num_candidates_evaluated_++;

  if (block_score > best_score_) {
    best_score_ = block_score;
  }

  return block_score;
}

kernel::KernelConfig *TritonSearchStrategy::select_best_config(
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
TritonSearchStrategy::optimize(kernel::Graph const &graph) {
  auto candidates = generate_candidates(graph);

  for (auto &candidate : candidates) {
    evaluate_candidate(candidate, graph);
  }

  auto *best = select_best_config(candidates);
  if (!best) {
    return nullptr;
  }

  return std::make_unique<kernel::triton::TritonKernelConfig>(
      *static_cast<kernel::triton::TritonKernelConfig *>(best));
}

std::string TritonSearchStrategy::get_statistics() const {
  std::ostringstream oss;
  oss << "Triton Search Statistics:\n";
  oss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  oss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  oss << "  Best score: " << best_score_ << "\n";
  return oss.str();
}

std::vector<std::tuple<int, int, int>>
TritonSearchStrategy::generate_block_size_configs(int m, int n, int k) {
  std::vector<std::tuple<int, int, int>> configs;

  // Common Triton block sizes
  std::vector<std::tuple<int, int, int>> sizes = {
      {32, 32, 32},
      {64, 64, 32},
      {128, 128, 32},
      {128, 256, 64},
      {256, 128, 64}
  };

  for (auto const &size : sizes) {
    configs.push_back(size);
  }

  return configs;
}

std::vector<int> TritonSearchStrategy::generate_warp_configs() {
  return {2, 4, 8};
}

std::vector<int> TritonSearchStrategy::generate_stage_configs() {
  return {2, 3, 4};
}

float TritonSearchStrategy::evaluate_block_efficiency(
    kernel::triton::TritonKernelConfig const &config) {
  // Score based on block size balance
  int total_elements = config.block_size_m * config.block_size_n;
  int threads = config.num_warps * 32;
  
  float elements_per_thread = static_cast<float>(total_elements) / threads;
  
  // Optimal range: 4-16 elements per thread
  float score = 1.0f;
  if (elements_per_thread < 4.0f) {
    score = elements_per_thread / 4.0f;
  } else if (elements_per_thread > 16.0f) {
    score = 16.0f / elements_per_thread;
  }
  
  return score;
}

bool TritonSearchStrategy::is_valid_config(
    kernel::triton::TritonKernelConfig const &config) {
  // Check block sizes are positive
  if (config.block_size_m <= 0 || config.block_size_n <= 0 ||
      config.block_size_k <= 0) {
    return false;
  }

  // Check warp count is valid
  if (config.num_warps < 1 || config.num_warps > 8) {
    return false;
  }

  return true;
}

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_TRITON_ENABLED





