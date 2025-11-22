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

#include "yirage/search/backend_strategies/mps_strategy.h"

#ifdef YIRAGE_BACKEND_MPS_ENABLED

#include "yirage/kernel/graph.h"
#include <algorithm>
#include <sstream>

namespace yirage {
namespace search {

MPSSearchStrategy::MPSSearchStrategy(int gpu_family)
    : gpu_family_(gpu_family > 0 ? gpu_family
                                  : kernel::mps::MPSOptimizer::detect_gpu_family()),
      gpu_cores_(kernel::mps::MPSOptimizer::get_gpu_core_count()) {}

bool MPSSearchStrategy::initialize(SearchConfig const &config) {
  config_ = config;
  num_candidates_generated_ = 0;
  num_candidates_evaluated_ = 0;
  best_score_ = 0.0f;
  return true;
}

std::vector<CandidateConfig>
MPSSearchStrategy::generate_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;

  // Get problem dimensions
  int m = 1024, n = 1024, k = 1024; // TODO: Extract from graph

  // Generate threadgroup configurations
  size_t problem_size = static_cast<size_t>(m) * n;
  auto threadgroup_configs = generate_threadgroup_configs(problem_size);

  // Generate tile configurations
  auto tile_configs = generate_tile_configs(m, n, k);

  // Generate memory patterns
  auto memory_patterns = generate_memory_patterns();

  // Combine configurations
  for (int tg_size : threadgroup_configs) {
    for (auto const &[tile_m, tile_n, tile_k] : tile_configs) {
      for (auto pattern : memory_patterns) {
        auto config = std::make_unique<kernel::mps::MPSKernelConfig>();

        config->threads_per_threadgroup = tg_size;
        config->tile_m = tile_m;
        config->tile_n = tile_n;
        config->tile_k = tile_k;
        config->access_pattern = pattern;
        config->gpu_family = gpu_family_;

        // Set grid dimensions
        config->grid_dim_x = (n + tile_n - 1) / tile_n;
        config->grid_dim_y = (m + tile_m - 1) / tile_m;
        config->grid_dim_z = 1;

        if (is_valid_config(*config)) {
          candidates.emplace_back(std::move(config));
        }
      }
    }
  }

  num_candidates_generated_ += candidates.size();
  return candidates;
}

float MPSSearchStrategy::evaluate_candidate(CandidateConfig &candidate,
                                            kernel::Graph const &graph) {
  auto *mps_config =
      static_cast<kernel::mps::MPSKernelConfig *>(candidate.config.get());

  // Compute score based on multiple metrics
  float gpu_util_score = evaluate_gpu_utilization(*mps_config);
  float memory_score = evaluate_memory_efficiency(*mps_config);
  float tg_memory_score = evaluate_threadgroup_memory(*mps_config);

  // Weighted combination
  float score = 0.4f * gpu_util_score + 
                0.3f * memory_score +
                0.3f * tg_memory_score;

  candidate.score = score;
  num_candidates_evaluated_++;

  if (score > best_score_) {
    best_score_ = score;
  }

  return score;
}

kernel::KernelConfig *MPSSearchStrategy::select_best_config(
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
MPSSearchStrategy::optimize(kernel::Graph const &graph) {
  // Generate candidates
  auto candidates = generate_candidates(graph);

  // Evaluate all candidates
  for (auto &candidate : candidates) {
    evaluate_candidate(candidate, graph);
  }

  // Select best
  auto *best = select_best_config(candidates);
  if (!best) {
    return nullptr;
  }

  // Return copy of best configuration
  return std::make_unique<kernel::mps::MPSKernelConfig>(
      *static_cast<kernel::mps::MPSKernelConfig *>(best));
}

std::string MPSSearchStrategy::get_statistics() const {
  std::ostringstream oss;
  oss << "MPS Search Statistics:\n";
  oss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  oss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  oss << "  Best score: " << best_score_ << "\n";
  oss << "  GPU family: " << gpu_family_ << "\n";
  oss << "  GPU cores: " << gpu_cores_ << "\n";
  return oss.str();
}

// Private helper methods

std::vector<int> MPSSearchStrategy::generate_threadgroup_configs(
    size_t problem_size) {
  std::vector<int> configs;

  // Try different threadgroup sizes (multiples of SIMD width)
  int simd_width = 32;
  for (int mult = 4; mult <= 32; mult *= 2) {
    int size = simd_width * mult;
    if (size <= 1024) { // Metal limit
      configs.push_back(size);
    }
  }

  return configs;
}

std::vector<std::tuple<int, int, int>>
MPSSearchStrategy::generate_tile_configs(int m, int n, int k) {
  std::vector<std::tuple<int, int, int>> configs;

  // Try different tile sizes optimized for threadgroup memory
  std::vector<int> tile_sizes = {16, 32, 48, 64};

  for (int tile : tile_sizes) {
    configs.emplace_back(
        std::min(m, tile),
        std::min(n, tile),
        std::min(k, tile));
  }

  return configs;
}

std::vector<kernel::mps::MemoryPattern>
MPSSearchStrategy::generate_memory_patterns() {
  std::vector<kernel::mps::MemoryPattern> patterns;

  // Try different memory access patterns
  patterns.push_back(kernel::mps::MemoryPattern::COALESCED);
  patterns.push_back(kernel::mps::MemoryPattern::TILED);

  return patterns;
}

float MPSSearchStrategy::evaluate_gpu_utilization(
    kernel::mps::MPSKernelConfig const &config) {
  // Estimate based on threadgroup size and GPU cores
  int total_threads = config.get_total_blocks() *
                     config.threads_per_threadgroup;
  
  // Assume each GPU core can handle ~1024 threads efficiently
  int ideal_threads = gpu_cores_ * 1024;
  
  float utilization = std::min(1.0f, 
      static_cast<float>(total_threads) / ideal_threads);
  
  return utilization;
}

float MPSSearchStrategy::evaluate_memory_efficiency(
    kernel::mps::MPSKernelConfig const &config) {
  // Score based on memory access pattern
  float pattern_score = 1.0f;
  
  switch (config.access_pattern) {
  case kernel::mps::MemoryPattern::COALESCED:
    pattern_score = 1.0f; // Best
    break;
  case kernel::mps::MemoryPattern::TILED:
    pattern_score = 0.85f; // Good
    break;
  case kernel::mps::MemoryPattern::STRIDED:
    pattern_score = 0.7f; // Acceptable
    break;
  }

  return pattern_score;
}

float MPSSearchStrategy::evaluate_threadgroup_memory(
    kernel::mps::MPSKernelConfig const &config) {
  // Check if tile sizes fit well in threadgroup memory
  size_t required_memory = (config.tile_m * config.tile_k +
                           config.tile_k * config.tile_n +
                           config.tile_m * config.tile_n) *
                          sizeof(float);

  float memory_ratio = static_cast<float>(required_memory) /
                      config.threadgroup_memory_size;

  // Penalize if over-allocated or under-utilized
  float score = 1.0f - std::abs(1.0f - memory_ratio);
  return std::max(0.0f, score);
}

bool MPSSearchStrategy::is_valid_config(
    kernel::mps::MPSKernelConfig const &config) {
  // Check threadgroup size limits
  if (config.threads_per_threadgroup < 32 ||
      config.threads_per_threadgroup > 1024) {
    return false;
  }

  // Check if threadgroup size is multiple of SIMD width
  if (config.threads_per_threadgroup % config.simd_width != 0) {
    return false;
  }

  // Check tile sizes
  if (config.tile_m <= 0 || config.tile_n <= 0 || config.tile_k <= 0) {
    return false;
  }

  return true;
}

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_MPS_ENABLED





