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
  int simd_width = 32;
  
  // Dynamically adjust range based on problem size
  int min_mult = (problem_size < 1024) ? 2 : 4;      // Small problems use smaller threadgroups
  int max_mult = (problem_size > 1048576) ? 32 : 20; // Large problems can use bigger ones
  
  // Generate all multiples of SIMD width in range
  for (int mult = min_mult; mult <= max_mult; mult++) {
    int size = simd_width * mult;
    if (size >= 32 && size <= 1024) { // Metal limits: 32-1024
      configs.push_back(size);
    }
  }
  
  // Add empirically good values if not already present
  std::vector<int> empirical = {64, 96, 128, 160, 192, 256, 320, 384, 512};
  for (int size : empirical) {
    if (size % simd_width == 0 && size >= 32 && size <= 1024) {
      // Add if not duplicate
      if (std::find(configs.begin(), configs.end(), size) == configs.end()) {
        configs.push_back(size);
      }
    }
  }
  
  // Sort and ensure we have at least some candidates
  std::sort(configs.begin(), configs.end());
  
  // Ensure we have reasonable candidates
  if (configs.empty()) {
    configs = {128, 256, 512}; // Fallback
  }
  
  return configs;  // Now returns 10-20 candidates instead of 4
}

std::vector<std::tuple<int, int, int>>
MPSSearchStrategy::generate_tile_configs(int m, int n, int k) {
  std::vector<std::tuple<int, int, int>> configs;
  
  // Threadgroup memory = 32KB
  const size_t tg_memory = 32 * 1024;
  const size_t dtype_size = 2;  // float16
  
  // Try different combinations of tile sizes
  // Adjust ranges based on problem dimensions
  std::vector<int> tile_m_sizes = {16, 32, 48, 64};
  std::vector<int> tile_n_sizes = {16, 32, 48, 64};
  std::vector<int> tile_k_sizes = {8, 16, 24, 32};
  
  // Add larger tiles for big matrices
  if (m >= 128) tile_m_sizes.push_back(96);
  if (m >= 192) tile_m_sizes.push_back(128);
  if (n >= 128) tile_n_sizes.push_back(96);
  if (n >= 192) tile_n_sizes.push_back(128);
  if (k >= 64) tile_k_sizes.push_back(48);
  if (k >= 96) tile_k_sizes.push_back(64);
  
  for (int tm : tile_m_sizes) {
    if (tm > m) continue;  // Skip if larger than dimension
    
    for (int tn : tile_n_sizes) {
      if (tn > n) continue;
      
      for (int tk : tile_k_sizes) {
        if (tk > k) continue;
        
        // Calculate required threadgroup memory
        // A: tm x tk, B: tk x tn (float16)
        // C: tm x tn (float32 accumulation)
        size_t memory_needed = (tm * tk + tk * tn) * dtype_size + 
                               tm * tn * sizeof(float);
        
        // Ensure fits in threadgroup memory (with 20% safety margin)
        if (memory_needed > tg_memory * 0.8f) {
          continue;
        }
        
        // Prefer balanced tile configurations
        float aspect_ratio = static_cast<float>(tm * tn) / (tk * tk);
        if (aspect_ratio < 0.25f || aspect_ratio > 4.0f) {
          continue;  // Skip very unbalanced tiles
        }
        
        configs.emplace_back(
            std::min(m, tm),
            std::min(n, tn),
            std::min(k, tk));
      }
    }
  }
  
  // Ensure at least one valid configuration
  if (configs.empty()) {
    configs.emplace_back(
        std::min(m, 32),
        std::min(n, 32),
        std::min(k, 16));
  }
  
  // Remove duplicates
  std::sort(configs.begin(), configs.end());
  configs.erase(std::unique(configs.begin(), configs.end()), configs.end());
  
  return configs;  // Now returns much more candidates
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
  // Use correct data type size (float16 = 2 bytes)
  const size_t dtype_size = 2;  // float16
  
  // Calculate actual memory needed for matrix tiles
  // Matrix A: tile_m x tile_k (float16)
  // Matrix B: tile_k x tile_n (float16)
  // Matrix C: tile_m x tile_n (float32 for accumulation)
  size_t memory_a = config.tile_m * config.tile_k * dtype_size;
  size_t memory_b = config.tile_k * config.tile_n * dtype_size;
  size_t memory_c = config.tile_m * config.tile_n * sizeof(float);
  
  // Reserve space for potential temporary variables
  size_t temp_memory = config.threads_per_threadgroup * sizeof(float);
  
  size_t total_required = memory_a + memory_b + memory_c + temp_memory;
  
  // Apple Silicon: 32KB threadgroup memory
  const size_t tg_memory = 32 * 1024;
  
  // Invalid if exceeds limit
  if (total_required > tg_memory) {
    return 0.0f;
  }
  
  // Calculate utilization
  float utilization = static_cast<float>(total_required) / tg_memory;
  
  // Optimal range: 60-80% (leave headroom but don't waste)
  float score = 1.0f;
  if (utilization >= 0.60f && utilization <= 0.80f) {
    score = 1.0f;  // Ideal range
  } else if (utilization > 0.80f && utilization <= 0.95f) {
    // Gradually penalize as we approach limit
    score = 0.9f - (utilization - 0.80f) * 2.0f;
  } else if (utilization < 0.60f && utilization >= 0.30f) {
    // Underutilization penalty
    score = 0.7f + utilization * 0.5f;
  } else if (utilization < 0.30f) {
    // Severe underutilization
    score = 0.5f;
  } else {
    // Over 95% is risky
    score = 0.3f;
  }
  
  // Bonus if tile configuration aligns well with threadgroup size
  int threads_needed = ((config.tile_m * config.tile_n + 31) / 32) * 32;
  if (threads_needed == config.threads_per_threadgroup) {
    score *= 1.1f;
  }
  
  return std::min(1.0f, score);
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





