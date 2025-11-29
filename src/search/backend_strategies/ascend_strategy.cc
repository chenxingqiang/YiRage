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
 */

#include "yirage/search/backend_strategies/ascend_strategy.h"

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED

#include "yirage/kernel/graph.h"
#include <algorithm>
#include <sstream>

namespace yirage {
namespace search {

AscendSearchStrategy::AscendSearchStrategy(int device_type)
    : device_type_(device_type > 0 ? device_type
                                   : kernel::ascend::AscendOptimizer::detect_device_type()) {
  ai_core_count_ = kernel::ascend::AscendOptimizer::get_ai_core_count();
  
  // L1 buffer size based on device type
  switch (device_type_) {
    case 0:  // Ascend 910
      l1_buffer_size_ = 256 * 1024;
      break;
    case 1:  // Ascend 910B
      l1_buffer_size_ = 512 * 1024;
      break;
    case 2:  // Ascend 310P
      l1_buffer_size_ = 128 * 1024;
      break;
    default:
      l1_buffer_size_ = 256 * 1024;
  }
}

bool AscendSearchStrategy::initialize(SearchConfig const &config) {
  config_ = config;
  num_candidates_generated_ = 0;
  num_candidates_evaluated_ = 0;
  best_score_ = 0.0f;
  return true;
}

std::vector<CandidateConfig>
AscendSearchStrategy::generate_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;

  // Get problem dimensions
  int m = 1024, n = 1024, k = 1024; // TODO: Extract from graph

  // Generate block configurations
  size_t problem_size = static_cast<size_t>(m) * n;
  auto block_configs = generate_block_configs(problem_size);

  // Generate tile configurations optimized for Cube operations
  auto tile_configs = generate_tile_configs(m, n, k);

  // Combine configurations
  for (int block_size : block_configs) {
    for (auto const &[tile_m, tile_n, tile_k] : tile_configs) {
      auto config = std::make_unique<kernel::ascend::AscendKernelConfig>();

      config->ai_cores_per_block = block_size;
      config->tile_m = tile_m;
      config->tile_n = tile_n;
      config->tile_k = tile_k;
      config->device_type = device_type_;
      config->l1_buffer_size = l1_buffer_size_;

      // Set grid dimensions
      config->blocks_per_grid_x = (n + tile_n - 1) / tile_n;
      config->blocks_per_grid_y = (m + tile_m - 1) / tile_m;

      // Enable optimizations based on problem characteristics
      config->use_cube_ops = (tile_m >= 16 && tile_n >= 16 && tile_k >= 16);
      config->use_vector_ops = true;
      config->enable_fusion = true;

      if (is_valid_config(*config)) {
        candidates.emplace_back(std::move(config));
      }
    }
  }

  num_candidates_generated_ += candidates.size();
  return candidates;
}

float AscendSearchStrategy::evaluate_candidate(CandidateConfig &candidate,
                                               kernel::Graph const &graph) {
  auto *ascend_config =
      static_cast<kernel::ascend::AscendKernelConfig *>(candidate.config.get());

  // Compute score based on multiple metrics
  float ai_core_score = evaluate_ai_core_utilization(*ascend_config);
  float l1_score = evaluate_l1_buffer_efficiency(*ascend_config);
  float cube_score = evaluate_cube_operation_fit(*ascend_config);

  // Weighted combination (similar to MPS strategy)
  float score = 0.4f * ai_core_score + 
                0.35f * l1_score +
                0.25f * cube_score;

  candidate.score = score;
  num_candidates_evaluated_++;

  if (score > best_score_) {
    best_score_ = score;
  }

  return score;
}

kernel::KernelConfig *AscendSearchStrategy::select_best_config(
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
AscendSearchStrategy::optimize(kernel::Graph const &graph) {
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
  return std::make_unique<kernel::ascend::AscendKernelConfig>(
      *static_cast<kernel::ascend::AscendKernelConfig *>(best));
}

std::string AscendSearchStrategy::get_statistics() const {
  std::ostringstream oss;
  oss << "Ascend Search Statistics:\n";
  oss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  oss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  oss << "  Best score: " << best_score_ << "\n";
  oss << "  Device type: " << device_type_ 
      << (device_type_ == 0 ? " (910)" : 
          device_type_ == 1 ? " (910B)" : " (310P)") << "\n";
  oss << "  AI Cores: " << ai_core_count_ << "\n";
  oss << "  L1 Buffer: " << (l1_buffer_size_ / 1024) << " KB\n";
  return oss.str();
}

// Private helper methods

std::vector<int> AscendSearchStrategy::generate_block_configs(
    size_t problem_size) {
  std::vector<int> configs;

  // Ascend AI Cores work best with power-of-2 block sizes
  // Try different block sizes based on problem size
  int max_cores = ai_core_count_;
  
  if (problem_size < 1024) {
    // Small problems: use fewer cores
    configs = {1, 2, 4};
  } else if (problem_size < 65536) {
    // Medium problems
    configs = {1, 2, 4, 8, 16};
  } else {
    // Large problems: try all reasonable block sizes
    configs = {1, 2, 4, 8, 16};
    if (max_cores >= 32) {
      configs.push_back(32);
    }
  }

  return configs;
}

std::vector<std::tuple<int, int, int>>
AscendSearchStrategy::generate_tile_configs(int m, int n, int k) {
  std::vector<std::tuple<int, int, int>> configs;
  
  // Ascend Cube unit works best with 16x multiples
  // Native Cube size is 16x16
  const size_t l1_buffer = l1_buffer_size_;
  const size_t dtype_size = 2;  // float16
  
  // Try different tile sizes (multiples of 16 for Cube)
  std::vector<int> tile_sizes = {16, 32, 48, 64};
  
  // Add larger tiles for 910B (more L1 buffer)
  if (device_type_ == 1) {  // 910B
    tile_sizes.push_back(96);
    tile_sizes.push_back(128);
  }
  
  for (int tm : tile_sizes) {
    if (tm > m) continue;
    
    for (int tn : tile_sizes) {
      if (tn > n) continue;
      
      for (int tk : tile_sizes) {
        if (tk > k) continue;
        
        // Calculate required L1 buffer
        // A: tm x tk, B: tk x tn (float16)
        // C: tm x tn (float32 accumulation for Cube)
        size_t memory_needed = (tm * tk + tk * tn) * dtype_size + 
                               tm * tn * sizeof(float);
        
        // Ensure fits in L1 buffer (80% utilization)
        if (memory_needed > l1_buffer * 0.8f) {
          continue;
        }
        
        // Prefer 16x multiples for Cube operations
        bool cube_aligned = (tm % 16 == 0) && (tn % 16 == 0) && (tk % 16 == 0);
        if (!cube_aligned && tm >= 32) {
          continue;  // Skip non-aligned large tiles
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
        std::min(m, 16),
        std::min(n, 16),
        std::min(k, 16));
  }
  
  // Remove duplicates
  std::sort(configs.begin(), configs.end());
  configs.erase(std::unique(configs.begin(), configs.end()), configs.end());
  
  return configs;
}

float AscendSearchStrategy::evaluate_ai_core_utilization(
    kernel::ascend::AscendKernelConfig const &config) {
  
  // Calculate total AI Cores used
  int total_cores = config.get_total_ai_cores();
  
  // Ideal: use all available AI Cores
  float utilization = std::min(1.0f,
      static_cast<float>(total_cores) / ai_core_count_);
  
  // Bonus for power-of-2 block sizes (better for Ascend scheduler)
  int block_size = config.ai_cores_per_block;
  if (block_size == 1 || block_size == 2 || block_size == 4 || 
      block_size == 8 || block_size == 16 || block_size == 32) {
    utilization *= 1.1f;
  }
  
  // Optimal block size: 4-16 AI Cores per block
  if (block_size >= 4 && block_size <= 16) {
    utilization *= 1.15f;
  }
  
  return std::min(1.3f, utilization);
}

float AscendSearchStrategy::evaluate_l1_buffer_efficiency(
    kernel::ascend::AscendKernelConfig const &config) {
  
  const size_t dtype_size = 2;  // float16
  
  // Calculate L1 buffer usage
  size_t a_size = config.tile_m * config.tile_k * dtype_size;
  size_t b_size = config.tile_k * config.tile_n * dtype_size;
  size_t c_size = config.tile_m * config.tile_n * sizeof(float);
  
  size_t total_needed = a_size + b_size + c_size;
  
  // Invalid if exceeds L1 buffer
  if (total_needed > l1_buffer_size_) {
    return 0.0f;
  }
  
  float utilization = static_cast<float>(total_needed) / l1_buffer_size_;
  
  // Optimal range: 60-85% (higher than MPS due to larger L1)
  float score = 1.0f;
  if (utilization >= 0.60f && utilization <= 0.85f) {
    score = 1.0f;  // Ideal range
  } else if (utilization > 0.85f && utilization <= 0.95f) {
    score = 0.85f - (utilization - 0.85f) * 3.0f;
  } else if (utilization < 0.60f && utilization >= 0.30f) {
    score = 0.6f + utilization * 0.7f;
  } else if (utilization < 0.30f) {
    score = 0.4f;  // Severe underutilization
  } else {
    score = 0.2f;  // Over 95% is risky
  }
  
  return std::min(1.0f, score);
}

float AscendSearchStrategy::evaluate_cube_operation_fit(
    kernel::ascend::AscendKernelConfig const &config) {
  
  if (!config.use_cube_ops) {
    return 0.7f;  // Vector-only operations
  }
  
  // Cube operations prefer 16x multiples
  bool m_aligned = (config.tile_m % 16 == 0);
  bool n_aligned = (config.tile_n % 16 == 0);
  bool k_aligned = (config.tile_k % 16 == 0);
  
  float score = 0.7f;  // Base score for Cube
  
  if (m_aligned && n_aligned && k_aligned) {
    score = 1.0f;  // Perfect alignment
  } else if (m_aligned && n_aligned) {
    score = 0.9f;  // Good alignment
  } else if (m_aligned || n_aligned) {
    score = 0.8f;  // Partial alignment
  }
  
  // Larger tiles benefit more from Cube
  size_t tile_size = config.tile_m * config.tile_n * config.tile_k;
  if (tile_size >= 16384) {  // 16K elements
    score *= 1.2f;
  } else if (tile_size >= 4096) {
    score *= 1.1f;
  }
  
  // 910B has improved Cube performance
  if (device_type_ == 1) {
    score *= 1.15f;
  }
  
  return std::min(1.3f, score);
}

bool AscendSearchStrategy::is_valid_config(
    kernel::ascend::AscendKernelConfig const &config) {
  
  // Check block size limits
  if (config.ai_cores_per_block < 1 || 
      config.ai_cores_per_block > ai_core_count_) {
    return false;
  }

  // Check tile sizes
  if (config.tile_m <= 0 || config.tile_n <= 0 || config.tile_k <= 0) {
    return false;
  }

  // Check L1 buffer usage
  size_t memory_needed = (config.tile_m * config.tile_k +
                         config.tile_k * config.tile_n) * 2 +
                        config.tile_m * config.tile_n * 4;
  
  if (memory_needed > l1_buffer_size_) {
    return false;
  }

  return true;
}

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED

