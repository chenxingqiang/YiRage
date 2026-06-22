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


#include "search/backend_strategies/cpu_strategy.h"

#ifdef YIRAGE_BACKEND_CPU_ENABLED

#include "kernel/graph.h"
#include <algorithm>
#include <sstream>
#include <thread>

namespace yirage {
namespace search {

CPUSearchStrategy::CPUSearchStrategy(int num_cores)
    : num_cores_(num_cores > 0 ? num_cores
                                : std::thread::hardware_concurrency()) {
  simd_type_ = kernel::cpu::CPUOptimizer::detect_simd_support();
}

bool CPUSearchStrategy::initialize(SearchConfig const &config) {
  config_ = config;
  num_candidates_generated_ = 0;
  num_candidates_evaluated_ = 0;
  best_score_ = 0.0f;
  return true;
}

std::vector<CandidateConfig>
CPUSearchStrategy::generate_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;

  // Extract problem dimensions from graph operators.
  // For MoE graphs the relevant GEMM is the expert projection
  // (tokens_per_expert × hidden_dim × intermediate_size), so we look for
  // the largest GEMM in the graph.
  int m = 1024, n = 1024, k = 1024;
  int num_experts = 1;   // detected from 3-D weight tensors
  int expert_tokens = 8; // typical tokens-per-expert for token-batching search

  if (!graph.operators.empty()) {
    for (auto const *op : graph.operators) {
      if (op->input_tensors.size() >= 2) {
        auto const &lhs = op->input_tensors[0];
        auto const &rhs = op->input_tensors[1];
        if (lhs.num_dims >= 2 && rhs.num_dims >= 2) {
          int cand_m = lhs.dim[lhs.num_dims - 2];
          int cand_k = lhs.dim[lhs.num_dims - 1];
          int cand_n = rhs.dim[rhs.num_dims - 1];
          // Keep the largest GEMM dimensions
          if (static_cast<size_t>(cand_m) * cand_n * cand_k >
              static_cast<size_t>(m) * n * k) {
            m = cand_m; k = cand_k; n = cand_n;
          }
          // Detect MoE weight tensors: 3-D means [num_experts, rows, cols]
          if (rhs.num_dims == 3) {
            num_experts = std::max(num_experts, (int)rhs.dim[0]);
          }
        }
      }
    }
  }

  // For MoE, explore expert-batch sizes that divide the token dimension.
  // Batching tokens for the same expert improves cache locality.
  std::vector<int> expert_batch_sizes = {1, 4, 8, 16};
  if (expert_tokens > 1) {
    expert_batch_sizes.push_back(expert_tokens);
  }

  // Generate tile configurations (also covers MoE expert-GEMM tile sizes)
  auto tile_configs = generate_tile_configs(m, n, k);

  // For MoE, also explore intermediate-size-aligned tiles
  if (num_experts > 1) {
    for (int bs : expert_batch_sizes) {
      // Tile_m = expert_batch_size; tile_n/k from standard candidates
      for (int tile : {64, 128, 256}) {
        tile_configs.emplace_back(bs, std::min(n, tile), std::min(k, tile));
      }
    }
  }

  // Generate thread configurations
  size_t problem_size = static_cast<size_t>(m) * n * k;
  auto thread_configs = generate_thread_configs(problem_size);

  // Generate SIMD configurations
  auto simd_configs = generate_simd_configs();

  // Combine configurations
  for (auto const &[tile_m, tile_n, tile_k] : tile_configs) {
    for (int num_threads : thread_configs) {
      for (auto simd : simd_configs) {
        auto config = std::make_unique<kernel::cpu::CPUKernelConfig>();

        config->tile_m = tile_m;
        config->tile_n = tile_n;
        config->tile_k = tile_k;
        config->num_threads = num_threads;
        config->simd_type = simd;

        // Compute micro-tiles based on L1 cache
        int micro_tile = std::max(4, tile_m / 8);
        config->micro_tile_m = micro_tile;
        config->micro_tile_n = micro_tile;

        // Set other parameters
        config->use_openmp = (num_threads > 1);
        config->use_prefetch = true;
        config->unroll_factor = 
            kernel::cpu::CPUOptimizer::compute_unroll_factor(k, simd);

        if (is_valid_config(*config)) {
          candidates.emplace_back(std::move(config));
        }
      }
    }
  }

  num_candidates_generated_ += candidates.size();
  return candidates;
}

float CPUSearchStrategy::evaluate_candidate(CandidateConfig &candidate,
                                            kernel::Graph const &graph) {
  auto *cpu_config =
      static_cast<kernel::cpu::CPUKernelConfig *>(candidate.config.get());

  // Get problem dimensions
  int m = 1024, n = 1024, k = 1024;
  size_t problem_size = static_cast<size_t>(m) * n * k;

  // Compute score based on multiple metrics
  float cache_score = evaluate_cache_efficiency(*cpu_config, m, n, k);
  float vectorization_score =
      evaluate_vectorization_efficiency(*cpu_config, problem_size);
  float load_balance_score =
      evaluate_load_balance(*cpu_config, problem_size);

  // Weighted combination
  float score = 0.4f * cache_score + 
                0.3f * vectorization_score +
                0.3f * load_balance_score;

  candidate.score = score;
  num_candidates_evaluated_++;

  if (score > best_score_) {
    best_score_ = score;
  }

  return score;
}

kernel::KernelConfig *CPUSearchStrategy::select_best_config(
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
CPUSearchStrategy::optimize(kernel::Graph const &graph) {
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
  return std::make_unique<kernel::cpu::CPUKernelConfig>(
      *static_cast<kernel::cpu::CPUKernelConfig *>(best));
}

std::string CPUSearchStrategy::get_statistics() const {
  std::ostringstream oss;
  oss << "CPU Search Statistics:\n";
  oss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  oss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  oss << "  Best score: " << best_score_ << "\n";
  oss << "  CPU cores: " << num_cores_ << "\n";
  oss << "  SIMD: " << kernel::cpu::simd_type_to_string(simd_type_) << "\n";
  return oss.str();
}

// Private helper methods

std::vector<std::tuple<int, int, int>>
CPUSearchStrategy::generate_tile_configs(int m, int n, int k) {
  std::vector<std::tuple<int, int, int>> configs;

  // Try different tile sizes optimized for different cache levels
  std::vector<int> tile_sizes = {32, 64, 128, 256};

  for (int tile : tile_sizes) {
    configs.emplace_back(
        std::min(m, tile),
        std::min(n, tile),
        std::min(k, tile));
  }

  return configs;
}

std::vector<int> CPUSearchStrategy::generate_thread_configs(
    size_t problem_size) {
  std::vector<int> configs;

  // Try power-of-2 thread counts up to num_cores
  for (int threads = 1; threads <= num_cores_; threads *= 2) {
    configs.push_back(threads);
  }

  // Also try num_cores itself if not power of 2
  if (num_cores_ > 0 &&
      (num_cores_ & (num_cores_ - 1)) != 0) { // Not power of 2
    configs.push_back(num_cores_);
  }

  return configs;
}

std::vector<kernel::cpu::SIMDType> CPUSearchStrategy::generate_simd_configs() {
  std::vector<kernel::cpu::SIMDType> configs;

  // Use detected SIMD type
  configs.push_back(simd_type_);

  // Also try scalar version for comparison
  configs.push_back(kernel::cpu::SIMDType::NONE);

  return configs;
}

float CPUSearchStrategy::evaluate_cache_efficiency(
    kernel::cpu::CPUKernelConfig const &config,
    int m, int n, int k) {
  // Use optimizer to estimate cache efficiency
  return kernel::cpu::CPUOptimizer::estimate_cache_efficiency(
      config, m, n, k, sizeof(float));
}

float CPUSearchStrategy::evaluate_vectorization_efficiency(
    kernel::cpu::CPUKernelConfig const &config,
    size_t problem_size) {
  // Use optimizer to estimate vectorization efficiency
  return kernel::cpu::CPUOptimizer::estimate_vectorization_efficiency(
      config, problem_size);
}

float CPUSearchStrategy::evaluate_load_balance(
    kernel::cpu::CPUKernelConfig const &config,
    size_t problem_size) {
  if (config.num_threads <= 1) {
    return 1.0f; // Perfect balance with single thread
  }

  // Check if work is evenly divisible
  size_t work_per_thread = problem_size / config.num_threads;
  size_t remainder = problem_size % config.num_threads;

  // Compute imbalance ratio
  float imbalance = static_cast<float>(remainder) / problem_size;

  return 1.0f - imbalance;
}

bool CPUSearchStrategy::is_valid_config(
    kernel::cpu::CPUKernelConfig const &config) {
  // Check basic constraints
  if (config.tile_m <= 0 || config.tile_n <= 0 || config.tile_k <= 0) {
    return false;
  }

  if (config.num_threads < 1 || config.num_threads > num_cores_ * 2) {
    return false;
  }

  return true;
}

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_CPU_ENABLED





