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

#include "yirage/kernel/common/kernel_interface.h"
#include "yirage/type.h"
#include <memory>
#include <vector>

namespace yirage {

// Forward declarations
namespace kernel {
class Graph;
}

namespace search {

/**
 * @brief Search configuration for kernel optimization
 */
struct SearchConfig {
  // General search parameters
  int max_iterations = 1000;
  float timeout_seconds = 600.0f;
  bool use_cache = true;
  int random_seed = 42;

  // Search strategy
  enum class Strategy {
    GREEDY,       // Greedy search
    BEAM,         // Beam search
    GENETIC,      // Genetic algorithm
    REINFORCEMENT // Reinforcement learning-based
  };
  Strategy strategy = Strategy::GREEDY;

  // Beam search parameters
  int beam_width = 10;

  // Genetic algorithm parameters
  int population_size = 50;
  float mutation_rate = 0.1f;
  float crossover_rate = 0.7f;

  // Performance sampling
  int num_warmup_iterations = 5;
  int num_profile_iterations = 10;

  virtual ~SearchConfig() = default;
};

/**
 * @brief Candidate kernel configuration with score
 */
struct CandidateConfig {
  std::unique_ptr<kernel::KernelConfig> config;
  float score = 0.0f; // Higher is better
  kernel::KernelMetrics metrics;

  CandidateConfig() = default;

  CandidateConfig(std::unique_ptr<kernel::KernelConfig> cfg, float s = 0.0f)
      : config(std::move(cfg)), score(s) {}
};

/**
 * @brief Abstract search strategy interface
 * 
 * Each backend implements this to provide hardware-specific
 * optimization strategies.
 */
class SearchStrategy {
public:
  virtual ~SearchStrategy() = default;

  /**
   * @brief Initialize search strategy
   * @param config Search configuration
   * @return true if initialization succeeded
   */
  virtual bool initialize(SearchConfig const &config) = 0;

  /**
   * @brief Generate candidate kernel configurations
   * @param graph Kernel graph to optimize
   * @return Vector of candidate configurations
   */
  virtual std::vector<CandidateConfig>
  generate_candidates(kernel::Graph const &graph) = 0;

  /**
   * @brief Evaluate a candidate configuration
   * @param candidate Candidate to evaluate
   * @param graph Kernel graph
   * @return Score (higher is better)
   */
  virtual float evaluate_candidate(CandidateConfig &candidate,
                                  kernel::Graph const &graph) = 0;

  /**
   * @brief Select the best configuration from candidates
   * @param candidates List of evaluated candidates
   * @return Pointer to best configuration (owned by candidate list)
   */
  virtual kernel::KernelConfig *
  select_best_config(std::vector<CandidateConfig> &candidates) = 0;

  /**
   * @brief Optimize kernel graph
   * @param graph Kernel graph to optimize
   * @return Best kernel configuration found
   */
  virtual std::unique_ptr<kernel::KernelConfig>
  optimize(kernel::Graph const &graph) = 0;

  /**
   * @brief Get backend type this strategy is for
   * @return Backend type
   */
  virtual type::BackendType get_backend_type() const = 0;

  /**
   * @brief Get search statistics
   * @return Statistics as string
   */
  virtual std::string get_statistics() const = 0;

protected:
  SearchConfig config_;
  int num_candidates_generated_ = 0;
  int num_candidates_evaluated_ = 0;
  float best_score_ = 0.0f;
};

/**
 * @brief Factory for creating backend-specific search strategies
 */
class SearchStrategyFactory {
public:
  /**
   * @brief Create a search strategy for a specific backend
   * @param backend Backend type
   * @param config Search configuration
   * @return Unique pointer to search strategy
   */
  static std::unique_ptr<SearchStrategy>
  create_strategy(type::BackendType backend, SearchConfig const &config);

  /**
   * @brief Check if a backend has a search strategy implemented
   * @param backend Backend type
   * @return true if strategy exists
   */
  static bool has_strategy(type::BackendType backend);
};

} // namespace search
} // namespace yirage





