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

#pragma once

#include "utils/json_utils.h"
#include "vector_types.h"

#include <chrono>
#include <string>
#include <vector>

namespace yirage {
namespace search {

/**
 * @brief Information about a candidate configuration explored during search
 *
 * This data is collected for each configuration evaluated,
 * enabling RL training from search trajectories.
 */
struct CandidateInfo {
  // Unique ID within the search
  int candidate_id = 0;

  // Configuration parameters
  dim3 grid_dim = {1, 1, 1};
  dim3 block_dim = {128, 1, 1};
  std::vector<int3> imaps;
  int3 omap = {0, 0, 0};
  int frange = 1;

  // Search context
  int search_depth = 0;
  int operator_count = 0;
  int kernel_level_ops = 0;
  int threadblock_level_ops = 0;
  int last_operator_type = 0; // Type of last added operator

  // Evaluation results
  bool verified = false;
  double fingerprint_time_ms = 0.0;
  double estimated_performance_ms = 0.0;
  std::string rejection_reason; // Why verification failed, if applicable

  // Timing
  double evaluation_time_ms = 0.0;

  json to_json() const;
  static CandidateInfo from_json(json const &j);
};

/**
 * @brief Aggregated feedback from a search run
 *
 * Contains all information needed to:
 * 1. Analyze search efficiency
 * 2. Train RL search policies
 * 3. Debug search issues
 */
struct SearchFeedback {
  // Partition information
  int partition_id = 0;
  int total_partitions = 1;

  // All candidates explored
  std::vector<CandidateInfo> candidates;

  // Valid graphs found (indices into candidates)
  std::vector<int> valid_candidate_ids;

  // Aggregate statistics
  int total_states_explored = 0;
  int valid_graphs_found = 0;
  int candidates_generated = 0;
  int candidates_verified = 0;
  int candidates_rejected = 0;

  // Timing
  double search_time_seconds = 0.0;
  double verification_time_seconds = 0.0;
  double generation_time_seconds = 0.0;

  // Search configuration used
  json search_config;

  // Best performance found
  double best_performance_ms = std::numeric_limits<double>::infinity();
  int best_candidate_id = -1;

  /**
   * @brief Add a candidate to the feedback
   */
  void add_candidate(CandidateInfo const &info);

  /**
   * @brief Mark a candidate as verified/valid
   */
  void mark_verified(int candidate_id, double performance_ms);

  /**
   * @brief Convert to JSON for serialization
   */
  json to_json() const;

  /**
   * @brief Create from JSON
   */
  static SearchFeedback from_json(json const &j);

  /**
   * @brief Merge feedback from multiple partitions
   */
  static SearchFeedback merge(std::vector<SearchFeedback> const &feedbacks);

  /**
   * @brief Get summary statistics as string
   */
  std::string get_summary() const;
};

/**
 * @brief Training sample extracted from search feedback
 *
 * Format suitable for RL training (state, action, reward, next_state, done)
 */
struct TrainingSample {
  // State features
  struct State {
    int search_depth;
    int operator_count;
    std::vector<int> grid_dim; // [x, y, z]
    std::vector<int> block_dim;
    int num_valid_found_so_far;
  } state;

  // Action taken
  struct Action {
    std::vector<std::vector<int>> imaps;
    std::vector<int> omap;
    int frange;
    int operator_type;
  } action;

  // Reward received
  double reward = 0.0;

  // Next state (null if terminal)
  bool has_next_state = false;
  State next_state;

  // Terminal flag
  bool done = false;

  json to_json() const;
};

/**
 * @brief Extract training samples from search feedback
 */
std::vector<TrainingSample>
extract_training_samples(SearchFeedback const &feedback,
                         double validity_reward = 1.0,
                         double invalid_penalty = -0.5,
                         double depth_penalty = -0.01);

} // namespace search
} // namespace yirage
