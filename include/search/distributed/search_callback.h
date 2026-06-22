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

#include "kernel/graph.h"
#include "search/distributed/search_feedback.h"
#include "search/search_context.h"
#include "vector_types.h"

#include <functional>
#include <memory>
#include <vector>

namespace yirage {
namespace search {

/**
 * @brief Callback interface for search events
 *
 * Implement this interface to monitor search progress,
 * collect training data, or implement custom early termination.
 */
class SearchCallback {
public:
  virtual ~SearchCallback() = default;

  /**
   * @brief Called when a new search state is explored
   * @param ctx Current search context
   * @param depth Search depth
   */
  virtual void on_state_explored(SearchContext const &ctx, int depth) {}

  /**
   * @brief Called when a candidate configuration is generated
   * @param grid_dim Grid dimensions
   * @param block_dim Block dimensions
   * @param imaps Input tensor mappings
   * @param omap Output tensor mapping
   * @param frange Forloop range
   */
  virtual void on_candidate_generated(dim3 grid_dim, dim3 block_dim,
                                      std::vector<int3> const &imaps,
                                      int3 omap, int frange) {}

  /**
   * @brief Called when verification completes for a candidate
   * @param graph The candidate kernel graph
   * @param verified Whether verification passed
   * @param fingerprint_time_ms Time spent on fingerprint verification
   * @param rejection_reason Reason for rejection (if !verified)
   */
  virtual void on_verification_result(kernel::Graph const &graph, bool verified,
                                      double fingerprint_time_ms,
                                      std::string const &rejection_reason = "") {
  }

  /**
   * @brief Called when a valid kernel graph is found
   * @param graph The valid kernel graph
   * @param estimated_performance Estimated execution time in ms
   */
  virtual void on_valid_graph_found(kernel::Graph const &graph,
                                    double estimated_performance) {}

  /**
   * @brief Called periodically during long searches
   * @param states_explored Number of states explored so far
   * @param valid_found Number of valid graphs found
   * @param elapsed_seconds Time elapsed
   * @return true to continue search, false to early terminate
   */
  virtual bool on_progress(int states_explored, int valid_found,
                           double elapsed_seconds) {
    return true; // Continue by default
  }

  /**
   * @brief Called when search completes
   * @param feedback Aggregated feedback data
   */
  virtual void on_search_completed(SearchFeedback const &feedback) {}
};

/**
 * @brief Callback that collects feedback data for RL training
 */
class FeedbackCollector : public SearchCallback {
public:
  FeedbackCollector(int partition_id = 0, int total_partitions = 1);

  void on_state_explored(SearchContext const &ctx, int depth) override;

  void on_candidate_generated(dim3 grid_dim, dim3 block_dim,
                              std::vector<int3> const &imaps, int3 omap,
                              int frange) override;

  void on_verification_result(kernel::Graph const &graph, bool verified,
                              double fingerprint_time_ms,
                              std::string const &rejection_reason) override;

  void on_valid_graph_found(kernel::Graph const &graph,
                            double estimated_performance) override;

  void on_search_completed(SearchFeedback const &feedback) override;

  /**
   * @brief Get collected feedback data
   */
  SearchFeedback const &get_feedback() const { return feedback_; }

  /**
   * @brief Reset collector for reuse
   */
  void reset();

private:
  SearchFeedback feedback_;
  CandidateInfo current_candidate_;
  int current_depth_ = 0;
};

/**
 * @brief Callback that logs search progress to console
 */
class ProgressLogger : public SearchCallback {
public:
  ProgressLogger(int log_interval = 1000, bool verbose = false);

  void on_state_explored(SearchContext const &ctx, int depth) override;
  void on_valid_graph_found(kernel::Graph const &graph,
                            double estimated_performance) override;
  bool on_progress(int states_explored, int valid_found,
                   double elapsed_seconds) override;

private:
  int log_interval_;
  bool verbose_;
  int states_since_log_ = 0;
};

/**
 * @brief Callback for early termination based on criteria
 */
class EarlyTerminator : public SearchCallback {
public:
  EarlyTerminator(int max_valid_graphs = 0,     // 0 = no limit
                  double max_time_seconds = 0,  // 0 = no limit
                  double target_performance = 0 // 0 = no target
  );

  bool on_progress(int states_explored, int valid_found,
                   double elapsed_seconds) override;

  void on_valid_graph_found(kernel::Graph const &graph,
                            double estimated_performance) override;

private:
  int max_valid_graphs_;
  double max_time_seconds_;
  double target_performance_;
  bool should_terminate_ = false;
  double best_performance_ = std::numeric_limits<double>::infinity();
};

/**
 * @brief Composite callback that chains multiple callbacks
 */
class CompositeCallback : public SearchCallback {
public:
  void add_callback(std::shared_ptr<SearchCallback> callback);

  void on_state_explored(SearchContext const &ctx, int depth) override;

  void on_candidate_generated(dim3 grid_dim, dim3 block_dim,
                              std::vector<int3> const &imaps, int3 omap,
                              int frange) override;

  void on_verification_result(kernel::Graph const &graph, bool verified,
                              double fingerprint_time_ms,
                              std::string const &rejection_reason) override;

  void on_valid_graph_found(kernel::Graph const &graph,
                            double estimated_performance) override;

  bool on_progress(int states_explored, int valid_found,
                   double elapsed_seconds) override;

  void on_search_completed(SearchFeedback const &feedback) override;

private:
  std::vector<std::shared_ptr<SearchCallback>> callbacks_;
};

} // namespace search
} // namespace yirage
