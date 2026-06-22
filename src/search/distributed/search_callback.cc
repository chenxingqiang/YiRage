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

#include "search/distributed/search_callback.h"

#include <iostream>
#include <iomanip>

namespace yirage {
namespace search {

// ============ FeedbackCollector ============

FeedbackCollector::FeedbackCollector(int partition_id, int total_partitions) {
  feedback_.partition_id = partition_id;
  feedback_.total_partitions = total_partitions;
}

void FeedbackCollector::on_state_explored(SearchContext const &ctx, int depth) {
  feedback_.total_states_explored++;
  current_depth_ = depth;
}

void FeedbackCollector::on_candidate_generated(
    dim3 grid_dim, dim3 block_dim,
    std::vector<int3> const &imaps, int3 omap, int frange) {
  
  current_candidate_ = CandidateInfo();
  current_candidate_.candidate_id = feedback_.candidates_generated;
  current_candidate_.grid_dim = grid_dim;
  current_candidate_.block_dim = block_dim;
  current_candidate_.imaps = imaps;
  current_candidate_.omap = omap;
  current_candidate_.frange = frange;
  current_candidate_.search_depth = current_depth_;
  
  feedback_.candidates_generated++;
}

void FeedbackCollector::on_verification_result(
    kernel::Graph const &graph, bool verified,
    double fingerprint_time_ms,
    std::string const &rejection_reason) {
  
  current_candidate_.verified = verified;
  current_candidate_.fingerprint_time_ms = fingerprint_time_ms;
  current_candidate_.rejection_reason = rejection_reason;
  
  // Count operators
  current_candidate_.operator_count = graph.operators.size();
  
  feedback_.candidates.push_back(current_candidate_);
  feedback_.candidates_verified++;
  
  if (!verified) {
    feedback_.candidates_rejected++;
  }
  
  feedback_.verification_time_seconds += fingerprint_time_ms / 1000.0;
}

void FeedbackCollector::on_valid_graph_found(
    kernel::Graph const &graph,
    double estimated_performance) {
  
  feedback_.valid_graphs_found++;
  
  // Update the last candidate
  if (!feedback_.candidates.empty()) {
    auto &last = feedback_.candidates.back();
    last.estimated_performance_ms = estimated_performance;
    
    feedback_.valid_candidate_ids.push_back(last.candidate_id);
    
    if (estimated_performance < feedback_.best_performance_ms) {
      feedback_.best_performance_ms = estimated_performance;
      feedback_.best_candidate_id = last.candidate_id;
    }
  }
}

void FeedbackCollector::on_search_completed(SearchFeedback const &) {
  // External code will set search_time_seconds
}

void FeedbackCollector::reset() {
  feedback_ = SearchFeedback();
  current_candidate_ = CandidateInfo();
  current_depth_ = 0;
}

// ============ ProgressLogger ============

ProgressLogger::ProgressLogger(int log_interval, bool verbose)
    : log_interval_(log_interval), verbose_(verbose) {}

void ProgressLogger::on_state_explored(SearchContext const &ctx, int depth) {
  states_since_log_++;
}

void ProgressLogger::on_valid_graph_found(
    kernel::Graph const &graph,
    double estimated_performance) {
  if (verbose_) {
    std::cout << "[Search] Valid graph found! "
              << "Ops: " << graph.operators.size()
              << ", Est. perf: " << estimated_performance << " ms"
              << std::endl;
  }
}

bool ProgressLogger::on_progress(
    int states_explored, int valid_found, double elapsed_seconds) {
  
  if (states_since_log_ >= log_interval_) {
    double rate = states_explored / elapsed_seconds;
    std::cout << "\r[Search] States: " << states_explored
              << ", Valid: " << valid_found
              << ", Time: " << std::fixed << std::setprecision(1) << elapsed_seconds << "s"
              << ", Rate: " << std::fixed << std::setprecision(0) << rate << " states/s"
              << std::flush;
    states_since_log_ = 0;
  }
  
  return true;
}

// ============ EarlyTerminator ============

EarlyTerminator::EarlyTerminator(
    int max_valid_graphs,
    double max_time_seconds,
    double target_performance)
    : max_valid_graphs_(max_valid_graphs),
      max_time_seconds_(max_time_seconds),
      target_performance_(target_performance) {}

bool EarlyTerminator::on_progress(
    int states_explored, int valid_found, double elapsed_seconds) {
  
  if (should_terminate_) {
    return false;
  }
  
  // Check max valid graphs
  if (max_valid_graphs_ > 0 && valid_found >= max_valid_graphs_) {
    std::cout << "\n[Search] Early termination: found " << valid_found
              << " valid graphs" << std::endl;
    should_terminate_ = true;
    return false;
  }
  
  // Check timeout
  if (max_time_seconds_ > 0 && elapsed_seconds >= max_time_seconds_) {
    std::cout << "\n[Search] Early termination: timeout after "
              << elapsed_seconds << "s" << std::endl;
    should_terminate_ = true;
    return false;
  }
  
  // Check target performance
  if (target_performance_ > 0 && best_performance_ <= target_performance_) {
    std::cout << "\n[Search] Early termination: target performance "
              << target_performance_ << "ms achieved" << std::endl;
    should_terminate_ = true;
    return false;
  }
  
  return true;
}

void EarlyTerminator::on_valid_graph_found(
    kernel::Graph const &graph,
    double estimated_performance) {
  
  if (estimated_performance < best_performance_) {
    best_performance_ = estimated_performance;
  }
}

// ============ CompositeCallback ============

void CompositeCallback::add_callback(std::shared_ptr<SearchCallback> callback) {
  callbacks_.push_back(callback);
}

void CompositeCallback::on_state_explored(SearchContext const &ctx, int depth) {
  for (auto &cb : callbacks_) {
    cb->on_state_explored(ctx, depth);
  }
}

void CompositeCallback::on_candidate_generated(
    dim3 grid_dim, dim3 block_dim,
    std::vector<int3> const &imaps, int3 omap, int frange) {
  
  for (auto &cb : callbacks_) {
    cb->on_candidate_generated(grid_dim, block_dim, imaps, omap, frange);
  }
}

void CompositeCallback::on_verification_result(
    kernel::Graph const &graph, bool verified,
    double fingerprint_time_ms,
    std::string const &rejection_reason) {
  
  for (auto &cb : callbacks_) {
    cb->on_verification_result(graph, verified, fingerprint_time_ms, rejection_reason);
  }
}

void CompositeCallback::on_valid_graph_found(
    kernel::Graph const &graph,
    double estimated_performance) {
  
  for (auto &cb : callbacks_) {
    cb->on_valid_graph_found(graph, estimated_performance);
  }
}

bool CompositeCallback::on_progress(
    int states_explored, int valid_found, double elapsed_seconds) {
  
  bool continue_search = true;
  for (auto &cb : callbacks_) {
    if (!cb->on_progress(states_explored, valid_found, elapsed_seconds)) {
      continue_search = false;
    }
  }
  return continue_search;
}

void CompositeCallback::on_search_completed(SearchFeedback const &feedback) {
  for (auto &cb : callbacks_) {
    cb->on_search_completed(feedback);
  }
}

} // namespace search
} // namespace yirage
