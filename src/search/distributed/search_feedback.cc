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

#include "search/distributed/search_feedback.h"

#include <cmath>
#include <sstream>

namespace yirage {
namespace search {

// ============ CandidateInfo ============

json CandidateInfo::to_json() const {
  json j;
  j["candidate_id"] = candidate_id;

  // Configuration
  j["grid_dim"] = {{"x", grid_dim.x}, {"y", grid_dim.y}, {"z", grid_dim.z}};
  j["block_dim"] = {{"x", block_dim.x}, {"y", block_dim.y}, {"z", block_dim.z}};

  json imap_arr = json::array();
  for (auto const &m : imaps) {
    imap_arr.push_back({{"x", m.x}, {"y", m.y}, {"z", m.z}});
  }
  j["imaps"] = imap_arr;

  j["omap"] = {{"x", omap.x}, {"y", omap.y}, {"z", omap.z}};
  j["frange"] = frange;

  // Search context
  j["search_depth"] = search_depth;
  j["operator_count"] = operator_count;
  j["kernel_level_ops"] = kernel_level_ops;
  j["threadblock_level_ops"] = threadblock_level_ops;

  // Results
  j["verified"] = verified;
  j["fingerprint_time_ms"] = fingerprint_time_ms;
  j["estimated_performance_ms"] = estimated_performance_ms;
  j["rejection_reason"] = rejection_reason;
  j["evaluation_time_ms"] = evaluation_time_ms;

  return j;
}

CandidateInfo CandidateInfo::from_json(json const &j) {
  CandidateInfo c;

  c.candidate_id = j.value("candidate_id", 0);

  // Configuration
  if (j.contains("grid_dim")) {
    auto const &g = j["grid_dim"];
    c.grid_dim.x = g.value("x", 1u);
    c.grid_dim.y = g.value("y", 1u);
    c.grid_dim.z = g.value("z", 1u);
  }

  if (j.contains("block_dim")) {
    auto const &b = j["block_dim"];
    c.block_dim.x = b.value("x", 128u);
    c.block_dim.y = b.value("y", 1u);
    c.block_dim.z = b.value("z", 1u);
  }

  if (j.contains("imaps")) {
    for (auto const &m : j["imaps"]) {
      int3 i;
      i.x = m.value("x", 0);
      i.y = m.value("y", 0);
      i.z = m.value("z", 0);
      c.imaps.push_back(i);
    }
  }

  if (j.contains("omap")) {
    auto const &o = j["omap"];
    c.omap.x = o.value("x", 0);
    c.omap.y = o.value("y", 0);
    c.omap.z = o.value("z", 0);
  }

  c.frange = j.value("frange", 1);

  // Search context
  c.search_depth = j.value("search_depth", 0);
  c.operator_count = j.value("operator_count", 0);
  c.kernel_level_ops = j.value("kernel_level_ops", 0);
  c.threadblock_level_ops = j.value("threadblock_level_ops", 0);

  // Results
  c.verified = j.value("verified", false);
  c.fingerprint_time_ms = j.value("fingerprint_time_ms", 0.0);
  c.estimated_performance_ms = j.value("estimated_performance_ms", 0.0);
  c.rejection_reason = j.value("rejection_reason", std::string(""));
  c.evaluation_time_ms = j.value("evaluation_time_ms", 0.0);

  return c;
}

// ============ SearchFeedback ============

void SearchFeedback::add_candidate(CandidateInfo const &info) {
  candidates.push_back(info);
  candidates_generated++;
}

void SearchFeedback::mark_verified(int candidate_id, double performance_ms) {
  valid_graphs_found++;
  valid_candidate_ids.push_back(candidate_id);

  if (performance_ms < best_performance_ms) {
    best_performance_ms = performance_ms;
    best_candidate_id = candidate_id;
  }

  // Update the candidate
  for (auto &c : candidates) {
    if (c.candidate_id == candidate_id) {
      c.verified = true;
      c.estimated_performance_ms = performance_ms;
      break;
    }
  }
}

json SearchFeedback::to_json() const {
  json j;

  j["partition_id"] = partition_id;
  j["total_partitions"] = total_partitions;

  // Candidates
  json cand_arr = json::array();
  for (auto const &c : candidates) {
    cand_arr.push_back(c.to_json());
  }
  j["candidates"] = cand_arr;

  j["valid_candidate_ids"] = valid_candidate_ids;

  // Statistics
  j["total_states_explored"] = total_states_explored;
  j["valid_graphs_found"] = valid_graphs_found;
  j["candidates_generated"] = candidates_generated;
  j["candidates_verified"] = candidates_verified;
  j["candidates_rejected"] = candidates_rejected;

  // Timing
  j["search_time_seconds"] = search_time_seconds;
  j["verification_time_seconds"] = verification_time_seconds;
  j["generation_time_seconds"] = generation_time_seconds;

  // Best result
  j["best_performance_ms"] = best_performance_ms;
  j["best_candidate_id"] = best_candidate_id;

  // Config
  j["search_config"] = search_config;

  return j;
}

SearchFeedback SearchFeedback::from_json(json const &j) {
  SearchFeedback f;

  f.partition_id = j.value("partition_id", 0);
  f.total_partitions = j.value("total_partitions", 1);

  // Candidates
  if (j.contains("candidates")) {
    for (auto const &c : j["candidates"]) {
      f.candidates.push_back(CandidateInfo::from_json(c));
    }
  }

  if (j.contains("valid_candidate_ids")) {
    f.valid_candidate_ids = j["valid_candidate_ids"].get<std::vector<int>>();
  }

  // Statistics
  f.total_states_explored = j.value("total_states_explored", 0);
  f.valid_graphs_found = j.value("valid_graphs_found", 0);
  f.candidates_generated = j.value("candidates_generated", 0);
  f.candidates_verified = j.value("candidates_verified", 0);
  f.candidates_rejected = j.value("candidates_rejected", 0);

  // Timing
  f.search_time_seconds = j.value("search_time_seconds", 0.0);
  f.verification_time_seconds = j.value("verification_time_seconds", 0.0);
  f.generation_time_seconds = j.value("generation_time_seconds", 0.0);

  // Best result
  f.best_performance_ms = j.value("best_performance_ms",
                                   std::numeric_limits<double>::infinity());
  f.best_candidate_id = j.value("best_candidate_id", -1);

  // Config
  if (j.contains("search_config")) {
    f.search_config = j["search_config"];
  }

  return f;
}

SearchFeedback SearchFeedback::merge(std::vector<SearchFeedback> const &feedbacks) {
  SearchFeedback merged;

  if (feedbacks.empty()) {
    return merged;
  }

  merged.total_partitions = feedbacks[0].total_partitions;
  merged.partition_id = -1; // Merged

  int candidate_offset = 0;

  for (auto const &f : feedbacks) {
    // Merge candidates with offset IDs
    for (auto c : f.candidates) {
      c.candidate_id += candidate_offset;
      merged.candidates.push_back(c);
    }

    // Merge valid IDs with offset
    for (int id : f.valid_candidate_ids) {
      merged.valid_candidate_ids.push_back(id + candidate_offset);
    }

    candidate_offset += f.candidates.size();

    // Aggregate statistics
    merged.total_states_explored += f.total_states_explored;
    merged.valid_graphs_found += f.valid_graphs_found;
    merged.candidates_generated += f.candidates_generated;
    merged.candidates_verified += f.candidates_verified;
    merged.candidates_rejected += f.candidates_rejected;

    // Max time (parallel execution)
    merged.search_time_seconds =
        std::max(merged.search_time_seconds, f.search_time_seconds);
    merged.verification_time_seconds += f.verification_time_seconds;
    merged.generation_time_seconds += f.generation_time_seconds;

    // Best result
    if (f.best_performance_ms < merged.best_performance_ms) {
      merged.best_performance_ms = f.best_performance_ms;
      merged.best_candidate_id = f.best_candidate_id + candidate_offset -
                                  static_cast<int>(f.candidates.size());
    }
  }

  return merged;
}

std::string SearchFeedback::get_summary() const {
  std::ostringstream oss;

  oss << "=== Search Feedback Summary ===" << std::endl;
  oss << "Partition: " << partition_id << "/" << total_partitions << std::endl;
  oss << "States explored: " << total_states_explored << std::endl;
  oss << "Candidates generated: " << candidates_generated << std::endl;
  oss << "Valid graphs found: " << valid_graphs_found << std::endl;
  oss << "Search time: " << search_time_seconds << "s" << std::endl;

  if (best_candidate_id >= 0) {
    oss << "Best performance: " << best_performance_ms << "ms" << std::endl;
    oss << "Best candidate ID: " << best_candidate_id << std::endl;
  }

  return oss.str();
}

// ============ TrainingSample ============

json TrainingSample::to_json() const {
  json j;

  // State
  j["state"] = {{"search_depth", state.search_depth},
                {"operator_count", state.operator_count},
                {"grid_dim", state.grid_dim},
                {"block_dim", state.block_dim},
                {"num_valid_found_so_far", state.num_valid_found_so_far}};

  // Action
  j["action"] = {{"imaps", action.imaps},
                 {"omap", action.omap},
                 {"frange", action.frange},
                 {"operator_type", action.operator_type}};

  j["reward"] = reward;
  j["done"] = done;

  if (has_next_state) {
    j["next_state"] = {{"search_depth", next_state.search_depth},
                       {"operator_count", next_state.operator_count},
                       {"grid_dim", next_state.grid_dim},
                       {"block_dim", next_state.block_dim}};
  } else {
    j["next_state"] = nullptr;
  }

  return j;
}

std::vector<TrainingSample>
extract_training_samples(SearchFeedback const &feedback,
                         double validity_reward,
                         double invalid_penalty,
                         double depth_penalty) {
  std::vector<TrainingSample> samples;

  int num_valid_found = 0;

  for (size_t i = 0; i < feedback.candidates.size(); ++i) {
    auto const &cand = feedback.candidates[i];
    TrainingSample sample;

    // State
    sample.state.search_depth = cand.search_depth;
    sample.state.operator_count = cand.operator_count;
    sample.state.grid_dim = {static_cast<int>(cand.grid_dim.x),
                             static_cast<int>(cand.grid_dim.y),
                             static_cast<int>(cand.grid_dim.z)};
    sample.state.block_dim = {static_cast<int>(cand.block_dim.x),
                              static_cast<int>(cand.block_dim.y),
                              static_cast<int>(cand.block_dim.z)};
    sample.state.num_valid_found_so_far = num_valid_found;

    // Action
    for (auto const &m : cand.imaps) {
      sample.action.imaps.push_back({m.x, m.y, m.z});
    }
    sample.action.omap = {cand.omap.x, cand.omap.y, cand.omap.z};
    sample.action.frange = cand.frange;
    sample.action.operator_type = cand.last_operator_type;

    // Reward
    sample.reward = depth_penalty * cand.search_depth;
    if (cand.verified) {
      sample.reward += validity_reward;
      if (cand.estimated_performance_ms > 0) {
        sample.reward += 1.0 / cand.estimated_performance_ms;
      }
      num_valid_found++;
    } else {
      sample.reward += invalid_penalty;
    }

    // Next state
    if (i + 1 < feedback.candidates.size()) {
      auto const &next = feedback.candidates[i + 1];
      sample.has_next_state = true;
      sample.next_state.search_depth = next.search_depth;
      sample.next_state.operator_count = next.operator_count;
      sample.next_state.grid_dim = {static_cast<int>(next.grid_dim.x),
                                    static_cast<int>(next.grid_dim.y),
                                    static_cast<int>(next.grid_dim.z)};
      sample.next_state.block_dim = {static_cast<int>(next.block_dim.x),
                                     static_cast<int>(next.block_dim.y),
                                     static_cast<int>(next.block_dim.z)};
      sample.done = false;
    } else {
      sample.has_next_state = false;
      sample.done = true;
    }

    samples.push_back(sample);
  }

  return samples;
}

} // namespace search
} // namespace yirage
