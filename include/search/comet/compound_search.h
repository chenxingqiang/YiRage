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

#include "kernel/compound_graph.h"
#include "kernel/graph.h"
#include "search/common/search_strategy.h"
#include "type.h"
#include <memory>
#include <vector>

namespace yirage {
namespace search {

// =============================================================================
// COMET Compound Operation Search
// =============================================================================
// This implements the search strategy for COMET's compound operations,
// exploring the design space of:
// - Fusion levels (which ops to fuse)
// - Tile sizes (M_tile, N_tile, K_tile)
// - Collective placement (where to insert collectives)
// - Scheduling strategy (sequential, pipelined, parallel)
//
// Reference: COMET paper (Negi et al.) - Section IV-C

/**
 * COMET-specific search configuration.
 * Extends base SearchConfig with compound operation parameters.
 */
struct COMETSearchConfig : public SearchConfig {
  // Fusion search parameters
  int max_fusion_depth = 5;        // Max ops to fuse
  bool enable_fusion_search = true;
  
  // Tile size search space
  std::vector<int> tile_sizes = {32, 64, 128, 256, 512};
  bool auto_tune_tiles = true;
  
  // Collective optimization
  bool optimize_collectives = true;
  int num_devices = 1;             // Number of distributed devices
  double noc_bandwidth_gbps = 600.0;  // Inter-device bandwidth
  
  // Scheduling search
  bool search_scheduling = true;
  std::vector<yirage::type::SchedulingStrategy> scheduling_options = {
    yirage::type::SCHED_SEQUENTIAL,
    yirage::type::SCHED_PIPELINED,
    yirage::type::SCHED_PARALLEL
  };
  
  // Cost model parameters
  double dram_bandwidth_gbps = 900.0;
  double onchip_bandwidth_gbps = 3000.0;
  double peak_tflops = 312.0;
  
  // Search objectives
  enum class Objective {
    MINIMIZE_LATENCY,
    MINIMIZE_ENERGY,
    BALANCE  // Weighted combination
  };
  Objective objective = Objective::MINIMIZE_LATENCY;
  double energy_weight = 0.3;  // For BALANCE objective
};

/**
 * Detected compound operation pattern.
 */
struct CompoundPattern {
  yirage::type::CompoundOpType type;
  std::vector<int> op_indices;  // Indices of ops in the pattern
  std::vector<int> input_dims;
  std::vector<int> output_dims;
  
  // Estimated benefit of fusion
  double memory_reduction_ratio;
  double latency_reduction_ratio;
};

/**
 * Compound operation search candidate.
 */
struct COMETCandidate {
  // Compound graph configuration
  std::shared_ptr<kernel::CompoundGraph> compound_graph;
  
  // Tile configuration
  std::vector<int> tile_dims;
  
  // Scheduling strategy
  yirage::type::SchedulingStrategy scheduling;
  
  // Cost estimates
  double latency_ns;
  double energy_pj;
  double memory_bytes;
  
  // Score (higher is better)
  double score;
  
  COMETCandidate()
      : scheduling(yirage::type::SCHED_PIPELINED),
        latency_ns(0), energy_pj(0), memory_bytes(0), score(0) {}
};

/**
 * COMET Search Strategy.
 * 
 * Implements compound operation search following the COMET paper.
 * The search explores:
 * 1. Pattern detection (GEMM-Softmax, GEMM-LayerNorm, Self-Attention, etc.)
 * 2. Fusion decisions (which patterns to fuse)
 * 3. Tile size optimization
 * 4. Collective placement for distributed execution
 * 5. Scheduling strategy selection
 */
class COMETSearchStrategy : public SearchStrategy {
public:
  COMETSearchStrategy();
  explicit COMETSearchStrategy(type::BackendType target_backend);
  ~COMETSearchStrategy() override;

  // SearchStrategy interface
  bool initialize(SearchConfig const &config) override;
  
  std::vector<CandidateConfig>
  generate_candidates(kernel::Graph const &graph) override;
  
  float evaluate_candidate(CandidateConfig &candidate,
                          kernel::Graph const &graph) override;
  
  kernel::KernelConfig *
  select_best_config(std::vector<CandidateConfig> &candidates) override;
  
  std::unique_ptr<kernel::KernelConfig>
  optimize(kernel::Graph const &graph) override;
  
  type::BackendType get_backend_type() const override;
  std::string get_statistics() const override;

  // COMET-specific search methods
  
  /**
   * Detect compound operation patterns in the graph.
   */
  std::vector<CompoundPattern> detect_patterns(kernel::Graph const &graph);
  
  /**
   * Generate COMET-specific candidates.
   */
  std::vector<COMETCandidate>
  generate_comet_candidates(kernel::Graph const &graph);
  
  /**
   * Evaluate a COMET candidate using the cost model.
   */
  double evaluate_comet_candidate(COMETCandidate &candidate);
  
  /**
   * Search for optimal tile sizes.
   */
  std::vector<int> search_tile_sizes(
      kernel::CompoundGraph const &compound_graph,
      std::vector<int> const &problem_dims);
  
  /**
   * Optimize collective placement.
   */
  void optimize_collective_placement(kernel::CompoundGraph &compound_graph);
  
  /**
   * Select optimal scheduling strategy.
   */
  yirage::type::SchedulingStrategy
  select_scheduling_strategy(kernel::CompoundGraph const &compound_graph);
  
  /**
   * Get the best COMET candidate found.
   */
  COMETCandidate const* get_best_comet_candidate() const;

private:
  COMETSearchConfig comet_config_;
  type::BackendType target_backend_;
  
  // Search state
  std::vector<COMETCandidate> comet_candidates_;
  COMETCandidate* best_comet_candidate_;
  
  // Statistics
  int patterns_detected_;
  int fusion_candidates_generated_;
  int tile_configs_evaluated_;
  
  // Helper methods
  bool is_gemm_softmax_pattern(kernel::Graph const &graph, int start_idx);
  bool is_gemm_layernorm_pattern(kernel::Graph const &graph, int start_idx);
  bool is_self_attention_pattern(kernel::Graph const &graph, int start_idx);
  bool is_gated_mlp_pattern(kernel::Graph const &graph, int start_idx);
  
  double compute_fusion_benefit(CompoundPattern const &pattern);
  std::vector<std::vector<int>> generate_tile_candidates();
};

/**
 * COMET Cost Model for search evaluation.
 * 
 * Implements COMET equations for latency and energy estimation.
 */
class COMETCostModel {
public:
  COMETCostModel();
  explicit COMETCostModel(COMETSearchConfig const &config);
  
  /**
   * Estimate total latency (COMET Eq. 1-7).
   */
  double estimate_latency_ns(kernel::CompoundGraph const &graph) const;
  
  /**
   * Estimate memory latency with data staging.
   */
  double estimate_memory_latency_ns(
      int64_t data_bytes,
      type::MemoryLevel src,
      type::MemoryLevel dst,
      int num_tiles) const;
  
  /**
   * Estimate collective latency.
   */
  double estimate_collective_latency_ns(
      type::CollectiveOpType type,
      int64_t data_bytes,
      int num_participants) const;
  
  /**
   * Estimate scheduling overhead.
   */
  double estimate_scheduling_overhead_ns(
      type::SchedulingStrategy strategy,
      int num_ops) const;
  
  /**
   * Estimate energy consumption.
   */
  double estimate_energy_pj(kernel::CompoundGraph const &graph) const;
  
  /**
   * Compute overall score for optimization.
   */
  double compute_score(
      double latency_ns,
      double energy_pj,
      COMETSearchConfig::Objective objective,
      double energy_weight) const;

private:
  double dram_bandwidth_gbps_;
  double onchip_bandwidth_gbps_;
  double noc_bandwidth_gbps_;
  double peak_tflops_;
  double noc_latency_ns_;
};

} // namespace search
} // namespace yirage
