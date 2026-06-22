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

#include "search/comet/compound_search.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <sstream>

namespace yirage {
namespace search {

// =============================================================================
// COMETSearchStrategy Implementation
// =============================================================================

COMETSearchStrategy::COMETSearchStrategy()
    : target_backend_(type::BT_CUDA),
      best_comet_candidate_(nullptr),
      patterns_detected_(0),
      fusion_candidates_generated_(0),
      tile_configs_evaluated_(0) {}

COMETSearchStrategy::COMETSearchStrategy(type::BackendType target_backend)
    : target_backend_(target_backend),
      best_comet_candidate_(nullptr),
      patterns_detected_(0),
      fusion_candidates_generated_(0),
      tile_configs_evaluated_(0) {}

COMETSearchStrategy::~COMETSearchStrategy() = default;

bool COMETSearchStrategy::initialize(SearchConfig const &config) {
  config_ = config;
  
  // Check if config is COMET-specific
  auto const* comet_cfg = dynamic_cast<COMETSearchConfig const*>(&config);
  if (comet_cfg) {
    comet_config_ = *comet_cfg;
  } else {
    // Use defaults
    comet_config_ = COMETSearchConfig();
    comet_config_.max_iterations = config.max_iterations;
    comet_config_.timeout_seconds = config.timeout_seconds;
    comet_config_.random_seed = config.random_seed;
  }
  
  // Reset state
  comet_candidates_.clear();
  best_comet_candidate_ = nullptr;
  patterns_detected_ = 0;
  fusion_candidates_generated_ = 0;
  tile_configs_evaluated_ = 0;
  
  return true;
}

std::vector<CompoundPattern>
COMETSearchStrategy::detect_patterns(kernel::Graph const &graph) {
  std::vector<CompoundPattern> patterns;
  
  // Get operator count
  auto const& operators = graph.operators;
  int num_ops = static_cast<int>(operators.size());
  
  // Track which ops have been consumed by a pattern
  std::set<int> used_ops;
  
  // Scan for compound operation patterns
  // Priority order: Self-Attention > GEMM-Softmax > GEMM-LayerNorm > Gated-MLP
  // (more complex patterns first to avoid partial matches)
  for (int i = 0; i < num_ops; ++i) {
    if (used_ops.count(i) > 0) {
      continue;  // Skip already consumed ops
    }
    
    // Try Self-Attention FIRST (most complex, includes GEMM-Softmax)
    if (is_self_attention_pattern(graph, i)) {
      CompoundPattern pattern;
      pattern.type = type::COMP_SELF_ATTENTION;
      int end_idx = std::min(i + 5, num_ops);
      for (int j = i; j < end_idx; ++j) {
        pattern.op_indices.push_back(j);
        used_ops.insert(j);
      }
      pattern.memory_reduction_ratio = 0.7;
      pattern.latency_reduction_ratio = 0.4;
      patterns.push_back(pattern);
      continue;
    }
    
    // Try GEMM-Softmax
    if (is_gemm_softmax_pattern(graph, i)) {
      CompoundPattern pattern;
      pattern.type = type::COMP_GEMM_SOFTMAX;
      int end_idx = std::min(i + 4, num_ops);
      for (int j = i; j < end_idx; ++j) {
        pattern.op_indices.push_back(j);
        used_ops.insert(j);
      }
      pattern.memory_reduction_ratio = 0.5;
      pattern.latency_reduction_ratio = 0.3;
      patterns.push_back(pattern);
      continue;
    }
    
    // Try GEMM-LayerNorm
    if (is_gemm_layernorm_pattern(graph, i)) {
      CompoundPattern pattern;
      pattern.type = type::COMP_GEMM_LAYERNORM;
      int end_idx = std::min(i + 5, num_ops);
      for (int j = i; j < end_idx; ++j) {
        pattern.op_indices.push_back(j);
        used_ops.insert(j);
      }
      pattern.memory_reduction_ratio = 0.5;
      pattern.latency_reduction_ratio = 0.25;
      patterns.push_back(pattern);
      continue;
    }
    
    // Try Gated MLP
    if (is_gated_mlp_pattern(graph, i)) {
      CompoundPattern pattern;
      pattern.type = type::COMP_GATED_MLP;
      int end_idx = std::min(i + 4, num_ops);
      for (int j = i; j < end_idx; ++j) {
        pattern.op_indices.push_back(j);
        used_ops.insert(j);
      }
      pattern.memory_reduction_ratio = 0.4;
      pattern.latency_reduction_ratio = 0.2;
      patterns.push_back(pattern);
      continue;
    }
  }
  
  patterns_detected_ = static_cast<int>(patterns.size());
  return patterns;
}

std::vector<COMETCandidate>
COMETSearchStrategy::generate_comet_candidates(kernel::Graph const &graph) {
  std::vector<COMETCandidate> candidates;
  
  // Detect patterns
  auto patterns = detect_patterns(graph);
  
  // Generate tile size candidates
  auto tile_candidates = generate_tile_candidates();
  
  // For each pattern, generate candidates with different configurations
  for (auto const& pattern : patterns) {
    for (auto const& tiles : tile_candidates) {
      for (auto scheduling : comet_config_.scheduling_options) {
        COMETCandidate candidate;
        
        // Build compound graph for this pattern
        auto compound_graph = std::make_shared<kernel::CompoundGraph>();
        
        // Configure based on pattern type
        std::vector<int> tile_vec = tiles;
        switch (pattern.type) {
          case type::COMP_GEMM_SOFTMAX:
            compound_graph->build_gemm_softmax(
                {tiles[0], tiles[1]}, {tiles[1], tiles[2]}, tile_vec);
            break;
          case type::COMP_GEMM_LAYERNORM:
            compound_graph->build_gemm_layernorm(
                {tiles[0], tiles[1]}, {tiles[1], tiles[2]}, tile_vec);
            break;
          case type::COMP_SELF_ATTENTION:
            compound_graph->build_self_attention(
                1, 8, tiles[0], tiles[1], tile_vec);
            break;
          case type::COMP_GATED_MLP:
            compound_graph->build_gated_mlp(
                1, tiles[0], tiles[1], tiles[2], tile_vec);
            break;
          default:
            continue;
        }
        
        compound_graph->set_scheduling_strategy(scheduling);
        
        candidate.compound_graph = compound_graph;
        candidate.tile_dims = tiles;
        candidate.scheduling = scheduling;
        
        // Evaluate candidate
        evaluate_comet_candidate(candidate);
        
        candidates.push_back(std::move(candidate));
        ++fusion_candidates_generated_;
        ++tile_configs_evaluated_;
      }
    }
  }
  
  comet_candidates_ = candidates;
  return candidates;
}

double COMETSearchStrategy::evaluate_comet_candidate(COMETCandidate &candidate) {
  if (!candidate.compound_graph) {
    candidate.score = 0.0;
    return 0.0;
  }
  
  // Use COMET cost model
  COMETCostModel cost_model(comet_config_);
  
  // Estimate latency and energy
  candidate.latency_ns = cost_model.estimate_latency_ns(*candidate.compound_graph);
  candidate.energy_pj = cost_model.estimate_energy_pj(*candidate.compound_graph);
  candidate.memory_bytes = static_cast<double>(
      candidate.compound_graph->get_total_memory_bytes());
  
  // Compute score
  candidate.score = cost_model.compute_score(
      candidate.latency_ns,
      candidate.energy_pj,
      comet_config_.objective,
      comet_config_.energy_weight);
  
  return candidate.score;
}

std::vector<int> COMETSearchStrategy::search_tile_sizes(
    kernel::CompoundGraph const &compound_graph,
    std::vector<int> const &problem_dims) {
  
  std::vector<int> best_tiles;
  double best_latency = std::numeric_limits<double>::max();
  
  COMETCostModel cost_model(comet_config_);
  
  // Search through tile size combinations
  for (int tile_m : comet_config_.tile_sizes) {
    for (int tile_n : comet_config_.tile_sizes) {
      for (int tile_k : comet_config_.tile_sizes) {
        // Check if tile sizes are valid for problem dims
        bool valid = true;
        if (problem_dims.size() >= 3) {
          valid = (tile_m <= problem_dims[0] &&
                   tile_n <= problem_dims[1] &&
                   tile_k <= problem_dims[2]);
        }
        
        if (!valid) continue;
        
        // Create temporary compound graph with these tiles
        kernel::CompoundGraph temp_graph = compound_graph;
        // (In a full implementation, we'd reconfigure the graph with new tiles)
        
        double latency = cost_model.estimate_latency_ns(temp_graph);
        
        if (latency < best_latency) {
          best_latency = latency;
          best_tiles = {tile_m, tile_n, tile_k};
        }
        
        ++tile_configs_evaluated_;
      }
    }
  }
  
  return best_tiles;
}

void COMETSearchStrategy::optimize_collective_placement(
    kernel::CompoundGraph &compound_graph) {
  
  if (comet_config_.num_devices <= 1) {
    return;  // No collective optimization needed
  }
  
  // Analyze data dependencies and insert collectives
  // This follows COMET's explicit collective representation
  
  auto root = compound_graph.get_root();
  if (!root) return;
  
  // For distributed execution, we need to:
  // 1. Determine which tiles need communication
  // 2. Insert appropriate collective ops (All-Reduce, All-Gather, etc.)
  // 3. Minimize communication volume
  
  // Simplified: Mark root node as needing All-Reduce for reduction ops
  if (root->collective_type != type::COLL_NONE) {
    // Already has collective, update participants
    root->collective_participants = comet_config_.num_devices;
  }
  
  // Recursively check children
  for (auto& child : root->children) {
    if (child->collective_type == type::COLL_NONE) {
      // Check if this tile's output is used across devices
      // If so, insert appropriate collective
      if (child->op_type == type::KN_REDUCTION_0_OP ||
          child->op_type == type::KN_REDUCTION_1_OP ||
          child->op_type == type::KN_REDUCTION_2_OP) {
        child->collective_type = type::COLL_ALL_REDUCE;
        child->collective_participants = comet_config_.num_devices;
      }
    }
  }
}

yirage::type::SchedulingStrategy
COMETSearchStrategy::select_scheduling_strategy(
    kernel::CompoundGraph const &compound_graph) {
  
  COMETCostModel cost_model(comet_config_);
  
  double best_latency = std::numeric_limits<double>::max();
  yirage::type::SchedulingStrategy best_strategy = yirage::type::SCHED_PIPELINED;
  
  for (auto strategy : comet_config_.scheduling_options) {
    // Create copy with different scheduling
    kernel::CompoundGraph temp_graph = compound_graph;
    temp_graph.set_scheduling_strategy(strategy);
    
    double latency = cost_model.estimate_latency_ns(temp_graph);
    
    if (latency < best_latency) {
      best_latency = latency;
      best_strategy = strategy;
    }
  }
  
  return best_strategy;
}

COMETCandidate const* COMETSearchStrategy::get_best_comet_candidate() const {
  return best_comet_candidate_;
}

// SearchStrategy interface implementation

std::vector<CandidateConfig>
COMETSearchStrategy::generate_candidates(kernel::Graph const &graph) {
  // Generate COMET candidates
  auto comet_cands = generate_comet_candidates(graph);
  
  // Convert to base CandidateConfig
  std::vector<CandidateConfig> candidates;
  for (auto const& cc : comet_cands) {
    CandidateConfig config;
    // Create kernel config from COMET candidate
    config.config = std::make_unique<kernel::KernelConfig>();
    config.score = static_cast<float>(cc.score);
    candidates.push_back(std::move(config));
  }
  
  num_candidates_generated_ = static_cast<int>(candidates.size());
  return candidates;
}

float COMETSearchStrategy::evaluate_candidate(
    CandidateConfig &candidate,
    kernel::Graph const &graph) {
  // Already evaluated during generation
  return candidate.score;
}

kernel::KernelConfig*
COMETSearchStrategy::select_best_config(
    std::vector<CandidateConfig> &candidates) {
  
  if (candidates.empty()) {
    return nullptr;
  }
  
  // Find best by score
  auto best_it = std::max_element(
      candidates.begin(), candidates.end(),
      [](CandidateConfig const& a, CandidateConfig const& b) {
        return a.score < b.score;
      });
  
  if (best_it != candidates.end()) {
    best_score_ = best_it->score;
    return best_it->config.get();
  }
  
  return nullptr;
}

std::unique_ptr<kernel::KernelConfig>
COMETSearchStrategy::optimize(kernel::Graph const &graph) {
  // Generate candidates
  auto candidates = generate_candidates(graph);
  
  if (candidates.empty()) {
    return nullptr;
  }
  
  // Select best
  auto* best = select_best_config(candidates);
  
  if (best) {
    // Find and update best COMET candidate
    if (!comet_candidates_.empty()) {
      auto best_cc = std::max_element(
          comet_candidates_.begin(), comet_candidates_.end(),
          [](COMETCandidate const& a, COMETCandidate const& b) {
            return a.score < b.score;
          });
      best_comet_candidate_ = &(*best_cc);
    }
    
    return std::make_unique<kernel::KernelConfig>(*best);
  }
  
  return nullptr;
}

type::BackendType COMETSearchStrategy::get_backend_type() const {
  return target_backend_;
}

std::string COMETSearchStrategy::get_statistics() const {
  std::stringstream ss;
  ss << "COMET Search Statistics:\n";
  ss << "  Patterns detected: " << patterns_detected_ << "\n";
  ss << "  Fusion candidates: " << fusion_candidates_generated_ << "\n";
  ss << "  Tile configs evaluated: " << tile_configs_evaluated_ << "\n";
  ss << "  Best score: " << best_score_ << "\n";
  
  if (best_comet_candidate_) {
    ss << "  Best latency: " << best_comet_candidate_->latency_ns << " ns\n";
    ss << "  Best energy: " << best_comet_candidate_->energy_pj << " pJ\n";
  }
  
  return ss.str();
}

// Pattern detection helpers

bool COMETSearchStrategy::is_gemm_softmax_pattern(
    kernel::Graph const &graph, int start_idx) {
  
  auto const& ops = graph.operators;
  if (start_idx + 1 >= static_cast<int>(ops.size())) {
    return false;
  }
  
  // Check for MATMUL followed by SOFTMAX-like ops (EXP, REDUCTION, DIV)
  auto* op1 = ops[start_idx];
  if (op1->op_type != type::KN_MATMUL_OP) {
    return false;
  }
  
  // Look for softmax pattern in following ops
  auto* op2 = ops[start_idx + 1];
  if (op2->op_type == type::KN_EXP_OP) {
    return true;  // Simplified check
  }
  
  return false;
}

bool COMETSearchStrategy::is_gemm_layernorm_pattern(
    kernel::Graph const &graph, int start_idx) {
  
  auto const& ops = graph.operators;
  if (start_idx + 1 >= static_cast<int>(ops.size())) {
    return false;
  }
  
  auto* op1 = ops[start_idx];
  if (op1->op_type != type::KN_MATMUL_OP) {
    return false;
  }
  
  // Look for layernorm pattern (reduction, sub, mul, add)
  auto* op2 = ops[start_idx + 1];
  if (op2->op_type == type::KN_REDUCTION_0_OP ||
      op2->op_type == type::KN_REDUCTION_1_OP ||
      op2->op_type == type::KN_REDUCTION_2_OP) {
    return true;  // Simplified check
  }
  
  return false;
}

bool COMETSearchStrategy::is_self_attention_pattern(
    kernel::Graph const &graph, int start_idx) {
  
  auto const& ops = graph.operators;
  if (start_idx + 4 >= static_cast<int>(ops.size())) {
    return false;
  }
  
  // Self-attention: Q*K^T, scale, softmax, *V
  // Simplified: look for two matmuls with softmax-like ops between
  auto* op1 = ops[start_idx];
  if (op1->op_type != type::KN_MATMUL_OP) {
    return false;
  }
  
  // Look for second matmul
  for (int i = start_idx + 1; i < std::min(start_idx + 5, 
                                           static_cast<int>(ops.size())); ++i) {
    if (ops[i]->op_type == type::KN_MATMUL_OP) {
      return true;  // Found pattern
    }
  }
  
  return false;
}

bool COMETSearchStrategy::is_gated_mlp_pattern(
    kernel::Graph const &graph, int start_idx) {
  
  auto const& ops = graph.operators;
  if (start_idx + 3 >= static_cast<int>(ops.size())) {
    return false;
  }
  
  // Gated MLP: Two parallel matmuls, activation, element-wise multiply, matmul
  auto* op1 = ops[start_idx];
  if (op1->op_type != type::KN_MATMUL_OP) {
    return false;
  }
  
  // Look for SiLU/GELU and multiply
  for (int i = start_idx + 1; i < std::min(start_idx + 4,
                                           static_cast<int>(ops.size())); ++i) {
    if (ops[i]->op_type == type::KN_SILU_OP ||
        ops[i]->op_type == type::KN_MUL_OP) {
      return true;
    }
  }
  
  return false;
}

double COMETSearchStrategy::compute_fusion_benefit(
    CompoundPattern const &pattern) {
  // Estimate benefit based on memory reduction and compute overlap
  double memory_benefit = pattern.memory_reduction_ratio * 0.6;
  double latency_benefit = pattern.latency_reduction_ratio * 0.4;
  return memory_benefit + latency_benefit;
}

std::vector<std::vector<int>> COMETSearchStrategy::generate_tile_candidates() {
  std::vector<std::vector<int>> candidates;
  
  // Generate common tile size combinations
  for (int m : comet_config_.tile_sizes) {
    for (int n : comet_config_.tile_sizes) {
      for (int k : comet_config_.tile_sizes) {
        candidates.push_back({m, n, k});
      }
    }
  }
  
  // Limit to reasonable number
  if (candidates.size() > 100) {
    candidates.resize(100);
  }
  
  return candidates;
}

// =============================================================================
// COMETCostModel Implementation
// =============================================================================

COMETCostModel::COMETCostModel()
    : dram_bandwidth_gbps_(900.0),
      onchip_bandwidth_gbps_(3000.0),
      noc_bandwidth_gbps_(600.0),
      peak_tflops_(312.0),
      noc_latency_ns_(100.0) {}

COMETCostModel::COMETCostModel(COMETSearchConfig const &config)
    : dram_bandwidth_gbps_(config.dram_bandwidth_gbps),
      onchip_bandwidth_gbps_(config.onchip_bandwidth_gbps),
      noc_bandwidth_gbps_(config.noc_bandwidth_gbps),
      peak_tflops_(config.peak_tflops),
      noc_latency_ns_(100.0) {}

double COMETCostModel::estimate_latency_ns(
    kernel::CompoundGraph const &graph) const {
  
  // Use the compound graph's built-in estimation
  return graph.estimate_latency_ns();
}

double COMETCostModel::estimate_memory_latency_ns(
    int64_t data_bytes,
    type::MemoryLevel src,
    type::MemoryLevel dst,
    int num_tiles) const {
  
  // COMET Eq. 4: Memory latency = bytes / bandwidth
  double bandwidth_gbps;
  
  if (src == type::MEM_DRAM || dst == type::MEM_DRAM) {
    bandwidth_gbps = dram_bandwidth_gbps_;
  } else {
    bandwidth_gbps = onchip_bandwidth_gbps_;
  }
  
  // Convert: GB/s to bytes/ns
  double bandwidth_bytes_per_ns = bandwidth_gbps * 1e9 / 1e9;
  
  double latency = static_cast<double>(data_bytes) / bandwidth_bytes_per_ns;
  
  // Add per-tile overhead
  latency += num_tiles * 10.0;  // 10ns per tile overhead
  
  return latency;
}

double COMETCostModel::estimate_collective_latency_ns(
    type::CollectiveOpType coll_type,
    int64_t data_bytes,
    int num_participants) const {
  
  if (num_participants <= 1) {
    return 0.0;
  }
  
  // COMET Eq. 5: Collective latency
  double bytes_per_ns = noc_bandwidth_gbps_ * 1e9 / 1e9;
  
  double alpha = noc_latency_ns_;  // Startup latency
  double beta = 1.0 / bytes_per_ns;  // Transfer time per byte
  
  double data_size = static_cast<double>(data_bytes);
  int n = num_participants;
  
  double latency;
  switch (coll_type) {
    case type::COLL_ALL_REDUCE:
      // Ring all-reduce: 2 * (n-1)/n * data * beta
      latency = alpha + 2.0 * (n - 1.0) / n * data_size * beta;
      break;
    case type::COLL_ALL_GATHER:
      // Ring all-gather: (n-1)/n * data * beta
      latency = alpha + (n - 1.0) / n * data_size * n * beta;
      break;
    case type::COLL_REDUCE_SCATTER:
      // Reduce-scatter: (n-1)/n * data * beta
      latency = alpha + (n - 1.0) / n * data_size * beta;
      break;
    case type::COLL_BROADCAST:
      // Broadcast: log2(n) * alpha + (n-1)/n * data * beta
      latency = std::log2(n) * alpha + (n - 1.0) / n * data_size * beta;
      break;
    case type::COLL_P2P:
      // Point-to-point: alpha + data * beta
      latency = alpha + data_size * beta;
      break;
    default:
      latency = 0.0;
  }
  
  return latency;
}

double COMETCostModel::estimate_scheduling_overhead_ns(
    type::SchedulingStrategy strategy,
    int num_ops) const {
  
  // COMET scheduling overhead estimation
  double base_overhead;
  
  switch (strategy) {
    case type::SCHED_SEQUENTIAL:
      base_overhead = 50.0;  // Minimal overhead
      break;
    case type::SCHED_PIPELINED:
      base_overhead = 200.0;  // Pipeline setup cost
      break;
    case type::SCHED_PARALLEL:
      base_overhead = 500.0;  // Parallel dispatch cost
      break;
    default:
      base_overhead = 100.0;
  }
  
  return base_overhead + num_ops * 5.0;
}

double COMETCostModel::estimate_energy_pj(
    kernel::CompoundGraph const &graph) const {
  
  return graph.estimate_energy_pj();
}

double COMETCostModel::compute_score(
    double latency_ns,
    double energy_pj,
    COMETSearchConfig::Objective objective,
    double energy_weight) const {
  
  // Normalize metrics (lower is better, so invert for score)
  double latency_score = 1.0 / (1.0 + latency_ns / 1e6);
  double energy_score = 1.0 / (1.0 + energy_pj / 1e9);
  
  switch (objective) {
    case COMETSearchConfig::Objective::MINIMIZE_LATENCY:
      return latency_score;
    case COMETSearchConfig::Objective::MINIMIZE_ENERGY:
      return energy_score;
    case COMETSearchConfig::Objective::BALANCE:
      return (1.0 - energy_weight) * latency_score + 
             energy_weight * energy_score;
    default:
      return latency_score;
  }
}

} // namespace search
} // namespace yirage
