/* Copyright 2023-2024 CMU
 * Copyright 2025 Chen Xingqiang (YiRage Project)
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

#include "kernel/device_tensor.h"
#include "type.h"
#include "utils/json_utils.h"
#include <memory>
#include <string>
#include <vector>

namespace yirage {
namespace kernel {

// Forward declarations
class Graph;
class KNOperator;

// =============================================================================
// COMET Compound Graph Representation
// =============================================================================
// This implements the tree-based representation from the COMET paper for
// modeling compound operations with explicit collectives.
//
// A CompoundGraph represents a compound operation (e.g., GEMM-Softmax,
// Self-Attention) as a tree of TileNodes, where each TileNode represents
// a tiled computation with associated memory staging and collective operations.
//
// Reference: "COMET: A Framework for Modeling Compound Operation Dataflows
// with Explicit Collectives" (Negi et al.)

/**
 * TileNode represents a tiled computation in the COMET tree.
 * 
 * Corresponds to T_i^j in the COMET paper, where:
 * - i is the operation index
 * - j is the tile index
 * 
 * Each TileNode has:
 * - Operation type (GEMM, Softmax, LayerNorm, etc.)
 * - Tile dimensions
 * - Memory staging level (DRAM, GB, IB/WB/OB)
 * - Optional collective operation
 * - Children nodes (for fused operations)
 */
struct TileNode {
  // Node identification
  int op_index;           // Operation index in the compound op
  int tile_index;         // Tile index within the operation
  std::string name;       // Human-readable name
  
  // Operation info
  yirage::type::KNOperatorType op_type;
  std::vector<int> tile_dims;  // Tile dimensions [M_tile, N_tile, K_tile]
  
  // Memory staging (COMET Section IV-A)
  yirage::type::MemoryLevel src_memory;  // Source memory level
  yirage::type::MemoryLevel dst_memory;  // Destination memory level
  
  // Collective operation (if any)
  bool has_collective;
  yirage::type::CollectiveOpType collective_type;
  yirage::type::CollectiveReduceOp collective_reduce_op;
  int collective_participants;
  
  // Tree structure
  std::vector<std::shared_ptr<TileNode>> children;
  TileNode* parent;
  
  // Data staging state (COMET Eq. 2)
  yirage::type::DataStagingState staging_state;
  
  // Cost model attributes
  int64_t compute_flops;       // FLOPs for this tile
  int64_t memory_bytes;        // Memory traffic for this tile
  double compute_latency_ns;   // Compute latency
  double memory_latency_ns;    // Memory latency
  double collective_latency_ns; // Collective latency
  
  TileNode()
      : op_index(0), tile_index(0), name(""),
        op_type(yirage::type::KN_UNKOWN),
        src_memory(yirage::type::MEM_DRAM),
        dst_memory(yirage::type::MEM_GLOBAL_BUFFER),
        has_collective(false),
        collective_type(yirage::type::COLL_NONE),
        collective_reduce_op(yirage::type::REDUCE_SUM),
        collective_participants(1),
        parent(nullptr),
        staging_state(yirage::type::STAGE_IDLE),
        compute_flops(0), memory_bytes(0),
        compute_latency_ns(0), memory_latency_ns(0),
        collective_latency_ns(0) {}
  
  // JSON serialization
  operator json() const;
};

void from_json(json const &j, TileNode &node);

/**
 * CompoundGraph represents a complete compound operation.
 * 
 * This is the top-level structure for COMET's tree-based representation.
 * It contains:
 * - The root TileNode (representing the fused operation)
 * - Scheduling strategy
 * - Cost model configuration
 */
class CompoundGraph {
public:
  CompoundGraph();
  CompoundGraph(yirage::type::CompoundOpType type);
  ~CompoundGraph();
  
  // Build compound operations
  void set_type(yirage::type::CompoundOpType type);
  void set_scheduling_strategy(yirage::type::SchedulingStrategy strategy);
  
  // Add operations to the compound graph
  void add_tile_node(std::shared_ptr<TileNode> node);
  void set_root(std::shared_ptr<TileNode> root);
  
  // Build predefined compound operations
  void build_gemm_softmax(
      std::vector<int> const &A_dims,
      std::vector<int> const &B_dims,
      std::vector<int> const &tile_dims,
      int num_devices = 1);
  
  void build_gemm_layernorm(
      std::vector<int> const &A_dims,
      std::vector<int> const &B_dims,
      std::vector<int> const &tile_dims,
      int num_devices = 1);
  
  void build_self_attention(
      int batch, int heads, int seq_len, int head_dim,
      std::vector<int> const &tile_dims,
      int num_devices = 1);
  
  void build_gated_mlp(
      int batch, int seq_len, int hidden_dim, int ff_dim,
      std::vector<int> const &tile_dims,
      int num_devices = 1);
  
  // Cost estimation (COMET Equations)
  double estimate_latency_ns() const;
  double estimate_energy_pj() const;
  
  // Latency breakdown
  double get_compute_latency_ns() const;
  double get_memory_latency_ns() const;
  double get_collective_latency_ns() const;
  double get_scheduling_overhead_ns() const;
  
  // Memory traffic
  int64_t get_total_memory_bytes() const;
  int64_t get_dram_traffic_bytes() const;
  int64_t get_onchip_traffic_bytes() const;
  
  // Convert to/from kernel Graph
  void lower_to_graph(Graph* graph) const;
  static CompoundGraph lift_from_graph(Graph const* graph);
  
  // Serialization
  operator json() const;
  static CompoundGraph from_json(json const &j);
  
  // Getters
  yirage::type::CompoundOpType get_type() const { return compound_type; }
  yirage::type::SchedulingStrategy get_strategy() const { return strategy; }
  std::shared_ptr<TileNode> get_root() const { return root; }
  std::vector<std::shared_ptr<TileNode>> const& get_nodes() const { return nodes; }
  
private:
  yirage::type::CompoundOpType compound_type;
  yirage::type::SchedulingStrategy strategy;
  std::shared_ptr<TileNode> root;
  std::vector<std::shared_ptr<TileNode>> nodes;
  
  // Hardware configuration for cost model
  double dram_bandwidth_gbps;
  double onchip_bandwidth_gbps;
  double peak_tflops;
  double noc_bandwidth_gbps;
  double noc_latency_ns;
  
  // Helper functions
  void compute_tile_costs(TileNode* node);
  double compute_ramp_up_latency(TileNode* node) const;
  double compute_steady_state_latency(TileNode* node) const;
  double compute_ramp_down_latency(TileNode* node) const;
};

} // namespace kernel
} // namespace yirage
