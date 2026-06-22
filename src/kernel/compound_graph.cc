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

#include "kernel/compound_graph.h"
#include "kernel/graph.h"
#include <algorithm>
#include <cmath>

namespace yirage {
namespace kernel {

// =============================================================================
// TileNode Implementation
// =============================================================================

TileNode::operator json() const {
  json j = {
    {"op_index", op_index},
    {"tile_index", tile_index},
    {"name", name},
    {"op_type", op_type},
    {"tile_dims", tile_dims},
    {"src_memory", src_memory},
    {"dst_memory", dst_memory},
    {"has_collective", has_collective},
    {"collective_type", collective_type},
    {"collective_reduce_op", collective_reduce_op},
    {"collective_participants", collective_participants},
    {"staging_state", staging_state},
    {"compute_flops", compute_flops},
    {"memory_bytes", memory_bytes},
    {"compute_latency_ns", compute_latency_ns},
    {"memory_latency_ns", memory_latency_ns},
    {"collective_latency_ns", collective_latency_ns}
  };
  
  // Serialize children
  json children_json = json::array();
  for (auto const& child : children) {
    children_json.push_back(*child);
  }
  j["children"] = children_json;
  
  return j;
}

void from_json(json const &j, TileNode &node) {
  node.op_index = j.at("op_index").get<int>();
  node.tile_index = j.at("tile_index").get<int>();
  node.name = j.at("name").get<std::string>();
  node.op_type = j.at("op_type").get<yirage::type::KNOperatorType>();
  node.tile_dims = j.at("tile_dims").get<std::vector<int>>();
  node.src_memory = j.at("src_memory").get<yirage::type::MemoryLevel>();
  node.dst_memory = j.at("dst_memory").get<yirage::type::MemoryLevel>();
  node.has_collective = j.at("has_collective").get<bool>();
  node.collective_type = j.at("collective_type").get<yirage::type::CollectiveOpType>();
  node.collective_reduce_op = j.at("collective_reduce_op").get<yirage::type::CollectiveReduceOp>();
  node.collective_participants = j.at("collective_participants").get<int>();
  node.staging_state = j.at("staging_state").get<yirage::type::DataStagingState>();
  node.compute_flops = j.at("compute_flops").get<int64_t>();
  node.memory_bytes = j.at("memory_bytes").get<int64_t>();
  node.compute_latency_ns = j.at("compute_latency_ns").get<double>();
  node.memory_latency_ns = j.at("memory_latency_ns").get<double>();
  node.collective_latency_ns = j.at("collective_latency_ns").get<double>();
  
  // Deserialize children
  for (auto const& child_json : j.at("children")) {
    auto child = std::make_shared<TileNode>();
    from_json(child_json, *child);
    child->parent = &node;
    node.children.push_back(child);
  }
}

// =============================================================================
// CompoundGraph Implementation
// =============================================================================

CompoundGraph::CompoundGraph()
    : compound_type(yirage::type::COMP_CUSTOM),
      strategy(yirage::type::SCHED_PIPELINED),
      root(nullptr),
      dram_bandwidth_gbps(900.0),    // Default: HBM2e
      onchip_bandwidth_gbps(3000.0), // Default: L2 cache
      peak_tflops(312.0),            // Default: A100
      noc_bandwidth_gbps(600.0),     // Default: NVLink
      noc_latency_ns(10.0) {}

CompoundGraph::CompoundGraph(yirage::type::CompoundOpType type)
    : CompoundGraph() {
  compound_type = type;
}

CompoundGraph::~CompoundGraph() {}

void CompoundGraph::set_type(yirage::type::CompoundOpType type) {
  compound_type = type;
}

void CompoundGraph::set_scheduling_strategy(yirage::type::SchedulingStrategy s) {
  strategy = s;
}

void CompoundGraph::add_tile_node(std::shared_ptr<TileNode> node) {
  nodes.push_back(node);
}

void CompoundGraph::set_root(std::shared_ptr<TileNode> r) {
  root = r;
}

// =============================================================================
// Build Predefined Compound Operations
// =============================================================================

void CompoundGraph::build_gemm_softmax(
    std::vector<int> const &A_dims,
    std::vector<int> const &B_dims,
    std::vector<int> const &tile_dims,
    int num_devices) {
  
  compound_type = yirage::type::COMP_GEMM_SOFTMAX;
  
  int M = A_dims[0];
  int K = A_dims[1];
  int N = B_dims[1];
  
  int M_tile = tile_dims.size() > 0 ? tile_dims[0] : 128;
  int N_tile = tile_dims.size() > 1 ? tile_dims[1] : 128;
  int K_tile = tile_dims.size() > 2 ? tile_dims[2] : 128;
  
  // Create root node (fused GEMM-Softmax)
  root = std::make_shared<TileNode>();
  root->name = "GEMM_Softmax_Root";
  root->op_index = 0;
  root->tile_index = 0;
  root->op_type = yirage::type::KN_MATMUL_OP;
  root->tile_dims = {M_tile, N_tile, K_tile};
  root->src_memory = yirage::type::MEM_DRAM;
  root->dst_memory = yirage::type::MEM_GLOBAL_BUFFER;
  
  // GEMM FLOPs: 2*M*N*K
  root->compute_flops = 2LL * M * N * K;
  // Memory: read A, B, write C (kept on chip)
  root->memory_bytes = (int64_t)(M * K + K * N) * 2;  // FP16
  
  // Add softmax child node
  auto softmax_node = std::make_shared<TileNode>();
  softmax_node->name = "Softmax";
  softmax_node->op_index = 1;
  softmax_node->tile_index = 0;
  softmax_node->op_type = yirage::type::KN_EXP_OP;  // Part of softmax
  softmax_node->tile_dims = {M_tile, N_tile};
  softmax_node->src_memory = yirage::type::MEM_GLOBAL_BUFFER;  // From GEMM output
  softmax_node->dst_memory = yirage::type::MEM_DRAM;
  
  // Softmax FLOPs: 5*M*N (max, sub, exp, sum, div)
  softmax_node->compute_flops = 5LL * M * N;
  // Memory: write D to DRAM
  softmax_node->memory_bytes = (int64_t)(M * N) * 2;  // FP16
  
  softmax_node->parent = root.get();
  root->children.push_back(softmax_node);
  
  // Add collective if distributed
  if (num_devices > 1) {
    softmax_node->has_collective = true;
    softmax_node->collective_type = yirage::type::COLL_ALL_REDUCE;
    softmax_node->collective_reduce_op = yirage::type::REDUCE_SUM;
    softmax_node->collective_participants = num_devices;
  }
  
  nodes.push_back(root);
  nodes.push_back(softmax_node);
  
  // Compute costs
  compute_tile_costs(root.get());
}

void CompoundGraph::build_gemm_layernorm(
    std::vector<int> const &A_dims,
    std::vector<int> const &B_dims,
    std::vector<int> const &tile_dims,
    int num_devices) {
  
  compound_type = yirage::type::COMP_GEMM_LAYERNORM;
  
  int M = A_dims[0];
  int K = A_dims[1];
  int N = B_dims[1];
  
  int M_tile = tile_dims.size() > 0 ? tile_dims[0] : 128;
  int N_tile = tile_dims.size() > 1 ? tile_dims[1] : 128;
  int K_tile = tile_dims.size() > 2 ? tile_dims[2] : 128;
  
  root = std::make_shared<TileNode>();
  root->name = "GEMM_LayerNorm_Root";
  root->op_index = 0;
  root->tile_index = 0;
  root->op_type = yirage::type::KN_MATMUL_OP;
  root->tile_dims = {M_tile, N_tile, K_tile};
  root->src_memory = yirage::type::MEM_DRAM;
  root->dst_memory = yirage::type::MEM_GLOBAL_BUFFER;
  root->compute_flops = 2LL * M * N * K;
  root->memory_bytes = (int64_t)(M * K + K * N) * 2;
  
  // LayerNorm child
  auto ln_node = std::make_shared<TileNode>();
  ln_node->name = "LayerNorm";
  ln_node->op_index = 1;
  ln_node->tile_index = 0;
  ln_node->op_type = yirage::type::KN_RMS_NORM_OP;
  ln_node->tile_dims = {M_tile, N_tile};
  ln_node->src_memory = yirage::type::MEM_GLOBAL_BUFFER;
  ln_node->dst_memory = yirage::type::MEM_DRAM;
  // LayerNorm FLOPs: 7*M*N (mean, sub, var, sqrt, div, scale, bias)
  ln_node->compute_flops = 7LL * M * N;
  ln_node->memory_bytes = (int64_t)(M * N) * 2;
  
  ln_node->parent = root.get();
  root->children.push_back(ln_node);
  
  if (num_devices > 1) {
    ln_node->has_collective = true;
    ln_node->collective_type = yirage::type::COLL_ALL_REDUCE;
    ln_node->collective_reduce_op = yirage::type::REDUCE_SUM;
    ln_node->collective_participants = num_devices;
  }
  
  nodes.push_back(root);
  nodes.push_back(ln_node);
  compute_tile_costs(root.get());
}

void CompoundGraph::build_self_attention(
    int batch, int heads, int seq_len, int head_dim,
    std::vector<int> const &tile_dims,
    int num_devices) {
  
  compound_type = yirage::type::COMP_SELF_ATTENTION;
  
  int S_tile = tile_dims.size() > 0 ? tile_dims[0] : 64;
  int D_tile = tile_dims.size() > 1 ? tile_dims[1] : 64;
  
  // Root: Q @ K^T
  root = std::make_shared<TileNode>();
  root->name = "QK_MatMul";
  root->op_index = 0;
  root->tile_index = 0;
  root->op_type = yirage::type::KN_MATMUL_OP;
  root->tile_dims = {S_tile, S_tile, D_tile};
  root->src_memory = yirage::type::MEM_DRAM;
  root->dst_memory = yirage::type::MEM_GLOBAL_BUFFER;
  // QK^T FLOPs: 2*B*H*S*S*D
  root->compute_flops = 2LL * batch * heads * seq_len * seq_len * head_dim;
  root->memory_bytes = (int64_t)(batch * heads * seq_len * head_dim * 2) * 2;  // Q and K
  
  // Softmax child
  auto softmax_node = std::make_shared<TileNode>();
  softmax_node->name = "Attention_Softmax";
  softmax_node->op_index = 1;
  softmax_node->tile_index = 0;
  softmax_node->op_type = yirage::type::KN_EXP_OP;
  softmax_node->tile_dims = {S_tile, S_tile};
  softmax_node->src_memory = yirage::type::MEM_GLOBAL_BUFFER;
  softmax_node->dst_memory = yirage::type::MEM_GLOBAL_BUFFER;  // Keep on chip for @V
  // Softmax FLOPs: 5*B*H*S*S
  softmax_node->compute_flops = 5LL * batch * heads * seq_len * seq_len;
  softmax_node->memory_bytes = 0;  // Stays on chip
  
  softmax_node->parent = root.get();
  root->children.push_back(softmax_node);
  
  // @V child
  auto av_node = std::make_shared<TileNode>();
  av_node->name = "Attn_V_MatMul";
  av_node->op_index = 2;
  av_node->tile_index = 0;
  av_node->op_type = yirage::type::KN_MATMUL_OP;
  av_node->tile_dims = {S_tile, D_tile, S_tile};
  av_node->src_memory = yirage::type::MEM_GLOBAL_BUFFER;
  av_node->dst_memory = yirage::type::MEM_DRAM;
  // A@V FLOPs: 2*B*H*S*D*S
  av_node->compute_flops = 2LL * batch * heads * seq_len * head_dim * seq_len;
  // Read V, write output
  av_node->memory_bytes = (int64_t)(batch * heads * seq_len * head_dim) * 2 * 2;
  
  av_node->parent = softmax_node.get();
  softmax_node->children.push_back(av_node);
  
  if (num_devices > 1) {
    av_node->has_collective = true;
    av_node->collective_type = yirage::type::COLL_ALL_REDUCE;
    av_node->collective_reduce_op = yirage::type::REDUCE_SUM;
    av_node->collective_participants = num_devices;
  }
  
  nodes.push_back(root);
  nodes.push_back(softmax_node);
  nodes.push_back(av_node);
  compute_tile_costs(root.get());
}

void CompoundGraph::build_gated_mlp(
    int batch, int seq_len, int hidden_dim, int ff_dim,
    std::vector<int> const &tile_dims,
    int num_devices) {
  
  compound_type = yirage::type::COMP_GATED_MLP;
  
  int M_tile = tile_dims.size() > 0 ? tile_dims[0] : 128;
  int N_tile = tile_dims.size() > 1 ? tile_dims[1] : 128;
  int K_tile = tile_dims.size() > 2 ? tile_dims[2] : 128;
  
  int M = batch * seq_len;
  
  // Root: X @ W_gate
  root = std::make_shared<TileNode>();
  root->name = "Gate_Proj";
  root->op_index = 0;
  root->tile_index = 0;
  root->op_type = yirage::type::KN_MATMUL_OP;
  root->tile_dims = {M_tile, N_tile, K_tile};
  root->src_memory = yirage::type::MEM_DRAM;
  root->dst_memory = yirage::type::MEM_GLOBAL_BUFFER;
  root->compute_flops = 2LL * M * ff_dim * hidden_dim;
  root->memory_bytes = (int64_t)(M * hidden_dim + hidden_dim * ff_dim) * 2;
  
  // SiLU activation
  auto silu_node = std::make_shared<TileNode>();
  silu_node->name = "SiLU";
  silu_node->op_index = 1;
  silu_node->tile_index = 0;
  silu_node->op_type = yirage::type::KN_SILU_OP;
  silu_node->tile_dims = {M_tile, N_tile};
  silu_node->src_memory = yirage::type::MEM_GLOBAL_BUFFER;
  silu_node->dst_memory = yirage::type::MEM_GLOBAL_BUFFER;
  silu_node->compute_flops = 4LL * M * ff_dim;  // SiLU: x * sigmoid(x)
  silu_node->memory_bytes = 0;  // On chip
  
  silu_node->parent = root.get();
  root->children.push_back(silu_node);
  
  // Up projection (parallel with gate)
  auto up_node = std::make_shared<TileNode>();
  up_node->name = "Up_Proj";
  up_node->op_index = 2;
  up_node->tile_index = 0;
  up_node->op_type = yirage::type::KN_MATMUL_OP;
  up_node->tile_dims = {M_tile, N_tile, K_tile};
  up_node->src_memory = yirage::type::MEM_DRAM;
  up_node->dst_memory = yirage::type::MEM_GLOBAL_BUFFER;
  up_node->compute_flops = 2LL * M * ff_dim * hidden_dim;
  up_node->memory_bytes = (int64_t)(hidden_dim * ff_dim) * 2;  // Just W_up (X shared)
  
  // Element-wise multiply
  auto mul_node = std::make_shared<TileNode>();
  mul_node->name = "Gate_Up_Mul";
  mul_node->op_index = 3;
  mul_node->tile_index = 0;
  mul_node->op_type = yirage::type::KN_MUL_OP;
  mul_node->tile_dims = {M_tile, N_tile};
  mul_node->src_memory = yirage::type::MEM_GLOBAL_BUFFER;
  mul_node->dst_memory = yirage::type::MEM_GLOBAL_BUFFER;
  mul_node->compute_flops = (int64_t)M * ff_dim;
  mul_node->memory_bytes = 0;
  
  mul_node->parent = silu_node.get();
  silu_node->children.push_back(mul_node);
  
  // Down projection
  auto down_node = std::make_shared<TileNode>();
  down_node->name = "Down_Proj";
  down_node->op_index = 4;
  down_node->tile_index = 0;
  down_node->op_type = yirage::type::KN_MATMUL_OP;
  down_node->tile_dims = {M_tile, K_tile, N_tile};
  down_node->src_memory = yirage::type::MEM_GLOBAL_BUFFER;
  down_node->dst_memory = yirage::type::MEM_DRAM;
  down_node->compute_flops = 2LL * M * hidden_dim * ff_dim;
  down_node->memory_bytes = (int64_t)(ff_dim * hidden_dim + M * hidden_dim) * 2;
  
  down_node->parent = mul_node.get();
  mul_node->children.push_back(down_node);
  
  if (num_devices > 1) {
    down_node->has_collective = true;
    down_node->collective_type = yirage::type::COLL_ALL_REDUCE;
    down_node->collective_reduce_op = yirage::type::REDUCE_SUM;
    down_node->collective_participants = num_devices;
  }
  
  nodes.push_back(root);
  nodes.push_back(silu_node);
  nodes.push_back(up_node);
  nodes.push_back(mul_node);
  nodes.push_back(down_node);
  compute_tile_costs(root.get());
}

// =============================================================================
// Cost Estimation (COMET Equations)
// =============================================================================

void CompoundGraph::compute_tile_costs(TileNode* node) {
  if (!node) return;
  
  // Compute latency = FLOPs / peak_throughput
  node->compute_latency_ns = (double)node->compute_flops / 
                             (peak_tflops * 1e3);  // TFLOPS -> ns
  
  // Memory latency based on source/destination levels
  double bandwidth_gbps = (node->src_memory == yirage::type::MEM_DRAM ||
                          node->dst_memory == yirage::type::MEM_DRAM)
                         ? dram_bandwidth_gbps
                         : onchip_bandwidth_gbps;
  
  // Memory latency = bytes / bandwidth (ns)
  node->memory_latency_ns = (double)node->memory_bytes / bandwidth_gbps;
  
  // Collective latency (if applicable)
  if (node->has_collective && node->collective_participants > 1) {
    int n = node->collective_participants;
    double factor = 2.0 * (n - 1) / n;  // Ring all-reduce
    int64_t collective_bytes = node->memory_bytes;
    node->collective_latency_ns = factor * collective_bytes / noc_bandwidth_gbps +
                                  2 * (n - 1) * noc_latency_ns;
  }
  
  // Recursively compute for children
  for (auto& child : node->children) {
    compute_tile_costs(child.get());
  }
}

double CompoundGraph::estimate_latency_ns() const {
  return get_compute_latency_ns() + get_memory_latency_ns() + 
         get_collective_latency_ns() + get_scheduling_overhead_ns();
}

double CompoundGraph::get_compute_latency_ns() const {
  double total = 0;
  for (auto const& node : nodes) {
    total += node->compute_latency_ns;
  }
  return total;
}

double CompoundGraph::get_memory_latency_ns() const {
  double total = 0;
  for (auto const& node : nodes) {
    total += node->memory_latency_ns;
  }
  return total;
}

double CompoundGraph::get_collective_latency_ns() const {
  double total = 0;
  for (auto const& node : nodes) {
    total += node->collective_latency_ns;
  }
  return total;
}

double CompoundGraph::get_scheduling_overhead_ns() const {
  // Simplified scheduling overhead based on strategy
  double overhead = 0;
  
  switch (strategy) {
    case yirage::type::SCHED_SEQUENTIAL:
      // No overlap, full stall between ops
      overhead = nodes.size() * 100;  // 100ns per op
      break;
    case yirage::type::SCHED_PIPELINED:
      // Some overlap, reduced stall
      overhead = nodes.size() * 20;   // 20ns per op
      break;
    case yirage::type::SCHED_PARALLEL:
      // Maximum overlap, minimal stall
      overhead = nodes.size() * 5;    // 5ns per op
      break;
  }
  
  return overhead;
}

int64_t CompoundGraph::get_total_memory_bytes() const {
  int64_t total = 0;
  for (auto const& node : nodes) {
    total += node->memory_bytes;
  }
  return total;
}

int64_t CompoundGraph::get_dram_traffic_bytes() const {
  int64_t total = 0;
  for (auto const& node : nodes) {
    if (node->src_memory == yirage::type::MEM_DRAM ||
        node->dst_memory == yirage::type::MEM_DRAM) {
      total += node->memory_bytes;
    }
  }
  return total;
}

int64_t CompoundGraph::get_onchip_traffic_bytes() const {
  return get_total_memory_bytes() - get_dram_traffic_bytes();
}

double CompoundGraph::estimate_energy_pj() const {
  // Simplified energy model
  double compute_energy = 0;
  double memory_energy = 0;
  
  for (auto const& node : nodes) {
    // ~1 pJ per FLOP
    compute_energy += node->compute_flops;
    
    // DRAM: ~10 pJ/bit, on-chip: ~1 pJ/bit
    if (node->src_memory == yirage::type::MEM_DRAM ||
        node->dst_memory == yirage::type::MEM_DRAM) {
      memory_energy += node->memory_bytes * 8 * 10;  // pJ
    } else {
      memory_energy += node->memory_bytes * 8 * 1;   // pJ
    }
  }
  
  return compute_energy + memory_energy;
}

// =============================================================================
// Serialization
// =============================================================================

CompoundGraph::operator json() const {
  json j = {
    {"compound_type", compound_type},
    {"strategy", strategy},
    {"dram_bandwidth_gbps", dram_bandwidth_gbps},
    {"onchip_bandwidth_gbps", onchip_bandwidth_gbps},
    {"peak_tflops", peak_tflops}
  };
  
  if (root) {
    j["root"] = *root;
  }
  
  return j;
}

CompoundGraph CompoundGraph::from_json(json const &j) {
  CompoundGraph cg;
  cg.compound_type = j.at("compound_type").get<yirage::type::CompoundOpType>();
  cg.strategy = j.at("strategy").get<yirage::type::SchedulingStrategy>();
  cg.dram_bandwidth_gbps = j.at("dram_bandwidth_gbps").get<double>();
  cg.onchip_bandwidth_gbps = j.at("onchip_bandwidth_gbps").get<double>();
  cg.peak_tflops = j.at("peak_tflops").get<double>();
  
  if (j.contains("root")) {
    cg.root = std::make_shared<TileNode>();
    yirage::kernel::from_json(j.at("root"), *cg.root);
  }
  
  return cg;
}

} // namespace kernel
} // namespace yirage
