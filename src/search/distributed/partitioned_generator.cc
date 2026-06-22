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

#include "search/distributed/partitioned_generator.h"
#include "utils/json_utils.h"

#include <cstring>

namespace yirage {
namespace search {

// =============================================================================
// PartitionedKernelGraphGenerator Implementation
// =============================================================================

PartitionedKernelGraphGenerator::PartitionedKernelGraphGenerator(
    kernel::Graph const &computation_graph,
    GeneratorConfig const &config,
    SearchPartition const &partition,
    SearchCallback *callback)
    : config_(config),
      partition_(partition),
      callback_(callback),
      start_time_(std::chrono::steady_clock::now()) {
  
  // Create a modified config that only explores the partition's ranges
  GeneratorConfig partition_config = config;
  
  // Override grid/block dims with partition ranges
  partition_config.grid_dim_to_explore.clear();
  for (auto const &g : partition.grid_dim_range) {
    partition_config.grid_dim_to_explore.push_back(g);
  }
  
  partition_config.block_dim_to_explore.clear();
  for (auto const &b : partition.block_dim_range) {
    partition_config.block_dim_to_explore.push_back(b);
  }
  
  // Use partition's fmap/frange if specified
  if (!partition.fmap_range.empty()) {
    partition_config.fmap_to_explore.clear();
    for (auto f : partition.fmap_range) {
      partition_config.fmap_to_explore.push_back(f);
    }
  }
  
  if (!partition.frange_range.empty()) {
    partition_config.frange_to_explore.clear();
    for (auto f : partition.frange_range) {
      partition_config.frange_to_explore.push_back(f);
    }
  }
  
  // Create the base generator with partition-specific config
  generator_ = std::make_unique<KernelGraphGenerator>(
      computation_graph,
      partition_config,
      nullptr,  // checkpoint filename
      false     // verbose
  );
  
  // Initialize feedback
  feedback_.partition_id = partition.partition_id;
  feedback_.total_partitions = partition.total_partitions;
}

PartitionedKernelGraphGenerator::~PartitionedKernelGraphGenerator() = default;

void PartitionedKernelGraphGenerator::generate_kernel_graphs_for_partition() {
  // Run the base generator
  generator_->generate_kernel_graphs();
  
  // Collect results
  generated_graphs_ = generator_->generated_graphs;
  
  // Update feedback
  auto end_time = std::chrono::steady_clock::now();
  feedback_.search_time_seconds = 
      std::chrono::duration<double>(end_time - start_time_).count();
  feedback_.valid_graphs_found = static_cast<int>(generated_graphs_.size());
  // Note: num_total_states is private, use generated_graphs size as estimate
  feedback_.total_states_explored = static_cast<int>(generated_graphs_.size());
  
  is_complete_ = true;
}

SearchFeedback PartitionedKernelGraphGenerator::get_feedback() const {
  return feedback_;
}

std::vector<json> const &PartitionedKernelGraphGenerator::get_generated_graphs() const {
  return generated_graphs_;
}

PartitionedKernelGraphGenerator::Statistics 
PartitionedKernelGraphGenerator::get_statistics() const {
  Statistics stats;
  // Note: num_total_states is private, use generated_graphs size
  stats.states_explored = static_cast<int>(generated_graphs_.size());
  stats.valid_graphs_found = static_cast<int>(generated_graphs_.size());
  
  auto now = std::chrono::steady_clock::now();
  stats.elapsed_seconds = 
      std::chrono::duration<double>(now - start_time_).count();
  
  return stats;
}

bool PartitionedKernelGraphGenerator::is_config_in_partition(
    dim3 grid_dim, 
    dim3 block_dim) const {
  return partition_.contains(grid_dim, block_dim);
}

// =============================================================================
// SearchPartition Implementation
// =============================================================================

std::vector<SearchPartition> SearchPartition::create_partitions(
    GeneratorConfig const &config, 
    int num_partitions) {
  
  std::vector<SearchPartition> partitions;
  
  // Get all grid dims to partition
  std::vector<dim3> all_grids = config.grid_dim_to_explore;
  if (all_grids.empty()) {
    all_grids.push_back({1, 1, 1});
  }
  
  // Calculate grids per partition
  size_t grids_per_partition = std::max(
      size_t(1), 
      (all_grids.size() + num_partitions - 1) / num_partitions
  );
  
  for (int i = 0; i < num_partitions; ++i) {
    SearchPartition p;
    p.partition_id = i;
    p.total_partitions = num_partitions;
    
    // Assign grid dims to this partition
    size_t start_idx = i * grids_per_partition;
    size_t end_idx = std::min(start_idx + grids_per_partition, all_grids.size());
    
    if (start_idx < all_grids.size()) {
      for (size_t j = start_idx; j < end_idx; ++j) {
        p.grid_dim_range.push_back(all_grids[j]);
      }
    }
    
    // Each partition gets all block dims
    p.block_dim_range = config.block_dim_to_explore;
    if (p.block_dim_range.empty()) {
      p.block_dim_range.push_back({128, 1, 1});
    }
    
    // Copy fmap/frange ranges
    p.fmap_range = config.fmap_to_explore;
    p.frange_range = config.frange_to_explore;
    
    // Estimate candidates
    p.estimated_candidates = 
        p.grid_dim_range.size() * 
        p.block_dim_range.size() * 
        std::max(size_t(1), p.frange_range.size());
    
    partitions.push_back(p);
  }
  
  return partitions;
}

bool SearchPartition::contains(dim3 grid_dim, dim3 block_dim) const {
  // Check if grid_dim is in range
  bool grid_found = false;
  for (auto const &g : grid_dim_range) {
    if (g.x == grid_dim.x && g.y == grid_dim.y && g.z == grid_dim.z) {
      grid_found = true;
      break;
    }
  }
  if (!grid_found) return false;
  
  // Check if block_dim is in range
  for (auto const &b : block_dim_range) {
    if (b.x == block_dim.x && b.y == block_dim.y && b.z == block_dim.z) {
      return true;
    }
  }
  
  return false;
}

json SearchPartition::to_json() const {
  json j;
  j["partition_id"] = partition_id;
  j["total_partitions"] = total_partitions;
  
  json grids = json::array();
  for (auto const &g : grid_dim_range) {
    grids.push_back({{"x", g.x}, {"y", g.y}, {"z", g.z}});
  }
  j["grid_dim_range"] = grids;
  
  json blocks = json::array();
  for (auto const &b : block_dim_range) {
    blocks.push_back({{"x", b.x}, {"y", b.y}, {"z", b.z}});
  }
  j["block_dim_range"] = blocks;
  
  j["fmap_range"] = fmap_range;
  j["frange_range"] = frange_range;
  j["estimated_candidates"] = estimated_candidates;
  
  return j;
}

SearchPartition SearchPartition::from_json(json const &j) {
  SearchPartition p;
  p.partition_id = j.value("partition_id", 0);
  p.total_partitions = j.value("total_partitions", 1);
  
  if (j.contains("grid_dim_range")) {
    for (auto const &g : j["grid_dim_range"]) {
      p.grid_dim_range.push_back({
          g.value("x", 1u),
          g.value("y", 1u),
          g.value("z", 1u)
      });
    }
  }
  
  if (j.contains("block_dim_range")) {
    for (auto const &b : j["block_dim_range"]) {
      p.block_dim_range.push_back({
          b.value("x", 128u),
          b.value("y", 1u),
          b.value("z", 1u)
      });
    }
  }
  
  if (j.contains("fmap_range")) {
    p.fmap_range = j["fmap_range"].get<std::vector<int>>();
  }
  if (j.contains("frange_range")) {
    p.frange_range = j["frange_range"].get<std::vector<int>>();
  }
  
  p.estimated_candidates = j.value("estimated_candidates", size_t(0));
  
  return p;
}

// =============================================================================
// PartitionConfig Implementation
// =============================================================================

json PartitionConfig::to_json() const {
  return {
      {"num_partitions", num_partitions},
      {"strategy", static_cast<int>(strategy)},
      {"balance_by_estimate", balance_by_estimate}
  };
}

PartitionConfig PartitionConfig::from_json(json const &j) {
  PartitionConfig c;
  c.num_partitions = j.value("num_partitions", 1);
  c.strategy = static_cast<PartitionStrategy>(j.value("strategy", 0));
  c.balance_by_estimate = j.value("balance_by_estimate", true);
  return c;
}

// =============================================================================
// C Interface Implementation
// =============================================================================

namespace partitioned_search_c {

char *create_partitions(int num_partitions, char const *config_json) {
  try {
    json config = json::parse(config_json);
    
    // Build GeneratorConfig from JSON
    GeneratorConfig gen_config;
    
    if (config.contains("grid_dims")) {
      for (auto const &g : config["grid_dims"]) {
        if (g.is_array()) {
          gen_config.grid_dim_to_explore.push_back({
              g[0].get<unsigned int>(),
              g[1].get<unsigned int>(),
              g[2].get<unsigned int>()
          });
        } else {
          gen_config.grid_dim_to_explore.push_back({
              g.value("x", 1u),
              g.value("y", 1u),
              g.value("z", 1u)
          });
        }
      }
    }
    
    if (config.contains("block_dims")) {
      for (auto const &b : config["block_dims"]) {
        if (b.is_array()) {
          gen_config.block_dim_to_explore.push_back({
              b[0].get<unsigned int>(),
              b[1].get<unsigned int>(),
              b[2].get<unsigned int>()
          });
        } else {
          gen_config.block_dim_to_explore.push_back({
              b.value("x", 128u),
              b.value("y", 1u),
              b.value("z", 1u)
          });
        }
      }
    }
    
    if (config.contains("fmaps")) {
      gen_config.fmap_to_explore = config["fmaps"].get<std::vector<int>>();
    }
    if (config.contains("franges")) {
      gen_config.frange_to_explore = config["franges"].get<std::vector<int>>();
    }
    
    // Create partitions
    auto partitions = SearchPartition::create_partitions(gen_config, num_partitions);
    
    // Convert to JSON array
    json result = json::array();
    for (auto const &p : partitions) {
      result.push_back(p.to_json());
    }
    
    // Allocate and copy string
    std::string result_str = result.dump();
    char *output = static_cast<char*>(malloc(result_str.size() + 1));
    std::strcpy(output, result_str.c_str());
    return output;
    
  } catch (std::exception const &e) {
    return nullptr;
  }
}

int search_partition(
    kernel::Graph const *input_graph,
    char const *partition_json,
    char const *config_json,
    bool collect_feedback,
    int max_num_graphs,
    kernel::Graph **new_graphs,
    char **feedback_json) {
  
  if (!input_graph || !partition_json || !config_json) {
    return 0;
  }
  
  try {
    // Parse partition
    json partition_j = json::parse(partition_json);
    SearchPartition partition = SearchPartition::from_json(partition_j);
    
    // Parse config
    json config_j = json::parse(config_json);
    GeneratorConfig config;
    
    // Apply partition ranges to config
    config.grid_dim_to_explore = partition.grid_dim_range;
    config.block_dim_to_explore = partition.block_dim_range;
    config.fmap_to_explore = partition.fmap_range;
    config.frange_to_explore = partition.frange_range;
    
    // Other config from JSON
    config.max_num_threadblock_graph_op = config_j.value("max_num_threadblock_graph_op", 9);
    config.max_num_kernel_graph_op = config_j.value("max_num_kernel_graph_op", 5);
    config.search_thread = config_j.value("search_thread", 1);
    
    // Create and run generator
    PartitionedKernelGraphGenerator generator(*input_graph, config, partition);
    generator.generate_kernel_graphs_for_partition();
    
    // Get results
    auto const &graphs = generator.get_generated_graphs();
    int num_graphs = std::min(static_cast<int>(graphs.size()), max_num_graphs);
    
    // Note: For now, we don't actually return the graphs as pointers
    // because the graph lifetime management is complex.
    // The caller should use the JSON representation instead.
    
    // Return feedback if requested
    if (collect_feedback && feedback_json) {
      auto feedback = generator.get_feedback();
      std::string feedback_str = feedback.to_json().dump();
      *feedback_json = static_cast<char*>(malloc(feedback_str.size() + 1));
      std::strcpy(*feedback_json, feedback_str.c_str());
    }
    
    return num_graphs;
    
  } catch (std::exception const &e) {
    return 0;
  }
}

void free_json_string(char *json_str) {
  if (json_str) {
    free(json_str);
  }
}

} // namespace partitioned_search_c

} // namespace search
} // namespace yirage
