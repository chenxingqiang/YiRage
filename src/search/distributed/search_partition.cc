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

#include "search/distributed/search_partition.h"

#include <algorithm>
#include <cmath>

namespace yirage {
namespace search {

std::vector<SearchPartition>
SearchPartition::create_partitions(GeneratorConfig const &config,
                                   int num_partitions) {
  std::vector<SearchPartition> partitions;

  if (num_partitions <= 0) {
    num_partitions = 1;
  }

  // Get the full configuration space
  auto const &grid_dims = config.grid_dim_to_explore;
  auto const &block_dims = config.block_dim_to_explore;
  auto const &imaps = config.imap_to_explore;
  auto const &omaps = config.omap_to_explore;
  auto const &fmaps = config.fmap_to_explore;
  auto const &franges = config.frange_to_explore;

  // Calculate total configurations
  size_t total_configs = grid_dims.size() * block_dims.size() *
                         std::max(size_t(1), imaps.size()) *
                         std::max(size_t(1), omaps.size()) *
                         std::max(size_t(1), franges.size());

  // Primary partitioning by grid_dim (usually has most impact on search)
  size_t grids_per_partition =
      (grid_dims.size() + num_partitions - 1) / num_partitions;

  for (int i = 0; i < num_partitions; ++i) {
    SearchPartition partition;
    partition.partition_id = i;
    partition.total_partitions = num_partitions;

    // Assign grid dimensions to this partition
    size_t start_idx = i * grids_per_partition;
    size_t end_idx = std::min(start_idx + grids_per_partition, grid_dims.size());

    if (start_idx >= grid_dims.size()) {
      // No more grid dims to assign - empty partition
      partition.estimated_candidates = 0;
      partitions.push_back(partition);
      continue;
    }

    for (size_t j = start_idx; j < end_idx; ++j) {
      partition.grid_dim_range.push_back(grid_dims[j]);
    }

    // Each partition gets all block_dims, imaps, omaps, franges
    partition.block_dim_range = block_dims;
    partition.imap_range = imaps;
    partition.omap_range = omaps;
    partition.fmap_range = fmaps;
    partition.frange_range = franges;

    // Estimate candidates in this partition
    partition.estimated_candidates = partition.grid_dim_range.size() *
                                     partition.block_dim_range.size() *
                                     std::max(size_t(1), partition.imap_range.size()) *
                                     std::max(size_t(1), partition.omap_range.size()) *
                                     std::max(size_t(1), partition.frange_range.size());

    partitions.push_back(partition);
  }

  return partitions;
}

bool SearchPartition::contains(dim3 grid_dim, dim3 block_dim) const {
  // Check if grid_dim is in our range
  bool grid_found = false;
  for (auto const &g : grid_dim_range) {
    if (g.x == grid_dim.x && g.y == grid_dim.y && g.z == grid_dim.z) {
      grid_found = true;
      break;
    }
  }

  if (!grid_found) {
    return false;
  }

  // Block dim check (if not empty)
  if (!block_dim_range.empty()) {
    bool block_found = false;
    for (auto const &b : block_dim_range) {
      if (b.x == block_dim.x && b.y == block_dim.y && b.z == block_dim.z) {
        block_found = true;
        break;
      }
    }
    if (!block_found) {
      return false;
    }
  }

  return true;
}

json SearchPartition::to_json() const {
  json j;
  j["partition_id"] = partition_id;
  j["total_partitions"] = total_partitions;
  j["estimated_candidates"] = estimated_candidates;

  // Grid dims
  json grid_arr = json::array();
  for (auto const &g : grid_dim_range) {
    grid_arr.push_back({{"x", g.x}, {"y", g.y}, {"z", g.z}});
  }
  j["grid_dim_range"] = grid_arr;

  // Block dims
  json block_arr = json::array();
  for (auto const &b : block_dim_range) {
    block_arr.push_back({{"x", b.x}, {"y", b.y}, {"z", b.z}});
  }
  j["block_dim_range"] = block_arr;

  // Imaps
  json imap_arr = json::array();
  for (auto const &m : imap_range) {
    imap_arr.push_back({{"x", m.x}, {"y", m.y}, {"z", m.z}});
  }
  j["imap_range"] = imap_arr;

  // Omaps
  json omap_arr = json::array();
  for (auto const &m : omap_range) {
    omap_arr.push_back({{"x", m.x}, {"y", m.y}, {"z", m.z}});
  }
  j["omap_range"] = omap_arr;

  // Fmaps and franges
  j["fmap_range"] = fmap_range;
  j["frange_range"] = frange_range;

  return j;
}

SearchPartition SearchPartition::from_json(json const &j) {
  SearchPartition p;

  p.partition_id = j.value("partition_id", 0);
  p.total_partitions = j.value("total_partitions", 1);
  p.estimated_candidates = j.value("estimated_candidates", size_t(0));

  // Grid dims
  if (j.contains("grid_dim_range")) {
    for (auto const &g : j["grid_dim_range"]) {
      dim3 d;
      d.x = g.value("x", 1u);
      d.y = g.value("y", 1u);
      d.z = g.value("z", 1u);
      p.grid_dim_range.push_back(d);
    }
  }

  // Block dims
  if (j.contains("block_dim_range")) {
    for (auto const &b : j["block_dim_range"]) {
      dim3 d;
      d.x = b.value("x", 128u);
      d.y = b.value("y", 1u);
      d.z = b.value("z", 1u);
      p.block_dim_range.push_back(d);
    }
  }

  // Imaps
  if (j.contains("imap_range")) {
    for (auto const &m : j["imap_range"]) {
      int3 i;
      i.x = m.value("x", 0);
      i.y = m.value("y", 0);
      i.z = m.value("z", 0);
      p.imap_range.push_back(i);
    }
  }

  // Omaps
  if (j.contains("omap_range")) {
    for (auto const &m : j["omap_range"]) {
      int3 i;
      i.x = m.value("x", 0);
      i.y = m.value("y", 0);
      i.z = m.value("z", 0);
      p.omap_range.push_back(i);
    }
  }

  // Fmaps and franges
  if (j.contains("fmap_range")) {
    p.fmap_range = j["fmap_range"].get<std::vector<int>>();
  }
  if (j.contains("frange_range")) {
    p.frange_range = j["frange_range"].get<std::vector<int>>();
  }

  return p;
}

json PartitionConfig::to_json() const {
  return {{"num_partitions", num_partitions},
          {"strategy", static_cast<int>(strategy)},
          {"balance_by_estimate", balance_by_estimate}};
}

PartitionConfig PartitionConfig::from_json(json const &j) {
  PartitionConfig c;
  c.num_partitions = j.value("num_partitions", 1);
  c.strategy = static_cast<PartitionStrategy>(j.value("strategy", 0));
  c.balance_by_estimate = j.value("balance_by_estimate", true);
  return c;
}

} // namespace search
} // namespace yirage
