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

#include "search/config.h"
#include "utils/json_utils.h"
#include "vector_types.h"

#include <vector>

namespace yirage {
namespace search {

/**
 * @brief Search space partition for distributed search
 *
 * Each partition represents a subset of the configuration space
 * that can be explored independently by a worker.
 */
struct SearchPartition {
  // Partition identification
  int partition_id = 0;
  int total_partitions = 1;

  // Configuration space ranges for this partition
  std::vector<dim3> grid_dim_range;
  std::vector<dim3> block_dim_range;
  std::vector<int3> imap_range;
  std::vector<int3> omap_range;
  std::vector<int> fmap_range;
  std::vector<int> frange_range;

  // Partition metadata
  size_t estimated_candidates = 0;

  /**
   * @brief Create partitions from full configuration space
   *
   * Divides the search space into roughly equal partitions.
   * Uses grid_dim as primary partitioning dimension.
   *
   * @param config Full search configuration
   * @param num_partitions Number of partitions to create
   * @return Vector of partitions
   */
  static std::vector<SearchPartition>
  create_partitions(GeneratorConfig const &config, int num_partitions);

  /**
   * @brief Check if a configuration belongs to this partition
   */
  bool contains(dim3 grid_dim, dim3 block_dim) const;

  /**
   * @brief Convert partition to JSON for serialization
   */
  json to_json() const;

  /**
   * @brief Create partition from JSON
   */
  static SearchPartition from_json(json const &j);
};

/**
 * @brief Partition strategy for load balancing
 */
enum class PartitionStrategy {
  // Partition by grid dimensions (default)
  BY_GRID_DIM,
  // Partition by block dimensions
  BY_BLOCK_DIM,
  // Partition by combined grid+block hash
  BY_CONFIG_HASH,
  // Round-robin assignment
  ROUND_ROBIN,
};

/**
 * @brief Configuration for partition creation
 */
struct PartitionConfig {
  int num_partitions = 1;
  PartitionStrategy strategy = PartitionStrategy::BY_GRID_DIM;
  bool balance_by_estimate = true; // Try to balance estimated work

  json to_json() const;
  static PartitionConfig from_json(json const &j);
};

} // namespace search
} // namespace yirage
