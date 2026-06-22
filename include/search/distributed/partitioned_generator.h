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
#include "search/config.h"
#include "search/distributed/search_callback.h"
#include "search/distributed/search_feedback.h"
#include "search/distributed/search_partition.h"
#include "search/search.h"
#include "utils/json_utils.h"

#include <atomic>
#include <memory>
#include <vector>

namespace yirage {
namespace search {

/**
 * @brief Kernel graph generator that operates on a partition of the search
 * space
 *
 * Used for distributed search where each worker handles a partition.
 * Extends the base KernelGraphGenerator with partition-aware logic.
 */
class PartitionedKernelGraphGenerator {
public:
  /**
   * @brief Construct a partitioned generator
   *
   * @param computation_graph Target computation graph to optimize
   * @param config Search configuration
   * @param partition Search partition to explore
   * @param callback Optional callback for feedback collection
   */
  PartitionedKernelGraphGenerator(kernel::Graph const &computation_graph,
                                  GeneratorConfig const &config,
                                  SearchPartition const &partition,
                                  SearchCallback *callback = nullptr);

  ~PartitionedKernelGraphGenerator();

  /**
   * @brief Execute search on the assigned partition
   *
   * Only explores configurations within the partition bounds.
   */
  void generate_kernel_graphs_for_partition();

  /**
   * @brief Get the search feedback
   */
  SearchFeedback get_feedback() const;

  /**
   * @brief Get generated graphs as JSON
   */
  std::vector<json> const &get_generated_graphs() const;

  /**
   * @brief Get search statistics
   */
  struct Statistics {
    int states_explored = 0;
    int candidates_generated = 0;
    int valid_graphs_found = 0;
    double elapsed_seconds = 0.0;
  };
  Statistics get_statistics() const;

  /**
   * @brief Check if search is complete
   */
  bool is_complete() const { return is_complete_; }

  /**
   * @brief Request early termination
   */
  void request_termination() { termination_requested_ = true; }

private:
  // Configuration
  GeneratorConfig config_;
  SearchPartition partition_;
  SearchCallback *callback_;

  // Internal generator (reuses existing implementation)
  std::unique_ptr<KernelGraphGenerator> generator_;

  // Partition filtering
  bool is_config_in_partition(dim3 grid_dim, dim3 block_dim) const;

  // Results
  std::vector<json> generated_graphs_;
  SearchFeedback feedback_;

  // State
  std::atomic<bool> is_complete_{false};
  std::atomic<bool> termination_requested_{false};
  std::chrono::time_point<std::chrono::steady_clock> start_time_;
};

/**
 * @brief C interface for partitioned search
 *
 * These functions are exposed to Python via Cython bindings.
 */
namespace partitioned_search_c {

/**
 * @brief Create search partitions
 *
 * @param num_partitions Number of partitions to create
 * @param config_json Search configuration as JSON string
 * @return JSON string containing partition array
 */
char *create_partitions(int num_partitions, char const *config_json);

/**
 * @brief Execute search on a single partition
 *
 * @param input_graph Input computation graph
 * @param partition_json Partition configuration as JSON
 * @param config_json Search configuration as JSON
 * @param collect_feedback Whether to collect feedback data
 * @param max_num_graphs Maximum number of graphs to return
 * @param new_graphs Output array for generated graphs
 * @param feedback_json Output: feedback data as JSON (if collect_feedback)
 * @return Number of graphs generated
 */
int search_partition(kernel::Graph const *input_graph, char const *partition_json,
                     char const *config_json, bool collect_feedback,
                     int max_num_graphs, kernel::Graph **new_graphs,
                     char **feedback_json);

/**
 * @brief Free JSON string allocated by create_partitions or search_partition
 */
void free_json_string(char *json_str);

} // namespace partitioned_search_c

} // namespace search
} // namespace yirage
