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
 *
 * This file is part of YiRage (Yi Revolutionary AGile Engine),
 * a derivative work based on Mirage by CMU.
 * Original Mirage Copyright 2023-2024 CMU.
 */


#pragma once

#include "yirage/kernel/mps/mps_kernel_config.h"
#include "yirage/search/common/search_strategy.h"

#ifdef YIRAGE_BACKEND_MPS_ENABLED

namespace yirage {
namespace search {

/**
 * @brief MPS (Metal Performance Shaders) search strategy
 * 
 * Optimizes kernel configurations for Apple Silicon GPUs by exploring:
 * - Threadgroup sizes
 * - Tile configurations
 * - Memory access patterns
 * - SIMD group utilization
 */
class MPSSearchStrategy : public SearchStrategy {
public:
  explicit MPSSearchStrategy(int gpu_family = 0);

  bool initialize(SearchConfig const &config) override;

  std::vector<CandidateConfig>
  generate_candidates(kernel::Graph const &graph) override;

  float evaluate_candidate(CandidateConfig &candidate,
                          kernel::Graph const &graph) override;

  kernel::KernelConfig *
  select_best_config(std::vector<CandidateConfig> &candidates) override;

  std::unique_ptr<kernel::KernelConfig>
  optimize(kernel::Graph const &graph) override;

  type::BackendType get_backend_type() const override {
    return type::BT_MPS;
  }

  std::string get_statistics() const override;

private:
  int gpu_family_;
  int gpu_cores_;

  // MPS-specific candidate generation

  /**
   * @brief Generate threadgroup size candidates
   * @param problem_size Problem size
   * @return Vector of threadgroup sizes to try
   */
  std::vector<int> generate_threadgroup_configs(size_t problem_size);

  /**
   * @brief Generate tile configuration candidates
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @return Vector of (tile_m, tile_n, tile_k) tuples
   */
  std::vector<std::tuple<int, int, int>> 
  generate_tile_configs(int m, int n, int k);

  /**
   * @brief Generate memory access pattern candidates
   * @return Vector of memory patterns to try
   */
  std::vector<kernel::mps::MemoryPattern> generate_memory_patterns();

  // MPS-specific evaluation metrics

  /**
   * @brief Evaluate GPU utilization
   * @param config Kernel configuration
   * @return Utilization score (0.0 - 1.0)
   */
  float evaluate_gpu_utilization(kernel::mps::MPSKernelConfig const &config);

  /**
   * @brief Evaluate memory efficiency
   * @param config Kernel configuration
   * @return Memory efficiency score (0.0 - 1.0)
   */
  float evaluate_memory_efficiency(kernel::mps::MPSKernelConfig const &config);

  /**
   * @brief Evaluate threadgroup memory usage
   * @param config Kernel configuration
   * @return Threadgroup memory score (0.0 - 1.0)
   */
  float evaluate_threadgroup_memory(
      kernel::mps::MPSKernelConfig const &config);

  /**
   * @brief Check if configuration is valid for Apple GPU
   * @param config Kernel configuration
   * @return true if valid
   */
  bool is_valid_config(kernel::mps::MPSKernelConfig const &config);
};

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_MPS_ENABLED





