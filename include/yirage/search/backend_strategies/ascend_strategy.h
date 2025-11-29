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

#include "yirage/kernel/ascend/ascend_kernel_config.h"
#include "yirage/search/common/search_strategy.h"

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED

namespace yirage {
namespace search {

/**
 * @brief Ascend NPU search strategy
 *
 * Optimizes kernel configurations for Huawei Ascend NPUs by exploring:
 * - AI Core block sizes
 * - Cube operation tile configurations
 * - L1 buffer usage patterns
 * - Vector/Cube operation selection
 */
class AscendSearchStrategy : public SearchStrategy {
public:
  explicit AscendSearchStrategy(int device_type = 0);

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
    return type::BT_ASCEND;
  }

  std::string get_statistics() const override;

private:
  int device_type_;  // 0=910, 1=910B, 2=310P
  int ai_core_count_;
  size_t l1_buffer_size_;

  // Ascend-specific candidate generation

  /**
   * @brief Generate AI Core block size candidates
   * @param problem_size Problem size
   * @return Vector of block sizes to try
   */
  std::vector<int> generate_block_configs(size_t problem_size);

  /**
   * @brief Generate tile configuration candidates for Cube operations
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @return Vector of (tile_m, tile_n, tile_k) tuples
   */
  std::vector<std::tuple<int, int, int>>
      generate_tile_configs(int m, int n, int k);

  // Ascend-specific evaluation metrics

  /**
   * @brief Evaluate AI Core utilization
   * @param config Kernel configuration
   * @return Utilization score (0.0 - 1.0)
   */
  float evaluate_ai_core_utilization(kernel::ascend::AscendKernelConfig const &config);

  /**
   * @brief Evaluate L1 buffer efficiency
   * @param config Kernel configuration
   * @return L1 buffer score (0.0 - 1.0)
   */
  float evaluate_l1_buffer_efficiency(kernel::ascend::AscendKernelConfig const &config);

  /**
   * @brief Evaluate Cube operation suitability
   * @param config Kernel configuration
   * @return Cube operation score (0.0 - 1.0)
   */
  float evaluate_cube_operation_fit(kernel::ascend::AscendKernelConfig const &config);

  /**
   * @brief Check if configuration is valid for Ascend NPU
   * @param config Kernel configuration
   * @return true if valid
   */
  bool is_valid_config(kernel::ascend::AscendKernelConfig const &config);
};

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED

