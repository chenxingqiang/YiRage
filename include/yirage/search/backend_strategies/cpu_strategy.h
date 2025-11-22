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

#include "yirage/kernel/cpu/cpu_kernel_config.h"
#include "yirage/search/common/search_strategy.h"

#ifdef YIRAGE_BACKEND_CPU_ENABLED

namespace yirage {
namespace search {

/**
 * @brief CPU-specific search strategy
 * 
 * Optimizes kernel configurations for CPUs by exploring:
 * - Cache blocking (tile sizes)
 * - Thread counts
 * - SIMD vectorization
 * - OpenMP scheduling
 */
class CPUSearchStrategy : public SearchStrategy {
public:
  explicit CPUSearchStrategy(int num_cores = 0);

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
    return type::BT_CPU;
  }

  std::string get_statistics() const override;

private:
  int num_cores_;
  kernel::cpu::SIMDType simd_type_;

  // CPU-specific candidate generation

  /**
   * @brief Generate tile size candidates for cache blocking
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @return Vector of (tile_m, tile_n, tile_k) tuples
   */
  std::vector<std::tuple<int, int, int>> 
  generate_tile_configs(int m, int n, int k);

  /**
   * @brief Generate thread count candidates
   * @param problem_size Problem size
   * @return Vector of thread counts to try
   */
  std::vector<int> generate_thread_configs(size_t problem_size);

  /**
   * @brief Generate SIMD configuration candidates
   * @return Vector of SIMD types to try
   */
  std::vector<kernel::cpu::SIMDType> generate_simd_configs();

  // CPU-specific evaluation metrics

  /**
   * @brief Evaluate cache utilization efficiency
   * @param config Kernel configuration
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @return Cache efficiency score (0.0 - 1.0)
   */
  float evaluate_cache_efficiency(kernel::cpu::CPUKernelConfig const &config,
                                  int m, int n, int k);

  /**
   * @brief Evaluate vectorization efficiency
   * @param config Kernel configuration
   * @param problem_size Problem size
   * @return Vectorization efficiency score (0.0 - 1.0)
   */
  float evaluate_vectorization_efficiency(
      kernel::cpu::CPUKernelConfig const &config,
      size_t problem_size);

  /**
   * @brief Evaluate load balance across threads
   * @param config Kernel configuration
   * @param problem_size Problem size
   * @return Load balance score (0.0 - 1.0)
   */
  float evaluate_load_balance(kernel::cpu::CPUKernelConfig const &config,
                              size_t problem_size);

  /**
   * @brief Check if configuration is valid
   * @param config Kernel configuration
   * @return true if valid
   */
  bool is_valid_config(kernel::cpu::CPUKernelConfig const &config);
};

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_CPU_ENABLED





