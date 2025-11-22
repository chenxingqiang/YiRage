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

#include "yirage/kernel/cuda/cuda_kernel_config.h"
#include "yirage/search/common/search_strategy.h"

#ifdef YIRAGE_BACKEND_CUDA_ENABLED

namespace yirage {
namespace search {

/**
 * @brief CUDA-specific search strategy
 * 
 * Optimizes kernel configurations for NVIDIA GPUs by exploring:
 * - Warp configurations
 * - Shared memory layouts
 * - Tensor Core usage
 * - Grid/block dimensions
 */
class CUDASearchStrategy : public SearchStrategy {
public:
  explicit CUDASearchStrategy(int compute_capability = 80);

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
    return type::BT_CUDA;
  }

  std::string get_statistics() const override;

private:
  int compute_capability_;
  bool has_tensor_cores_;

  // CUDA-specific candidate generation

  /**
   * @brief Generate warp configuration candidates
   * @param problem_size Problem size
   * @return Vector of warp counts to try
   */
  std::vector<int> generate_warp_configs(size_t problem_size);

  /**
   * @brief Generate shared memory configuration candidates
   * @param data_size Size of data
   * @return Vector of shared memory sizes
   */
  std::vector<size_t> generate_smem_configs(size_t data_size);

  /**
   * @brief Generate Tensor Core configuration candidates
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @return Vector of Tensor Core configs
   */
  std::vector<kernel::cuda::CUDAKernelConfig>
  generate_tensor_core_configs(int m, int n, int k);

  /**
   * @brief Generate grid/block dimension candidates
   * @param m M dimension
   * @param n N dimension
   * @return Vector of grid/block configs
   */
  std::vector<std::pair<dim3, dim3>> generate_grid_block_configs(int m,
                                                                  int n);

  // CUDA-specific evaluation metrics

  /**
   * @brief Evaluate GPU occupancy
   * @param config Kernel configuration
   * @return Occupancy score (0.0 - 1.0)
   */
  float evaluate_occupancy(kernel::cuda::CUDAKernelConfig const &config);

  /**
   * @brief Evaluate memory access efficiency
   * @param config Kernel configuration
   * @return Memory efficiency score (0.0 - 1.0)
   */
  float evaluate_memory_efficiency(
      kernel::cuda::CUDAKernelConfig const &config);

  /**
   * @brief Evaluate compute throughput potential
   * @param config Kernel configuration
   * @return Compute throughput score (0.0 - 1.0)
   */
  float evaluate_compute_throughput(
      kernel::cuda::CUDAKernelConfig const &config);

  /**
   * @brief Evaluate shared memory bank conflicts
   * @param config Kernel configuration
   * @return Conflict penalty (0.0 = no conflicts, 1.0 = maximum conflicts)
   */
  float evaluate_bank_conflicts(kernel::cuda::CUDAKernelConfig const &config);

  /**
   * @brief Check if configuration is valid for hardware
   * @param config Kernel configuration
   * @return true if valid
   */
  bool is_valid_config(kernel::cuda::CUDAKernelConfig const &config);
};

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_CUDA_ENABLED





