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
 * This file is part of YiRage (Yi Revolutionary AGile Engine)
 * 
 * MACA Search Strategy
 * 
 * Search strategy optimized for MetaX MACA GPU hardware.
 * Key differences from CUDA:
 * - warpSize = 64 (vs NVIDIA's 32)
 * - Different register and shared memory limits
 * - Specific tile sizes optimized for MACA architecture
 */

#pragma once

#include "yirage/search/common/search_strategy.h"
#include "yirage/kernel/maca/maca_kernel_config.h"
#include "yirage/type.h"

#ifdef YIRAGE_BACKEND_MACA_ENABLED

namespace yirage {
namespace search {

/**
 * @brief MACA-specific kernel configuration
 * 
 * Extends base KernelConfig with MACA-specific parameters
 */
struct MACAKernelConfig : public kernel::KernelConfig {
  // Warp configuration (MACA uses 64-thread warps)
  int warp_size = 64;
  int num_warps = 4;

  // Shared memory configuration
  kernel::maca::MACAMatmulConfig matmul_config;
  
  // Optimization flags
  bool use_tensor_cores = false;
  bool use_vectorized_load = true;
  
  MACAKernelConfig() { backend_type = type::BT_MACA; }
  
  // Get number of warps (with 64-thread warp size)
  int get_num_warps() const {
    return (get_total_threads() + warp_size - 1) / warp_size;
  }
};

/**
 * @brief Search strategy for MetaX MACA backend
 * 
 * This strategy considers MACA hardware characteristics when
 * searching for optimal kernel configurations:
 * - 64-thread warps (vs NVIDIA's 32)
 * - 131K registers per block
 * - 64KB shared memory per block
 * - 104 SMs on C500
 */
class MACASearchStrategy : public SearchStrategy {
public:
  MACASearchStrategy();
  explicit MACASearchStrategy(int compute_capability);
  
  virtual ~MACASearchStrategy() = default;

  // ============================================================
  // SearchStrategy interface implementation
  // ============================================================
  
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
    return type::BT_MACA;
  }

  std::string get_statistics() const override;

  // ============================================================
  // MACA-specific methods
  // ============================================================
  
  /**
   * @brief Get architecture configuration
   */
  kernel::maca::MACAArchConfig const &get_arch_config() const {
    return arch_config_;
  }

  /**
   * @brief Set architecture configuration
   */
  void set_arch_config(kernel::maca::MACAArchConfig const &config) {
    arch_config_ = config;
  }

  /**
   * @brief Check if tensor cores should be used
   */
  bool should_use_tensor_cores(int M, int N, int K, 
                               type::DataType dtype) const;

private:
  kernel::maca::MACAArchConfig arch_config_;
  int compute_capability_;
  bool has_tensor_cores_;

  // MACA-specific candidate generation
  
  /**
   * @brief Generate warp configuration candidates
   * @note MACA uses 64-thread warps, so valid warp counts differ from NVIDIA
   */
  std::vector<int> generate_warp_configs(size_t problem_size);

  /**
   * @brief Generate shared memory configuration candidates
   */
  std::vector<size_t> generate_smem_configs(size_t data_size);

  /**
   * @brief Generate grid/block dimension candidates
   * @note Block sizes must be multiples of 64 (warpSize)
   */
  std::vector<std::pair<dim3, dim3>> generate_grid_block_configs(int m, int n);

  /**
   * @brief Generate matmul tile configurations
   */
  std::vector<kernel::maca::MACAMatmulConfig>
      generate_matmul_configs(int m, int n, int k);

  // MACA-specific evaluation metrics

  /**
   * @brief Evaluate GPU occupancy for MACA
   * @note Occupancy calculation differs due to 64-thread warps
   */
  float evaluate_occupancy(MACAKernelConfig const &config);

  /**
   * @brief Evaluate memory access efficiency
   */
  float evaluate_memory_efficiency(MACAKernelConfig const &config);

  /**
   * @brief Evaluate compute throughput potential
   */
  float evaluate_compute_throughput(MACAKernelConfig const &config);

  /**
   * @brief Check if configuration is valid for MACA hardware
   */
  bool is_valid_config(MACAKernelConfig const &config);
};

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED
