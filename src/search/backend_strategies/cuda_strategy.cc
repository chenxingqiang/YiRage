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


#include "yirage/search/backend_strategies/cuda_strategy.h"

#ifdef YIRAGE_BACKEND_CUDA_ENABLED

#include "yirage/kernel/graph.h"
#include <algorithm>
#include <sstream>

namespace yirage {
namespace search {

CUDASearchStrategy::CUDASearchStrategy(int compute_capability)
    : compute_capability_(compute_capability),
      has_tensor_cores_(compute_capability >= 70) {}

bool CUDASearchStrategy::initialize(SearchConfig const &config) {
  config_ = config;
  num_candidates_generated_ = 0;
  num_candidates_evaluated_ = 0;
  best_score_ = 0.0f;
  return true;
}

std::vector<CandidateConfig>
CUDASearchStrategy::generate_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;

  // Get problem dimensions (simplified for now)
  int m = 1024, n = 1024, k = 1024; // TODO: Extract from graph

  // Generate different warp configurations
  auto warp_configs = generate_warp_configs(m * n);

  // Generate Tensor Core configurations if available
  std::vector<kernel::cuda::CUDAKernelConfig> tc_configs;
  if (has_tensor_cores_) {
    tc_configs = generate_tensor_core_configs(m, n, k);
  }

  // Generate grid/block configurations
  auto grid_block_configs = generate_grid_block_configs(m, n);

  // Combine configurations
  for (int num_warps : warp_configs) {
    for (auto const &[grid, block] : grid_block_configs) {
      auto config = std::make_unique<kernel::cuda::CUDAKernelConfig>();

      config->grid_dim_x = grid.x;
      config->grid_dim_y = grid.y;
      config->grid_dim_z = grid.z;

      config->block_dim_x = block.x;
      config->block_dim_y = block.y;
      config->block_dim_z = block.z;

      config->num_warps = num_warps;
      config->compute_capability = compute_capability_;

      // Try different shared memory layouts
      for (auto layout : {kernel::cuda::SmemLayout::ROW_MAJOR,
                         kernel::cuda::SmemLayout::SWIZZLED}) {
        auto cfg_copy = std::make_unique<kernel::cuda::CUDAKernelConfig>(*config);
        cfg_copy->smem_layout = layout;

        if (is_valid_config(*cfg_copy)) {
          candidates.emplace_back(std::move(cfg_copy));
        }
      }
    }
  }

  // Add Tensor Core configurations
  for (auto const &tc_config : tc_configs) {
    auto config = std::make_unique<kernel::cuda::CUDAKernelConfig>(tc_config);
    if (is_valid_config(*config)) {
      candidates.emplace_back(std::move(config));
    }
  }

  num_candidates_generated_ += candidates.size();
  return candidates;
}

float CUDASearchStrategy::evaluate_candidate(CandidateConfig &candidate,
                                             kernel::Graph const &graph) {
  auto *cuda_config =
      static_cast<kernel::cuda::CUDAKernelConfig *>(candidate.config.get());

  // Compute score based on multiple metrics
  float occupancy_score = evaluate_occupancy(*cuda_config);
  float memory_score = evaluate_memory_efficiency(*cuda_config);
  float compute_score = evaluate_compute_throughput(*cuda_config);
  float conflict_penalty = evaluate_bank_conflicts(*cuda_config);

  // Weighted combination
  float score = 0.3f * occupancy_score + 
                0.3f * memory_score +
                0.3f * compute_score - 
                0.1f * conflict_penalty;

  candidate.score = score;
  num_candidates_evaluated_++;

  if (score > best_score_) {
    best_score_ = score;
  }

  return score;
}

kernel::KernelConfig *CUDASearchStrategy::select_best_config(
    std::vector<CandidateConfig> &candidates) {
  if (candidates.empty()) {
    return nullptr;
  }

  auto best_it = std::max_element(
      candidates.begin(), candidates.end(),
      [](CandidateConfig const &a, CandidateConfig const &b) {
        return a.score < b.score;
      });

  return best_it->config.get();
}

std::unique_ptr<kernel::KernelConfig>
CUDASearchStrategy::optimize(kernel::Graph const &graph) {
  // Generate candidates
  auto candidates = generate_candidates(graph);

  // Evaluate all candidates
  for (auto &candidate : candidates) {
    evaluate_candidate(candidate, graph);
  }

  // Select best
  auto *best = select_best_config(candidates);
  if (!best) {
    return nullptr;
  }

  // Return copy of best configuration
  return std::make_unique<kernel::cuda::CUDAKernelConfig>(
      *static_cast<kernel::cuda::CUDAKernelConfig *>(best));
}

std::string CUDASearchStrategy::get_statistics() const {
  std::ostringstream oss;
  oss << "CUDA Search Statistics:\n";
  oss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  oss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  oss << "  Best score: " << best_score_ << "\n";
  oss << "  Compute capability: " << compute_capability_ << "\n";
  oss << "  Tensor Cores: " << (has_tensor_cores_ ? "Yes" : "No") << "\n";
  return oss.str();
}

// Private helper methods

std::vector<int> CUDASearchStrategy::generate_warp_configs(
    size_t problem_size) {
  std::vector<int> configs;

  // Try power-of-2 warp counts
  for (int warps = 4; warps <= 32; warps *= 2) {
    configs.push_back(warps);
  }

  return configs;
}

std::vector<size_t> CUDASearchStrategy::generate_smem_configs(
    size_t data_size) {
  std::vector<size_t> configs;

  // Try different shared memory sizes
  size_t base_size = 16 * 1024; // 16 KB
  for (int mult = 1; mult <= 6; ++mult) {
    configs.push_back(base_size * mult);
  }

  return configs;
}

std::vector<kernel::cuda::CUDAKernelConfig>
CUDASearchStrategy::generate_tensor_core_configs(int m, int n, int k) {
  std::vector<kernel::cuda::CUDAKernelConfig> configs;

  if (!has_tensor_cores_) {
    return configs;
  }

  // Generate different Tensor Core tile sizes
  std::vector<std::tuple<int, int, int>> tile_sizes = {
      {16, 16, 16}, // Volta/Turing
      {16, 8, 16},  // Ampere
      {16, 8, 8},   // Ampere alternative
  };

  for (auto const &[mma_m, mma_n, mma_k] : tile_sizes) {
    kernel::cuda::CUDAKernelConfig config;
    config.use_tensor_core = true;
    config.mma_m = mma_m;
    config.mma_n = mma_n;
    config.mma_k = mma_k;
    config.compute_capability = compute_capability_;

    // Set grid/block dims
    kernel::cuda::CUDAOptimizer::optimize_grid_block_dims(
        m, n, k, compute_capability_, config);

    configs.push_back(config);
  }

  return configs;
}

std::vector<std::pair<dim3, dim3>>
CUDASearchStrategy::generate_grid_block_configs(int m, int n) {
  std::vector<std::pair<dim3, dim3>> configs;

  // Try different block sizes
  std::vector<int> block_sizes = {128, 256, 512};

  for (int block_size : block_sizes) {
    dim3 block(block_size, 1, 1);

    // Calculate grid based on problem size
    int grid_x = (n + block_size - 1) / block_size;
    int grid_y = (m + block_size - 1) / block_size;
    dim3 grid(grid_x, grid_y, 1);

    configs.emplace_back(grid, block);
  }

  return configs;
}

float CUDASearchStrategy::evaluate_occupancy(
    kernel::cuda::CUDAKernelConfig const &config) {
  // Use optimizer to estimate occupancy
  int estimated_registers = 64; // Rough estimate
  return kernel::cuda::CUDAOptimizer::estimate_occupancy(config,
                                                        estimated_registers);
}

float CUDASearchStrategy::evaluate_memory_efficiency(
    kernel::cuda::CUDAKernelConfig const &config) {
  // Score based on memory coalescing and shared memory usage
  float coalescing_score = 1.0f; // Assume good coalescing for now

  // Penalize if shared memory is underutilized or over-allocated
  size_t optimal_smem = 48 * 1024; // 48 KB target
  float smem_ratio =
      static_cast<float>(config.shared_memory_size) / optimal_smem;
  float smem_score = 1.0f - std::abs(1.0f - smem_ratio);

  return 0.5f * coalescing_score + 0.5f * smem_score;
}

float CUDASearchStrategy::evaluate_compute_throughput(
    kernel::cuda::CUDAKernelConfig const &config) {
  // Score based on Tensor Core usage and warp utilization
  float tc_score = config.use_tensor_core ? 1.0f : 0.7f;

  // Higher warp count generally better up to a point
  float warp_score = std::min(1.0f, config.num_warps / 32.0f);

  return 0.6f * tc_score + 0.4f * warp_score;
}

float CUDASearchStrategy::evaluate_bank_conflicts(
    kernel::cuda::CUDAKernelConfig const &config) {
  // Swizzled layout avoids conflicts
  if (config.smem_layout == kernel::cuda::SmemLayout::SWIZZLED) {
    return 0.0f; // No penalty
  }

  // Other layouts may have conflicts
  return 0.3f; // Moderate penalty
}

bool CUDASearchStrategy::is_valid_config(
    kernel::cuda::CUDAKernelConfig const &config) {
  // Check basic constraints
  if (config.get_total_threads() == 0) {
    return false;
  }

  // Check shared memory doesn't exceed limit
  size_t max_smem = (compute_capability_ >= 80) ? 164 * 1024 : 96 * 1024;
  if (config.shared_memory_size > max_smem) {
    return false;
  }

  // Check thread count limits
  int max_threads = 1024;
  if (config.get_total_threads() > max_threads) {
    return false;
  }

  return true;
}

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_CUDA_ENABLED





