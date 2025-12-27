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
 * MACA Search Strategy Implementation
 * 
 * Search strategy optimized for MetaX MACA GPU hardware.
 * Key optimizations:
 * - 64-thread warp configuration
 * - MACA-specific tile sizes
 * - Occupancy calculation for MACA architecture
 */

#include "yirage/search/backend_strategies/maca_strategy.h"

#ifdef YIRAGE_BACKEND_MACA_ENABLED

#include "yirage/kernel/graph.h"
#include <algorithm>
#include <sstream>

namespace yirage {
namespace search {

// MACA hardware constants
namespace {
  constexpr int MACA_WARP_SIZE = 64;
  constexpr int MACA_MAX_THREADS_PER_BLOCK = 1024;
  constexpr int MACA_MAX_WARPS_PER_SM = 32;  // 2048 / 64
  constexpr int MACA_MAX_BLOCKS_PER_SM = 16;
  constexpr int MACA_REGISTERS_PER_SM = 131072;
  constexpr size_t MACA_SHARED_MEM_PER_BLOCK = 65536;  // 64 KB
  constexpr int MACA_SM_COUNT = 104;  // MetaX C500
}

MACASearchStrategy::MACASearchStrategy()
    : arch_config_(kernel::maca::get_maca_arch_config(100)),
      compute_capability_(100),  // MetaX C500 = 10.0
      has_tensor_cores_(true) {}

MACASearchStrategy::MACASearchStrategy(int compute_capability)
    : arch_config_(kernel::maca::get_maca_arch_config(compute_capability)),
      compute_capability_(compute_capability),
      has_tensor_cores_(compute_capability >= 100) {}

bool MACASearchStrategy::initialize(SearchConfig const &config) {
  config_ = config;
  num_candidates_generated_ = 0;
  num_candidates_evaluated_ = 0;
  best_score_ = 0.0f;
  return true;
}

namespace {
// Extract problem dimensions from kernel graph
void extract_maca_problem_dimensions(kernel::Graph const &graph, 
                                     int &m, int &n, int &k) {
  m = 1; n = 1; k = 1;
  
  for (const auto &op : graph.operators) {
    if (op->op_type == type::KN_MATMUL_OP) {
      if (op->input_tensors.size() >= 2) {
        auto const &A = op->input_tensors[0];
        auto const &B = op->input_tensors[1];
        if (A.num_dims >= 2 && B.num_dims >= 2) {
          m = std::max(m, A.dim[A.num_dims - 2]);
          k = std::max(k, A.dim[A.num_dims - 1]);
          n = std::max(n, B.dim[B.num_dims - 1]);
        }
      }
    }
    for (const auto &t : op->output_tensors) {
      if (t.num_dims >= 2) {
        m = std::max(m, t.dim[t.num_dims - 2]);
        n = std::max(n, t.dim[t.num_dims - 1]);
      }
    }
  }
  
  // MACA 64-thread warps prefer larger dimensions
  if (m < 64) m = 64;
  if (n < 64) n = 64;
  if (k < 32) k = 32;
}
} // anonymous namespace

std::vector<CandidateConfig>
MACASearchStrategy::generate_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;

  // Extract problem dimensions from graph
  int m, n, k;
  extract_maca_problem_dimensions(graph, m, n, k);

  // Generate different warp configurations for MACA's 64-thread warps
  auto warp_configs = generate_warp_configs(m * n);

  // Generate grid/block configurations (block sizes must be multiple of 64)
  auto grid_block_configs = generate_grid_block_configs(m, n);

  // Generate matmul tile configurations
  auto matmul_configs = generate_matmul_configs(m, n, k);

  // Combine configurations
  for (int num_warps : warp_configs) {
    for (auto const &[grid, block] : grid_block_configs) {
      for (auto const &matmul_cfg : matmul_configs) {
        auto config = std::make_unique<MACAKernelConfig>();

        config->grid_dim_x = grid.x;
        config->grid_dim_y = grid.y;
        config->grid_dim_z = grid.z;

        config->block_dim_x = block.x;
        config->block_dim_y = block.y;
        config->block_dim_z = block.z;

        config->warp_size = MACA_WARP_SIZE;
        config->num_warps = num_warps;
        config->matmul_config = matmul_cfg;
        config->use_tensor_cores = has_tensor_cores_ && matmul_cfg.use_tensor_cores;
        config->use_vectorized_load = matmul_cfg.use_vectorized_load;

        if (is_valid_config(*config)) {
          candidates.emplace_back(std::move(config));
        }
      }
    }
  }

  num_candidates_generated_ += candidates.size();
  return candidates;
}

float MACASearchStrategy::evaluate_candidate(CandidateConfig &candidate,
                                             kernel::Graph const &graph) {
  auto *maca_config = static_cast<MACAKernelConfig *>(candidate.config.get());

  // Compute score based on multiple metrics
  float occupancy_score = evaluate_occupancy(*maca_config);
  float memory_score = evaluate_memory_efficiency(*maca_config);
  float compute_score = evaluate_compute_throughput(*maca_config);

  // Weighted combination
  float score = 0.35f * occupancy_score + 
                0.35f * memory_score +
                0.30f * compute_score;

  candidate.score = score;
  num_candidates_evaluated_++;

  if (score > best_score_) {
    best_score_ = score;
  }

  return score;
}

kernel::KernelConfig *MACASearchStrategy::select_best_config(
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
MACASearchStrategy::optimize(kernel::Graph const &graph) {
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
  return std::make_unique<MACAKernelConfig>(
      *static_cast<MACAKernelConfig *>(best));
}

std::string MACASearchStrategy::get_statistics() const {
  std::ostringstream oss;
  oss << "MACA Search Statistics:\n";
  oss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  oss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  oss << "  Best score: " << best_score_ << "\n";
  oss << "  Compute capability: " << compute_capability_ << "\n";
  oss << "  Warp size: " << MACA_WARP_SIZE << " (vs NVIDIA's 32)\n";
  oss << "  Tensor Cores: " << (has_tensor_cores_ ? "Yes" : "No") << "\n";
  return oss.str();
}

bool MACASearchStrategy::should_use_tensor_cores(int M, int N, int K,
                                                  type::DataType dtype) const {
  // Tensor cores work best with:
  // - FP16 or BF16 data types
  // - Dimensions aligned to 16
  // - Sufficiently large matrices
  
  bool aligned = (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0);
  bool large_enough = (M * N * K) >= (128 * 128 * 128);
  bool supported_type = (dtype == type::DT_FLOAT16 || 
                        dtype == type::DT_BFLOAT16);
  
  return has_tensor_cores_ && aligned && large_enough && supported_type;
}

// ============================================================
// Private helper methods
// ============================================================

std::vector<int> MACASearchStrategy::generate_warp_configs(size_t problem_size) {
  std::vector<int> configs;

  // MACA uses 64-thread warps, so valid warp counts differ
  // Max warps per block = 1024 / 64 = 16
  // Try power-of-2 warp counts
  for (int warps = 2; warps <= 16; warps *= 2) {
    configs.push_back(warps);
  }

  return configs;
}

std::vector<size_t> MACASearchStrategy::generate_smem_configs(size_t data_size) {
  std::vector<size_t> configs;

  // MACA has 64KB shared memory per block
  // Try different sizes up to max
  for (size_t size = 16 * 1024; size <= MACA_SHARED_MEM_PER_BLOCK; size += 16 * 1024) {
    configs.push_back(size);
  }

  return configs;
}

std::vector<std::pair<dim3, dim3>>
MACASearchStrategy::generate_grid_block_configs(int m, int n) {
  std::vector<std::pair<dim3, dim3>> configs;

  // Block sizes must be multiples of 64 (MACA warp size)
  std::vector<int> block_sizes = {64, 128, 256, 512, 1024};

  for (int block_size : block_sizes) {
    dim3 block(block_size, 1, 1);

    // Calculate grid based on problem size
    int grid_x = (n + block_size - 1) / block_size;
    int grid_y = (m + block_size - 1) / block_size;
    dim3 grid(grid_x, grid_y, 1);

    configs.emplace_back(grid, block);
  }

  // Also try 2D block configurations
  std::vector<std::pair<int, int>> block_2d = {
    {64, 1},    // 1 warp
    {64, 2},    // 2 warps
    {64, 4},    // 4 warps
    {128, 2},   // 4 warps, different shape
    {256, 2},   // 8 warps
  };

  for (auto const &[bx, by] : block_2d) {
    dim3 block(bx, by, 1);
    int grid_x = (n + bx - 1) / bx;
    int grid_y = (m + by - 1) / by;
    dim3 grid(grid_x, grid_y, 1);
    configs.emplace_back(grid, block);
  }

  return configs;
}

std::vector<kernel::maca::MACAMatmulConfig>
MACASearchStrategy::generate_matmul_configs(int m, int n, int k) {
  std::vector<kernel::maca::MACAMatmulConfig> configs;

  // Generate configurations optimized for MACA's 64-thread warps
  struct TileConfig {
    int tile_m, tile_n, tile_k;
    int warp_m, warp_n;
    int num_stages;
  };

  std::vector<TileConfig> tile_configs = {
    // Small tiles for small problems
    {64, 64, 32, 32, 64, 2},
    {64, 128, 32, 32, 64, 2},
    
    // Medium tiles
    {128, 128, 32, 64, 64, 3},
    {128, 256, 32, 64, 64, 3},
    
    // Large tiles for large problems
    {256, 128, 32, 64, 64, 4},
    {256, 256, 32, 64, 64, 4},
  };

  for (auto const &tc : tile_configs) {
    kernel::maca::MACAMatmulConfig config;
    config.tile_m = tc.tile_m;
    config.tile_n = tc.tile_n;
    config.tile_k = tc.tile_k;
    config.warp_m = tc.warp_m;
    config.warp_n = tc.warp_n;
    config.num_stages = tc.num_stages;
    
    // Check if this config fits in shared memory
    size_t smem_needed = 
        (tc.tile_m * tc.tile_k + tc.tile_k * tc.tile_n) * 
        sizeof(float) * tc.num_stages;
    
    if (smem_needed <= MACA_SHARED_MEM_PER_BLOCK) {
      config.use_tensor_cores = has_tensor_cores_;
      config.use_vectorized_load = true;
      configs.push_back(config);
    }
  }

  return configs;
}

float MACASearchStrategy::evaluate_occupancy(MACAKernelConfig const &config) {
  // Calculate occupancy for MACA's 64-thread warps
  int threads_per_block = config.get_total_threads();
  int warps_per_block = (threads_per_block + MACA_WARP_SIZE - 1) / MACA_WARP_SIZE;
  
  // Calculate blocks limited by warps
  int blocks_by_warps = MACA_MAX_WARPS_PER_SM / warps_per_block;
  
  // Calculate blocks limited by registers (estimate 64 regs per thread)
  int regs_per_thread = 64;
  int regs_per_block = regs_per_thread * threads_per_block;
  int blocks_by_regs = MACA_REGISTERS_PER_SM / regs_per_block;
  
  // Calculate blocks limited by shared memory
  size_t smem_per_block = config.shared_memory_size;
  int blocks_by_smem = (smem_per_block > 0) ? 
      MACA_SHARED_MEM_PER_BLOCK / smem_per_block : 
      MACA_MAX_BLOCKS_PER_SM;
  
  // Take minimum
  int blocks_per_sm = std::min({blocks_by_warps, blocks_by_regs, 
                                blocks_by_smem, MACA_MAX_BLOCKS_PER_SM});
  
  // Calculate occupancy
  int active_warps = blocks_per_sm * warps_per_block;
  return static_cast<float>(active_warps) / MACA_MAX_WARPS_PER_SM;
}

float MACASearchStrategy::evaluate_memory_efficiency(MACAKernelConfig const &config) {
  // Score based on memory access patterns
  float vectorized_score = config.use_vectorized_load ? 1.0f : 0.7f;
  
  // Penalize if shared memory is underutilized
  float smem_utilization = static_cast<float>(config.shared_memory_size) / 
                           MACA_SHARED_MEM_PER_BLOCK;
  float smem_score = (smem_utilization > 0.1f && smem_utilization < 0.9f) ? 
                     1.0f : 0.8f;
  
  // MACA has wide memory bus (4096-bit), favor larger transactions
  int threads_per_block = config.get_total_threads();
  float coalescing_score = (threads_per_block % 64 == 0) ? 1.0f : 0.7f;
  
  return 0.4f * vectorized_score + 0.3f * smem_score + 0.3f * coalescing_score;
}

float MACASearchStrategy::evaluate_compute_throughput(MACAKernelConfig const &config) {
  // Score based on Tensor Core usage and warp utilization
  float tc_score = config.use_tensor_cores ? 1.0f : 0.7f;

  // Higher warp count generally better for MACA (max 16 warps per block)
  float warp_score = std::min(1.0f, config.num_warps / 16.0f);
  
  // Tile size efficiency
  float tile_score = 0.8f;
  if (config.matmul_config.tile_m >= 128 && config.matmul_config.tile_n >= 128) {
    tile_score = 1.0f;
  }

  return 0.4f * tc_score + 0.3f * warp_score + 0.3f * tile_score;
}

bool MACASearchStrategy::is_valid_config(MACAKernelConfig const &config) {
  // Check basic constraints
  int total_threads = config.get_total_threads();
  
  if (total_threads == 0 || total_threads > MACA_MAX_THREADS_PER_BLOCK) {
    return false;
  }

  // Thread count must be multiple of warp size (64)
  if (total_threads % MACA_WARP_SIZE != 0) {
    return false;
  }

  // Check shared memory doesn't exceed limit
  if (config.shared_memory_size > MACA_SHARED_MEM_PER_BLOCK) {
    return false;
  }

  // Check block dimension limits
  if (config.block_dim_x > 1024 || 
      config.block_dim_y > 1024 || 
      config.block_dim_z > 64) {
    return false;
  }

  return true;
}

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_MACA_ENABLED

