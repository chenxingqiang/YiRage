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


#include "yirage/kernel/nki/nki_kernel_config.h"

#ifdef YIRAGE_BACKEND_NKI_ENABLED

#include <algorithm>

namespace yirage {
namespace kernel {
namespace nki {

void NKIOptimizer::compute_optimal_tiles(int m, int n, int k,
                                        NKIKernelConfig &config) {
  // NeuronCore has specific tile size preferences
  // Typically works best with multiples of 128
  
  // Optimal sizes based on SBUF capacity
  size_t sbuf_size = config.sbuf_size;
  size_t element_size = config.use_bf16 ? 2 : 4; // BF16 or FP32
  
  // SBUF holds: A_tile (m x k) + B_tile (k x n) + C_tile (m x n)
  size_t total_elements = sbuf_size / element_size / 3;
  
  // Neuron prefers K dimension to be large for better compute utilization
  // Typical optimal: tile_k = 512, tile_m = tile_n = 128
  config.tile_k = std::min(k, 512);
  
  // Compute M and N tiles based on remaining SBUF
  int remaining = static_cast<int>(std::sqrt(total_elements - config.tile_k * config.tile_k));
  config.tile_m = std::min(m, std::max(128, (remaining / 128) * 128));
  config.tile_n = std::min(n, std::max(128, (remaining / 128) * 128));
}

size_t NKIOptimizer::optimize_sbuf_usage(int tile_m, int tile_n, int tile_k) {
  // Calculate required SBUF size
  size_t element_size = 2; // BF16
  size_t required = (tile_m * tile_k + tile_k * tile_n + tile_m * tile_n) *
                   element_size;
  
  // Add padding for alignment
  size_t alignment = 128;
  required = ((required + alignment - 1) / alignment) * alignment;
  
  return required;
}

NKIKernelConfig::ScheduleStrategy
NKIOptimizer::select_schedule_strategy(int m, int k) {
  // For large problems, use pipelined or async DMA
  if (m >= 1024 && k >= 1024) {
    return NKIKernelConfig::ScheduleStrategy::ASYNC_DMA;
  } else if (m >= 512 || k >= 512) {
    return NKIKernelConfig::ScheduleStrategy::PIPELINED;
  }
  
  return NKIKernelConfig::ScheduleStrategy::SEQUENTIAL;
}

void NKIOptimizer::optimize_for_neuron(int m, int n, int k,
                                      NKIKernelConfig &config) {
  // Compute optimal tile sizes
  compute_optimal_tiles(m, n, k, config);
  
  // Select scheduling strategy
  config.schedule_strategy = select_schedule_strategy(m, k);
  
  // Enable double buffering for better DMA overlap
  config.use_double_buffering = (m >= 512 && k >= 512);
  
  // Use BF16 for better performance on Neuron
  config.use_bf16 = true;
  
  // Optimize SBUF usage
  config.sbuf_size = optimize_sbuf_usage(config.tile_m, config.tile_n,
                                        config.tile_k);
  
  // Set grid dimensions
  config.grid_dim_x = (n + config.tile_n - 1) / config.tile_n;
  config.grid_dim_y = (m + config.tile_m - 1) / config.tile_m;
  config.grid_dim_z = 1;
}

} // namespace nki
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_NKI_ENABLED





