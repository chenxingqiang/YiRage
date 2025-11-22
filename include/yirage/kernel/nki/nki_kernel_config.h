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

#include "yirage/kernel/common/kernel_interface.h"

#ifdef YIRAGE_BACKEND_NKI_ENABLED

namespace yirage {
namespace kernel {
namespace nki {

/**
 * @brief NKI (Neuron Kernel Interface) specific configuration
 * 
 * For AWS Inferentia and Trainium chips
 */
struct NKIKernelConfig : public KernelConfig {
  // NeuronCore configuration
  int num_neuron_cores = 1;
  
  // Tile configuration for NeuronCore
  int tile_m = 128;
  int tile_n = 128;
  int tile_k = 512;
  
  // SBUF (State Buffer) size - Neuron's on-chip memory
  size_t sbuf_size = 24 * 1024 * 1024; // 24 MB
  
  // PSUM (Partial Sum) buffer size
  size_t psum_size = 2 * 1024 * 1024; // 2 MB
  
  // DMA configuration
  int num_dma_channels = 4;
  bool use_double_buffering = true;
  
  // Collective communication
  bool use_collective_comm = false;
  int num_devices = 1;
  
  // Data type
  bool use_bf16 = true; // Neuron optimized for BF16
  
  // Instruction scheduling
  enum class ScheduleStrategy {
    SEQUENTIAL,
    PIPELINED,
    ASYNC_DMA
  };
  ScheduleStrategy schedule_strategy = ScheduleStrategy::PIPELINED;

  NKIKernelConfig() { backend_type = type::BT_NKI; }
};

/**
 * @brief NKI kernel optimizer
 */
class NKIOptimizer {
public:
  /**
   * @brief Compute optimal tile sizes for NeuronCore
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param config Output configuration
   */
  static void compute_optimal_tiles(int m, int n, int k,
                                    NKIKernelConfig &config);

  /**
   * @brief Optimize SBUF usage
   * @param tile_m Tile M size
   * @param tile_n Tile N size
   * @param tile_k Tile K size
   * @return Optimal SBUF allocation
   */
  static size_t optimize_sbuf_usage(int tile_m, int tile_n, int tile_k);

  /**
   * @brief Select optimal scheduling strategy
   * @param m M dimension
   * @param k K dimension
   * @return Optimal schedule strategy
   */
  static NKIKernelConfig::ScheduleStrategy 
  select_schedule_strategy(int m, int k);

  /**
   * @brief Optimize for Neuron architecture
   * @param m M dimension
   * @param n N dimension
   * @param k K dimension
   * @param config Configuration to optimize
   */
  static void optimize_for_neuron(int m, int n, int k,
                                 NKIKernelConfig &config);
};

} // namespace nki
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_NKI_ENABLED





