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

#include "search/comet/compound_search.h"
#include "type.h"

namespace yirage {
namespace search {

/**
 * Backend-specific COMET configuration factory.
 * 
 * Each hardware backend has different characteristics that affect
 * COMET's cost model and search space:
 * - Memory bandwidth
 * - Compute throughput
 * - On-chip memory size
 * - Collective communication capabilities
 */
class BackendCOMETConfig {
public:
  /**
   * Get COMET configuration for a specific backend.
   */
  static COMETSearchConfig get_config(type::BackendType backend);
  
  /**
   * Get CUDA-optimized COMET config.
   * - High DRAM bandwidth (900+ GB/s for H100)
   * - Large shared memory
   * - NVLink for collectives
   */
  static COMETSearchConfig get_cuda_config(int compute_capability = 80);
  
  /**
   * Get CPU-optimized COMET config.
   * - Lower bandwidth, larger caches
   * - Cache-aware tile sizes
   * - OpenMP parallelism
   */
  static COMETSearchConfig get_cpu_config(int num_cores = 0);
  
  /**
   * Get ROCm-optimized COMET config (AMD GPUs).
   * - HBM2e bandwidth
   * - LDS (Local Data Share) size
   * - Infinity Fabric for collectives
   */
  static COMETSearchConfig get_rocm_config();
  
  /**
   * Get Ascend-optimized COMET config (Huawei NPUs).
   * - AI Core architecture
   * - Cube units for matrix ops
   * - HCCS interconnect
   */
  static COMETSearchConfig get_ascend_config();
  
  /**
   * Get TPU-optimized COMET config (Google TPUs).
   * - Systolic array architecture
   * - Very high bandwidth HBM
   * - ICI interconnect
   */
  static COMETSearchConfig get_tpu_config();
  
  /**
   * Get MACA-optimized COMET config (MetaX GPUs).
   * - Similar to CUDA architecture
   */
  static COMETSearchConfig get_maca_config();
  
  /**
   * Get XPU-optimized COMET config (Intel GPUs).
   * - Xe architecture
   * - Shared local memory
   */
  static COMETSearchConfig get_xpu_config();
  
  /**
   * Get MPS-optimized COMET config (Apple Silicon).
   * - Unified memory architecture
   * - Metal compute units
   */
  static COMETSearchConfig get_mps_config();
  
  /**
   * Get FPGA-optimized COMET config.
   * - Custom dataflow
   * - Lower clock, higher parallelism
   */
  static COMETSearchConfig get_fpga_config();
};

/**
 * COMET-enhanced search strategy wrapper.
 * 
 * Wraps any backend-specific search strategy with COMET's
 * compound operation optimization.
 */
class COMETEnhancedStrategy : public SearchStrategy {
public:
  /**
   * Create COMET-enhanced strategy for a backend.
   * 
   * @param backend Target backend
   * @param base_strategy Underlying backend strategy (takes ownership)
   */
  COMETEnhancedStrategy(
      type::BackendType backend,
      std::unique_ptr<SearchStrategy> base_strategy);
  
  ~COMETEnhancedStrategy() override;
  
  bool initialize(SearchConfig const &config) override;
  
  std::vector<CandidateConfig>
  generate_candidates(kernel::Graph const &graph) override;
  
  float evaluate_candidate(CandidateConfig &candidate,
                          kernel::Graph const &graph) override;
  
  kernel::KernelConfig *
  select_best_config(std::vector<CandidateConfig> &candidates) override;
  
  std::unique_ptr<kernel::KernelConfig>
  optimize(kernel::Graph const &graph) override;
  
  type::BackendType get_backend_type() const override;
  std::string get_statistics() const override;
  
  /**
   * Get the COMET search strategy.
   */
  COMETSearchStrategy* get_comet_strategy() { return &comet_strategy_; }
  
  /**
   * Get the underlying backend strategy.
   */
  SearchStrategy* get_base_strategy() { return base_strategy_.get(); }

private:
  type::BackendType backend_;
  std::unique_ptr<SearchStrategy> base_strategy_;
  COMETSearchStrategy comet_strategy_;
  COMETSearchConfig comet_config_;
  
  // Statistics
  int compound_patterns_found_;
  int compound_optimizations_applied_;
};

} // namespace search
} // namespace yirage
