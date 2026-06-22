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

#include "search/comet/backend_comet_config.h"
#include <sstream>
#include <thread>

namespace yirage {
namespace search {

// =============================================================================
// BackendCOMETConfig - Factory for backend-specific configurations
// =============================================================================

COMETSearchConfig BackendCOMETConfig::get_config(type::BackendType backend) {
  switch (backend) {
    case type::BT_CUDA:
      return get_cuda_config();
    case type::BT_CPU:
      return get_cpu_config();
    case type::BT_ROCM:
      return get_rocm_config();
    case type::BT_ASCEND:
      return get_ascend_config();
    case type::BT_TPU:
      return get_tpu_config();
    case type::BT_MACA:
      return get_maca_config();
    case type::BT_XPU:
      return get_xpu_config();
    case type::BT_MPS:
      return get_mps_config();
    case type::BT_FPGA:
      return get_fpga_config();
    default:
      // Default config for unknown backends
      return COMETSearchConfig();
  }
}

COMETSearchConfig BackendCOMETConfig::get_cuda_config(int compute_capability) {
  COMETSearchConfig config;
  
  // NVIDIA GPU characteristics
  if (compute_capability >= 90) {
    // H100 / Hopper
    config.dram_bandwidth_gbps = 3350.0;  // HBM3
    config.onchip_bandwidth_gbps = 33000.0;  // ~33 TB/s shared mem bandwidth
    config.peak_tflops = 989.0;  // FP16 Tensor Core
    config.noc_bandwidth_gbps = 900.0;  // NVLink 4.0
    config.tile_sizes = {64, 128, 256};  // Optimized for Hopper
  } else if (compute_capability >= 80) {
    // A100 / Ampere
    config.dram_bandwidth_gbps = 2039.0;  // HBM2e
    config.onchip_bandwidth_gbps = 19000.0;
    config.peak_tflops = 312.0;  // FP16 Tensor Core
    config.noc_bandwidth_gbps = 600.0;  // NVLink 3.0
    config.tile_sizes = {64, 128, 256};
  } else if (compute_capability >= 70) {
    // V100 / Volta
    config.dram_bandwidth_gbps = 900.0;  // HBM2
    config.onchip_bandwidth_gbps = 12000.0;
    config.peak_tflops = 125.0;
    config.noc_bandwidth_gbps = 300.0;  // NVLink 2.0
    config.tile_sizes = {32, 64, 128};
  } else {
    // Older GPUs
    config.dram_bandwidth_gbps = 480.0;
    config.onchip_bandwidth_gbps = 8000.0;
    config.peak_tflops = 12.0;
    config.noc_bandwidth_gbps = 32.0;  // PCIe
    config.tile_sizes = {32, 64};
  }
  
  config.scheduling_options = {
    type::SCHED_PIPELINED,  // Best for CUDA async copy
    type::SCHED_PARALLEL
  };
  
  config.max_fusion_depth = 6;  // CUDA supports deep fusion
  config.optimize_collectives = true;
  
  return config;
}

COMETSearchConfig BackendCOMETConfig::get_cpu_config(int num_cores) {
  COMETSearchConfig config;
  
  // Detect CPU cores if not specified
  if (num_cores <= 0) {
    num_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (num_cores <= 0) num_cores = 4;  // Fallback
  }
  
  // CPU characteristics (assume modern server CPU)
  config.dram_bandwidth_gbps = 200.0;  // DDR5 ~200 GB/s typical
  config.onchip_bandwidth_gbps = 1000.0;  // L3 cache bandwidth
  config.peak_tflops = 2.0 * num_cores / 8.0;  // AVX-512 estimate
  config.noc_bandwidth_gbps = 50.0;  // Inter-socket bandwidth
  
  // Cache-friendly tile sizes (L1: 32KB, L2: 256KB-1MB, L3: shared)
  config.tile_sizes = {32, 64, 128, 256};  // Cache-aligned
  
  config.scheduling_options = {
    type::SCHED_SEQUENTIAL,  // Often best for CPU cache
    type::SCHED_PARALLEL     // OpenMP parallelism
  };
  
  config.max_fusion_depth = 4;  // Less aggressive than GPU
  config.num_devices = 1;  // Single socket default
  config.optimize_collectives = false;  // Usually single machine
  
  return config;
}

COMETSearchConfig BackendCOMETConfig::get_rocm_config() {
  COMETSearchConfig config;
  
  // AMD MI300X characteristics
  config.dram_bandwidth_gbps = 5300.0;  // HBM3 (8 stacks)
  config.onchip_bandwidth_gbps = 25000.0;  // LDS bandwidth
  config.peak_tflops = 1307.0;  // FP16
  config.noc_bandwidth_gbps = 896.0;  // Infinity Fabric
  
  config.tile_sizes = {64, 128, 256};
  
  config.scheduling_options = {
    type::SCHED_PIPELINED,
    type::SCHED_PARALLEL
  };
  
  config.max_fusion_depth = 6;
  config.optimize_collectives = true;
  
  return config;
}

COMETSearchConfig BackendCOMETConfig::get_ascend_config() {
  COMETSearchConfig config;
  
  // Huawei Ascend 910B characteristics
  config.dram_bandwidth_gbps = 1600.0;  // HBM2e
  config.onchip_bandwidth_gbps = 12000.0;  // AI Core local memory
  config.peak_tflops = 320.0;  // FP16
  config.noc_bandwidth_gbps = 392.0;  // HCCS
  
  config.tile_sizes = {64, 128, 256};  // Cube unit friendly
  
  config.scheduling_options = {
    type::SCHED_PIPELINED,
    type::SCHED_SEQUENTIAL
  };
  
  config.max_fusion_depth = 5;
  config.optimize_collectives = true;
  
  return config;
}

COMETSearchConfig BackendCOMETConfig::get_tpu_config() {
  COMETSearchConfig config;
  
  // Google TPU v5e characteristics
  config.dram_bandwidth_gbps = 1600.0;  // HBM2e
  config.onchip_bandwidth_gbps = 20000.0;  // VMEM bandwidth
  config.peak_tflops = 197.0;  // BF16
  config.noc_bandwidth_gbps = 1600.0;  // ICI (Inter-chip Interconnect)
  
  // TPU prefers larger tiles for systolic array
  config.tile_sizes = {128, 256, 512};
  
  config.scheduling_options = {
    type::SCHED_PIPELINED  // TPU XLA compiler handles scheduling
  };
  
  config.max_fusion_depth = 8;  // XLA aggressive fusion
  config.optimize_collectives = true;  // TPU pods need collectives
  
  return config;
}

COMETSearchConfig BackendCOMETConfig::get_maca_config() {
  COMETSearchConfig config;
  
  // MetaX MXC500 characteristics (similar to NVIDIA)
  config.dram_bandwidth_gbps = 2000.0;  // HBM2e
  config.onchip_bandwidth_gbps = 15000.0;
  config.peak_tflops = 256.0;  // FP16
  config.noc_bandwidth_gbps = 400.0;
  
  config.tile_sizes = {64, 128, 256};
  
  config.scheduling_options = {
    type::SCHED_PIPELINED,
    type::SCHED_PARALLEL
  };
  
  config.max_fusion_depth = 5;
  config.optimize_collectives = true;
  
  return config;
}

COMETSearchConfig BackendCOMETConfig::get_xpu_config() {
  COMETSearchConfig config;
  
  // Intel Data Center GPU Max (Ponte Vecchio) characteristics
  config.dram_bandwidth_gbps = 3200.0;  // HBM2e (128 GB)
  config.onchip_bandwidth_gbps = 10000.0;  // SLM bandwidth
  config.peak_tflops = 420.0;  // FP16 (with XMX)
  config.noc_bandwidth_gbps = 200.0;  // Xe Link
  
  config.tile_sizes = {32, 64, 128, 256};
  
  config.scheduling_options = {
    type::SCHED_PIPELINED,
    type::SCHED_PARALLEL
  };
  
  config.max_fusion_depth = 5;
  config.optimize_collectives = true;
  
  return config;
}

COMETSearchConfig BackendCOMETConfig::get_mps_config() {
  COMETSearchConfig config;
  
  // Apple M3 Max characteristics (unified memory)
  config.dram_bandwidth_gbps = 400.0;  // Unified memory
  config.onchip_bandwidth_gbps = 2000.0;  // GPU cache
  config.peak_tflops = 14.2;  // FP32
  config.noc_bandwidth_gbps = 400.0;  // Same as memory (unified)
  
  // Smaller tiles for mobile efficiency
  config.tile_sizes = {32, 64, 128};
  
  config.scheduling_options = {
    type::SCHED_SEQUENTIAL,  // Unified memory benefits from sequential
    type::SCHED_PIPELINED
  };
  
  config.max_fusion_depth = 4;
  config.num_devices = 1;  // Single device
  config.optimize_collectives = false;
  
  return config;
}

COMETSearchConfig BackendCOMETConfig::get_fpga_config() {
  COMETSearchConfig config;
  
  // Generic FPGA characteristics (Xilinx Alveo U250)
  config.dram_bandwidth_gbps = 77.0;  // DDR4
  config.onchip_bandwidth_gbps = 500.0;  // BRAM bandwidth
  config.peak_tflops = 4.0;  // Depends on design
  config.noc_bandwidth_gbps = 100.0;  // PCIe + QSFP
  
  // FPGA prefers power-of-2 tiles matching DSP blocks
  config.tile_sizes = {16, 32, 64, 128};
  
  config.scheduling_options = {
    type::SCHED_PIPELINED  // FPGA dataflow is inherently pipelined
  };
  
  config.max_fusion_depth = 10;  // FPGA can do very deep fusion
  config.optimize_collectives = false;
  
  return config;
}

// =============================================================================
// COMETEnhancedStrategy - Wrapper combining COMET with backend strategies
// =============================================================================

COMETEnhancedStrategy::COMETEnhancedStrategy(
    type::BackendType backend,
    std::unique_ptr<SearchStrategy> base_strategy)
    : backend_(backend),
      base_strategy_(std::move(base_strategy)),
      compound_patterns_found_(0),
      compound_optimizations_applied_(0) {
  
  // Get backend-specific COMET configuration
  comet_config_ = BackendCOMETConfig::get_config(backend);
  comet_strategy_ = COMETSearchStrategy(backend);
}

COMETEnhancedStrategy::~COMETEnhancedStrategy() = default;

bool COMETEnhancedStrategy::initialize(SearchConfig const &config) {
  config_ = config;
  
  // Initialize COMET strategy with backend-specific config
  if (!comet_strategy_.initialize(comet_config_)) {
    return false;
  }
  
  // Initialize base strategy if available
  if (base_strategy_ && !base_strategy_->initialize(config)) {
    return false;
  }
  
  compound_patterns_found_ = 0;
  compound_optimizations_applied_ = 0;
  
  return true;
}

std::vector<CandidateConfig>
COMETEnhancedStrategy::generate_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> all_candidates;
  
  // Step 1: Detect compound patterns with COMET
  auto patterns = comet_strategy_.detect_patterns(graph);
  compound_patterns_found_ = static_cast<int>(patterns.size());
  
  // Step 2: Generate COMET candidates for compound operations
  if (!patterns.empty()) {
    auto comet_candidates = comet_strategy_.generate_comet_candidates(graph);
    
    // Convert to CandidateConfig
    for (auto const& cc : comet_candidates) {
      CandidateConfig config;
      config.config = std::make_unique<kernel::KernelConfig>();
      config.score = static_cast<float>(cc.score);
      all_candidates.push_back(std::move(config));
    }
    
    compound_optimizations_applied_ = static_cast<int>(comet_candidates.size());
  }
  
  // Step 3: Generate backend-specific candidates for non-compound ops
  if (base_strategy_) {
    auto base_candidates = base_strategy_->generate_candidates(graph);
    
    // Merge candidates
    for (auto& bc : base_candidates) {
      all_candidates.push_back(std::move(bc));
    }
  }
  
  num_candidates_generated_ = static_cast<int>(all_candidates.size());
  return all_candidates;
}

float COMETEnhancedStrategy::evaluate_candidate(
    CandidateConfig &candidate,
    kernel::Graph const &graph) {
  
  // Delegate to base strategy if available
  if (base_strategy_) {
    return base_strategy_->evaluate_candidate(candidate, graph);
  }
  
  // Otherwise use score from generation
  return candidate.score;
}

kernel::KernelConfig*
COMETEnhancedStrategy::select_best_config(
    std::vector<CandidateConfig> &candidates) {
  
  if (candidates.empty()) {
    return nullptr;
  }
  
  // Find best by score
  auto best_it = std::max_element(
      candidates.begin(), candidates.end(),
      [](CandidateConfig const& a, CandidateConfig const& b) {
        return a.score < b.score;
      });
  
  if (best_it != candidates.end()) {
    best_score_ = best_it->score;
    return best_it->config.get();
  }
  
  return nullptr;
}

std::unique_ptr<kernel::KernelConfig>
COMETEnhancedStrategy::optimize(kernel::Graph const &graph) {
  // Generate and evaluate candidates
  auto candidates = generate_candidates(graph);
  
  if (candidates.empty()) {
    return nullptr;
  }
  
  // Evaluate all
  for (auto& c : candidates) {
    evaluate_candidate(c, graph);
    ++num_candidates_evaluated_;
  }
  
  // Select best
  auto* best = select_best_config(candidates);
  
  if (best) {
    return std::make_unique<kernel::KernelConfig>(*best);
  }
  
  return nullptr;
}

type::BackendType COMETEnhancedStrategy::get_backend_type() const {
  return backend_;
}

std::string COMETEnhancedStrategy::get_statistics() const {
  std::stringstream ss;
  ss << "COMET-Enhanced Strategy Statistics:\n";
  ss << "  Backend: " << static_cast<int>(backend_) << "\n";
  ss << "  Compound patterns found: " << compound_patterns_found_ << "\n";
  ss << "  Compound optimizations: " << compound_optimizations_applied_ << "\n";
  ss << "  Total candidates: " << num_candidates_generated_ << "\n";
  ss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  ss << "  Best score: " << best_score_ << "\n";
  
  // Include COMET strategy stats
  ss << "\n" << comet_strategy_.get_statistics();
  
  // Include base strategy stats if available
  if (base_strategy_) {
    ss << "\nBase Strategy:\n" << base_strategy_->get_statistics();
  }
  
  return ss.str();
}

} // namespace search
} // namespace yirage
