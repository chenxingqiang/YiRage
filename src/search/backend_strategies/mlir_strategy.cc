/* Copyright 2025 YiRage Team
 * MLIR Search Strategy Implementation
 */

#include "search/backend_strategies/mlir_strategy.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>

namespace yirage {
namespace search {

// =============================================================================
// MLIRSearchStrategy Implementation
// =============================================================================

MLIRSearchStrategy::MLIRSearchStrategy() {
  // Default to CPU target with LLVM lowering
  target_backend_ = type::BT_CPU;
  dialect_ = type::MLIR_DIALECT_LINALG;
  target_dialect_ = type::MLIR_DIALECT_LLVM;
  
  // Default optimization config
  opt_config_.enable_tiling = true;
  opt_config_.enable_fusion = true;
  opt_config_.enable_vectorization = true;
  opt_config_.enable_parallelization = true;
}

bool MLIRSearchStrategy::initialize(SearchConfig const &config) {
  config_ = config;
  
  // Configure based on any target hints in config
  configure_for_target(target_backend_);
  
  return true;
}

void MLIRSearchStrategy::set_target_backend(type::BackendType target) {
  target_backend_ = target;
  configure_for_target(target);
}

void MLIRSearchStrategy::set_dialect(type::MLIRDialect dialect) {
  dialect_ = dialect;
}

void MLIRSearchStrategy::configure_for_target(type::BackendType target) {
  switch (target) {
    case type::BT_CUDA:
      target_dialect_ = type::MLIR_DIALECT_NVVM;
      opt_config_.tile_size_m = 128;
      opt_config_.tile_size_n = 128;
      opt_config_.tile_size_k = 32;
      opt_config_.vector_width = 4;  // float4
      opt_config_.num_threads = 256;  // CUDA threads per block
      break;
      
    case type::BT_ROCM:
      target_dialect_ = type::MLIR_DIALECT_ROCDL;
      opt_config_.tile_size_m = 128;
      opt_config_.tile_size_n = 128;
      opt_config_.tile_size_k = 16;
      opt_config_.vector_width = 4;
      opt_config_.num_threads = 256;
      break;
      
    case type::BT_XPU:
      target_dialect_ = type::MLIR_DIALECT_SPIRV;
      opt_config_.tile_size_m = 64;
      opt_config_.tile_size_n = 64;
      opt_config_.tile_size_k = 16;
      opt_config_.vector_width = 8;  // SIMD16
      opt_config_.num_threads = 256;
      break;
      
    case type::BT_TPU:
      // TPU uses StableHLO/MHLO, not direct MLIR lowering
      dialect_ = type::MLIR_DIALECT_STABLEHLO;
      opt_config_.tile_size_m = 128;  // MXU size
      opt_config_.tile_size_n = 128;
      opt_config_.tile_size_k = 128;
      opt_config_.enable_vectorization = false;  // TPU handles this
      break;
      
    case type::BT_CPU:
    default:
      target_dialect_ = type::MLIR_DIALECT_LLVM;
      opt_config_.tile_size_m = 64;
      opt_config_.tile_size_n = 64;
      opt_config_.tile_size_k = 32;
      opt_config_.vector_width = 8;  // AVX2
      opt_config_.num_threads = 8;
      break;
  }
}

std::string MLIRSearchStrategy::get_lowering_pipeline() const {
  return type::get_mlir_lowering_pipeline(type::BT_MLIR, target_backend_);
}

std::vector<CandidateConfig>
MLIRSearchStrategy::generate_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;
  
  // Generate candidates based on dialect
  switch (dialect_) {
    case type::MLIR_DIALECT_LINALG:
      candidates = generate_linalg_candidates(graph);
      break;
    default:
      candidates = generate_linalg_candidates(graph);  // Default to linalg
      break;
  }
  
  num_candidates_generated_ = candidates.size();
  return candidates;
}

std::vector<CandidateConfig>
MLIRSearchStrategy::generate_linalg_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;
  
  // Tile size combinations to explore
  std::vector<int> tile_sizes_m = {32, 64, 128, 256};
  std::vector<int> tile_sizes_n = {32, 64, 128, 256};
  std::vector<int> tile_sizes_k = {16, 32, 64};
  
  // Filter based on target
  if (target_backend_ == type::BT_TPU) {
    tile_sizes_m = {128};  // MXU fixed size
    tile_sizes_n = {128};
    tile_sizes_k = {128};
  }
  
  for (int tm : tile_sizes_m) {
    for (int tn : tile_sizes_n) {
      for (int tk : tile_sizes_k) {
        auto config = std::make_unique<kernel::KernelConfig>();
        config->backend_type = type::BT_MLIR;
        
        // Store tile sizes in config
        config->tile_sizes = {tm, tn, tk};
        
        // Set vectorization width based on target
        config->vector_width = opt_config_.vector_width;
        
        // Parallelization
        config->num_threads = opt_config_.num_threads;
        
        candidates.emplace_back(std::move(config), 0.0f);
      }
    }
  }
  
  return candidates;
}

float MLIRSearchStrategy::evaluate_candidate(CandidateConfig &candidate,
                                            kernel::Graph const &graph) {
  num_candidates_evaluated_++;
  return evaluate_for_target(candidate, graph);
}

float MLIRSearchStrategy::evaluate_for_target(CandidateConfig &candidate,
                                             kernel::Graph const &graph) {
  float score = 0.0f;
  
  auto& config = *candidate.config;
  int tm = config.tile_sizes.size() > 0 ? config.tile_sizes[0] : 64;
  int tn = config.tile_sizes.size() > 1 ? config.tile_sizes[1] : 64;
  int tk = config.tile_sizes.size() > 2 ? config.tile_sizes[2] : 32;
  
  // Base score from tile efficiency
  float tile_efficiency = 1.0f;
  
  switch (target_backend_) {
    case type::BT_CUDA:
    case type::BT_ROCM: {
      // GPU: larger tiles generally better, but must fit in shared memory
      int smem_per_tile = (tm * tk + tk * tn) * sizeof(float);
      int max_smem = 48 * 1024;  // 48KB typical
      
      if (smem_per_tile <= max_smem) {
        tile_efficiency = float(tm * tn) / (128 * 128);  // Normalize to max
      } else {
        tile_efficiency = 0.1f;  // Penalty for exceeding SMEM
      }
      
      // Warp efficiency (32 threads)
      if (tn % 32 == 0) tile_efficiency *= 1.2f;
      break;
    }
    
    case type::BT_TPU: {
      // TPU: must match MXU size (128x128)
      if (tm == 128 && tn == 128 && tk == 128) {
        tile_efficiency = 1.0f;
      } else {
        tile_efficiency = 0.5f;  // Suboptimal tile size
      }
      break;
    }
    
    case type::BT_XPU: {
      // Intel XPU: XMX prefers 8x16 tiles
      if (tm % 8 == 0 && tn % 16 == 0) {
        tile_efficiency = 1.0f;
      } else {
        tile_efficiency = 0.7f;
      }
      break;
    }
    
    case type::BT_CPU:
    default: {
      // CPU: L1/L2 cache friendly tiles
      int l1_size = 32 * 1024;  // 32KB typical
      int tile_bytes = (tm * tk + tk * tn + tm * tn) * sizeof(float);
      
      if (tile_bytes <= l1_size) {
        tile_efficiency = 1.0f;
      } else if (tile_bytes <= 256 * 1024) {  // L2
        tile_efficiency = 0.8f;
      } else {
        tile_efficiency = 0.5f;
      }
      
      // Vectorization alignment
      if (tn % config.vector_width == 0) tile_efficiency *= 1.1f;
      break;
    }
  }
  
  score = tile_efficiency * 100.0f;
  
  // Fusion bonus
  if (opt_config_.enable_fusion) {
    score *= 1.1f;
  }
  
  candidate.score = score;
  best_score_ = std::max(best_score_, score);
  
  return score;
}

kernel::KernelConfig *
MLIRSearchStrategy::select_best_config(std::vector<CandidateConfig> &candidates) {
  if (candidates.empty()) return nullptr;
  
  auto best = std::max_element(candidates.begin(), candidates.end(),
      [](const CandidateConfig &a, const CandidateConfig &b) {
        return a.score < b.score;
      });
  
  return best->config.get();
}

std::unique_ptr<kernel::KernelConfig>
MLIRSearchStrategy::optimize(kernel::Graph const &graph) {
  auto candidates = generate_candidates(graph);
  
  for (auto &candidate : candidates) {
    evaluate_candidate(candidate, graph);
  }
  
  auto best = std::max_element(candidates.begin(), candidates.end(),
      [](const CandidateConfig &a, const CandidateConfig &b) {
        return a.score < b.score;
      });
  
  if (best != candidates.end()) {
    return std::move(best->config);
  }
  
  return nullptr;
}

std::string MLIRSearchStrategy::generate_mlir_code(
    kernel::KernelConfig const &config) const {
  std::ostringstream ss;
  
  int tm = config.tile_sizes.size() > 0 ? config.tile_sizes[0] : 64;
  int tn = config.tile_sizes.size() > 1 ? config.tile_sizes[1] : 64;
  int tk = config.tile_sizes.size() > 2 ? config.tile_sizes[2] : 32;
  
  ss << "// Generated MLIR (Linalg dialect)\n";
  ss << "// Target: " << type::backend_type_to_string(target_backend_) << "\n";
  ss << "// Tile sizes: " << tm << "x" << tn << "x" << tk << "\n\n";
  
  ss << "func.func @matmul_tiled(\n";
  ss << "    %A: tensor<?x?xf32>,\n";
  ss << "    %B: tensor<?x?xf32>,\n";
  ss << "    %C: tensor<?x?xf32>) -> tensor<?x?xf32> {\n";
  ss << "  %c0 = arith.constant 0 : index\n";
  ss << "  %c" << tm << " = arith.constant " << tm << " : index\n";
  ss << "  %c" << tn << " = arith.constant " << tn << " : index\n";
  ss << "  %c" << tk << " = arith.constant " << tk << " : index\n\n";
  
  ss << "  // Tiled linalg.matmul\n";
  ss << "  %result = scf.forall (%i, %j) in (%c" << tm << ", %c" << tn << ") {\n";
  ss << "    // Extract tiles\n";
  ss << "    // Compute local matmul\n";
  ss << "    // Store result\n";
  ss << "  }\n";
  ss << "  return %result : tensor<?x?xf32>\n";
  ss << "}\n";
  
  return ss.str();
}

std::string MLIRSearchStrategy::get_statistics() const {
  std::ostringstream ss;
  ss << "MLIR Search Statistics:\n";
  ss << "  Target backend: " << type::backend_type_to_string(target_backend_) << "\n";
  ss << "  Dialect: " << type::get_mlir_dialect_name(dialect_) << "\n";
  ss << "  Target dialect: " << type::get_mlir_dialect_name(target_dialect_) << "\n";
  ss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  ss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  ss << "  Best score: " << best_score_ << "\n";
  ss << "  Lowering pipeline: " << get_lowering_pipeline() << "\n";
  return ss.str();
}

// =============================================================================
// StableHLOSearchStrategy Implementation
// =============================================================================

StableHLOSearchStrategy::StableHLOSearchStrategy() : MLIRSearchStrategy() {
  dialect_ = type::MLIR_DIALECT_STABLEHLO;
  target_backend_ = type::BT_TPU;  // Default target
  configure_for_target(target_backend_);
}

std::vector<CandidateConfig>
StableHLOSearchStrategy::generate_hlo_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;
  
  // StableHLO focuses on high-level fusion, not low-level tiling
  // Generate candidates based on fusion strategies
  
  auto config = std::make_unique<kernel::KernelConfig>();
  config->backend_type = type::BT_STABLEHLO;
  config->tile_sizes = {128, 128, 128};  // TPU MXU
  
  if (enable_sharding_) {
    // Add sharding configuration
  }
  
  candidates.emplace_back(std::move(config), 0.0f);
  
  return candidates;
}

std::string StableHLOSearchStrategy::get_statistics() const {
  std::ostringstream ss;
  ss << "StableHLO Search Statistics:\n";
  ss << "  Target backend: " << type::backend_type_to_string(target_backend_) << "\n";
  ss << "  SPMD enabled: " << (enable_spmd_ ? "yes" : "no") << "\n";
  ss << "  Sharding enabled: " << (enable_sharding_ ? "yes" : "no") << "\n";
  ss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  ss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  ss << "  Best score: " << best_score_ << "\n";
  return ss.str();
}

// =============================================================================
// TOSASearchStrategy Implementation
// =============================================================================

TOSASearchStrategy::TOSASearchStrategy() : MLIRSearchStrategy() {
  dialect_ = type::MLIR_DIALECT_TOSA;
  target_backend_ = type::BT_CPU;  // TOSA is portable, default to CPU
  configure_for_target(target_backend_);
}

std::vector<CandidateConfig>
TOSASearchStrategy::generate_tosa_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;
  
  // TOSA is designed for portability, so we generate
  // candidates that work well across different targets
  
  std::vector<int> tile_sizes;
  switch (profile_) {
    case TOSA_BASE:
      tile_sizes = {16, 32};  // Small tiles for edge
      break;
    case TOSA_MAIN:
      tile_sizes = {32, 64, 128};
      break;
    case TOSA_FULL:
      tile_sizes = {32, 64, 128, 256};
      break;
  }
  
  for (int ts : tile_sizes) {
    auto config = std::make_unique<kernel::KernelConfig>();
    config->backend_type = type::BT_TOSA;
    config->tile_sizes = {ts, ts, ts};
    candidates.emplace_back(std::move(config), 0.0f);
  }
  
  return candidates;
}

std::string TOSASearchStrategy::get_statistics() const {
  std::ostringstream ss;
  ss << "TOSA Search Statistics:\n";
  ss << "  Profile: " << (profile_ == TOSA_BASE ? "base" : 
                         profile_ == TOSA_MAIN ? "main" : "full") << "\n";
  ss << "  Target backend: " << type::backend_type_to_string(target_backend_) << "\n";
  ss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  ss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  ss << "  Best score: " << best_score_ << "\n";
  return ss.str();
}

// =============================================================================
// LinalgSearchStrategy Implementation
// =============================================================================

LinalgSearchStrategy::LinalgSearchStrategy() : MLIRSearchStrategy() {
  dialect_ = type::MLIR_DIALECT_LINALG;
  tiling_strategy_ = TILE_AUTOTUNE;
}

std::vector<CandidateConfig>
LinalgSearchStrategy::generate_tiled_candidates(kernel::Graph const &graph) {
  std::vector<CandidateConfig> candidates;
  
  switch (tiling_strategy_) {
    case TILE_NONE: {
      auto config = std::make_unique<kernel::KernelConfig>();
      config->backend_type = type::BT_LINALG;
      config->tile_sizes = {};  // No tiling
      candidates.emplace_back(std::move(config), 0.0f);
      break;
    }
    
    case TILE_FIXED: {
      auto config = std::make_unique<kernel::KernelConfig>();
      config->backend_type = type::BT_LINALG;
      config->tile_sizes = {opt_config_.tile_size_m, 
                           opt_config_.tile_size_n, 
                           opt_config_.tile_size_k};
      candidates.emplace_back(std::move(config), 0.0f);
      break;
    }
    
    case TILE_DYNAMIC:
    case TILE_AUTOTUNE:
    default:
      return generate_linalg_candidates(graph);
  }
  
  return candidates;
}

std::string LinalgSearchStrategy::get_statistics() const {
  std::ostringstream ss;
  ss << "Linalg Search Statistics:\n";
  ss << "  Tiling strategy: " << (tiling_strategy_ == TILE_NONE ? "none" :
                                  tiling_strategy_ == TILE_FIXED ? "fixed" :
                                  tiling_strategy_ == TILE_DYNAMIC ? "dynamic" : 
                                  "autotune") << "\n";
  ss << "  Target backend: " << type::backend_type_to_string(target_backend_) << "\n";
  ss << "  Candidates generated: " << num_candidates_generated_ << "\n";
  ss << "  Candidates evaluated: " << num_candidates_evaluated_ << "\n";
  ss << "  Best score: " << best_score_ << "\n";
  return ss.str();
}

}  // namespace search
}  // namespace yirage
