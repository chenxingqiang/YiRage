/* Copyright 2025 YiRage Team
 * MLIR Search Strategy - Universal IR for multi-target compilation
 */

#pragma once

#include "search/common/search_strategy.h"
#include "type.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace yirage {
namespace search {

/**
 * @brief MLIR-based search strategy for universal kernel optimization
 * 
 * Supports multiple lowering paths:
 *   - StableHLO/MHLO → TPU/GPU/CPU
 *   - TOSA → portable targets
 *   - Linalg → tiled/fused execution
 *   - Generic MLIR → LLVM/NVVM/ROCDL/SPIRV
 */
class MLIRSearchStrategy : public SearchStrategy {
public:
  MLIRSearchStrategy();
  ~MLIRSearchStrategy() override = default;

  bool initialize(SearchConfig const &config) override;

  std::vector<CandidateConfig>
  generate_candidates(kernel::Graph const &graph) override;

  float evaluate_candidate(CandidateConfig &candidate,
                          kernel::Graph const &graph) override;

  kernel::KernelConfig *
  select_best_config(std::vector<CandidateConfig> &candidates) override;

  std::unique_ptr<kernel::KernelConfig>
  optimize(kernel::Graph const &graph) override;

  type::BackendType get_backend_type() const override { return type::BT_MLIR; }

  std::string get_statistics() const override;

  // MLIR-specific methods
  
  /**
   * @brief Set the target hardware backend for MLIR lowering
   */
  void set_target_backend(type::BackendType target);
  
  /**
   * @brief Set the MLIR dialect to use for code generation
   */
  void set_dialect(type::MLIRDialect dialect);
  
  /**
   * @brief Get the lowering pass pipeline for current configuration
   */
  std::string get_lowering_pipeline() const;
  
  /**
   * @brief Generate MLIR code for a kernel configuration
   */
  std::string generate_mlir_code(kernel::KernelConfig const &config) const;

protected:
  // Target backend (CUDA, ROCm, CPU, TPU, etc.)
  type::BackendType target_backend_ = type::BT_CPU;
  
  // MLIR dialect for intermediate representation
  type::MLIRDialect dialect_ = type::MLIR_DIALECT_LINALG;
  
  // Target-specific dialect for final lowering
  type::MLIRDialect target_dialect_ = type::MLIR_DIALECT_LLVM;
  
  // Optimization settings
  struct MLIROptConfig {
    bool enable_tiling = true;
    bool enable_fusion = true;
    bool enable_vectorization = true;
    bool enable_parallelization = true;
    int tile_size_m = 64;
    int tile_size_n = 64;
    int tile_size_k = 32;
    int vector_width = 8;
    int num_threads = 4;
  };
  MLIROptConfig opt_config_;
  
  // Generate tiled linalg candidates
  std::vector<CandidateConfig> generate_linalg_candidates(
      kernel::Graph const &graph);
  
  // Evaluate based on target backend
  float evaluate_for_target(CandidateConfig &candidate,
                           kernel::Graph const &graph);
  
  // Configure optimization for target
  void configure_for_target(type::BackendType target);
};

/**
 * @brief StableHLO search strategy (XLA/TPU compatible)
 */
class StableHLOSearchStrategy : public MLIRSearchStrategy {
public:
  StableHLOSearchStrategy();
  
  type::BackendType get_backend_type() const override { 
    return type::BT_STABLEHLO; 
  }
  
  std::string get_statistics() const override;

protected:
  // StableHLO-specific optimizations
  bool enable_sharding_ = false;
  bool enable_spmd_ = false;
  
  std::vector<CandidateConfig> generate_hlo_candidates(
      kernel::Graph const &graph);
};

/**
 * @brief TOSA search strategy (portable, edge-focused)
 */
class TOSASearchStrategy : public MLIRSearchStrategy {
public:
  TOSASearchStrategy();
  
  type::BackendType get_backend_type() const override { 
    return type::BT_TOSA; 
  }
  
  std::string get_statistics() const override;

protected:
  // TOSA profile (base, main, full)
  enum TOSAProfile {
    TOSA_BASE,    // Minimal ops, int8 focus
    TOSA_MAIN,    // Standard ops, fp32/fp16
    TOSA_FULL     // All ops including extensions
  };
  TOSAProfile profile_ = TOSA_MAIN;
  
  std::vector<CandidateConfig> generate_tosa_candidates(
      kernel::Graph const &graph);
};

/**
 * @brief Linalg search strategy (structured ops, tiling)
 */
class LinalgSearchStrategy : public MLIRSearchStrategy {
public:
  LinalgSearchStrategy();
  
  type::BackendType get_backend_type() const override { 
    return type::BT_LINALG; 
  }
  
  std::string get_statistics() const override;

protected:
  // Tiling strategies
  enum TilingStrategy {
    TILE_NONE,
    TILE_FIXED,
    TILE_DYNAMIC,
    TILE_AUTOTUNE
  };
  TilingStrategy tiling_strategy_ = TILE_AUTOTUNE;
  
  std::vector<CandidateConfig> generate_tiled_candidates(
      kernel::Graph const &graph);
};

}  // namespace search
}  // namespace yirage
