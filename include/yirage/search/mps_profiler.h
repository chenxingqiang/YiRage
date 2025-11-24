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

#include "yirage/kernel/mps/mps_kernel_config.h"
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>

#ifdef YIRAGE_BACKEND_MPS_ENABLED

namespace yirage {
namespace search {

/**
 * @brief Profile-guided optimization for MPS kernels
 * 
 * Records actual kernel performance and uses historical data
 * to guide future optimization decisions.
 */
class MPSProfiler {
public:
  /**
   * @brief Profile entry recording kernel execution
   */
  struct ProfileEntry {
    kernel::mps::MPSKernelConfig config;
    float actual_time_ms;
    float estimated_score;
    size_t problem_m;
    size_t problem_n;
    size_t problem_k;
    int gpu_family;
    int gpu_cores;
    std::string kernel_type;  // "matmul", "attention", etc.
  };
  
  MPSProfiler() : enabled_(true) {}
  
  /**
   * @brief Record a kernel execution result
   */
  void record_execution(kernel::mps::MPSKernelConfig const &config,
                       float execution_time_ms,
                       float estimated_score,
                       size_t m, size_t n, size_t k,
                       std::string const &kernel_type);
  
  /**
   * @brief Suggest optimal configuration based on history
   * @param m M dimension
   * @param n N dimension  
   * @param k K dimension
   * @param kernel_type Type of kernel
   * @return Suggested configuration or nullptr if no history
   */
  std::unique_ptr<kernel::mps::MPSKernelConfig>
      suggest_config(size_t m, size_t n, size_t k,
                    std::string const &kernel_type);
  
  /**
   * @brief Find similar problems from history
   */
  std::vector<ProfileEntry>
      find_similar_problems(size_t m, size_t n, size_t k,
                           std::string const &kernel_type,
                           float tolerance = 0.2f);
  
  /**
   * @brief Save profile database to file
   */
  bool save_to_file(std::string const &filename);
  
  /**
   * @brief Load profile database from file
   */
  bool load_from_file(std::string const &filename);
  
  /**
   * @brief Clear all history
   */
  void clear_history();
  
  /**
   * @brief Get statistics
   */
  std::string get_statistics() const;
  
  /**
   * @brief Enable/disable profiling
   */
  void set_enabled(bool enabled) { enabled_ = enabled; }
  bool is_enabled() const { return enabled_; }

private:
  bool enabled_;
  std::vector<ProfileEntry> history_;
  
  // Cache for quick lookup: "kernel_type_mxnxk" -> best config
  std::unordered_map<std::string, kernel::mps::MPSKernelConfig> cache_;
  
  /**
   * @brief Compute similarity score between two problem sizes
   */
  float compute_similarity(size_t m1, size_t n1, size_t k1,
                          size_t m2, size_t n2, size_t k2);
  
  /**
   * @brief Generate cache key
   */
  std::string make_cache_key(size_t m, size_t n, size_t k,
                            std::string const &kernel_type);
};

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_MPS_ENABLED

