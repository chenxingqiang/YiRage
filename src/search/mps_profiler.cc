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

#include "yirage/search/mps_profiler.h"

#ifdef YIRAGE_BACKEND_MPS_ENABLED

#include <algorithm>
#include <cmath>
#include <sstream>

namespace yirage {
namespace search {

void MPSProfiler::record_execution(
    kernel::mps::MPSKernelConfig const &config,
    float execution_time_ms,
    float estimated_score,
    size_t m, size_t n, size_t k,
    std::string const &kernel_type) {
  
  if (!enabled_) return;
  
  ProfileEntry entry;
  entry.config = config;
  entry.actual_time_ms = execution_time_ms;
  entry.estimated_score = estimated_score;
  entry.problem_m = m;
  entry.problem_n = n;
  entry.problem_k = k;
  entry.gpu_family = config.gpu_family;
  entry.kernel_type = kernel_type;
  
  history_.push_back(entry);
  
  // Update cache for this problem size
  std::string cache_key = make_cache_key(m, n, k, kernel_type);
  
  // Only cache if this is better than existing
  auto it = cache_.find(cache_key);
  if (it == cache_.end() || 
      execution_time_ms < history_[it->second.gpu_family].actual_time_ms) {
    cache_[cache_key] = config;
  }
}

std::unique_ptr<kernel::mps::MPSKernelConfig>
MPSProfiler::suggest_config(size_t m, size_t n, size_t k,
                           std::string const &kernel_type) {
  
  // Try exact match first
  std::string cache_key = make_cache_key(m, n, k, kernel_type);
  auto it = cache_.find(cache_key);
  if (it != cache_.end()) {
    return std::make_unique<kernel::mps::MPSKernelConfig>(it->second);
  }
  
  // Find similar problems
  auto similar = find_similar_problems(m, n, k, kernel_type, 0.3f);
  
  if (similar.empty()) {
    return nullptr;  // No suggestions
  }
  
  // Get best from similar problems
  auto best_it = std::min_element(
      similar.begin(), similar.end(),
      [](ProfileEntry const &a, ProfileEntry const &b) {
        return a.actual_time_ms < b.actual_time_ms;
      });
  
  return std::make_unique<kernel::mps::MPSKernelConfig>(best_it->config);
}

std::vector<MPSProfiler::ProfileEntry>
MPSProfiler::find_similar_problems(size_t m, size_t n, size_t k,
                                  std::string const &kernel_type,
                                  float tolerance) {
  
  std::vector<ProfileEntry> similar;
  
  for (auto const &entry : history_) {
    if (entry.kernel_type != kernel_type) continue;
    
    float similarity = compute_similarity(
        m, n, k,
        entry.problem_m, entry.problem_n, entry.problem_k);
    
    if (similarity >= (1.0f - tolerance)) {
      similar.push_back(entry);
    }
  }
  
  return similar;
}

float MPSProfiler::compute_similarity(size_t m1, size_t n1, size_t k1,
                                      size_t m2, size_t n2, size_t k2) {
  // Compute normalized distance in log space
  auto log_dist = [](size_t a, size_t b) {
    float log_a = std::log2(static_cast<float>(a) + 1.0f);
    float log_b = std::log2(static_cast<float>(b) + 1.0f);
    return std::abs(log_a - log_b);
  };
  
  float dist_m = log_dist(m1, m2);
  float dist_n = log_dist(n1, n2);
  float dist_k = log_dist(k1, k2);
  
  // Average distance
  float avg_dist = (dist_m + dist_n + dist_k) / 3.0f;
  
  // Convert to similarity (0-1, where 1 is identical)
  float similarity = std::exp(-avg_dist);
  
  return similarity;
}

std::string MPSProfiler::make_cache_key(size_t m, size_t n, size_t k,
                                       std::string const &kernel_type) {
  std::ostringstream oss;
  oss << kernel_type << "_" << m << "x" << n << "x" << k;
  return oss.str();
}

bool MPSProfiler::save_to_file(std::string const &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) return false;
  
  // Write header
  file << "# YiRage MPS Profile Database\n";
  file << "# kernel_type,m,n,k,tg_size,tile_m,tile_n,tile_k,"
       << "time_ms,score,gpu_family,gpu_cores\n";
  
  // Write entries
  for (auto const &entry : history_) {
    file << entry.kernel_type << ","
         << entry.problem_m << "," << entry.problem_n << "," << entry.problem_k << ","
         << entry.config.threads_per_threadgroup << ","
         << entry.config.tile_m << "," << entry.config.tile_n << ","
         << entry.config.tile_k << ","
         << entry.actual_time_ms << "," << entry.estimated_score << ","
         << entry.gpu_family << "," << entry.gpu_cores << "\n";
  }
  
  file.close();
  return true;
}

bool MPSProfiler::load_from_file(std::string const &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) return false;
  
  std::string line;
  // Skip header lines
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    
    // Parse CSV line
    // TODO: Implement CSV parsing
    // For now, just skip
  }
  
  file.close();
  return true;
}

void MPSProfiler::clear_history() {
  history_.clear();
  cache_.clear();
}

std::string MPSProfiler::get_statistics() const {
  std::ostringstream oss;
  oss << "MPS Profiler Statistics:\n";
  oss << "  Total executions recorded: " << history_.size() << "\n";
  oss << "  Cached configurations: " << cache_.size() << "\n";
  oss << "  Status: " << (enabled_ ? "Enabled" : "Disabled") << "\n";
  
  if (!history_.empty()) {
    // Find best time
    auto best = std::min_element(
        history_.begin(), history_.end(),
        [](ProfileEntry const &a, ProfileEntry const &b) {
          return a.actual_time_ms < b.actual_time_ms;
        });
    
    oss << "  Best time: " << best->actual_time_ms << " ms\n";
    oss << "  Best config: tg=" << best->config.threads_per_threadgroup
        << " tile=(" << best->config.tile_m << "," 
        << best->config.tile_n << "," << best->config.tile_k << ")\n";
  }
  
  return oss.str();
}

} // namespace search
} // namespace yirage

#endif // YIRAGE_BACKEND_MPS_ENABLED

