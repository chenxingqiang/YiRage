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


#include "yirage/kernel/cpu/cpu_kernel_config.h"

#ifdef YIRAGE_BACKEND_CPU_ENABLED

#include <algorithm>
#include <cmath>
#include <sstream>
#include <thread>

#ifdef __x86_64__
#include <cpuid.h>
#endif

namespace yirage {
namespace kernel {
namespace cpu {

SIMDType CPUOptimizer::detect_simd_support() {
#ifdef __x86_64__
  unsigned int eax, ebx, ecx, edx;

  // Check for AVX-512
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    if (ebx & (1 << 16)) { // AVX512F
      return SIMDType::AVX512;
    }
  }

  // Check for AVX2
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    if (ebx & (1 << 5)) { // AVX2
      return SIMDType::AVX2;
    }
  }

  // Check for AVX
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    if (ecx & (1 << 28)) { // AVX
      return SIMDType::AVX;
    }

    // Check for SSE4.2
    if (ecx & (1 << 20)) {
      return SIMDType::SSE4_2;
    }

    // Check for SSE4.1
    if (ecx & (1 << 19)) {
      return SIMDType::SSE4_1;
    }

    // Check for SSE3
    if (ecx & (1 << 0)) {
      return SIMDType::SSE3;
    }

    // Check for SSE2
    if (edx & (1 << 26)) {
      return SIMDType::SSE2;
    }

    // Check for SSE
    if (edx & (1 << 25)) {
      return SIMDType::SSE;
    }
  }
#endif

  return SIMDType::NONE;
}

std::string CPUOptimizer::get_cpu_features() {
  std::ostringstream oss;
  SIMDType simd = detect_simd_support();
  oss << "SIMD: " << simd_type_to_string(simd) << ", ";
  oss << "Cores: " << std::thread::hardware_concurrency();
  return oss.str();
}

void CPUOptimizer::compute_optimal_tiles(int m, int n, int k,
                                        size_t element_size,
                                        CPUKernelConfig &config) {
  // Compute tile sizes to fit in L1/L2 cache
  size_t l1_size = config.l1_cache_size;
  size_t l2_size = config.l2_cache_size;

  // We want: tile_m * tile_k + tile_k * tile_n + tile_m * tile_n <= L2_size
  // Simplified: optimize for L2 cache
  size_t total_elements = l2_size / element_size / 3; // Divide by 3 matrices

  // Try to keep tiles square-ish for better cache reuse
  int tile_size = static_cast<int>(std::sqrt(total_elements));

  // Round down to multiple of SIMD width
  int vector_width = config.get_vector_bytes() / element_size;
  tile_size = (tile_size / vector_width) * vector_width;

  // Clamp to reasonable range
  tile_size = std::max(16, std::min(256, tile_size));

  config.tile_m = std::min(m, tile_size);
  config.tile_n = std::min(n, tile_size);
  config.tile_k = std::min(k, tile_size);

  // Compute micro-tile sizes for L1 cache
  int micro_size = static_cast<int>(std::sqrt(l1_size / element_size / 3));
  micro_size = (micro_size / vector_width) * vector_width;
  micro_size = std::max(4, std::min(32, micro_size));

  config.micro_tile_m = micro_size;
  config.micro_tile_n = micro_size;
}

int CPUOptimizer::compute_optimal_threads(size_t problem_size, int num_cores,
                                         bool memory_bound) {
  if (num_cores <= 0) {
    num_cores = std::thread::hardware_concurrency();
  }

  if (memory_bound) {
    // For memory-bound workloads, use fewer threads to avoid memory bandwidth
    // saturation
    return std::max(1, num_cores / 2);
  } else {
    // For compute-bound workloads, use all cores
    // But ensure enough work per thread
    size_t min_work_per_thread = 1024; // Minimum elements per thread
    int max_useful_threads =
        static_cast<int>(problem_size / min_work_per_thread);
    return std::max(1, std::min(num_cores, max_useful_threads));
  }
}

float CPUOptimizer::estimate_cache_efficiency(CPUKernelConfig const &config,
                                              int m, int n, int k,
                                              size_t element_size) {
  // Estimate based on whether working set fits in cache
  size_t working_set =
      (config.tile_m * config.tile_k + config.tile_k * config.tile_n +
       config.tile_m * config.tile_n) *
      element_size;

  float l1_ratio = static_cast<float>(working_set) / config.l1_cache_size;
  float l2_ratio = static_cast<float>(working_set) / config.l2_cache_size;
  float l3_ratio = static_cast<float>(working_set) / config.l3_cache_size;

  // Weighted cache hit rate
  float hit_rate = 0.0f;
  if (l1_ratio <= 1.0f) {
    hit_rate = 0.99f; // L1 hit
  } else if (l2_ratio <= 1.0f) {
    hit_rate = 0.95f; // L2 hit
  } else if (l3_ratio <= 1.0f) {
    hit_rate = 0.85f; // L3 hit
  } else {
    hit_rate = 0.5f; // Memory access
  }

  return hit_rate;
}

float CPUOptimizer::estimate_vectorization_efficiency(
    CPUKernelConfig const &config, size_t data_size) {
  if (config.simd_type == SIMDType::NONE) {
    return 1.0f; // Scalar performance baseline
  }

  int vector_width = config.get_vector_bytes() / 4; // Assume float32
  size_t vectorized_elements = (data_size / vector_width) * vector_width;
  size_t scalar_elements = data_size - vectorized_elements;

  float vectorized_ratio =
      static_cast<float>(vectorized_elements) / data_size;
  float speedup = static_cast<float>(vector_width);

  // Account for unaligned accesses
  float alignment_penalty = (config.alignment == 64) ? 1.0f : 0.9f;

  return (vectorized_ratio * speedup + (1.0f - vectorized_ratio)) *
         alignment_penalty;
}

void CPUOptimizer::optimize_for_cpu(int m, int n, int k,
                                   CPUKernelConfig &config) {
  // Detect SIMD support
  config.simd_type = detect_simd_support();

  // Set vector width based on SIMD type
  switch (config.simd_type) {
  case SIMDType::AVX512:
    config.vector_width = 16; // 16 floats
    config.alignment = 64;
    break;
  case SIMDType::AVX2:
  case SIMDType::AVX:
    config.vector_width = 8; // 8 floats
    config.alignment = 32;
    break;
  case SIMDType::SSE4_2:
  case SIMDType::SSE4_1:
  case SIMDType::SSE3:
  case SIMDType::SSE2:
  case SIMDType::SSE:
    config.vector_width = 4; // 4 floats
    config.alignment = 16;
    break;
  default:
    config.vector_width = 1;
    config.alignment = 4;
    break;
  }

  // Compute optimal tile sizes
  compute_optimal_tiles(m, n, k, sizeof(float), config);

  // Compute optimal thread count
  size_t problem_size = static_cast<size_t>(m) * n * k;
  int num_cores = std::thread::hardware_concurrency();
  config.num_threads = compute_optimal_threads(problem_size, num_cores, false);

  // Set unroll factor
  config.unroll_factor = compute_unroll_factor(k, config.simd_type);
}

int CPUOptimizer::compute_unroll_factor(int loop_size, SIMDType simd_type) {
  // Base unroll factor on SIMD width
  int base_unroll = 4;

  switch (simd_type) {
  case SIMDType::AVX512:
    base_unroll = 8;
    break;
  case SIMDType::AVX2:
  case SIMDType::AVX:
    base_unroll = 4;
    break;
  case SIMDType::SSE4_2:
  case SIMDType::SSE4_1:
  case SIMDType::SSE3:
  case SIMDType::SSE2:
  case SIMDType::SSE:
    base_unroll = 2;
    break;
  default:
    base_unroll = 1;
    break;
  }

  // Adjust based on loop size
  if (loop_size < 16) {
    base_unroll = std::min(base_unroll, 2);
  }

  return base_unroll;
}

} // namespace cpu
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_CPU_ENABLED





