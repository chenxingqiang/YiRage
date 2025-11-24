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


#include "yirage/kernel/mps/mps_kernel_config.h"

#ifdef YIRAGE_BACKEND_MPS_ENABLED

#include <algorithm>
#include <cmath>

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

namespace yirage {
namespace kernel {
namespace mps {

int MPSOptimizer::detect_gpu_family() {
#ifdef __APPLE__
  // Try to detect Apple Silicon generation
  char model[256];
  size_t len = sizeof(model);
  if (sysctlbyname("hw.model", model, &len, NULL, 0) == 0) {
    std::string model_str(model);
    // M1: Family 7, M2: Family 8, M3: Family 9
    if (model_str.find("Mac14") != std::string::npos) {
      return 8; // M2
    } else if (model_str.find("Mac15") != std::string::npos) {
      return 9; // M3
    } else if (model_str.find("Mac13") != std::string::npos) {
      return 7; // M1
    }
  }
  
  // Default to M2
  return 8;
#else
  return 0;
#endif
}

int MPSOptimizer::get_gpu_core_count() {
#ifdef __APPLE__
  // Try to detect specific Mac model to get accurate GPU core count
  char model[256];
  size_t len = sizeof(model);
  if (sysctlbyname("hw.model", model, &len, NULL, 0) == 0) {
    std::string model_str(model);
    
    // M1 series (Family 7)
    // Mac13,x models
    if (model_str.find("Mac13,1") != std::string::npos) {
      return 24;  // M1 Max (24 or 32 cores)
    } else if (model_str.find("Mac13,2") != std::string::npos) {
      return 16;  // M1 Pro (14 or 16 cores)
    } else if (model_str.find("Mac13,") != std::string::npos) {
      return 8;   // M1 (7 or 8 cores)
    } else if (model_str.find("Mac14,13") != std::string::npos ||
               model_str.find("Mac14,14") != std::string::npos ||
               model_str.find("Mac14,15") != std::string::npos) {
      return 64;  // M1 Ultra (48 or 64 cores)
    }
    
    // M2 series (Family 8)
    // Mac14,x and Mac15,x models
    if (model_str.find("Mac14,6") != std::string::npos ||
        model_str.find("Mac14,10") != std::string::npos) {
      return 38;  // M2 Max (30 or 38 cores)
    } else if (model_str.find("Mac14,5") != std::string::npos ||
               model_str.find("Mac14,9") != std::string::npos ||
               model_str.find("Mac14,12") != std::string::npos) {
      return 19;  // M2 Pro (16 or 19 cores)
    } else if (model_str.find("Mac14,") != std::string::npos) {
      return 10;  // M2 (8 or 10 cores)
    } else if (model_str.find("Mac15,4") != std::string::npos ||
               model_str.find("Mac15,5") != std::string::npos) {
      return 76;  // M2 Ultra (60 or 76 cores)
    }
    
    // M3 series (Family 9)
    // Mac15,x and Mac16,x models
    if (model_str.find("Mac15,7") != std::string::npos ||
        model_str.find("Mac15,10") != std::string::npos ||
        model_str.find("Mac15,11") != std::string::npos) {
      return 40;  // M3 Max (30 or 40 cores)
    } else if (model_str.find("Mac15,3") != std::string::npos ||
               model_str.find("Mac15,6") != std::string::npos ||
               model_str.find("Mac15,8") != std::string::npos) {
      return 18;  // M3 Pro (14 or 18 cores)
    } else if (model_str.find("Mac15,") != std::string::npos) {
      return 10;  // M3 (10 cores)
    }
    
    // M4 series (Family 10 - future)
    if (model_str.find("Mac16,") != std::string::npos) {
      // M4 detection based on sub-model
      if (model_str.find("Mac16,6") != std::string::npos ||
          model_str.find("Mac16,7") != std::string::npos) {
        return 40;  // M4 Max (estimated)
      } else if (model_str.find("Mac16,3") != std::string::npos ||
                 model_str.find("Mac16,4") != std::string::npos) {
        return 20;  // M4 Pro (estimated)
      }
      return 10;  // M4 base (estimated)
    }
  }
  
  // Fallback: use GPU family
  int family = detect_gpu_family();
  switch (family) {
  case 7:  // M1
    return 8;
  case 8:  // M2
    return 10;
  case 9:  // M3
    return 10;
  case 10: // M4
    return 10;
  default:
    return 8;
  }
#else
  return 0;
#endif
}

int MPSOptimizer::compute_optimal_threadgroup_size(size_t problem_size,
                                                   int gpu_family) {
  // Apple GPUs work well with threadgroup sizes that are multiples of SIMD
  // width
  int simd_width = 32;

  // Start with a reasonable base size
  int base_size = 256; // Common choice

  // Adjust based on problem size
  if (problem_size < 1024) {
    base_size = 128;
  } else if (problem_size > 1024 * 1024) {
    base_size = 512;
  }

  // Ensure it's a multiple of SIMD width
  base_size = ((base_size + simd_width - 1) / simd_width) * simd_width;

  // Clamp to valid range (typically 32-1024 for Metal)
  return std::max(32, std::min(1024, base_size));
}

void MPSOptimizer::compute_optimal_tiles(int m, int n, int k, int gpu_family,
                                        MPSKernelConfig &config) {
  // Threadgroup memory is similar to CUDA shared memory
  size_t tg_memory = config.threadgroup_memory_size;

  // Element size (assume float32)
  size_t element_size = 4;

  // We want: tile_m * tile_k + tile_k * tile_n + tile_m * tile_n <=
  // tg_memory
  size_t total_elements = tg_memory / element_size / 3;

  // Keep tiles square-ish for better reuse
  int tile_size = static_cast<int>(std::sqrt(total_elements));

  // Align to SIMD width
  int simd_width = 32;
  tile_size = (tile_size / simd_width) * simd_width;

  // Clamp to reasonable range
  tile_size = std::max(16, std::min(64, tile_size));

  config.tile_m = std::min(m, tile_size);
  config.tile_n = std::min(n, tile_size);
  config.tile_k = std::min(k, tile_size);
}

MemoryPattern MPSOptimizer::select_memory_pattern(size_t data_size,
                                                  int stride) {
  // Coalesced access is best when stride is 1
  if (stride == 1) {
    return MemoryPattern::COALESCED;
  }

  // For small data with irregular access, use tiled pattern
  if (data_size < 4096 && stride > 16) {
    return MemoryPattern::TILED;
  }

  // Otherwise use strided
  return MemoryPattern::STRIDED;
}

float MPSOptimizer::estimate_memory_bandwidth(MPSKernelConfig const &config,
                                              size_t bytes_accessed,
                                              float execution_time_ms) {
  if (execution_time_ms <= 0.0f) {
    return 0.0f;
  }

  // Convert to GB/s
  float seconds = execution_time_ms / 1000.0f;
  float gigabytes = bytes_accessed / (1024.0f * 1024.0f * 1024.0f);
  return gigabytes / seconds;
}

void MPSOptimizer::optimize_for_apple_silicon(int m, int n, int k,
                                              MPSKernelConfig &config) {
  // Detect GPU family
  config.gpu_family = detect_gpu_family();

  // Set optimal threadgroup size
  size_t problem_size = static_cast<size_t>(m) * n;
  config.threads_per_threadgroup =
      compute_optimal_threadgroup_size(problem_size, config.gpu_family);

  // Compute optimal tile sizes
  compute_optimal_tiles(m, n, k, config.gpu_family, config);

  // Set SIMD width based on GPU family
  config.simd_width = 32; // Standard for Apple GPUs

  // Set threadgroup memory based on GPU family
  if (config.gpu_family >= 8) {
    // M2 and later have more threadgroup memory
    config.threadgroup_memory_size = 64 * 1024; // 64 KB
  } else {
    config.threadgroup_memory_size = 32 * 1024; // 32 KB
  }

  // Enable fast math for better performance
  config.use_fast_math = true;

  // Set grid dimensions
  int tile_m = config.tile_m;
  int tile_n = config.tile_n;
  config.grid_dim_x = (n + tile_n - 1) / tile_n;
  config.grid_dim_y = (m + tile_m - 1) / tile_m;
  config.grid_dim_z = 1;
}

} // namespace mps
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_MPS_ENABLED





