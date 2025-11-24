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

#include "yirage/kernel/ascend/ascend_kernel_config.h"

#ifdef YIRAGE_BACKEND_ASCEND_ENABLED

#include <algorithm>
#include <cmath>

#ifdef __ASCEND__
#include "acl/acl.h"
#include "acl/acl_rt.h"
#endif

namespace yirage {
namespace kernel {
namespace ascend {

int AscendOptimizer::detect_device_type() {
#ifdef __ASCEND__
  const char *soc_name = aclrtGetSocName();
  if (soc_name) {
    std::string soc(soc_name);
    if (soc.find("Ascend910B") != std::string::npos || 
        soc.find("910B") != std::string::npos) {
      return 1;  // 910B
    } else if (soc.find("Ascend910") != std::string::npos) {
      return 0;  // 910
    } else if (soc.find("Ascend310P") != std::string::npos ||
               soc.find("310P") != std::string::npos) {
      return 2;  // 310P
    }
  }
#endif
  return 0;  // Default to 910
}

int AscendOptimizer::get_ai_core_count() {
#ifdef __ASCEND__
  int device_type = detect_device_type();
  
  // Could also query via ACL API if needed
  switch (device_type) {
    case 0:  // Ascend 910
      return 32;
    case 1:  // Ascend 910B
      return 32;
    case 2:  // Ascend 310P
      return 8;
    default:
      return 32;
  }
#else
  return 32;  // Default
#endif
}

int AscendOptimizer::compute_optimal_block_size(size_t problem_size,
                                                int device_type) {
  // Ascend NPUs work well with moderate block sizes
  // Unlike GPUs, NPUs benefit from smaller blocks for better scheduling
  
  int base_size = 8;  // Good default
  
  // Adjust based on problem size
  if (problem_size < 1024) {
    base_size = 2;
  } else if (problem_size < 4096) {
    base_size = 4;
  } else if (problem_size > 1048576) {
    base_size = 16;
  }
  
  // 310P has fewer cores
  if (device_type == 2) {
    base_size = std::min(base_size, 4);
  }
  
  return base_size;
}

void AscendOptimizer::compute_optimal_tiles(int m, int n, int k, 
                                           int device_type,
                                           AscendKernelConfig &config) {
  // Ascend Cube unit native size is 16x16
  // Optimal tiles are multiples of 16
  
  // L1 buffer size
  size_t l1_size = (device_type == 1) ? 512 * 1024 : 256 * 1024;
  
  // Start with 16x16 (native Cube size)
  int tile_m = 16, tile_n = 16, tile_k = 16;
  
  // Try to increase tile size while fitting in L1
  auto try_tile = [&](int tm, int tn, int tk) -> bool {
    size_t mem = (tm * tk + tk * tn) * 2 + tm * tn * 4;  // float16 + float32
    return mem <= l1_size * 0.75f;  // 75% utilization target
  };
  
  // Try larger tiles
  for (int mult = 2; mult <= 8; mult++) {
    int new_tile = 16 * mult;
    if (try_tile(new_tile, new_tile, new_tile)) {
      if (new_tile <= m && new_tile <= n && new_tile <= k) {
        tile_m = tile_n = tile_k = new_tile;
      }
    } else {
      break;  // Too large, stop
    }
  }
  
  config.tile_m = std::min(m, tile_m);
  config.tile_n = std::min(n, tile_n);
  config.tile_k = std::min(k, tile_k);
}

} // namespace ascend
} // namespace kernel
} // namespace yirage

#endif // YIRAGE_BACKEND_ASCEND_ENABLED

