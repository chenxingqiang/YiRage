/* Copyright 2025 YiRage Project
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
 * MACA Threadblock Matmul Fingerprinter
 * Adapted for MetaX MACA backend with 64-thread warps
 */

#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_MACA

#include "yirage/threadblock/smem_tensor.h"
#include "yirage/utils/maca_helper.h"
#include "yirage/utils/fingerprint_functions.h"
#include "yirage/utils/static_switch.h"

#ifndef MACA_DEVICE
#define MACA_DEVICE __device__
#endif

namespace yirage {
namespace threadblock {

class TBMatmulFingerprinter {
public:
  MACA_DEVICE
  TBMatmulFingerprinter(FPType *A_ptr,
                        FPType *B_ptr,
                        FPType *C_ptr,
                        int a_m_size,
                        int c_n_size,
                        int a_k_size,
                        int thread_id,
                        int num_threads) {
    // Note that we assume all tensors are in row-major layouts
    // when computing fingerprints
    int num_elements = a_m_size * c_n_size;
    int b_n_size = c_n_size;
    for (int i = thread_id; i < num_elements; i += num_threads) {
      FPType result = 0;
      int m = i / c_n_size;
      int n = i % c_n_size;
      for (int k = 0; k < a_k_size; k++) {
        FPType a = A_ptr[m * a_k_size + k];
        FPType b = B_ptr[k * b_n_size + n];
        FPType ab = compute_mul_fingerprint(a, b);
        result = compute_add_fingerprint(result, ab);
      }
      C_ptr[i] = result;
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_MACA

