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
 * Ascend Threadblock Matmul Fingerprinter
 * Adapted for Huawei Ascend NPU with Cube Unit (16x16 native tiles)
 */

#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ASCEND

#include "yirage/utils/ascend_helper.h"
#include "yirage/utils/fingerprint_functions.h"
#include "yirage/utils/static_switch.h"

#ifndef ASCEND_AICORE
#ifdef __ASCEND__
#define ASCEND_AICORE __aicore__
#else
#define ASCEND_AICORE
#endif
#endif

namespace yirage {
namespace threadblock {

using namespace yirage::type;
using namespace yirage::utils;

class TBMatmulFingerprinter {
public:
  ASCEND_AICORE
  TBMatmulFingerprinter(FPType *A_ptr,
                        FPType *B_ptr,
                        FPType *C_ptr,
                        int m,
                        int n,
                        int k,
                        int thread_id,
                        int num_threads) {
    // Simple matmul fingerprint using linear thread mapping
    // Ascend Cube unit would typically use 16x16 tiles
    for (int i = thread_id; i < m * n; i += num_threads) {
      int row = i / n;
      int col = i % n;
      FPType result = 0;
      for (int p = 0; p < k; p++) {
        FPType a_val = A_ptr[row * k + p];
        FPType b_val = B_ptr[p * n + col];
        FPType mul_result = compute_mul_fingerprint(a_val, b_val);
        result = compute_add_fingerprint(result, mul_result);
      }
      C_ptr[i] = result;
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND

