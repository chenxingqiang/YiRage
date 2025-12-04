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
 * Ascend Threadblock RMS Norm Fingerprinter
 * Adapted for Huawei Ascend NPU with Vector Unit
 */

#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ASCEND

#include "yirage/utils/ascend_helper.h"
#include "yirage/utils/fingerprint_functions.h"

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

class TBRmsNormFingerprinter {
public:
  ASCEND_AICORE
  TBRmsNormFingerprinter(FPType *input_ptr,
                         FPType *output_ptr,
                         int smem_offset,
                         int num_elements,
                         int norm_size,
                         FPType *div_p_lookup_table,
                         FPType *div_q_lookup_table,
                         FPType *sqrt_p_lookup_table,
                         FPType *sqrt_q_lookup_table,
                         int thread_id,
                         int num_threads) {
    // RMS normalization
    // 1. Compute sum of squares
    // 2. Compute mean
    // 3. Compute sqrt (RMS)
    // 4. Normalize each element
    
    int num_samples = num_elements / norm_size;
    
    for (int s = thread_id; s < num_samples; s += num_threads) {
      // Compute sum of squares for this sample
      FPType square_sum = 0;
      for (int i = 0; i < norm_size; i++) {
        FPType x = input_ptr[smem_offset + s * norm_size + i];
        FPType x_sq = compute_mul_fingerprint(x, x);
        square_sum = compute_add_fingerprint(square_sum, x_sq);
      }
      
      // Compute mean
      FPType n = static_cast<FPType>(norm_size % config::FP_PQ);
      FPType mean = compute_div_fingerprint(square_sum, n,
                                            div_p_lookup_table, div_q_lookup_table);
      
      // Compute RMS (sqrt of mean)
      FPType rms = compute_sqrt_fingerprint(mean, sqrt_p_lookup_table, sqrt_q_lookup_table);
      
      // Normalize each element
      for (int i = 0; i < norm_size; i++) {
        FPType x = input_ptr[smem_offset + s * norm_size + i];
        output_ptr[smem_offset + s * norm_size + i] = 
            compute_div_fingerprint(x, rms, div_p_lookup_table, div_q_lookup_table);
      }
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND

