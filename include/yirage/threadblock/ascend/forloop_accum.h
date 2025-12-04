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
 * Ascend Threadblock Forloop Accumulator Fingerprinter
 * Adapted for Huawei Ascend NPU
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

class TBForloopAccumFingerprinter {
public:
  ASCEND_AICORE
  TBForloopAccumFingerprinter(TBOperatorType type,
                              FPType *input_ptr,
                              FPType *output_ptr,
                              int input_smem_offset,
                              int output_smem_offset,
                              int num_elements,
                              FPType *div_p_lookup_table,
                              FPType *div_q_lookup_table,
                              int forloop_dim,
                              int forloop_range,
                              int thread_id,
                              int num_threads) {
    // Accumulate values across forloop iterations
    DISPATCH_TBOPERATOR_TYPE(type, [&] {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        FPType input_val = input_ptr[input_smem_offset + i];
        FPType output_val = output_ptr[output_smem_offset + i];
        
        // Apply accumulation based on operator type
        if constexpr (TB_OP_TYPE == TB_FORLOOP_ACCUM_NO_RED_OP) {
          // No reduction, just copy
          output_ptr[output_smem_offset + i] = input_val;
        } else if constexpr (TB_OP_TYPE == TB_FORLOOP_ACCUM_RED_LD_SUM_OP ||
                            TB_OP_TYPE == TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP) {
          // Sum reduction
          output_ptr[output_smem_offset + i] = 
              compute_add_fingerprint(output_val, input_val);
        } else if constexpr (TB_OP_TYPE == TB_FORLOOP_ACCUM_RED_LD_MEAN_OP) {
          // Mean reduction (sum then divide by range)
          FPType sum = compute_add_fingerprint(output_val, input_val);
          FPType divisor = static_cast<FPType>(forloop_range);
          output_ptr[output_smem_offset + i] = 
              compute_div_fingerprint(sum, divisor, 
                                      div_p_lookup_table, div_q_lookup_table);
        } else if constexpr (TB_OP_TYPE == TB_FORLOOP_ACCUM_RED_LD_RMS_OP) {
          // RMS reduction (square, sum, sqrt)
          FPType squared = compute_mul_fingerprint(input_val, input_val);
          output_ptr[output_smem_offset + i] = 
              compute_add_fingerprint(output_val, squared);
        }
      }
    });
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND

