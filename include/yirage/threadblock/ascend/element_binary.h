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
 * Ascend Threadblock Element Binary Fingerprinter
 * Adapted for Huawei Ascend NPU with Vector Unit
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

class TBElementBinaryFingerprinter {
public:
  ASCEND_AICORE
  TBElementBinaryFingerprinter(TBOperatorType type,
                               FPType *input1_ptr,
                               FPType *input2_ptr,
                               FPType *output_ptr,
                               int input1_smem_offset,
                               int input2_smem_offset,
                               int output_smem_offset,
                               int num_elements,
                               FPType *div_p_lookup_table,
                               FPType *div_q_lookup_table,
                               int thread_id,
                               int num_threads) {
    // Ascend Vector unit handles element-wise binary operations
    DISPATCH_TBOPERATOR_TYPE(type, [&] {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        output_ptr[output_smem_offset + i] =
            compute_binary_fingerprint<TB_OP_TYPE>(
                input1_ptr[input1_smem_offset + i],
                input2_ptr[input2_smem_offset + i],
                div_p_lookup_table,
                div_q_lookup_table);
      }
    });
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND

