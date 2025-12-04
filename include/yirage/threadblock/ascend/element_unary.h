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
 * Ascend Threadblock Element Unary Fingerprinter
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

class TBElementUnaryFingerprinter {
public:
  ASCEND_AICORE
  TBElementUnaryFingerprinter(TBOperatorType type,
                              FPType *input_ptr,
                              FPType *output_ptr,
                              int smem_offset,
                              int num_elements,
                              FPType *exp_lookup_table,
                              int thread_id,
                              int num_threads) {
    // Ascend Vector unit handles element-wise operations efficiently
    DISPATCH_TBOPERATOR_TYPE(type, [&] {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        output_ptr[smem_offset + i] =
            compute_unary_fingerprint<TB_OP_TYPE>(input_ptr[smem_offset + i],
                                                  exp_lookup_table);
      }
    });
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND

