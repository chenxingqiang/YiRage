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
 * Ascend Threadblock Concat Fingerprinter
 * Adapted for Huawei Ascend NPU
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

class TBConcatFingerprinter {
public:
  ASCEND_AICORE
  TBConcatFingerprinter(FPType *input_ptr,
                        FPType *output_ptr,
                        int input_smem_offset,
                        int output_smem_offset,
                        int num_elements,
                        int thread_id,
                        int num_threads) {
    // Simple copy for concat operation
    for (int i = thread_id; i < num_elements; i += num_threads) {
      output_ptr[output_smem_offset + i] = input_ptr[input_smem_offset + i];
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND

