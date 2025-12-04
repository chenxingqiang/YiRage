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
 * Ascend Threadblock AllReduce Fingerprinter
 * Adapted for Huawei Ascend NPU with HCCL support
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

class TBAllReduceFingerprinter {
public:
  ASCEND_AICORE
  TBAllReduceFingerprinter(FPType *input_ptr,
                           FPType *output_ptr,
                           int smem_offset,
                           int num_elements,
                           int reduction_op,  // 0=SUM, 1=MEAN, 2=MAX, 3=MIN
                           int thread_id,
                           int num_threads) {
    // For fingerprint, AllReduce behaves like identity (single device)
    // In real execution, HCCL would handle multi-device communication
    for (int i = thread_id; i < num_elements; i += num_threads) {
      output_ptr[smem_offset + i] = input_ptr[smem_offset + i];
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND

