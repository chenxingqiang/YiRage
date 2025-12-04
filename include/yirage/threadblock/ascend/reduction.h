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
 * Ascend Threadblock Reduction Fingerprinter
 * Adapted for Huawei Ascend NPU with AI Core architecture
 */

#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ASCEND

#include "yirage/utils/ascend_helper.h"
#include "yirage/utils/fingerprint_functions.h"
#include "yirage/utils/static_switch.h"

// Ascend uses __aicore__ for AI Core functions
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

class TBReductionFingerprinter {
public:
  ASCEND_AICORE
  TBReductionFingerprinter(yirage::type::TBOperatorType type,
                           FPType *input_ptr,
                           FPType *output_ptr,
                           int output_num_elements,
                           int reduction_degree,
                           int inner_range,
                           int thread_id,
                           int num_threads) {
    // for a reduction_dim, inner_range is calculated as follows
    // inner_range = 1;
    // for (int i = reduction_dim; i < num_dims; i++)
    //   inner_range *= output.dim[i]
    // Note that the reduction_dim size itself is included in inner_range

    // input position = (i / inner_range) * (inner_range * reduction_degree)
    // + i % inner_range + k * inner_range
    for (int i = thread_id; i < output_num_elements; i += num_threads) {
      int pos = (i / inner_range) * (inner_range * reduction_degree) +
                i % inner_range;
      FPType result = 0;
      for (int k = 0; k < reduction_degree; k++) {
        result = compute_add_fingerprint(result, input_ptr[pos]);
        pos += inner_range;
      }
      output_ptr[i] = result;
    }
  };
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND

