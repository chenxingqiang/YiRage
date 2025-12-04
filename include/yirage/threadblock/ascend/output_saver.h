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
 * Ascend Threadblock Output Saver Fingerprinter
 * Adapted for Huawei Ascend NPU with L1 Buffer
 */

#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ASCEND

#include "yirage/utils/ascend_helper.h"
#include "yirage/utils/fingerprint_functions.h"
#include "yirage/threadblock/ascend/input_loader.h"  // For MatrixCoord

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

class TBOutputSaverFingerprinter {
public:
  ASCEND_AICORE
  TBOutputSaverFingerprinter(TBOperatorType type,
                             FPType *smem_ptr,
                             FPType *dmem_ptr,
                             int smem_offset,
                             int global_offset,
                             MatrixCoord stensor_matrix_shape,
                             MatrixCoord dtensor_matrix_shape,
                             int dtensor_layout,
                             int forloop_dim,
                             int forloop_range,
                             FPType *div_p_lookup_table,
                             FPType *div_q_lookup_table,
                             int thread_id,
                             int num_threads) {
    // Save data from L1 buffer to device memory
    int num_elements = stensor_matrix_shape.row * stensor_matrix_shape.column;
    
    for (int i = thread_id; i < num_elements; i += num_threads) {
      int row = i / stensor_matrix_shape.column;
      int col = i % stensor_matrix_shape.column;
      
      int global_idx;
      if (dtensor_layout == 0) {  // Row-major
        global_idx = global_offset + row * dtensor_matrix_shape.column + col;
      } else {  // Column-major
        global_idx = global_offset + col * dtensor_matrix_shape.row + row;
      }
      
      FPType value = smem_ptr[smem_offset + i];
      
      // Apply reduction if needed
      if (type == TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP) {
        // Sum reduction across forloop iterations
        value = compute_add_fingerprint(dmem_ptr[global_idx], value);
      }
      
      dmem_ptr[global_idx] = value;
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND

