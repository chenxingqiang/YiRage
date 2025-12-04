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
 * Ascend Threadblock Input Loader Fingerprinter
 * Adapted for Huawei Ascend NPU with L1 Buffer
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

// MatrixCoord for Ascend (similar to cutlass::MatrixCoord)
struct MatrixCoord {
  int row;
  int column;
  
  ASCEND_AICORE MatrixCoord() : row(0), column(0) {}
  ASCEND_AICORE MatrixCoord(int r, int c) : row(r), column(c) {}
};

class TBInputLoaderFingerprinter {
public:
  ASCEND_AICORE
  TBInputLoaderFingerprinter(FPType *dmem_ptr,
                             FPType *smem_ptr,
                             int smem_offset,
                             int global_offset,
                             MatrixCoord dtensor_matrix_shape,
                             MatrixCoord stensor_matrix_shape,
                             int dtensor_layout,
                             int thread_id,
                             int num_threads) {
    // Load data from device memory to L1 buffer (shared memory equivalent)
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
      
      smem_ptr[smem_offset + i] = dmem_ptr[global_idx];
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_ASCEND

