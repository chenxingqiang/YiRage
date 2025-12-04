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
 * MACA Threadblock Output Saver Fingerprinter
 * Adapted for MetaX MACA backend with 64-thread warps
 */

#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_MACA

#include "yirage/utils/maca_helper.h"
#include "yirage/type.h"
#include "yirage/layout.h"
#include "yirage/threadblock/maca/input_loader.h" // For MatrixCoord

#ifndef MACA_DEVICE
#define MACA_DEVICE __device__
#endif

namespace yirage {
namespace threadblock {

using namespace yirage::type;

class TBOutputSaverFingerprinter {
public:
  MACA_DEVICE
  TBOutputSaverFingerprinter(yirage::type::FPType *dtensor_ptr,
                             yirage::type::FPType *stensor_ptr,
                             int2 dtensor_matrix_shape,
                             int2 stensor_matrix_shape,
                             yirage::layout::DmemLayout dtensor_layout,
                             yirage::layout::SmemLayout stensor_layout,
                             int thread_id,
                             int num_threads,
                             MatrixCoord matrix_offset,
                             int global_offset) {
    yirage::type::FPType *smem_ptr = stensor_ptr;
    yirage::type::FPType *dmem_ptr = dtensor_ptr + global_offset;
    int num_elements = stensor_matrix_shape.x * stensor_matrix_shape.y;
    int smem_num_column = stensor_matrix_shape.y;
    int dmem_num_column = dtensor_matrix_shape.y;
    for (int idx = thread_id; idx < num_elements; idx += num_threads) {
      int dmem_row_idx = matrix_offset.row() + idx / smem_num_column;
      int dmem_column_idx = matrix_offset.column() + idx % smem_num_column;
      assert(dmem_column_idx < dmem_num_column);
      dmem_ptr[dmem_row_idx * dmem_num_column + dmem_column_idx] =
          smem_ptr[idx];
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_MACA

