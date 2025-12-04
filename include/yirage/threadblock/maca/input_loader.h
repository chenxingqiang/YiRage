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
 * MACA Threadblock Input Loader Fingerprinter
 * Adapted for MetaX MACA backend with 64-thread warps
 */

#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_MACA

#include "yirage/utils/maca_helper.h"
#include "yirage/type.h"
#include "yirage/layout.h"

#ifndef MACA_DEVICE
#define MACA_DEVICE __device__
#endif

namespace yirage {
namespace threadblock {

// Simple MatrixCoord replacement for MACA (cutlass::MatrixCoord equivalent)
struct MatrixCoord {
  int row_idx;
  int col_idx;
  
  MACA_DEVICE MatrixCoord() : row_idx(0), col_idx(0) {}
  MACA_DEVICE MatrixCoord(int r, int c) : row_idx(r), col_idx(c) {}
  MACA_DEVICE int row() const { return row_idx; }
  MACA_DEVICE int column() const { return col_idx; }
};

class TBInputLoaderFingerprinter {
public:
  MACA_DEVICE
  TBInputLoaderFingerprinter(yirage::type::FPType *dtensor_ptr,
                             yirage::type::FPType *stensor_ptr,
                             int2 dtensor_matrix_shape,
                             int2 stensor_matrix_shape,
                             yirage::layout::DmemLayout dlayout,
                             yirage::layout::SmemLayout slayout,
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
      smem_ptr[idx] =
          dmem_ptr[dmem_row_idx * dmem_num_column + dmem_column_idx];
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_MACA

