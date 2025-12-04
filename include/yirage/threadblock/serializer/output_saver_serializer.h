/* Copyright 2023-2024 CMU
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
 */

#pragma once

#if defined(YIRAGE_BACKEND_USE_CUDA) || defined(YIRAGE_FINGERPRINT_USE_MACA) || defined(YIRAGE_FINGERPRINT_USE_ASCEND)

#include "yirage/layout.h"
#include "yirage/type.h"
#ifdef YIRAGE_BACKEND_USE_CUDA
#include <vector_types.h>
#elif defined(YIRAGE_FINGERPRINT_USE_MACA)
#include <common/__clang_maca_vector_types.h>
#endif

namespace yirage {
namespace threadblock {

CUTLASS_HOST_DEVICE
void deserialize_output_saver_parameters(
    int const *params,
    int &param_idx,
    int3 &output_matrix_row_offset_block_stride,
    int3 &output_matrix_column_offset_block_stride,
    int &output_matrix_row_offset_forloop_stride,
    int &output_matrix_column_offset_forloop_stride,
    int3 &global_offset_block_stride,
    int &global_offset_forloop_stride,
    int2 &dtensor_matrix_shape,
    int2 &stensor_matrix_shape,
    yirage::layout::DmemLayout &dlayout,
    yirage::layout::SmemLayout &slayout,
    int &output_smem_offset,
    yirage::type::TBEpilogueType &epilogue) {
  output_matrix_row_offset_block_stride.x = params[param_idx++];
  output_matrix_row_offset_block_stride.y = params[param_idx++];
  output_matrix_row_offset_block_stride.z = params[param_idx++];

  output_matrix_column_offset_block_stride.x = params[param_idx++];
  output_matrix_column_offset_block_stride.y = params[param_idx++];
  output_matrix_column_offset_block_stride.z = params[param_idx++];

  output_matrix_row_offset_forloop_stride = params[param_idx++];
  output_matrix_column_offset_forloop_stride = params[param_idx++];

  global_offset_block_stride.x = params[param_idx++];
  global_offset_block_stride.y = params[param_idx++];
  global_offset_block_stride.z = params[param_idx++];
  global_offset_forloop_stride = params[param_idx++];

  dtensor_matrix_shape.x = params[param_idx++];
  dtensor_matrix_shape.y = params[param_idx++];

  stensor_matrix_shape.x = params[param_idx++];
  stensor_matrix_shape.y = params[param_idx++];

  dlayout = (yirage::layout::DmemLayout)params[param_idx++];
  slayout = (yirage::layout::SmemLayout)params[param_idx++];
  output_smem_offset = params[param_idx++];
  epilogue = static_cast<yirage::type::TBEpilogueType>(params[param_idx++]);
}

inline void serialize_output_saver_parameters(
    int *params,
    int &param_idx,
    int3 output_matrix_row_offset_block_stride,
    int3 output_matrix_column_offset_block_stride,
    int output_matrix_row_offset_forloop_stride,
    int output_matrix_column_offset_forloop_stride,
    int3 global_offset_block_stride,
    int global_offset_forloop_stride,
    int2 dtensor_matrix_shape,
    int2 stensor_matrix_shape,
    yirage::layout::DmemLayout dlayout,
    yirage::layout::SmemLayout slayout,
    int output_smem_offset,
    yirage::type::TBEpilogueType epilogue) {
  params[param_idx++] = output_matrix_row_offset_block_stride.x;
  params[param_idx++] = output_matrix_row_offset_block_stride.y;
  params[param_idx++] = output_matrix_row_offset_block_stride.z;

  params[param_idx++] = output_matrix_column_offset_block_stride.x;
  params[param_idx++] = output_matrix_column_offset_block_stride.y;
  params[param_idx++] = output_matrix_column_offset_block_stride.z;

  params[param_idx++] = output_matrix_row_offset_forloop_stride;
  params[param_idx++] = output_matrix_column_offset_forloop_stride;

  params[param_idx++] = global_offset_block_stride.x;
  params[param_idx++] = global_offset_block_stride.y;
  params[param_idx++] = global_offset_block_stride.z;
  params[param_idx++] = global_offset_forloop_stride;

  params[param_idx++] = dtensor_matrix_shape.x;
  params[param_idx++] = dtensor_matrix_shape.y;

  params[param_idx++] = stensor_matrix_shape.x;
  params[param_idx++] = stensor_matrix_shape.y;

  params[param_idx++] = dlayout;
  params[param_idx++] = slayout;
  params[param_idx++] = output_smem_offset;
  params[param_idx++] = epilogue;
  assert(param_idx <= NewKernelParams::MAX_NUM_PARAMETERS);
}

} // namespace threadblock
} // namespace yirage
#endif
