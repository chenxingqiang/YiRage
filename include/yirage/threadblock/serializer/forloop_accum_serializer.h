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

#if defined(YIRAGE_BACKEND_USE_CUDA) || defined(YIRAGE_FINGERPRINT_USE_MACA)

#include "yirage/layout.h"
#include "yirage/type.h"
#ifdef YIRAGE_BACKEND_USE_CUDA
#include <vector_types.h>
#elif defined(YIRAGE_FINGERPRINT_USE_MACA)
#include <common/__clang_maca_vector_types.h>
#endif

namespace yirage {
namespace threadblock {

inline CUTLASS_HOST_DEVICE
void deserialize_forloop_accum_parameters(int const *params,
                                          int &param_idx,
                                          int &accum_num_elements,
                                          int &per_iter_reduction_degree,
                                          int &inner_range,
                                          int &input_smem_offset,
                                          int &accum_smem_offset) {
  accum_num_elements = params[param_idx++];
  per_iter_reduction_degree = params[param_idx++];
  inner_range = params[param_idx++];
  input_smem_offset = params[param_idx++];
  accum_smem_offset = params[param_idx++];
}

inline void serialize_forloop_accum_parameters(int *params,
                                               int &param_idx,
                                               int accum_num_elements,
                                               int per_iter_reduction_degree,
                                               int inner_range,
                                               int input_smem_offset,
                                               int accum_smem_offset) {
  params[param_idx++] = accum_num_elements;
  params[param_idx++] = per_iter_reduction_degree;
  params[param_idx++] = inner_range;
  params[param_idx++] = input_smem_offset;
  params[param_idx++] = accum_smem_offset;
  assert(param_idx <= NewKernelParams::MAX_NUM_PARAMETERS);
}

} // namespace threadblock
} // namespace yirage
#endif
