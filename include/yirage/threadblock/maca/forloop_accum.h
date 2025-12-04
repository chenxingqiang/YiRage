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
 * MACA Threadblock Forloop Accum Fingerprinter
 * Adapted for MetaX MACA backend with 64-thread warps
 */

#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_MACA

#include "yirage/utils/maca_helper.h"
#include "yirage/utils/fingerprint_functions.h"

#ifndef MACA_DEVICE
#define MACA_DEVICE __device__
#endif

namespace yirage {
namespace threadblock {

using namespace yirage::type;
using namespace yirage::config;
using namespace yirage::utils;

class TBForloopAccumFingerprinter {
public:
  MACA_DEVICE
  TBForloopAccumFingerprinter(TBOperatorType type,
                              FPType *input_ptr,
                              FPType *output_ptr,
                              FPType *div_p_lookup_table,
                              FPType *div_q_lookup_table,
                              FPType *sqrt_p_lookup_table,
                              FPType *sqrt_q_lookup_table,
                              int output_num_elements,
                              int per_iter_reduction_degree,
                              int inner_range,
                              int num_forloop_iters,
                              bool reset_output,
                              bool post_process,
                              int thread_id,
                              int num_threads) {
    // For non-reduction accumulation: inner_range = 1, reduction_degree = 1
    if (type == yirage::type::TB_FORLOOP_ACCUM_NO_RED_OP) {
      for (int idx = thread_id; idx < output_num_elements; idx += num_threads) {
        FPType old_output = reset_output ? 0 : output_ptr[idx];
        output_ptr[idx] = compute_add_fingerprint(old_output, input_ptr[idx]);
      }
    } else if (type == TB_FORLOOP_ACCUM_RED_LD_SUM_OP ||
               type == TB_FORLOOP_ACCUM_RED_LD_MEAN_OP ||
               type == TB_FORLOOP_ACCUM_RED_LD_RMS_OP ||
               type == TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP) {
      bool compute_square = false;
      if (type == TB_FORLOOP_ACCUM_RED_LD_RMS_OP) {
        compute_square = true;
      }
      for (int idx = thread_id; idx < output_num_elements; idx += num_threads) {
        int pos =
            (idx / inner_range) * (inner_range * per_iter_reduction_degree) +
            idx % inner_range;
        FPType result = 0;
        for (int k = 0; k < per_iter_reduction_degree; k++) {
          FPType x = input_ptr[pos];
          if (compute_square) {
            x = compute_mul_fingerprint(x, x);
          }
          result = compute_add_fingerprint(result, x);
          pos += inner_range;
        }
        FPType old_output = reset_output ? 0 : output_ptr[idx];
        output_ptr[idx] = compute_add_fingerprint(old_output, result);
        if (post_process && type == TB_FORLOOP_ACCUM_RED_LD_MEAN_OP) {
          FPType x = output_ptr[idx];
          FPType n = (num_forloop_iters * per_iter_reduction_degree) % FP_PQ;
          FPType z = compute_div_fingerprint(
              x, n, div_p_lookup_table, div_q_lookup_table);
          output_ptr[idx] = z;
        }
        if (post_process && type == TB_FORLOOP_ACCUM_RED_LD_RMS_OP) {
          FPType x = output_ptr[idx];
          FPType n = (num_forloop_iters * per_iter_reduction_degree) % FP_PQ;
          FPType z = compute_div_fingerprint(
              x, n, div_p_lookup_table, div_q_lookup_table);
          z = compute_sqrt_fingerprint(
              z, sqrt_p_lookup_table, sqrt_q_lookup_table);
          output_ptr[idx] = z;
        }
      }
    } else {
      assert(false && "Unsupported accum type");
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_MACA

