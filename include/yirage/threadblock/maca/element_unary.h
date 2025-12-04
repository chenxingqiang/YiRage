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
 * MACA Threadblock Element Unary Fingerprinter
 * Adapted for MetaX MACA backend with 64-thread warps
 */

#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_MACA

#include "yirage/utils/maca_helper.h"
#include "yirage/utils/fingerprint_functions.h"
#include <cmath>

#ifndef MACA_DEVICE
#define MACA_DEVICE __device__
#endif

namespace yirage {
// MACA constant memory for clamp bounds
__constant__ float CLAMP_MIN_MAX_DEVICE[2];

namespace threadblock {

using namespace yirage::type;
using namespace yirage::utils;

class TBElementUnaryFingerPrinter {
public:
  MACA_DEVICE
  TBElementUnaryFingerPrinter(yirage::type::TBOperatorType type,
                              FPType *exp_lookup_table,
                              FPType *sqrt_p_lookup_table,
                              FPType *sqrt_q_lookup_table,
                              FPType *base_ptr,
                              int num_elements,
                              int thread_id,
                              int num_threads) {
    if (type == yirage::type::TB_EXP_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_exp_fingerprint(base_ptr[i], exp_lookup_table);
      }
    } else if (type == yirage::type::TB_SQUARE_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_square_fingerprint(base_ptr[i]);
      }
    } else if (type == yirage::type::TB_SQRT_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_sqrt_fingerprint(
            base_ptr[i], sqrt_p_lookup_table, sqrt_q_lookup_table);
      }
    } else if (type == yirage::type::TB_SILU_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_silu_fingerprint(base_ptr[i], exp_lookup_table);
      }
    } else if (type == yirage::type::TB_RELU_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_relu_fingerprint(base_ptr[i]);
      }
    } else if (type == yirage::type::TB_CLAMP_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_clamp_fingerprint(base_ptr[i]);
      }
    } else if (type == yirage::type::TB_GELU_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[i] = compute_gelu_fingerprint(base_ptr[i], exp_lookup_table);
      }
    } else {
      assert(false && "Unimplemented");
    }
  }
};

} // namespace threadblock
} // namespace yirage

#endif // YIRAGE_FINGERPRINT_USE_MACA

