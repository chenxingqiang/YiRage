/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_TPU

#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

class TBRMSNormFingerprinter {
public:
  void compute(FPType *input_ptr, FPType *weight_ptr, FPType *output_ptr,
               int batch_size, int hidden_size, FPType eps) {
    for (int b = 0; b < batch_size; b++) {
      FPType sum_sq = 0;
      for (int h = 0; h < hidden_size; h++) {
        FPType val = input_ptr[b * hidden_size + h];
        sum_sq = compute_add_fingerprint(sum_sq, compute_mul_fingerprint(val, val));
      }
      FPType rms = compute_sqrt_fingerprint(sum_sq / hidden_size + eps);
      for (int h = 0; h < hidden_size; h++) {
        int idx = b * hidden_size + h;
        output_ptr[idx] = compute_mul_fingerprint(
            compute_div_fingerprint(input_ptr[idx], rms), weight_ptr[h]);
      }
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif
