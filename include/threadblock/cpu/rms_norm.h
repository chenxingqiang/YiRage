/* Copyright 2025 YiRage Team */
#pragma once
#include "type.h"
#ifdef YIRAGE_FINGERPRINT_USE_CPU
#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"
namespace yirage { namespace threadblock {
class TBRMSNormFingerprinter {
public:
  void compute(type::FPType *input_ptr, type::FPType *weight_ptr, type::FPType *output_ptr,
               int batch_size, int hidden_size, type::FPType eps) {
    #pragma omp parallel for if(batch_size > 16)
    for (int b = 0; b < batch_size; b++) {
      type::FPType sum_sq = 0;
      for (int h = 0; h < hidden_size; h++) {
        type::FPType val = input_ptr[b * hidden_size + h];
        sum_sq = utils::compute_add_fingerprint(sum_sq, utils::compute_mul_fingerprint(val, val));
      }
      type::FPType rms = utils::compute_sqrt_fingerprint(sum_sq / hidden_size + eps);
      for (int h = 0; h < hidden_size; h++) {
        int idx = b * hidden_size + h;
        output_ptr[idx] = utils::compute_mul_fingerprint(compute_div_fingerprint(input_ptr[idx], rms), weight_ptr[h]);
      }
    }
  }
};
}}
#endif
