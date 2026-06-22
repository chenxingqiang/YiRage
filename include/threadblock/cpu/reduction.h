/* Copyright 2025 YiRage Team */
#pragma once
#include "type.h"
#ifdef YIRAGE_FINGERPRINT_USE_CPU
#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"
namespace yirage { namespace threadblock {
class TBReductionFingerprinter {
public:
  void compute(type::FPType *input_ptr, type::FPType *output_ptr, int outer_size, int reduce_size) {
    #pragma omp parallel for if(outer_size > 64)
    for (int i = 0; i < outer_size; i++) {
      type::FPType result = 0;
      for (int j = 0; j < reduce_size; j++)
        result = utils::compute_add_fingerprint(result, input_ptr[i * reduce_size + j]);
      output_ptr[i] = result;
    }
  }
};
}}
#endif
