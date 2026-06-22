/* Copyright 2025 YiRage Team */
#pragma once
#include "type.h"
#ifdef YIRAGE_FINGERPRINT_USE_CPU
#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"
namespace yirage { namespace threadblock {
class TBForloopAccumFingerprinter {
public:
  void compute(type::FPType *accum_ptr, type::FPType *input_ptr, int num_elements) {
    #pragma omp parallel for if(num_elements > 1024)
    for (int i = 0; i < num_elements; i++)
      accum_ptr[i] = utils::compute_add_fingerprint(accum_ptr[i], input_ptr[i]);
  }
};
}}
#endif
