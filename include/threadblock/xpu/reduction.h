/* Copyright 2025 YiRage Team */
#pragma once
#ifdef YIRAGE_FINGERPRINT_USE_XPU
#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"
namespace yirage { namespace threadblock {
class TBReductionFingerprinter {
public:
  void compute(FPType *input_ptr, FPType *output_ptr, int outer_size, int reduce_size,
               int thread_id = 0, int num_threads = 1) {
    for (int i = thread_id; i < outer_size; i += num_threads) {
      FPType result = 0;
      for (int j = 0; j < reduce_size; j++)
        result = compute_add_fingerprint(result, input_ptr[i * reduce_size + j]);
      output_ptr[i] = result;
    }
  }
};
}}
#endif
