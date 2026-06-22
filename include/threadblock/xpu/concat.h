/* Copyright 2025 YiRage Team */
#pragma once
#ifdef YIRAGE_FINGERPRINT_USE_XPU
#include "threadblock/smem_tensor.h"
namespace yirage { namespace threadblock {
class TBConcatFingerprinter {
public:
  void compute(FPType **input_ptrs, int *input_sizes, int num_inputs, FPType *output_ptr,
               int thread_id = 0, int num_threads = 1) {
    int offset = 0;
    for (int n = 0; n < num_inputs; n++) {
      for (int i = thread_id; i < input_sizes[n]; i += num_threads)
        output_ptr[offset + i] = input_ptrs[n][i];
      offset += input_sizes[n];
    }
  }
};
}}
#endif
