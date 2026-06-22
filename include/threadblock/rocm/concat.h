/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ROCM

#include "threadblock/smem_tensor.h"

namespace yirage {
namespace threadblock {

class TBConcatFingerprinter {
public:
  __device__
  TBConcatFingerprinter(FPType **input_ptrs, int *input_sizes, int num_inputs,
                        FPType *output_ptr, int thread_id, int num_threads) {
    int offset = 0;
    for (int n = 0; n < num_inputs; n++) {
      for (int i = thread_id; i < input_sizes[n]; i += num_threads) {
        output_ptr[offset + i] = input_ptrs[n][i];
      }
      offset += input_sizes[n];
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_FINGERPRINT_USE_ROCM
