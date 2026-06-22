/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ROCM

#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

class TBReductionFingerprinter {
public:
  __device__
  TBReductionFingerprinter(FPType *input_ptr, FPType *output_ptr,
                           int outer_size, int reduce_size,
                           int thread_id, int num_threads) {
    for (int i = thread_id; i < outer_size; i += num_threads) {
      FPType result = 0;
      for (int j = 0; j < reduce_size; j++) {
        result = compute_add_fingerprint(result, input_ptr[i * reduce_size + j]);
      }
      output_ptr[i] = result;
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_FINGERPRINT_USE_ROCM
