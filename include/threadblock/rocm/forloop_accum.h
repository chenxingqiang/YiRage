/* Copyright 2025 YiRage Team */
#pragma once

#ifdef YIRAGE_FINGERPRINT_USE_ROCM

#include "threadblock/smem_tensor.h"
#include "utils/fingerprint_functions.h"

namespace yirage {
namespace threadblock {

class TBForloopAccumFingerprinter {
public:
  __device__
  TBForloopAccumFingerprinter(FPType *accum_ptr, FPType *input_ptr,
                               int num_elements, int thread_id, int num_threads) {
    for (int i = thread_id; i < num_elements; i += num_threads) {
      accum_ptr[i] = compute_add_fingerprint(accum_ptr[i], input_ptr[i]);
    }
  }
};

}  // namespace threadblock
}  // namespace yirage

#endif  // YIRAGE_FINGERPRINT_USE_ROCM
